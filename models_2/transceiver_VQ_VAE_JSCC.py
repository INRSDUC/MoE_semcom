
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Iterable, Callable, Dict, Any, Optional, Union
# from utils_oke import PowerNormalize, Channels
import math
from transformers import AutoModel
from range_coder import RangeEncoder, RangeDecoder
import numpy as np
import os
import torch.distributed as dist

def _gray(x: torch.Tensor) -> torch.Tensor:
    return x ^ (x >> 1)

class VQExpertEMA(nn.Module):
    def __init__(self, K, code_dim, beta=0.25, decay=0.99, eps=1e-5, ddp_sync=True):
        super().__init__()
        self.K = int(K)
        self.code_dim = int(code_dim)
        self.beta = float(beta)
        self.decay = float(decay)
        self.eps = float(eps)
        self.ddp_sync = bool(ddp_sync)

        self.codebook = nn.Embedding(self.K, self.code_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)
        self.codebook.weight.requires_grad_(False)

        self.register_buffer("cluster_size", torch.zeros(self.K))
        self.register_buffer("embed_avg", self.codebook.weight.detach().clone())

        # ---- labeling tables (NEW) ----
        self.k_bits = int(math.ceil(math.log2(self.K)))
        self.n_labels = 1 << self.k_bits

        # idx -> label (only size K)
        self.register_buffer("idx_to_label", torch.arange(self.K, dtype=torch.long))

        # label -> idx (size 2^k_bits, so flips are always valid)
        l2i = torch.zeros(self.n_labels, dtype=torch.long)
        # identity for first K labels by default
        l2i[:self.K] = torch.arange(self.K, dtype=torch.long)
        self.register_buffer("label_to_idx", l2i)
        
        # Precompute label bits to avoid calling int_to_bits() repeatedly in forward
        # Will be filled after rebuild_labeling()
        self.register_buffer("label_bits", torch.zeros(self.K, self.k_bits, dtype=torch.float32))
    @torch.no_grad()
    def rebuild_labeling(self, method: str = "pca_gray", sync_ddp: bool = True):
        """
        Builds idx_to_label and label_to_idx so that small Hamming changes correspond
        (roughly) to small changes in codebook space.

        DDP-safe: rank0 computes, then broadcasts tables to all ranks.
        """
        device = self.codebook.weight.device
        K = self.K
        n_labels = self.n_labels

        # who computes?
        use_ddp = sync_ddp and dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if use_ddp else 0

        if rank == 0:
            w = self.codebook.weight.detach().float()  # [K,C]
            w = w - w.mean(dim=0, keepdim=True)

            if method == "pca_gray":
                # 1D ordering by top singular direction
                _, _, Vh = torch.linalg.svd(w, full_matrices=False)
                score = w @ Vh[0]  # [K]
                order = torch.argsort(score)
            else:
                # fallback: random fixed projection ordering
                g = torch.randn(w.size(1), device=device)
                score = w @ g
                order = torch.argsort(score)

            # rank in [0..K-1] for each idx
            ranks = torch.empty(K, device=device, dtype=torch.long)
            ranks[order] = torch.arange(K, device=device, dtype=torch.long)

            labels = _gray(ranks)  # [K] in [0..2^k-1]

            idx_to_label = labels.long()

            label_to_idx = torch.zeros(n_labels, device=device, dtype=torch.long)
            label_to_idx[idx_to_label] = torch.arange(K, device=device, dtype=torch.long)

        else:
            idx_to_label = torch.empty(K, device=device, dtype=torch.long)
            label_to_idx = torch.empty(n_labels, device=device, dtype=torch.long)

        if use_ddp:
            dist.broadcast(idx_to_label, src=0)
            dist.broadcast(label_to_idx, src=0)

        self.idx_to_label.copy_(idx_to_label)
        self.label_to_idx.copy_(label_to_idx)
        
        # Precompute label_bits for all K labels (avoid int_to_bits in forward)
        # Import inline to avoid circular dependency
        label_bits_computed = int_to_bits(idx_to_label, self.k_bits).float()  # [K, k_bits]
        self.label_bits.copy_(label_bits_computed)
    @torch.no_grad()
    def _ddp_all_reduce_(self, x: torch.Tensor):
        if self.ddp_sync and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
        return x
    def forward(self, z_e: torch.Tensor):
        B, N, C = z_e.shape
        assert C == self.code_dim

        z = z_e.reshape(-1, C)
        T = z.shape[0]
        z_fp32 = z.float()

        # Default outputs (works for T=0 too)
        z_q_st = z_e  # empty stays empty
        indices = torch.empty((B, N), device=z_e.device, dtype=torch.long)
        vq_loss = z_e.new_tensor(0.0)

        if T > 0:
            codebook_w = self.codebook.weight.float()  # [K,C]

            z2 = (z_fp32 ** 2).sum(dim=1, keepdim=True)      # [T,1]
            e2 = (codebook_w ** 2).sum(dim=1).unsqueeze(0)   # [1,K]
            dist = z2 - 2.0 * (z_fp32 @ codebook_w.t()) + e2 # [T,K]

            idx = torch.argmin(dist, dim=1)                  # [T]
            z_q_fp32 = codebook_w.index_select(0, idx)       # [T,C]
            z_q = z_q_fp32.to(dtype=z.dtype)

            vq_loss = self.beta * F.mse_loss(z_fp32, z_q_fp32.detach())

            z_st = z + (z_q - z).detach()
            z_q_st = z_st.view(B, N, C)
            indices = idx.view(B, N)

        # EMA update (must be DDP-safe)
        if self.training:
            with torch.no_grad():
                # Always create counts/embed_sum, even if T==0
                counts = torch.zeros(self.K, device=z_e.device, dtype=torch.float32)
                embed_sum = torch.zeros(self.K, C, device=z_e.device, dtype=torch.float32)

                if T > 0:
                    counts_local = torch.bincount(indices.reshape(-1), minlength=self.K).to(torch.float32)
                    embed_sum.index_add_(0, indices.reshape(-1), z_fp32)
                    counts.copy_(counts_local)

                # Sync across ranks (everyone calls this)
                self._ddp_all_reduce_(counts)
                self._ddp_all_reduce_(embed_sum)

                # Only update if ANY tokens globally hit this expert this step
                if counts.sum().item() > 0:
                    self.cluster_size.mul_(self.decay).add_(counts * (1.0 - self.decay))
                    self.embed_avg.mul_(self.decay).add_(embed_sum * (1.0 - self.decay))

                    # Safe normalization
                    n = self.cluster_size.sum().clamp_min(1.0)
                    cluster_size = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
                    new_codebook = self.embed_avg / cluster_size.unsqueeze(1)

                    self.codebook.weight.data.copy_(new_codebook.to(self.codebook.weight.dtype))

        return z_q_st, indices, vq_loss

    def soft_assign(self, z_e: torch.Tensor, temperature: float = 1.0):
        """
        Soft (differentiable) codebook assignment.
        
        Args:
            z_e: [B, N, C] or [T, C]
            temperature: softmax temperature for sharpness
        
        Returns:
            q_k: [B, N, K] or [T, K] soft assignment weights q(k), sum to 1 over K dim
            z_soft: [B, N, C] or [T, C] soft reconstruction as weighted sum of codebook
        """
        original_shape = z_e.shape
        if z_e.ndim == 3:
            B, N, C = z_e.shape
            z_flat = z_e.reshape(-1, C)
        else:
            z_flat = z_e
        
        T, C = z_flat.shape
        codebook_w = self.codebook.weight.float()  # [K, C]
        K = codebook_w.shape[0]
        
        # Compute distances: [T, K]
        z2 = (z_flat ** 2).sum(dim=1, keepdim=True)
        e2 = (codebook_w ** 2).sum(dim=1).unsqueeze(0)
        dist = z2 - 2.0 * (z_flat @ codebook_w.t()) + e2  # [T, K]
        
        # Soft assignment: q(k) = softmax(-dist / T)
        q_k = torch.softmax(-dist / max(temperature, 1e-6), dim=1)  # [T, K]
        
        # Soft reconstruction: z_soft = sum_k q(k) * e_k
        z_soft = q_k @ codebook_w  # [T, C]
        
        # Reshape back to original shape
        if len(original_shape) == 3:
            B, N, C = original_shape
            z_soft = z_soft.reshape(B, N, C)
            q_k = q_k.reshape(B, N, K)
        
        return q_k, z_soft

class VQMoE(nn.Module):
    """
    Mixture of EMA-VQ experts with per-example expert selection.

    Args:
        code_dims: list[int], K_r per expert.
        code_dim: latent dimension C.
        beta: commitment loss weight.
        decay: EMA decay.
    """
    def __init__(self, code_dims, code_dim: int, beta: float = 0.25, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.num_experts = len(code_dims)
        self.code_dims = list(code_dims)
        self.code_dim = code_dim

        self.experts = nn.ModuleList(
            [VQExpertEMA(k, code_dim, beta=beta, decay=decay, eps=eps, ddp_sync=True) for k in self.code_dims]
        )

        self.bits_per_expert = [int(math.ceil(math.log2(k))) for k in self.code_dims]
 
    def forward(self, z_e: torch.Tensor, expert_idx: torch.Tensor):
        """
        Args:
            z_e:        [B, N, C]
            expert_idx: [B] in [0, num_experts-1]

        Returns:
            z_q:      [B, N, C]
            indices:  [B, N]   (indices are per-expert, range [0, K_r-1])
            vq_loss:  scalar (weighted avg across used experts)
        """
        B, N, C = z_e.shape
        assert expert_idx.shape[0] == B
        assert C == self.code_dim

        z_q = torch.zeros_like(z_e)
        indices = torch.zeros(B, N, dtype=torch.long, device=z_e.device)

        vq_losses = []
        weights = []

        total_tokens = float(B * N)

        for r, expert in enumerate(self.experts):
            mask = (expert_idx == r)
            z_in = z_e[mask]  # may be [0, N, C] on this rank

            z_q_r, idx_r, vq_loss_r = expert(z_in)  # ALWAYS called on every rank because ema ddp fuckery

            if mask.any():
                z_q[mask] = z_q_r
                indices[mask] = idx_r

            # weighting can stay the same; optional improvement below
            Br = z_in.shape[0]
            frac = (Br * N) / total_tokens
            vq_losses.append(vq_loss_r)
            weights.append(frac)


        if vq_losses:
            vq_loss_total = sum(w * l for w, l in zip(weights, vq_losses))
        else:
            vq_loss_total = z_e.new_tensor(0.0)

        return z_q, indices, vq_loss_total
class ModeRouter(nn.Module):
    """
    Factorized router:
      phi -> shared MLP -> (logits_expert [B,R], logits_phy [B,M])

    We still provide joint logits/probs [B, J=R*M] for compatibility,
    but we ALSO expose marginals so you can do Switch-style balancing properly.
    """
    def __init__(
        self,
        in_dim: int,
        num_experts: int,
        num_phy_modes: int,
        hidden_dims=(128, 128),
        temperature: float = 1.0,
        dropout: float = 0.0,
        noisy_gate_std: float = 0.0,   # e.g. 1.0 during training helps exploration
        eps: float = 1e-9,
    ):
        super().__init__()
        self.temperature = temperature
        self.R = num_experts
        self.M = num_phy_modes
        self.J = self.R * self.M
        self.noisy_gate_std = noisy_gate_std
        self.eps = eps

        self.in_norm = nn.LayerNorm(in_dim)

        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        self.trunk = nn.Sequential(*layers)

        self.expert_head = nn.Linear(last, self.R)
        self.phy_head = nn.Linear(last, self.M)

    def forward(self, phi: torch.Tensor, temperature: float | None = None, return_aux: bool = False):
        """
        Returns (by default):
            joint_logits: [B, J]
            joint_probs:  [B, J]

        If return_aux=True, also returns:
            aux dict with logits/probs for expert+phy marginals.
        """
        phi = self.in_norm(phi)
        h = self.trunk(phi)

        logits_e = self.expert_head(h)  # [B,R]
        logits_m = self.phy_head(h)     # [B,M]

        # Noisy gating (Switch-style trick) to avoid sticky argmax collapse
        if self.training and self.noisy_gate_std > 0:
            logits_e = logits_e + torch.randn_like(logits_e) * self.noisy_gate_std
            logits_m = logits_m + torch.randn_like(logits_m) * self.noisy_gate_std

        tau = temperature if temperature is not None else self.temperature

        p_e = F.softmax(logits_e / tau, dim=-1)  # [B,R]
        p_m = F.softmax(logits_m / tau, dim=-1)  # [B,M]

        # Joint logits for compatibility (gumbel over joint is still possible)
        joint_logits = (logits_e.unsqueeze(2) + logits_m.unsqueeze(1)).reshape(-1, self.J)  # [B,J]

        # Joint probs as product of marginals (stable + easy to marginalize)
        joint_probs = (p_e.unsqueeze(2) * p_m.unsqueeze(1)).reshape(-1, self.J)             # [B,J]

        # inside router / stats
        p_e_lb = F.softmax(logits_e, dim=-1)  # no /tau
        importance = p_e_lb.mean(dim=0)       # or sum then normalize


        if return_aux:
            aux = {
                "logits_e": logits_e, "logits_m": logits_m,
                "p_e": p_e, "p_m": p_m,
            }
            return joint_logits, joint_probs, aux

        return joint_logits, joint_probs

    @torch.no_grad()
    def select_mode(self, phi: torch.Tensor):
        # deterministic joint selection
        joint_logits, _ = self.forward(phi, return_aux=False)
        return torch.argmax(joint_logits, dim=-1)  # [B]

class JointModeMapper:
    """
    Helper to map global mode index j in [0, J-1] to:
      - VQ expert index r in [0, R-1]
      - PHY mode index m in [0, M-1]
    assuming a simple cartesian grid: J = R * M.
    """
    def __init__(self, num_experts: int, num_phy_modes: int):
        self.R = num_experts
        self.M = num_phy_modes
        self.J = num_experts * num_phy_modes

    def split(self, mode_idx: torch.Tensor):
        """
        Args:
            mode_idx: [B] global mode indices j

        Returns:
            expert_idx: [B]
            phy_idx:    [B]
        """
        expert_idx = mode_idx // self.M
        phy_idx = mode_idx % self.M
        return expert_idx, phy_idx



# ---------- bit helpers ----------

def int_to_bits(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Convert integer tensor to binary bits (MSB-first).

    Args:
        x: integer tensor of any shape.
        num_bits: number of bits per entry.

    Returns:
        bits: same shape as x with extra dim [num_bits], dtype=float32 in {0,1}.
    """
    x = x.long()
    device = x.device
    shape = x.shape
    shifts = torch.arange(num_bits - 1, -1, -1, device=device)
    bits = (x.unsqueeze(-1) >> shifts) & 1  # [..., num_bits]
    return bits.view(*shape, num_bits).float()


def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    """
    Convert bits (MSB-first) to integer tensor.

    Args:
        bits: [..., num_bits] float or int in {0,1}.

    Returns:
        ints: [...] long tensor.
    """
    bits = bits.long()
    device = bits.device
    num_bits = bits.shape[-1]
    weights = (2 ** torch.arange(num_bits - 1, -1, -1, device=device)).long()
    ints = (bits * weights).sum(dim=-1)
    return ints


# ---------- QAM constellation helpers ----------

def qam_constellation(M: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Generate a square QAM constellation with MAX power 1.

    Rationale: scaling by max energy (outermost point) makes the peak TX power
    identical across different QAM orders. This helps lower-order QAM be more
    reliable under the same peak-power constraint.

    Natural (not Gray) labeling: index -> (row, col) on a grid.

    Args:
        M: constellation size (must be a perfect square: 4,16,64,...)
        device: optional device.

    Returns:
        const: [M, 2] tensor of (I, Q) points.
    """
    if device is None:
        device = torch.device("cpu")

    m_side = int(math.sqrt(M))
    assert m_side * m_side == M, "M must be a perfect square for this QAM helper."

    levels = torch.arange(-(m_side - 1), m_side + 1, 2, device=device).float()  # e.g. [-3,-1,1,3]
    xs, ys = torch.meshgrid(levels, levels, indexing='ij')
    points = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # [M,2]

    # Normalize MAX power to 1 (peak/outermost point)
    max_power = (points ** 2).sum(dim=-1).max()
    points = points / torch.sqrt(max_power + 1e-9)
    return points  # [M,2]


def soft_bits_to_constellation(
    soft_bits: torch.Tensor,
    constellation: torch.Tensor,
    bps: int,
    temperature: float = 1.0,
    cand_bits: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Map soft bits [0,1] to continuous constellation points via soft mixture (FAST VERSION).
    Uses L2 distance in bit space with temperature control.
    
    Args:
        soft_bits: [n_sym, bps] soft bits in [0,1]
        constellation: [M, 2] QAM constellation points
        bps: bits per symbol
        temperature: controls sharpness (lower = more discrete, higher = more uniform)
    
    Returns:
        x_sym: [n_sym, 2] continuous I/Q points (soft blend over constellation)
    """
    n_sym, _ = soft_bits.shape
    M = constellation.shape[0]
    device = soft_bits.device
    
    # Fast: use Euclidean distance in bit space (no expensive log operations)
    # IMPORTANT for speed: allow passing precomputed cand_bits (e.g. self.sym_bits_table[m]).
    if cand_bits is None:
        cand = torch.arange(M, device=device)  # [M]
        cand_bits = int_to_bits(cand, bps).float()  # [M, bps]
    else:
        cand_bits = cand_bits.to(device=device, dtype=torch.float32)
    
    # dist[i,j] = ||soft_bits[i] - cand_bits[j]||^2
    dist = ((soft_bits[:, None, :] - cand_bits[None, :, :]) ** 2).sum(dim=-1)  # [n_sym, M]
    
    # Soft weights via softmax of negative distance WITH TEMPERATURE
    # Lower temp → sharper (more like argmin), higher temp → softer (more uniform)
    temp = max(float(temperature), 1e-6)  # avoid division by zero
    w_sym = torch.softmax(-dist / temp, dim=1)  # [n_sym, M]
    
    # Weighted sum of constellation points
    x_sym = w_sym @ constellation  # [n_sym, 2]
    
    return x_sym


def soft_index_to_bits(q_k: torch.Tensor, expert_label_bits: torch.Tensor, num_bits: int) -> torch.Tensor:
    """
    Convert soft VQ assignments q(k) to soft bit probabilities using precomputed label_bits.
    
    Args:
        q_k: [B, N, K] or [N, K] soft assignment weights
        expert_label_bits: [K, num_bits] precomputed label bits from expert.label_bits
        num_bits: number of bits per label (for validation)
    
    Returns:
        p_bits: [..., num_bits] soft bit probabilities p(b_i=1)
    """
    # Expected bit probability: E[b_i] = sum_k q(k) * bit_i(k)
    original_shape = q_k.shape[:-1]
    q_flat = q_k.reshape(-1, q_k.shape[-1])  # [*, K]
    p_bits_flat = q_flat @ expert_label_bits  # [*, num_bits]
    p_bits = p_bits_flat.reshape(*original_shape, num_bits)
    
    return p_bits


def soft_embed_from_llr_hamming1(
    llr_bits: torch.Tensor,
    codebook_weight: torch.Tensor,
    label_to_idx: torch.Tensor,
    temp: float = 1.0,
):
    """
    Soft embedding using Hamming radius-1 neighborhood around hard label.

    llr_bits:      [N, k]  where llr = log P(bit=0)/P(bit=1)
    codebook_weight: [K, C] (expert.codebook.weight)
    label_to_idx:  [K] maps label_int -> codebook_idx
    temp:          softmax temperature (lower = sharper)

    returns:
      z_soft: [N, C]
    """
    assert llr_bits.dim() == 2
    N, k = llr_bits.shape
    device = llr_bits.device
    K, C = codebook_weight.shape
    assert label_to_idx.numel() == K

    # Convert LLRs -> per-bit probabilities
    # If llr = log P0/P1, then P1 = sigmoid(-llr), P0 = sigmoid(llr)
    llr = llr_bits.float()
    p1 = torch.sigmoid(-llr)  # [N,k]
    p0 = 1.0 - p1

    # Hard label center (bit=1 if llr<0)
    hard_bits = (llr < 0).long()
    hard_label = bits_to_int(hard_bits)  # [N]

    # Candidates = {hard_label} U {hard_label ^ (1<<b)}
    flips = (1 << torch.arange(k, device=device, dtype=torch.long))  # [k]
    cand_labels = torch.cat(
        [hard_label[:, None], hard_label[:, None] ^ flips[None, :]],
        dim=1
    )  # [N, 1+k]

    # Candidate bits
    cand_bits = int_to_bits(cand_labels, k).float()  # [N,1+k,k] in {0,1}

    # Log-prob under independent bits:
    # log p(label) = sum_j [ b*log(p1) + (1-b)*log(p0) ]
    logp = (
        cand_bits * torch.log(p1[:, None, :] + 1e-12) +
        (1.0 - cand_bits) * torch.log(p0[:, None, :] + 1e-12)
    ).sum(dim=-1)  # [N,1+k]

    # Normalize -> weights
    w = torch.softmax(logp / max(temp, 1e-6), dim=1)  # [N,1+k]

    # Map labels -> codebook indices, then gather embeddings
    cand_idx = label_to_idx[cand_labels]          # [N,1+k]
    z_cand = codebook_weight[cand_idx]            # [N,1+k,C]

    # Expected embedding
    z_soft = (w[:, :, None] * z_cand).sum(dim=1)  # [N,C]
    return z_soft

class RoBERTaEncoder(nn.Module):
    def __init__(self, d_model, freeze_bert):
        super().__init__()
        self.roberta = AutoModel.from_pretrained("roberta-base")
        self.projection = nn.Linear(768, d_model)  # optional: project to your internal dimension

        if freeze_bert:
            for param in self.roberta.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]  # shape [B, 768]
        return self.projection(cls_token)               # shape [B, d_model]



class MoETransceiverVQ(nn.Module):
    def __init__(self, code_dims, code_dim, phy_M_list, router_in_dim, router_hidden=(128,128), vq_beta=0.25,
                 chan_dim=64, use_surrogate=True):
        super().__init__()
        self.vq_moe = VQMoE(code_dims=code_dims, code_dim=code_dim, beta=vq_beta)
        self.R = self.vq_moe.num_experts

        self.phy_M_list = list(phy_M_list)
        self.M = len(self.phy_M_list)
        self.bits_per_phy = [int(math.log2(M)) for M in self.phy_M_list]

        self.constellations = [qam_constellation(M) for M in self.phy_M_list]
        self.sym_bits_table = [int_to_bits(torch.arange(M), int(math.log2(M))) for M in self.phy_M_list]

        # Constellation average power after scaling.
        # NOTE: We normalize constellations by MAX power (=1). For a fixed SNR definition
        # that assumes unit average signal power, we must scale noise variance by the
        # constellation's average power to keep the effective SNR consistent across M.
        const_avg_pow = []
        for const in self.constellations:
            const_avg_pow.append(float((const ** 2).sum(dim=-1).mean().item()))
        self.register_buffer("const_avg_pow_tensor", torch.tensor(const_avg_pow, dtype=torch.float32))

        self.J = self.R * self.M
        self.mode_router = ModeRouter(
            in_dim=router_in_dim,
            num_experts=self.R,
            num_phy_modes=self.M,
            hidden_dims=router_hidden,
            temperature=1.0,
            dropout=0.0,
            noisy_gate_std=1.0,
        )
        self.mode_mapper = JointModeMapper(num_experts=self.R, num_phy_modes=self.M)

        # ---- NEW: differentiable channel/code surrogate ----
        self.use_surrogate = use_surrogate
        self.chan_dim = chan_dim   # "channel code" dimension per token

        # simple MLP encoder/decoder, you can deepen if you like
        self.channel_encoder = nn.Sequential(
            nn.Linear(code_dim, chan_dim),
            nn.LayerNorm(chan_dim),
            nn.GELU(),
        )
        self.channel_decoder = nn.Sequential(
            nn.Linear(chan_dim, code_dim),
            nn.GELU(),
        )

        # optional per-mode SNR scaling (so router can shape effective robustness)
        self.mode_snr_db = nn.Parameter(torch.zeros(self.J))  # learnable offset per mode
        # you still pass a global SNR; this offsets it per mode
        self.soft_temp = 0.7
        # bits_per_expert is currently a Python list inside VQMoE
        self.register_buffer(
            "bits_per_expert_tensor",
            torch.tensor(self.vq_moe.bits_per_expert, dtype=torch.float32)
        )  # shape [R]

        self.register_buffer(
            "bits_per_phy_tensor",
            torch.tensor(self.bits_per_phy, dtype=torch.float32)
        )  # shape [M]
        # routing/symbol temperature (can be annealed externally during training)
        self.routing_temp = 1.0
        # Debug flag to print a single batch of TX/RX examples
        self._debug_logged = False
        self._last_epoch_logged = -1  # Track which epoch we last logged
        self.symbol_temp = 1.0
        # VQ softening temperature (anneal during training)
        self.vq_temp = 1.0
        # whether to use soft (mixture) transmit symbols during training
        self.soft_tx_symbols = False
        # whether to use soft VQ path during training (recommended)
        self.soft_vq_path = False

    def _awgn(self, x: torch.Tensor, noise_var) -> torch.Tensor:
        # x: [*, chan_dim]
        var = float(noise_var.item()) if torch.is_tensor(noise_var) else float(noise_var)
        var = max(var, 1e-12)
        std = math.sqrt(var)
        eps = torch.randn_like(x)
        return x + std * eps


    @staticmethod
    def soft_demod_llr(y_sym: torch.Tensor, const: torch.Tensor, bits_table: torch.Tensor, noise_var: float):
        """
        y_sym: [n_sym, 2]
        const: [M, 2]
        bits_table: [M, bps] in {0,1} (float/int)
        returns llr [n_sym, bps] where llr = log P(b=0)/P(b=1)
        VECTORIZED: eliminates Python loop for 3-4x speedup
        """
        var = float(noise_var)
        var = max(var, 1e-12)

        d2 = ((y_sym[:, None, :] - const[None, :, :]) ** 2).sum(dim=-1)  # [n_sym, M]
        metric = -d2 / (2.0 * var)                                        # [n_sym, M]

        bits_table = (bits_table > 0.5)  # bool [M, bps]
        bps = bits_table.shape[1]
        n_sym = y_sym.shape[0]
        
        # Vectorized: expand metric to [n_sym, M, bps], mask by bits, then logsumexp
        metric_exp = metric[:, :, None].expand(-1, -1, bps)  # [n_sym, M, bps]
        bits_exp = bits_table[None, :, :].expand(n_sym, -1, -1)  # [n_sym, M, bps]
        
        NEG_INF = torch.tensor(-1e9, device=metric.device, dtype=metric.dtype)
        metric_b0 = metric_exp.clone()
        metric_b1 = metric_exp.clone()
        metric_b0[bits_exp] = NEG_INF      # mask where bit=1
        metric_b1[~bits_exp] = NEG_INF     # mask where bit=0
        
        m0 = torch.logsumexp(metric_b0, dim=1)  # [n_sym, bps]
        m1 = torch.logsumexp(metric_b1, dim=1)  # [n_sym, bps]
        llr = m0 - m1
        
        # Monitor LLR magnitude to detect saturation/underflow
        llr_max = llr.abs().max().item()
        llr_mean = llr.abs().mean().item()
        if llr_max > 100:
            pass  # print(f"[soft_demod] LLR too large: max={llr_max:.2f}, mean={llr_mean:.2f}")
        
        return llr

    @staticmethod
    def soft_embed_radius2(llr_bits: torch.Tensor, codebook: torch.Tensor, label_to_idx: torch.Tensor, temp: float = 1.0):
        """
        Soft embedding with RADIUS-1 neighborhood (fast version).
        Uses single-bit flips only (not radius-2 pairs) for 2-3x speedup.
        Still fully differentiable and gives good results.

        llr_bits:     [N, k] (log P0/P1) for LABEL bits
        codebook:     [K, C]
        label_to_idx: [K] mapping label_int -> codebook index
        returns:      [N, C]
        """
        N, k = llr_bits.shape
        device = llr_bits.device
        K, C = codebook.shape

        llr = llr_bits.float()
        p1 = torch.sigmoid(-llr)          # P(bit=1) [N, k]
        p0 = 1.0 - p1                     # P(bit=0) [N, k]

        hard_bits = (llr < 0).float()     # [N, k]
        center = bits_to_int(hard_bits)   # [N]

        # Radius-1: single-bit flips only (much faster than radius-2)
        flips1 = (1 << torch.arange(k-1, -1, -1, device=device, dtype=torch.long))  # [k]
        
        cand = torch.cat([
            center[:, None],                 # radius 0 (hard decision)
            center[:, None] ^ flips1[None],  # radius 1 (single-bit flips)
        ], dim=1)  # [N, 1+k]

        # Compute logp with fully vectorized bit extraction (NO PYTHON LOOP)
        # Extract all bits at once: [N, 1+k] candidates, [k] bit positions
        shifts = torch.arange(k-1, -1, -1, device=device, dtype=torch.long)  # [k]
        bit_vals = ((cand.long()[:, :, None] >> shifts[None, None, :]) & 1).float()  # [N, 1+k, k]
        
        # Compute log probabilities: E[log p(bits|label)] = sum_i log p(bit_i)
        # bit_vals: [N, 1+k, k], p1: [N, k], p0: [N, k]
        log_p1 = torch.log(p1 + 1e-12)  # [N, k]
        log_p0 = torch.log(p0 + 1e-12)  # [N, k]
        
        # For each candidate, sum log probs over bits
        # Expand log_p1, log_p0 to [N, k] -> broadcast with [N, 1+k, k]
        # bit_vals[:, j, i] is the i-th bit of candidate j
        # We want logp[n, j] = sum_i bit_vals[n, j, i] * log_p1[n, i] + (1 - bit_vals[n, j, i]) * log_p0[n, i]
        logp = (bit_vals * log_p1[:, None, :]).sum(dim=-1) + \
               ((1.0 - bit_vals) * log_p0[:, None, :]).sum(dim=-1)  # [N, 1+k]

        w = torch.softmax(logp / max(temp, 1e-6), dim=1)  # [N, 1+k]

        cand_idx = label_to_idx[cand]          # [N, 1+k]
        z_cand = codebook[cand_idx]            # [N, 1+k, C]
        z_soft = (w[:, :, None] * z_cand).sum(dim=1)  # [N, C]
        return z_soft
    def forward(self, H, phi, noise_var, channel_type="awgn", hard_routing=True, gumbel_tau=1.0):
        B, N, C = H.shape
        device = H.device

        logits, probs, aux = self.mode_router(
            phi,
            temperature=gumbel_tau if self.training else None,
            return_aux=True
        )
        logits_e, logits_m = aux["logits_e"], aux["logits_m"]
        p_e, p_m = aux["p_e"], aux["p_m"]

        # Soft routing during training with annealable temperature
        if self.training:
            tau = gumbel_tau if gumbel_tau is not None else float(self.routing_temp)
            ge = F.gumbel_softmax(logits_e, tau=tau, hard=False, dim=-1)  # soft weights
            gm = F.gumbel_softmax(logits_m, tau=tau, hard=False, dim=-1)
            # still expose a hard mode index for routing into VQ (keeps expert selection explicit)
            expert_idx = ge.argmax(dim=-1)
            phy_idx = gm.argmax(dim=-1)
            routing_tau_used = tau
        else:
            # at eval/inference respect hard_routing flag
            if hard_routing:
                # Deterministic hard routing (no Gumbel noise at eval)
                expert_idx = p_e.argmax(dim=-1)
                phy_idx = p_m.argmax(dim=-1)
                ge = F.one_hot(expert_idx, num_classes=self.R).float()
                gm = F.one_hot(phy_idx, num_classes=self.M).float()
            else:
                expert_idx = p_e.argmax(dim=-1)
                phy_idx = p_m.argmax(dim=-1)
            routing_tau_used = gumbel_tau

        mode_idx = expert_idx * self.M + phy_idx  # [B]

        # ---- VQ quantization (still uses ST inside VQ) ----
        z_q, indices, vq_loss = self.vq_moe(H, expert_idx=expert_idx)  # z_q: [B,N,C]

        bits_per_expert = self.bits_per_expert_tensor   # [R] float32 tensor
        bits_per_phy = self.bits_per_phy_tensor  

        # containers (filled differently depending on path)
        H_hat = torch.zeros_like(H)
        H_hat_soft = torch.zeros_like(H)  # Track soft reconstruction separately for consistency loss
        z_q_soft = torch.zeros_like(H)   # CRITICAL: keep soft quantization separate from hard z_q
        soft_path_mask = torch.zeros(B, dtype=torch.bool, device=device)  # Track which samples use soft path
        vq_entropy_loss = torch.tensor(0.0, device=device)  # Entropy regularization for diverse routing
        bits_per_block, syms_per_block = [], []
        consistency_loss = torch.tensor(0.0, device=device)
            # ---- NEW: approximate per-mode bits/symbols for EXPECTED rate ----
        # bits per block for (expert r, mode m) ≈ N * k_r
        mode_bits = N * bits_per_expert[:, None].expand(self.R, self.M)  # [R,M]
        # approximate symbols = bits / bps
        mode_syms = mode_bits / bits_per_phy[None, :]                    # [R,M]

        mode_bits_flat = mode_bits.reshape(-1)   # [J]
        mode_syms_flat = mode_syms.reshape(-1)   # [J]

        # probs: [B, J]; expected bits/syms per block (fully differentiable)
        exp_bits_per_block = (probs * mode_bits_flat[None, :]).sum(dim=1)  # [B]
        exp_syms_per_block = (probs * mode_syms_flat[None, :]).sum(dim=1)  # [B]

        if  self.use_surrogate:
            # ----- FULLY DIFFERENTIABLE SURROGATE JSCC -----
            # channel code dimension per token: self.chan_dim

            for b in range(B):
                    r = int(expert_idx[b].item())
                    m = int(phy_idx[b].item())
                    k_r = bits_per_expert[r]
                    bps = bits_per_phy[m]

                    # approximate payload bits for accounting / rate loss
                    B_bits = N * k_r
                    bits_per_block.append(B_bits)

                    # approximate symbols from "digital view" (for logging / rate penalty)
                    n_sym = B_bits / bps
                    syms_per_block.append(n_sym)

                    # channel encoder: map latent tokens to channel vectors
                    # z_q[b]: [N, C] -> [N, chan_dim]
                    x = self.channel_encoder(z_q[b])  # [N, chan_dim]

                    # energy normalize (optional but recommended)
                    # keep average power per token ~ 1
                    x = x / (x.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8)

                    # effective SNR per sample/mode:
                    # global SNR from noise_var, plus learnable per-mode offset
                    if torch.is_tensor(noise_var) and noise_var.numel() > 1:
                        var_b = noise_var.view(-1)[b]
                    else:
                        var_b = noise_var

                    # you can optionally map "mode_snr_db" to a per-mode scaling of var_b
                    # here we leave var_b as is for simplicity.

                    # AWGN on channel representation
                    y = self._awgn(x, var_b)  # [N, chan_dim]

                    # channel decoder back to latent space
                    H_hat[b] = self.channel_decoder(y)  # [N, C]

            bits_per_block = torch.tensor(bits_per_block, device=device, dtype=torch.float32)
            syms_per_block = torch.tensor(syms_per_block, device=device, dtype=torch.float32)







                # since VQ already has ST inside, we DO NOT need another ST here.
                # Let gradients flow from H_hat -> z_q -> encoder via VQ-ST.
            H_noisy_st = H_hat

        else:
                ge = F.gumbel_softmax(logits_e, tau=gumbel_tau, hard=True, dim=-1) 
                gm = F.gumbel_softmax(logits_m, tau=gumbel_tau, hard=True, dim=-1) 
                expert_idx = ge.argmax(dim=-1) 
                phy_idx = gm.argmax(dim=-1) 
                mode_idx = expert_idx * self.M + phy_idx # VQ 
                z_q, indices, vq_loss = self.vq_moe(H, expert_idx=expert_idx) 
                bits_per_expert = self.vq_moe.bits_per_expert 
                bits_per_phy = self.bits_per_phy 
                H_hat = torch.zeros_like(H) 
                H_hat_soft_list = [torch.zeros(N, H.shape[-1], device=device) for _ in range(B)]
                bits_per_block, syms_per_block = [], [] 
                
                # Group samples by (expert_idx, phy_idx) for efficient batch processing
                # This vectorizes across samples with the same routing
                unique_modes = torch.unique(mode_idx)
                for mode in unique_modes:
                    mask = mode_idx == mode
                    batch_idxs = torch.where(mask)[0]  # indices of samples with this mode
                    
                    if len(batch_idxs) == 0:
                        continue
                    
                    # Decode mode to (expert, phy)
                    r = int((mode // self.M).item())
                    m = int((mode % self.M).item())
                    k_r = bits_per_expert[r]
                    bps = bits_per_phy[m]
                    
                    expert = self.vq_moe.experts[r]
                    const = self.constellations[m].to(device)
                    
                    # Extract batch samples with this mode
                    # z_q[batch_idxs]: [len(batch_idxs), N, C]
                    # indices[batch_idxs]: [len(batch_idxs), N]
                    n_batch = len(batch_idxs)
                    z_q_batch = z_q[batch_idxs]  # [n_batch, N, C]
                    indices_batch = indices[batch_idxs]  # [n_batch, N]
                    # Use encoder outputs (pre-quantized) for soft assignments
                    H_batch = H[batch_idxs]  # [n_batch, N, C]
                    
                    # ============================================
                    # SOFT VQ PATH: fully differentiable pipeline (VECTORIZED)
                    # ============================================
                    # Use soft path during training AND eval if enabled (ensures consistent forward pass)
                    # This helps hard path learn from soft path gradients during backprop
                    if self.soft_vq_path:
                        # 1. Soft VQ assignment for all samples in batch: [n_batch, N, C] -> [n_batch, N, K]
                        # Process each sample separately (expert.soft_assign expects [N, C])
                        # IMPORTANT: use encoder outputs `H_batch` (pre-quantized) NOT the hard `z_q_batch`
                        q_k_list = []
                        for i in range(n_batch):
                            q_k_i, _ = expert.soft_assign(H_batch[i], temperature=self.vq_temp)
                            q_k_list.append(q_k_i)
                        q_k_batch = torch.stack(q_k_list, dim=0)  # [n_batch, N, K]
                        
                        # Monitor VQ entropy to detect collapse (low entropy = collapsed/degenerate)
                        vq_ent = -(q_k_batch * torch.log(q_k_batch + 1e-12)).sum(dim=-1).mean()
                        # print(f"[soft_vq] vq_entropy={vq_ent.item():.4f} (target >0.5 for diverse routing)")
                        
                        # ENTROPY REGULARIZATION: penalize low entropy (encourage diverse routing)
                        # Only apply during training to avoid numerical issues at eval
                        entropy_per_sample = -(q_k_batch * torch.log(q_k_batch + 1e-12)).sum(dim=-1)  # [n_batch, N]
                        if self.training:
                            entropy_loss_batch = -entropy_per_sample.mean()  # Negate: loss = -entropy (minimize = maximize entropy)
                            vq_entropy_loss = vq_entropy_loss + entropy_loss_batch.detach()  # Accumulate for diagnostics
                        # This is a differentiable soft blend of codebook entries weighted by soft assignments
                        # KEEP SEPARATE from hard z_q (don't overwrite z_q)
                        codebook = expert.codebook.weight  # [K, C]
                        # q_k_batch: [n_batch, N, K], codebook: [K, C]
                        # z_q_soft_batch: [n_batch, N, C] = soft blend
                        z_q_soft_batch = torch.einsum('bnk,kc->bnc', q_k_batch, codebook)  # [n_batch, N, C]
                        # Store soft quantization SEPARATELY for later use in straight-through bridge
                        for i, b_idx in enumerate(batch_idxs):
                            z_q_soft[b_idx] = z_q_soft_batch[i]  # Keep z_q_soft separate; don't touch z_q
                            soft_path_mask[b_idx] = True  # Mark this sample as using soft path
                        
                        # 2. Convert soft VQ to soft bits: [n_batch, N, K] -> [n_batch, N*k_r]
                        # IMPORTANT: compute bit encoding on-the-fly from idx_to_label to ensure correctness
                        # (avoid relying on potentially-unpopulated expert.label_bits)
                        label_bits_fresh = int_to_bits(expert.idx_to_label, k_r).float()  # [K, k_r]
                        p_bits_batch_list = []
                        for i in range(n_batch):
                            # Compute soft bit probabilities: q(k) @ label_bits(k)
                            # q_k_batch[i]: [N, K], label_bits_fresh: [K, k_r] -> p_bits: [N, k_r]
                            p_bits_i = q_k_batch[i] @ label_bits_fresh  # [N, k_r]
                            p_bits_batch_list.append(p_bits_i)
                        # All samples have same bit structure, so can stack
                        p_bits_batch_flat = torch.cat([p.view(-1) for p in p_bits_batch_list], dim=0)  # [n_batch*N*k_r]
                        
                        # Reshape to symbol format for all samples
                        B_bits_total = p_bits_batch_flat.numel()
                        n_sym_total = math.ceil(B_bits_total / (n_batch * bps))  # symbols per sample
                        
                        # Process all samples' symbols in one batch
                        padded_total = n_batch * n_sym_total * bps
                        if padded_total > B_bits_total:
                            p_bits_batch_flat = torch.cat([p_bits_batch_flat, 
                                                           torch.zeros(padded_total - B_bits_total, device=device)], dim=0)
                        p_bits_sym_all = p_bits_batch_flat.view(n_batch * n_sym_total, bps)  # [n_batch*n_sym, bps]
                        
                        # 3. Map all soft bits to soft constellation (VECTORIZED over symbols)
                        # Use symbol_temp to control sharpness; reuse cached bit table for speed.
                        x_sym_all = soft_bits_to_constellation(
                            p_bits_sym_all,
                            const,
                            bps,
                            temperature=self.symbol_temp,
                            cand_bits=self.sym_bits_table[m],
                        )  # [n_batch*n_sym, 2]
                        
                        # 4. AWGN channel for all symbols (FULLY VECTORIZED - no per-symbol loop!)
                        var_batch = noise_var.view(-1)[batch_idxs] if (torch.is_tensor(noise_var) and noise_var.numel() > 1) else noise_var
                        # Use average noise var for all symbols (OR expand if needed)
                        avg_noise_var = var_batch.mean().item() if torch.is_tensor(var_batch) else float(var_batch)
                        # Keep SNR consistent w.r.t. average constellation power
                        avg_noise_var_eff = avg_noise_var * float(self.const_avg_pow_tensor[m].item())
                        y_sym_all = self._awgn(x_sym_all, avg_noise_var_eff)  # [n_batch*n_sym, 2] - VECTORIZED!
                        # Log only first batch of each epoch to track annealing
                        if not self._debug_logged:
                            # Power checks
                            sym_pow = (x_sym_all ** 2).sum(dim=-1)
                            avg_pow = sym_pow.mean().item()
                            max_pow = sym_pow.max().item()
                            const_pow = (const ** 2).sum(dim=-1)
                            const_max_pow = const_pow.max().item()
                            const_avg_pow = const_pow.mean().item()

                            print("[soft_path] vq_T={:.3f} sym_T={:.3f} soft_T={:.3f} | m={} M={} bps={}".format(
                                self.vq_temp, self.symbol_temp, self.soft_temp, m, int(self.phy_M_list[m]), bps
                            ))
                            print("  Const avg_pow={:.4f} max_pow={:.4f} (max should be 1.0)".format(const_avg_pow, const_max_pow))
                            print("  TX x[:3]={}".format(x_sym_all[:3].detach().cpu().numpy()))
                            print("  TX avg_pow={:.4f} max_pow={:.4f} (max should be <= 1.0)".format(avg_pow, max_pow))
                            print("  Soft bits[:3]={}".format(p_bits_sym_all[:3].detach().cpu().numpy()))
                            self._debug_logged = True
                        
                        # 5. Soft demodulation for all symbols
                        llr_sym_all = self.soft_demod_llr(
                            y_sym=y_sym_all,
                            const=const,
                            bits_table=self.sym_bits_table[m].to(device),
                            noise_var=avg_noise_var_eff,
                        )
                        
                        # 6. Reshape LLRs back to per-sample labels and embed
                        for i, b_idx in enumerate(batch_idxs):
                            llr_flat_i = llr_sym_all[i*n_sym_total:(i+1)*n_sym_total].reshape(-1)[:N*k_r]
                            llr_label_i = llr_flat_i.view(N, k_r)
                            
                            codebook = expert.codebook.weight
                            H_hat_soft_list[b_idx] = self.soft_embed_radius2(llr_bits=llr_label_i, 
                                                                             codebook=codebook,
                                                                             label_to_idx=expert.label_to_idx,
                                                                             temp=self.soft_temp)
                            H_hat[b_idx] = H_hat_soft_list[b_idx]
                            bits_per_block.append(N * k_r)
                            syms_per_block.append(n_sym_total)
                    
                    # ============================================
                    # HARD VQ PATH: discrete baseline (VECTORIZED)
                    # ============================================
                    else:
                        # All samples in this batch use same (expert, phy)
                        # indices_batch: [n_batch, N]
                        label_int_batch = expert.idx_to_label[indices_batch]  # [n_batch, N]
                        bits_batch = int_to_bits(label_int_batch.view(-1), k_r).view(n_batch, N, k_r)  # [n_batch, N, k_r]
                        
                        # Flatten bits for symbol mapping
                        bits_flat = bits_batch.view(-1)  # [n_batch*N*k_r]
                        B_bits_total = bits_flat.numel()
                        n_sym_total = math.ceil(B_bits_total / (n_batch * bps))
                        
                        padded_total = n_batch * n_sym_total * bps
                        if padded_total > B_bits_total:
                            bits_flat = torch.cat([bits_flat, torch.zeros(padded_total - B_bits_total, device=device)], dim=0)
                        bits_sym_all = bits_flat.view(n_batch * n_sym_total, bps)  # [n_batch*n_sym, bps]
                        
                        # Map bits to constellation points (VECTORIZED)
                        weights = (2 ** torch.arange(bps - 1, -1, -1, device=device, dtype=torch.float32))
                        sym_ints = (bits_sym_all * weights[None, :]).sum(dim=-1).long().clamp(0, self.phy_M_list[m] - 1)
                        x_sym_all = const[sym_ints]  # [n_batch*n_sym, 2]
                        
                        # Soft TX symbols if enabled
                        if self.soft_tx_symbols and self.training:
                            cand_bits = int_to_bits(torch.arange(self.phy_M_list[m], device=device), bps).float()
                            dist = (bits_sym_all[:, None, :] - cand_bits[None, :, :]).abs().sum(dim=-1)
                            temp_sym = max(self.symbol_temp, 1e-6)
                            w_sym = torch.softmax(-dist / temp_sym, dim=1)
                            x_sym_all = (w_sym[:, :, None] * const[None, :, :]).sum(dim=1)
                        
                        # AWGN channel (FULLY VECTORIZED - no per-symbol loop!)
                        var_batch = noise_var.view(-1)[batch_idxs] if (torch.is_tensor(noise_var) and noise_var.numel() > 1) else noise_var
                        avg_noise_var = var_batch.mean().item() if torch.is_tensor(var_batch) else float(var_batch)
                        # Keep SNR consistent w.r.t. average constellation power
                        avg_noise_var_eff = avg_noise_var * float(self.const_avg_pow_tensor[m].item())
                        y_sym_all = self._awgn(x_sym_all, avg_noise_var_eff)  # [n_batch*n_sym, 2] - VECTORIZED!
                        if not self._debug_logged:
                            # Log one batch of hard-path TX/RX symbols for inspection
                            print("[debug][hard] m={} x[:3]={} y[:3]={}".format(
                                m,
                                x_sym_all[:3].detach().cpu().numpy(),
                                y_sym_all[:3].detach().cpu().numpy(),
                            ))
                            self._debug_logged = True
                        
                        # Demodulation
                        llr_sym_all = self.soft_demod_llr(
                            y_sym=y_sym_all,
                            const=const,
                            bits_table=self.sym_bits_table[m].to(device),
                            noise_var=avg_noise_var_eff,
                        )
                        
                        # Embedding for all samples
                        for i, b_idx in enumerate(batch_idxs):
                            llr_flat_i = llr_sym_all[i*n_sym_total:(i+1)*n_sym_total].reshape(-1)[:N*k_r]
                            llr_label_i = llr_flat_i.view(N, k_r)
                            
                            codebook = expert.codebook.weight
                            H_hat[b_idx] = self.soft_embed_radius2(llr_bits=llr_label_i,
                                                                  codebook=codebook,
                                                                  label_to_idx=expert.label_to_idx,
                                                                  temp=self.soft_temp)
                            bits_per_block.append(N * k_r)
                            syms_per_block.append(n_sym_total)
                
                # Copy soft_hat_list to H_hat_soft if using soft path
                if self.soft_vq_path and self.training:
                    H_hat_soft = torch.stack(H_hat_soft_list, dim=0)  # [B, N, C]
                
                # ===== Normalize H_hat per-sample to match pre-quantized z_q statistics =====
                # VECTORIZED: compute statistics for all samples at once
                try:
                    H_hat_mean = H_hat.mean(dim=1, keepdim=True).detach()  # [B, 1, C]
                    H_hat_std = H_hat.std(dim=1, keepdim=True).detach()    # [B, 1, C]
                    z_q_mean = z_q.mean(dim=1, keepdim=True)              # [B, 1, C]
                    z_q_std = z_q.std(dim=1, keepdim=True)                # [B, 1, C]
                    # Normalize H_hat: (H_hat - mean) / std * (z_q_std) + z_q_mean [VECTORIZED]
                    H_hat = (H_hat - H_hat_mean) / (H_hat_std + 1e-6) * (z_q_std + 1e-6) + z_q_mean
                except Exception:
                    pass

                # Straight-through estimator bridge
                # CRITICAL: Use z_q_soft for soft path samples, z_q (hard) for hard path samples
                z_q_for_st = torch.where(soft_path_mask[:, None, None], z_q_soft, z_q)  # [B, N, C]
                H_noisy_st = H_hat + z_q_for_st - z_q_for_st.detach()
        
        latent_mse = ((H_noisy_st - H) ** 2).mean(dim=(1, 2))      # [B]
        
        latent_mse = ((H_noisy_st - H) ** 2).mean(dim=(1, 2))      # [B]
        channel_mse = ((H_hat - z_q) ** 2).mean(dim=(1, 2))        # [B]
        rate_bits = bits_per_block
        Ns_eff = syms_per_block

        if isinstance(rate_bits, list):
            rate_bits = torch.tensor(rate_bits, dtype=torch.float32, device=H_noisy_st.device)
        if isinstance(Ns_eff, list):
            Ns_eff = torch.tensor(Ns_eff, dtype=torch.float32, device=H_noisy_st.device)
        stats = dict(
            logits=logits,
            probs=probs,
            expert_probs=probs.view(B, self.R, self.M).sum(dim=2),
            phy_probs=probs.view(B, self.R, self.M).sum(dim=1),
            mode_idx=mode_idx,
            expert_idx=expert_idx,
            phy_idx=phy_idx,
            vq_loss=vq_loss,
            vq_entropy_loss=vq_entropy_loss,  # NEW: entropy regularization for soft path
            bits_per_block=rate_bits,
            syms_per_block=Ns_eff,
            routing_tau=routing_tau_used,
            exp_bits_per_block=exp_bits_per_block,   # NEW, differentiable
            exp_syms_per_block=exp_syms_per_block,  
            latent_mse=latent_mse,
            channel_mse=channel_mse,
            H_hat=H_hat,
            z_q=z_q,
        )
        return H_noisy_st, stats


    # def forward(self, H, phi, noise_var, channel_type="awgn", hard_routing=True, gumbel_tau=1.0):
    #     B, N, C = H.shape
    #     device = H.device

    #     logits, probs, aux = self.mode_router(phi, temperature=gumbel_tau if self.training else None, return_aux=True)
    #     logits_e, logits_m = aux["logits_e"], aux["logits_m"]
    #     p_e, p_m = aux["p_e"], aux["p_m"]

    # 
    #         phy_probs=phy_probs,
    #         mode_idx=mode_idx,
    #         expert_idx=expert_idx,
    #         phy_idx=phy_idx,
    #         vq_loss=vq_loss,
    #         bits_per_block=bits_per_block,
    #         syms_per_block=syms_per_block,
    #         exp_bits_per_block=exp_bits_per_block,
    #         exp_syms_per_block=exp_syms_per_block,
    #         latent_mse=latent_mse,
    #         channel_mse=channel_mse,
    #         H_hat=H_hat.detach(),
    #         z_q=z_q.detach(),
    #     )
    #     return H_noisy_st, stats


class MODJSCC_MoE_Cls_VQ(nn.Module):
    """
    Semantic JSCC with MoE + VQ for CLASSIFICATION (e.g., SST-2).

    Pipeline:
    - Input sentences → RoBERTa encoder → semantic embedding y [B, d_model].
    - Project y into N latent tokens H [B, N, d_model].
    - Build router feature phi (logSNR + semantic difficulty stats of y).
    - MoETransceiverVQ:
        - VQ MoE selects an expert, quantizes H, maps indices to bits and QAM.
        - Channel corrupts symbols, RX demaps + dequantizes to H_hat [B, N, d_model].
        - VQ-VAE style straight-through: returns H_st, which is H_hat forward,
          but gradients flow as if it were the pre-channel quantized latent.
    - Classification head takes pooled H_st and outputs logits over sentiment labels.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_labels: int = 2,
        freeze_bert: bool = False,
        N: int = 9,
        # vq_codebook_sizes=(128, 256, 512, 1024, 2048),
        vq_codebook_sizes=(128,),
        # phy_M_list=(4, 16, 64, 256, 1024),
        phy_M_list=(4,),
        load_surrogate: bool = True,
        surrogate_ckpt_path: str = '/home/necphy/ducjunior/RoBERTa_MoE/checkpoints/JSCC_MoE_MoE_Cls_teacher/best_model.pt',
    ):
        super().__init__()

        self.d_model = d_model
        self.num_labels = num_labels
        self.N = N
        self.vq_codebook_sizes = vq_codebook_sizes
        self.phy_M_list = phy_M_list

        # ----- Semantic encoder -----
        self.encoder = RoBERTaEncoder(d_model=d_model, freeze_bert=freeze_bert)

        # ----- VQ MoE + PHY transceiver -----
        # IMPORTANT: MoETransceiverVQ should return H_st (VQ-VAE ST across channel),
        # not the raw post-channel H_hat. See earlier transceiver fix.
        self.pre_vq_norm = nn.LayerNorm(d_model)
        # self.transceiver = MoETransceiverVQ(
        #     code_dims=list(vq_codebook_sizes),  # K_r per expert
        #     code_dim=d_model,                   # latent dim
        #     phy_M_list=list(phy_M_list),        # QAM sizes
        #     router_in_dim=6,                    # feature dim (defined below)
        #     use_surrogate=True
        #     router_hidden=(128, 128),
        #     vq_beta=0.25,
        # )


        # Analog (teacher / reference) path
        self.transceiver_analog = MoETransceiverVQ(
            code_dims=list(vq_codebook_sizes),  # K_r per expert
            code_dim=d_model,                   # latent dim
            phy_M_list=list(phy_M_list),        # QAM sizes
            router_in_dim=6,                    # feature dim (defined below)
            use_surrogate=True,
            router_hidden=(128, 128),
            vq_beta=0.25,
        )
        # Optionally load pretrained surrogate transceiver weights
        if load_surrogate:
            try:
                ckpt_analog = torch.load(surrogate_ckpt_path, map_location='cpu')
                full_state = ckpt_analog.get("model_state_dict", ckpt_analog)
                transceiver_state = {}
                for k, v in full_state.items():
                    if k.startswith("transceiver."):
                        # strip the "transceiver." prefix for loading into the submodule
                        new_k = k[len("transceiver."):]
                        transceiver_state[new_k] = v
                self.transceiver_analog.load_state_dict(transceiver_state, strict=True)
                for p in self.transceiver_analog.parameters():
                    p.requires_grad = False
            except Exception:
                print(f"Warning: analog (surrogate) checkpoint not loaded from {surrogate_ckpt_path}; continuing without pretrained analog.")
        self.transceiver_dig  = MoETransceiverVQ(            code_dims=list(vq_codebook_sizes),  # K_r per expert
            code_dim=d_model,                   # latent dim
            phy_M_list=list(phy_M_list),        # QAM sizes
            router_in_dim=6,                    # feature dim (defined below)
            use_surrogate=False,
            router_hidden=(128, 128),
            vq_beta=0.25,
        )

        # Project sentence embedding y into N latent tokens [B, N, d_model]
        self.latent_proj = nn.Linear(d_model, N * d_model)

        # ----- Classification head -----
        self.cls_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels),
        )

    # --- Router features: semantic difficulty + channel ---
    @staticmethod
    def _router_feat(logsnr: torch.Tensor, y: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns phi: [B, 6]
        """
        eps = 1e-8

        # L2 norm and log-norm
        y_norm = torch.norm(y, dim=1, keepdim=True)                 # [B,1]
        log_norm = torch.log(y_norm + eps)                          # [B,1]

        # Softmax entropy
        p = F.softmax(y, dim=1)
        entropy = -(p * torch.log(p + eps)).sum(dim=1, keepdim=True)

        # Normalized variance
        y_unit = y / (y_norm + eps)
        var_y = y_unit.var(dim=1, unbiased=False, keepdim=True)

        # length feature
        length = attention_mask.sum(dim=1, keepdim=True).float()    # [B,1]
        log_len = torch.log(length + 1.0)                           # [B,1]

        # L1/L2 ratio
        y_l1 = torch.norm(y, p=1, dim=1, keepdim=True)
        l1_over_l2 = y_l1 / (y_norm + eps)

        phi = torch.cat([logsnr, log_norm, entropy, var_y, log_len, l1_over_l2], dim=1)
        return phi
    def forward(
        self,
        input_ids,
        attention_mask,
        n_var,
        channel: str = "AWGN",
        return_probs: bool = False,
    ):
        B = input_ids.size(0)
        device = input_ids.device

        # 1) Encode sentence to semantic vector y
        y = self.encoder(input_ids, attention_mask)  # [B, d_model]


        y = self.pre_vq_norm(y) 
        # 2) Build SNR feature
        if torch.is_tensor(n_var):
            n_var = n_var.to(device)
            if n_var.ndim == 0:
                logsnr_val = torch.log(1.0 / n_var).item()
                logsnr = torch.full((B, 1), logsnr_val, device=device)
            else:
                n_var_flat = n_var.view(B, -1)[:, 0]
                logsnr = torch.log(1.0 / n_var_flat).view(-1, 1)
        else:
            logsnr = torch.full((B, 1), math.log(1.0 / n_var), device=device)

        # 3) Build router feature phi
        phi = self._router_feat(logsnr, y, attention_mask)  # [B, 4]

        # 4) Build latent tokens H: [B, N, d_model]
        H = self.latent_proj(y).view(B, self.N, self.d_model)

        # 5) Transceiver: JSCC with MoE + VQ (returns VQ-VAE ST latent H_st)
        # H_st, stats_tx = self.transceiver(
        #     H,
        #     phi,
        #     noise_var=n_var,
        #     channel_type=channel,
        #     hard_routing=True,
        # )  # H_st: [B, N, d_model]

        with torch.no_grad():
            H_analog, stats_analog = self.transceiver_analog(
                H, phi, noise_var=n_var, channel_type=channel, hard_routing=True
        )  # [B,N,C]

        # --- STUDENT: digital JSCC (gradient) ---
        H_dig, stats_dig = self.transceiver_dig(
            H, phi, noise_var=n_var, channel_type=channel, hard_routing=True
        )  # [B,N,C]

        # 3) Pool + classifier (shared head)
        feat_analog = H_analog.mean(dim=1)
        feat_dig  = H_dig.mean(dim=1)
        logits_analog = self.cls_head(feat_analog)   # teacher logits (soft target)
        logits_dig  = self.cls_head(feat_dig)    # student logits

        # Build rate/routing outputs expected by training loop
        rate_bits = stats_dig.get("bits_per_block", None)
        R = self.transceiver_dig.R if hasattr(self, "transceiver_dig") else self.transceiver_analog.R
        route_hard_tx = F.one_hot(stats_dig.get("expert_idx", torch.zeros(B, dtype=torch.long, device=device)), num_classes=R).float()
        Ns_eff = stats_dig.get("syms_per_block", None)

        # Build stats for training: use digital stats but include teacher logits for distillation
        stats_tx = dict(stats_dig)
        stats_tx["logits_soft"] = logits_analog
        # provide encoder-level and pooled features for plotting/diagnostics
        stats_tx["y_target"] = y.detach()
        stats_tx["feat_pooled"] = feat_dig.detach()

        return logits_dig, rate_bits, route_hard_tx, Ns_eff, stats_tx
    # else:
    #     return logits_dig, stats_dig

        # 6) Pool over latent tokens (mean pool) → [B, d_model]
        # feat_pooled = H_st.mean(dim=1)

        # # 7) Classification logits
        # logits = self.cls_head(feat_pooled)  # [B, num_labels]

        # # 8) Rate & routing info
        # rate_bits = stats_tx["bits_per_block"]          # [B]
        # R = self.transceiver.R
        # route_hard_tx = F.one_hot(stats_tx["expert_idx"], num_classes=R).float()
        # Ns_eff = stats_tx["syms_per_block"]     # [B]

        # # For convenience / later tasks
        # stats_tx["y_target"] = y.detach()
        # stats_tx["feat_pooled"] = feat_pooled

        # if return_probs:
        #     route_probs = stats_tx["probs"]  # [B, J]
        #     return logits, rate_bits, route_hard_tx, Ns_eff, route_probs, None, stats_tx
        # else:
        #     return logits, rate_bits, route_hard_tx, Ns_eff, stats_tx

from transformers import BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class MODJSCC_MoE_TextRec_VQ(nn.Module):
    """
    Text reconstruction with a generation model (BART) + MoETransceiverVQ in the bottleneck.

    Encoder side:
      input_ids -> BART encoder -> E [B, L, d]
      compress E -> H [B, N, d] via learned query attention pooling
      router feat phi from pooled y + logsnr
      H -> MoETransceiverVQ -> H_st [B, N, d] (ST across channel)

    Decoder side:
      BART decoder cross-attends to H_st and reconstructs text (labels / teacher forcing).
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        freeze_encoder: bool = False,
        latent_dim=256,
        freeze_decoder: bool = False,
        N: int = 9,
        vq_codebook_sizes=(0,),   # MUST be tuple
        phy_M_list=(0,),            # MUST be tuple
        router_in_dim: int = 6,
    ):
        super().__init__()
        self.N = N
        self.latent_dim = latent_dim
        # ---- Generator backbone ----
        self.vq_codebook_sizes = vq_codebook_sizes
        self.phy_M_list = phy_M_list
        self.gen = BartForConditionalGeneration.from_pretrained(model_name)
        d_model = self.gen.config.d_model
        self.d_model = d_model

        if freeze_encoder:
            for p in self.gen.model.encoder.parameters():
                p.requires_grad = False
        if freeze_decoder:
            for p in self.gen.model.decoder.parameters():
                p.requires_grad = False



         # ---- project encoder states down to latent_dim ----
        self.enc_down = nn.Linear(d_model, latent_dim)
        self.enc_up   = nn.Linear(latent_dim, d_model)

        # ---- learned query pooling in latent_dim ----
        self.latent_queries = nn.Parameter(torch.randn(N, latent_dim) * 0.02)
        self.pool_ln = nn.LayerNorm(latent_dim)       # <-- FIX (was 768)
        self.pre_vq_norm = nn.LayerNorm(latent_dim)  

        # ---- MoE + VQ + PHY transceiver ----
        # self.pre_vq_norm = nn.LayerNorm(d_model)
        self.transceiver = MoETransceiverVQ(
            code_dims=list(vq_codebook_sizes),
            code_dim=latent_dim,
            phy_M_list=list(phy_M_list),
            router_in_dim=router_in_dim,
            router_hidden=(128, 128),
            vq_beta=0.25,
        )

    # --- Router features: semantic difficulty + channel ---
    @staticmethod
    def _router_feat(logsnr: torch.Tensor, y: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns phi: [B, 6]
        """
        eps = 1e-8

        # L2 norm and log-norm
        y_norm = torch.norm(y, dim=1, keepdim=True)                 # [B,1]
        log_norm = torch.log(y_norm + eps)                          # [B,1]

        # Softmax entropy
        p = F.softmax(y, dim=1)
        entropy = -(p * torch.log(p + eps)).sum(dim=1, keepdim=True)

        # Normalized variance
        y_unit = y / (y_norm + eps)
        var_y = y_unit.var(dim=1, unbiased=False, keepdim=True)

        # length feature
        length = attention_mask.sum(dim=1, keepdim=True).float()    # [B,1]
        log_len = torch.log(length + 1.0)                           # [B,1]

        # L1/L2 ratio
        y_l1 = torch.norm(y, p=1, dim=1, keepdim=True)
        l1_over_l2 = y_l1 / (y_norm + eps)

        phi = torch.cat([logsnr, log_norm, entropy, var_y, log_len, l1_over_l2], dim=1)
        return phi

    def _build_logsnr(self, n_var, B: int, device):
        if torch.is_tensor(n_var):
            n_var = n_var.to(device)
            if n_var.ndim == 0:
                logsnr_val = torch.log(1.0 / n_var).item()
                return torch.full((B, 1), logsnr_val, device=device)
            n_var_flat = n_var.view(B, -1)[:, 0]
            return torch.log(1.0 / n_var_flat).view(-1, 1)
        return torch.full((B, 1), math.log(1.0 / n_var), device=device)

    # def _query_pool(self, E: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    #     """
    #     E: [B, L, d], attention_mask: [B, L] (1=keep, 0=pad)
    #     Returns H: [B, N, d]
    #     """
    #     B, L, d = E.shape
    #     Q = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # [B,N,d]

    #     # scaled dot-product attention: attn = softmax(Q K^T)
    #     K = E
    #     V = E

    #     scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(d)  # [B,N,L]

    #     # mask out padding tokens in encoder sequence
    #     if attention_mask is not None:
    #         mask = attention_mask.unsqueeze(1).expand(-1, self.N, -1)  # [B,N,L]
    #         scores = scores.masked_fill(mask == 0, -1e9)

    #     attn = torch.softmax(scores, dim=-1)         # [B,N,L]
    #     H = torch.matmul(attn, V)                    # [B,N,d]
    #     H = self.pool_ln(H)
    #     return H
    def _query_pool(self, E_lat: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L, d = E_lat.shape  # d should be latent_dim

        Q = self.latent_queries.unsqueeze(0).expand(B, -1, -1)  # [B, N, d]

        # sanity check to avoid silent pain
        assert Q.size(-1) == d, (Q.shape, E_lat.shape)

        scores = torch.matmul(Q, E_lat.transpose(1, 2)) / math.sqrt(d)  # [B, N, L]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).expand(-1, self.N, -1)  # [B,N,L]
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)         # [B,N,L]
        H = torch.matmul(attn, E_lat)                # [B,N,d]
        H = self.pool_ln(H)
        return H
    def forward(
        self,
        input_ids,
        attention_mask,
        n_var,
        channel: str = "AWGN",
        labels=None,                 # IMPORTANT: provide target text token ids here
        decoder_input_ids=None,       # optional (usually omit if using labels)
        return_probs: bool = False,
        hard_routing: bool = True, gumbel_tau: float = 3.0
    ):
        """
        Returns:
          outputs: HuggingFace seq2seq outputs (loss/logits/etc.)
          rate_bits:     [B]
          route_hard_tx: [B, R]
          Ns_eff:        [B]
          (optional) route_probs: [B, J]
          stats_tx: dict
        """
        device = input_ids.device
        B = input_ids.size(0)

        # 1) Encoder token states
        enc_out = self.gen.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        E = enc_out.last_hidden_state                   # [B, L, d]
        E_lat = self.enc_down(E)                      # [B,L,latent_dim]

        # 2) Pool a sentence-level y for routing features
        # (masked mean pool)
        if attention_mask is None:
            y = E_lat.mean(dim=1)
        else:
            m = attention_mask.unsqueeze(-1).float()
            y = (E_lat * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))

        y = self.pre_vq_norm(y)

        # 3) Router feature phi
        logsnr = self._build_logsnr(n_var, B, device)    # [B,1]
        phi = self._router_feat(logsnr, y, attention_mask)               # [B,4]

        # 4) Compress encoder states -> N latent tokens
        H = self._query_pool(E_lat, attention_mask)          # [B, N, d]

        # 5) Channel JSCC bottleneck (MoE + VQ)
        H_st, stats_tx = self.transceiver(
            H,
            phi,
            noise_var=n_var,
            channel_type=channel,
            hard_routing=hard_routing,
            gumbel_tau=gumbel_tau,
        )
        H_st = self.enc_up(H_st)                         # back to d_model

        # 6) Feed H_st as encoder memory to the decoder
        enc_mem = BaseModelOutput(last_hidden_state=H_st)         # BART expects this type
        enc_mem_mask = torch.ones((B, self.N), device=device)     # all valid tokens

        outputs = self.gen(
            encoder_outputs=enc_mem,
            attention_mask=enc_mem_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True,
        )

        # 7) Rate & routing info
        rate_bits = stats_tx["bits_per_block"]                    # [B]
        R = self.transceiver.R
        route_hard_tx = F.one_hot(stats_tx["expert_idx"], num_classes=R).float()
        Ns_eff = stats_tx["syms_per_block"].long()                # [B]

        if return_probs:
            route_probs = stats_tx.get("probs", None)
            return outputs, rate_bits, route_hard_tx, Ns_eff, route_probs, stats_tx

        return outputs, rate_bits, route_hard_tx, Ns_eff, stats_tx

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        attention_mask,
        n_var,
        channel: str = "AWGN",
        hard_routing: bool = True,
        **gen_kwargs,
    ):
        """
        Inference-time generation (beam search, etc.) from JSCC-corrupted memory.
        Pass usual HF generate kwargs: max_length, num_beams, do_sample, etc.
        """
        device = input_ids.device
        B = input_ids.size(0)

        enc_out = self.gen.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        E = enc_out.last_hidden_state
        E = self.enc_down(E)        
        if attention_mask is None:
            y = E.mean(dim=1)
        else:
            m = attention_mask.unsqueeze(-1).float()
            y = (E * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))

        y = self.pre_vq_norm(y)
        logsnr = self._build_logsnr(n_var, B, device)
        phi = self._router_feat(logsnr, y, attention_mask)

        H = self._query_pool(E, attention_mask)
        H_st, _ = self.transceiver(
            H, phi, noise_var=n_var, channel_type=channel, hard_routing=hard_routing
        )

        enc_mem = BaseModelOutput(last_hidden_state=H_st)
        enc_mem_mask = torch.ones((B, self.N), device=device)

        return self.gen.generate(
            encoder_outputs=enc_mem,
            attention_mask=enc_mem_mask,
            **gen_kwargs,
        )
