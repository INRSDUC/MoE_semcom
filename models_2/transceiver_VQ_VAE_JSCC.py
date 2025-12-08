
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Iterable, Callable, Dict, Any, Optional, Union
from utils_oke import PowerNormalize, Channels
import math
from transformers import AutoModel
from range_coder import RangeEncoder, RangeDecoder
import numpy as np
import os

class VQExpert(nn.Module):
    def __init__(self, num_codes: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.beta = beta

        self.codebook = nn.Embedding(num_codes, code_dim)
        # nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)
        nn.init.uniform_(self.codebook.weight,
                         -1.0 / math.sqrt(code_dim),
                         +1.0 / math.sqrt(code_dim))

    def forward(self, z_e: torch.Tensor):
        """
        Args:
            z_e: [B, N, C]
        Returns:
            z_q_st: [B, N, C] (straight-through)
            indices: [B, N]
            vq_loss: scalar
        """
        B, N, C = z_e.shape
        assert C == self.code_dim

        z_e_flat = z_e.reshape(-1, C)               # [BN, C]
        codebook = self.codebook.weight             # [K, C]

        z_sq = (z_e_flat ** 2).sum(dim=1, keepdim=True)     # [BN, 1]
        e_sq = (codebook ** 2).sum(dim=1).unsqueeze(0)      # [1, K]
        dists = z_sq + e_sq - 2 * z_e_flat @ codebook.t()   # [BN, K]

        indices = torch.argmin(dists, dim=1)        # [BN]
        z_q_flat = F.embedding(indices, codebook)   # [BN, C]

        z_q = z_q_flat.view(B, N, C)
        indices = indices.view(B, N)

        # Straight-through
        z_q_st = z_e + (z_q - z_e).detach()

        codebook_loss = F.mse_loss(z_q.detach(), z_e, reduction="mean")
        commit_loss = F.mse_loss(z_q, z_e.detach(), reduction="mean")
        vq_loss = (codebook_loss + self.beta * commit_loss) / self.code_dim
        # if self.debug:
        # with torch.no_grad():
        #         print(
        #             f"[VQExpert] ||z_e|| mean={z_e.abs().mean():.3f}, "
        #             f"max={z_e.abs().max():.3f}, "
        #             f"codebook_loss={codebook_loss.item():.3f}, "
        #             f"commit_loss={commit_loss.item():.3f}, "
        #             f"vq_loss={vq_loss.item():.3f}"
        #         )


        return z_q_st, indices, vq_loss


class VQMoE(nn.Module):
    """
    Mixture of VQ experts, now supporting per-example expert selection.

    Args:
        code_dims: list[int], K_r per expert.
        code_dim: latent dimension C.
        beta: commitment loss weight.
    """
    def __init__(self, code_dims, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_experts = len(code_dims)
        self.code_dims = list(code_dims)
        self.code_dim = code_dim
        self.experts = nn.ModuleList(
            [VQExpert(k, code_dim, beta) for k in code_dims]
        )

        # convenience: bits per index per expert
        self.bits_per_expert = [int(math.ceil(math.log2(k))) for k in self.code_dims]

    def forward(self, z_e: torch.Tensor, expert_idx: torch.Tensor):
        """
        Args:
            z_e:        [B, N, C]
            expert_idx: [B] int tensor in [0, num_experts-1]

        Returns:
            z_q:      [B, N, C]
            indices:  [B, N]
            vq_loss:  scalar (sum loss over used experts)
        """
        B, N, C = z_e.shape
        assert expert_idx.shape[0] == B

        z_q = torch.zeros_like(z_e)
        indices = torch.zeros(B, N, dtype=torch.long, device=z_e.device)
        vq_loss_total = z_e.new_tensor(0.0)
        vq_losses = []
        weights = []
        for r, expert in enumerate(self.experts):
            mask = (expert_idx == r)
            if not mask.any():
                continue
            z_in = z_e[mask]  # [B_r, N, C]
            z_q_r, idx_r, vq_loss_r = expert(z_in)
            z_q[mask] = z_q_r
            indices[mask] = idx_r
            # vq_loss_total = vq_loss_total + vq_loss_r
            frac = z_in.numel() / z_e.numel()
            vq_loss_total = vq_loss_total + frac * vq_loss_r
            vq_losses.append(vq_loss_r)
            weights.append(frac)
        if vq_losses:
            vq_loss_total = sum(w * l for w, l in zip(weights, vq_losses))
        else:
            vq_loss_total = z_e.new_tensor(0.0)
        return z_q, indices, vq_loss_total

class ModeRouter(nn.Module):
    """
    Mode router g_rt(phi; beta).

    Takes a feature vector phi (semantic summary + CSI) and outputs
    logits over J joint modes (each mode = (VQ expert, PHY mode)).
    """
    def __init__(
        self,
        in_dim: int,
        num_modes: int,
        hidden_dims=(128, 128),
        temperature: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        layers = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        layers.append(nn.Linear(last_dim, num_modes))  # logits over J modes
        self.net = nn.Sequential(*layers)

    def forward(self, phi: torch.Tensor, temperature: float | None = None):
        """
        Args:
            phi: [B, D_in] feature vectors per block.
            temperature: optional override of softmax temperature.

        Returns:
            logits: [B, J]
            probs:  [B, J] softmax over modes (for training)
        """
        logits = self.net(phi)  # [B, J]
        tau = temperature if temperature is not None else self.temperature
        probs = F.softmax(logits / tau, dim=-1)
        return logits, probs

    @torch.no_grad()
    def select_mode(self, phi: torch.Tensor):
        """
        Hard selection for inference.

        Args:
            phi: [B, D_in]

        Returns:
            mode_idx: [B] argmax over modes
        """
        logits = self.net(phi)
        mode_idx = torch.argmax(logits, dim=-1)
        return mode_idx
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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Generate a square QAM constellation with average power 1.

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

    # Normalize average power to 1
    avg_power = (points ** 2).sum(dim=-1).mean()
    points = points / torch.sqrt(avg_power + 1e-9)
    return points  # [M,2]

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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoETransceiverVQ(nn.Module):
    """
    Transmission model using VQ MoE + mode router.

    System:
    - Latent tokens H [B, N, C] are quantized by one of R VQ experts (semantic rate).
    - A mode router selects a joint mode j = (expert r, PHY mode m) per block.
    - VQ indices are mapped to bits, then to QAM symbols and sent over a channel.
    - Symbols are demapped back to bits/indices and to quantized latents H_hat.

    Notes:
    - No FEC is implemented; bits are mapped directly to QAM.
    - QAM mapping is natural (not Gray), but normalized to unit average power.
    - Channel is simple AWGN; Rayleigh/Rician can be added later.
    """

    def __init__(
        self,
        code_dims,              # list[int] K_r per expert
        code_dim: int,          # latent dim C
        phy_M_list,             # list[int] QAM orders M_m, e.g. [4, 16, 64]
        router_in_dim: int,     # feature dim D_in for router
        router_hidden=(128, 128),
        vq_beta: float = 0.25,
    ):
        super().__init__()
        # --- VQ MoE ---
        self.vq_moe = VQMoE(code_dims=code_dims, code_dim=code_dim, beta=vq_beta)
        self.R = self.vq_moe.num_experts

        # --- PHY modes ---
        self.phy_M_list = list(phy_M_list)
        self.M = len(self.phy_M_list)
        self.bits_per_phy = [int(math.log2(M)) for M in self.phy_M_list]

        # constellations as plain python list; moved to device at forward
        self.constellations = [qam_constellation(M) for M in self.phy_M_list]

        # --- Router over joint modes J = R * M ---
        self.J = self.R * self.M
        self.mode_router = ModeRouter(
            in_dim=router_in_dim,
            num_modes=self.J,
            hidden_dims=router_hidden,
            temperature=1.0,
        )
        self.mode_mapper = JointModeMapper(num_experts=self.R, num_phy_modes=self.M)

    def _awgn(self, x: torch.Tensor, noise_var) -> torch.Tensor:
        """
        Simple AWGN channel: x + N(0, noise_var).

        noise_var can be:
          - Python float
          - scalar tensor
          - vector tensor (we will take the first element)
        """
        if torch.is_tensor(noise_var):
            # ensure scalar variance
            if noise_var.numel() == 1:
                var = noise_var.item()
            else:
                # fallback: use first element (caller should ideally slice beforehand)
                var = noise_var.view(-1)[0].item()
        else:
            var = float(noise_var)

        std = math.sqrt(max(var, 0.0))
        noise = torch.randn_like(x) * std
        return x + noise



    def forward(
        self,
        H: torch.Tensor,             # [B, N, C] latent tokens from semantic encoder
        phi: torch.Tensor,           # [B, D_in] router features (e.g. SNR, semantic stats)
        noise_var: float | torch.Tensor,
        channel_type: str = "awgn",
        hard_routing: bool = True,
    ):
        """
        Returns:
            H_hat: [B, N, C] recovered latent tokens (post channel)
            stats: dict with:
                - logits: [B, J]
                - probs: [B, J]
                - mode_idx: [B]
                - expert_idx: [B]
                - phy_idx: [B]
                - vq_loss: scalar
                - bits_per_block: [B]
                - syms_per_block: [B]
        """
        B, N, C = H.shape
        device = H.device

        # --- 1) Router: pick joint mode j per block ---
        logits, probs = self.mode_router(phi)
        if hard_routing or not self.training:
            mode_idx = torch.argmax(probs, dim=-1)  # [B]
        else:
            # (optional) sample with Gumbel-softmax; here we just use argmax for simplicity
            mode_idx = torch.argmax(probs, dim=-1)

        expert_idx, phy_idx = self.mode_mapper.split(mode_idx)  # [B], [B]

        # --- 2) VQ MoE: quantize latents with selected expert ---
        z_q, indices, vq_loss = self.vq_moe(H, expert_idx=expert_idx)  # [B,N,C], [B,N]

        # --- 3) Per-example TX/RX over channel (bits -> QAM -> channel -> RX) ---

        bits_per_expert = self.vq_moe.bits_per_expert  # list[int]
        bits_per_phy = self.bits_per_phy               # list[int]

        H_hat = torch.zeros_like(H)
        bits_per_block = []
        syms_per_block = []

        for b in range(B):
            r = int(expert_idx[b].item())
            m = int(phy_idx[b].item())

            k_r = bits_per_expert[r]      # bits per VQ index
            bps = bits_per_phy[m]         # bits per QAM symbol

            idx_b = indices[b]            # [N]
            # --- TX side: indices -> bits ---
            bits_b = int_to_bits(idx_b, k_r)          # [N, k_r] in {0,1}
            B_bits = bits_b.numel()                   # scalar

            n_sym = math.ceil(B_bits / bps)
            syms_per_block.append(n_sym)
            bits_per_block.append(B_bits)

            padded_len = n_sym * bps
            bits_flat = bits_b.view(-1)
            if padded_len > B_bits:
                pad = torch.zeros(padded_len - B_bits, device=device)
                bits_flat = torch.cat([bits_flat, pad], dim=0)
            bits_sym = bits_flat.view(n_sym, bps)     # [n_sym, bps]

            # --- map bits -> QAM ---
            weights = (2 ** torch.arange(bps - 1, -1, -1, device=device)).float()
            sym_ints = (bits_sym * weights).sum(dim=-1).long().clamp(
                0, self.phy_M_list[m] - 1
            )  # [n_sym]

            const = self.constellations[m].to(device)  # [M_m, 2]
            x_sym = const[sym_ints]                    # [n_sym, 2]

            # --- channel ---
            if channel_type.lower() == "awgn":
                y_sym = self._awgn(x_sym, noise_var)
            else:
                # TODO: extend to Rayleigh/Rician; for now, treat as AWGN
                y_sym = self._awgn(x_sym, noise_var)

            # --- demap QAM -> bits ---
            # brute-force ML demapper
            const_exp = const.unsqueeze(0)                  # [1, M, 2]
            y_exp = y_sym.unsqueeze(1)                      # [n_sym, 1, 2]
            d2 = ((y_exp - const_exp) ** 2).sum(dim=-1)     # [n_sym, M]
            sym_hat = torch.argmin(d2, dim=-1)              # [n_sym]

            bits_hat_sym = int_to_bits(sym_hat, bps)        # [n_sym, bps]
            bits_hat_flat = bits_hat_sym.view(-1)[:B_bits]  # truncate padding
            bits_hat = bits_hat_flat.view(N, k_r)           # [N, k_r]

            # --- bits -> indices_hat -> codewords ---
            idx_hat = bits_to_int(bits_hat)                 # [N]
            codebook = self.vq_moe.experts[r].codebook.weight  # [K_r, C]
            H_hat[b] = F.embedding(idx_hat, codebook)       # [N, C]
        H_noisy_st = z_q + (H_hat - z_q).detach()  # [B, N, C]

        bits_per_block = torch.tensor(bits_per_block, dtype=torch.float32, device=device)
        syms_per_block = torch.tensor(syms_per_block, dtype=torch.float32, device=device)

        stats = dict(
            logits=logits,
            probs=probs,
            mode_idx=mode_idx,
            expert_idx=expert_idx,
            phy_idx=phy_idx,
            vq_loss=vq_loss,
            bits_per_block=bits_per_block,
            syms_per_block=syms_per_block,
            H_hat=H_hat.detach(),   # optional: for logging
            z_q=z_q.detach(),       # optional: for logging
        )

        return H_noisy_st, stats
    
def split_into_subsentences(
    input_ids,
    attention_mask,
    max_len: int,
    min_len: int,
    pad_token_id: int,
    keep_special: bool = True,
):
    """
    Split a tokenized sentence into contiguous subsentences (blocks).

    Args:
        input_ids:     1D LongTensor [T] (token ids for one sentence)
        attention_mask:1D LongTensor [T] (0/1 mask)
        max_len:       L, maximum tokens per subsentence (including specials if keep_special=True)
        min_len:       L_min, minimum *non-pad* tokens per subsentence (best-effort)
        pad_token_id:  tokenizer.pad_token_id
        keep_special:  if True, keeps leading <s> and trailing </s> in each block; if False, strips them.

    Returns:
        blocks_ids:   [B_s, max_len] LongTensor
        blocks_mask:  [B_s, max_len] LongTensor
    """
    # 1) strip padding
    valid = attention_mask.bool()
    ids = input_ids[valid]
    T = ids.size(0)

    # Optionally drop special tokens at ends (e.g. <s>, </s>)
    if not keep_special and T > 2:
        ids = ids[1:-1]  # drop first and last
        T = ids.size(0)

    # 2) naive chunking into max_len pieces
    chunks = []
    for start in range(0, T, max_len):
        end = min(start + max_len, T)
        chunk = ids[start:end]
        chunks.append(chunk)

    # 3) optional: merge last short chunk with previous if too small
    if len(chunks) >= 2 and chunks[-1].size(0) < min_len:
        last = chunks.pop()
        prev = chunks.pop()
        merged = torch.cat([prev, last], dim=0)
        # if merged too long, just keep as is and let it exceed min_len but not max_len
        if merged.size(0) <= max_len:
            chunks.append(merged)
        else:
            # fallback: keep prev as is, pad last later
            chunks.append(prev)
            chunks.append(last)

    # 4) pad all chunks to max_len
    blocks = []
    masks = []
    for c in chunks:
        Lc = c.size(0)
        if Lc < max_len:
            pad = torch.full((max_len - Lc,), pad_token_id, dtype=torch.long, device=c.device)
            c_padded = torch.cat([c, pad], dim=0)
            m_padded = torch.cat([
                torch.ones(Lc, dtype=torch.long, device=c.device),
                torch.zeros(max_len - Lc, dtype=torch.long, device=c.device),
            ], dim=0)
        else:
            c_padded = c[:max_len]
            m_padded = torch.ones(max_len, dtype=torch.long, device=c.device)
        blocks.append(c_padded)
        masks.append(m_padded)

    blocks_ids = torch.stack(blocks, dim=0)   # [B_s, max_len]
    blocks_mask = torch.stack(masks, dim=0)   # [B_s, max_len]
    return blocks_ids, blocks_mask

class MODJSCC_MoE_Recon_VQ(nn.Module):
    """
System model (rewired version):
- Input sentences are tokenized, embedded, and passed through a semantic encoder to produce a d_model latent vector.
- We treat this vector as a single latent token and feed it to a VQ MoE with R experts (different codebook sizes → different semantic bitrates).
- A mode router, given a small feature vector (log SNR + simple stats of the latent), selects a joint mode: (VQ expert, modulation scheme).
- The selected expert’s indices are mapped to bits, then to QAM symbols and sent over a (currently AWGN) channel via MoETransceiverVQ.
- At the receiver, symbols are demapped back to VQ indices, dequantized to latents, and passed through a reconstruction head to recover the semantic embedding.
    """

    def __init__(self,
                 d_model=256,
                 freeze_bert=False,
                 # ----- new semantic/PHY configuration -----
                #  vq_codebook_sizes=(128, 256, 512, 1024, 2048),  # K_r per VQ expert
                #  M_list=(4, 16, 64, 256, 1024),                  # QAM orders for PHY modes
                 vq_codebook_sizes=(2048),  # K_r per VQ expert
                #  M_list=(4, 16, 64, 256, 1024),                  # QAM orders for PHY modes

                 # ----- legacy args kept for compatibility (currently unused) -----
                 N_s_base=64,
                 N_z=8,
                 M_z=2,
                 ns_modes=(0.1, 0.25, 0.5, 0.75, 1.0, 1.25),
                 force_expert: int = None,
                 T_max=16,
                 pilot_P=10,
                 r_c=1,
                 O_hdr_bits=48,
                 use_soft_hdr=True):
        super().__init__()

        self.d_model = d_model
        self.force_expert = force_expert

        # ----- Semantic encoder -----
        self.encoder = RoBERTaEncoder(d_model=d_model, freeze_bert=freeze_bert)

        # ----- Reconstruction head -----
        # We keep your original 2*d_model → d_model head and just feed a dummy sigma.
        self.recon_head = nn.Sequential(
            nn.Linear(2 * d_model, 512),
            nn.ReLU(),
            nn.Linear(512, d_model)
        )

        # ----- VQ MoE + mode router transmission model -----
        # Assumes you have MoETransceiverVQ defined as in the previous message.
        self.transceiver = MoETransceiverVQ(
            code_dims=list(vq_codebook_sizes),  # K_r per expert
            code_dim=d_model,                   # latent dim C
            phy_M_list=list(M_list),           # QAM orders M_m
            router_in_dim=4,                   # [logSNR, mean|y|, mean(y^2), var(y)]
            router_hidden=(128, 128),
            vq_beta=0.25,
        )

    # --------- router feature builder (simple, no hyperprior) ---------
    @staticmethod
        # @staticmethod
    def _router_feat(logsnr: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Build a 4D feature vector per sample, focused on semantic difficulty:
        - log SNR                         (channel condition)
        - log ||y||_2                     (embedding magnitude / salience)
        - entropy(softmax(y))             (semantic uncertainty / complexity)
        - var(y / ||y||_2)                (shape / spread of semantic info)

        Args:
            logsnr: [B, 1]
            y:      [B, d_model]  semantic embedding

        Returns:
            phi: [B, 4]
        """
        eps = 1e-8

        # L2 norm of embedding
        y_norm = torch.norm(y, dim=1, keepdim=True)  # [B,1]
        log_norm = torch.log(y_norm + eps)           # [B,1]

        # Semantic "uncertainty": entropy over softmax(y)
        p = F.softmax(y, dim=1)                      # [B, d_model]
        entropy = -(p * torch.log(p + eps)).sum(dim=1, keepdim=True)  # [B,1]

        # Normalize embedding and look at variance across dims
        y_unit = y / (y_norm + eps)
        var_y = y_unit.var(dim=1, unbiased=False, keepdim=True)       # [B,1]

        # Final feature vector
        phi = torch.cat([logsnr, log_norm, entropy, var_y], dim=1)    # [B,4]
        return phi

    # --------- forward ---------
    def forward(self, input_ids, attention_mask, n_var,
                channel: str = 'AWGN',
                return_probs: bool = False,
                snr_seq=None,  # kept for signature compatibility, not used
                tau: float = 1.0):  # kept for signature compatibility
        """
        Outputs (reconstruction task, compatible shape with old version):
          recon         [B, d_model]
          rate_bits     [B]       -- now: actual bits sent per block
          route_hard_tx [B, R]    -- one-hot over VQ experts used
          Ns_eff        [B]       -- effective # channel symbols used
          (if return_probs=True)
              route_probs [B, J]  -- soft probs over joint modes
              None               -- placeholder to match old API
              stats dict         -- includes vq_loss, etc.
          (else)
              stats dict
        """
        B = input_ids.size(0)
        device = input_ids.device

        # 1) Semantic encoder: target embedding to reconstruct
        y = self.encoder(input_ids, attention_mask)  # [B, d_model]

        # Treat as a single latent token: H [B, 1, d_model]
        H = y.unsqueeze(1)

        # 2) Build SNR feature
        if torch.is_tensor(n_var):
            n_var = n_var.to(device)
            if n_var.ndim == 0:
                logsnr_val = torch.log(1.0 / n_var).item()
                logsnr = torch.full((B, 1), logsnr_val, device=device)
            else:
                # assume shape [B] or [B, ...]
                n_var_flat = n_var.view(B, -1)[:, 0]
                logsnr = torch.log(1.0 / n_var_flat).view(-1, 1)
        else:
            logsnr = torch.full((B, 1), math.log(1.0 / n_var), device=device)

        # 3) Router features
        phi = self._router_feat(logsnr, y)  # [B, 4]

        # 4) Transmission through VQ MoE + QAM channel
        H_hat, stats_tx = self.transceiver(
            H,
            phi,
            noise_var=n_var,
            channel_type=channel,
            hard_routing=True,
        )  # H_hat: [B, 1, d_model]

        # Pool over tokens (here N=1, so just squeeze)
        feat_pooled = H_hat.squeeze(1)  # [B, d_model]

        # 5) Dummy sigma_rec (we're not using hyperprior anymore)
        sigma_rec = torch.zeros_like(feat_pooled)

        # 6) Reconstruction head to predict y_hat
        recon = self.recon_head(torch.cat([feat_pooled, sigma_rec], dim=1))  # [B, d_model]

        # 7) "Rate" = actual number of bits transmitted (per block)
        rate_bits = stats_tx["bits_per_block"]  # [B]

        # 8) One-hot of selected VQ expert (semantic mode)
        R = self.transceiver.R  # number of VQ experts
        route_hard_tx = F.one_hot(stats_tx["expert_idx"], num_classes=R).float()  # [B, R]

        # 9) Effective number of channel symbols used
        Ns_eff = stats_tx["syms_per_block"].long()  # [B]

        # 10) Expose some targets + features in stats dict (like old sched)
        stats_tx["y_target"] = y.detach()
        stats_tx["feat_pooled"] = feat_pooled

        if return_probs:
            # Soft probs over joint modes J = R * M
            route_probs = stats_tx["probs"]  # [B, J]
            return recon, rate_bits, route_hard_tx, Ns_eff, route_probs, None, stats_tx
        else:
            return recon, rate_bits, route_hard_tx, Ns_eff, stats_tx
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        vq_codebook_sizes=(128, 256, 512, 1024, 2048),
        # vq_codebook_sizes=(2048),
        phy_M_list=(4, 16, 64, 256, 1024),
        # phy_M_list=(64),
    ):
        super().__init__()

        self.d_model = d_model
        self.num_labels = num_labels
        self.N = N

        # ----- Semantic encoder -----
        self.encoder = RoBERTaEncoder(d_model=d_model, freeze_bert=freeze_bert)

        # ----- VQ MoE + PHY transceiver -----
        # IMPORTANT: MoETransceiverVQ should return H_st (VQ-VAE ST across channel),
        # not the raw post-channel H_hat. See earlier transceiver fix.
        self.pre_vq_norm = nn.LayerNorm(d_model)
        self.transceiver = MoETransceiverVQ(
            code_dims=list(vq_codebook_sizes),  # K_r per expert
            code_dim=d_model,                   # latent dim
            phy_M_list=list(phy_M_list),        # QAM sizes
            router_in_dim=4,                    # feature dim (defined below)
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
    def _router_feat(logsnr: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        4D feature per sample, focused on semantic difficulty:
        - log SNR
        - log ||y||_2
        - entropy(softmax(y))
        - var(y / ||y||_2)

        Args:
            logsnr: [B, 1]
            y:      [B, d_model]
        Returns:
            phi: [B, 4]
        """
        eps = 1e-8

        # L2 norm and log-norm
        y_norm = torch.norm(y, dim=1, keepdim=True)                # [B,1]
        log_norm = torch.log(y_norm + eps)                         # [B,1]

        # Softmax entropy as a proxy for semantic uncertainty/complexity
        p = F.softmax(y, dim=1)                                    # [B, d_model]
        entropy = -(p * torch.log(p + eps)).sum(dim=1, keepdim=True)  # [B,1]

        # Normalized embedding variance
        y_unit = y / (y_norm + eps)
        var_y = y_unit.var(dim=1, unbiased=False, keepdim=True)    # [B,1]

        phi = torch.cat([logsnr, log_norm, entropy, var_y], dim=1) # [B,4]
        return phi

    def forward(
        self,
        input_ids,
        attention_mask,
        n_var,
        channel: str = "AWGN",
        return_probs: bool = False,
    ):
        """
        Args:
            input_ids:      [B, L]
            attention_mask: [B, L]
            n_var:          scalar or [B] noise variance
            channel:        'AWGN' (for now)
            return_probs:   if True, also return joint-mode probabilities.

        Returns:
            logits:        [B, num_labels]
            rate_bits:     [B]       actual bits per block
            route_hard_tx: [B, R]    one-hot over VQ experts
            Ns_eff:        [B]       number of channel symbols
            (if return_probs)
                route_probs: [B, J]
                None         (for API compatibility)
                stats        dict
            else:
                stats        dict
        """
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
        phi = self._router_feat(logsnr, y)  # [B, 4]

        # 4) Build latent tokens H: [B, N, d_model]
        H = self.latent_proj(y).view(B, self.N, self.d_model)

        # 5) Transceiver: JSCC with MoE + VQ (returns VQ-VAE ST latent H_st)
        H_st, stats_tx = self.transceiver(
            H,
            phi,
            noise_var=n_var,
            channel_type=channel,
            hard_routing=True,
        )  # H_st: [B, N, d_model]

        # 6) Pool over latent tokens (mean pool) → [B, d_model]
        feat_pooled = H_st.mean(dim=1)

        # 7) Classification logits
        logits = self.cls_head(feat_pooled)  # [B, num_labels]

        # 8) Rate & routing info
        rate_bits = stats_tx["bits_per_block"]          # [B]
        R = self.transceiver.R
        route_hard_tx = F.one_hot(stats_tx["expert_idx"], num_classes=R).float()
        Ns_eff = stats_tx["syms_per_block"].long()      # [B]

        # For convenience / later tasks
        stats_tx["y_target"] = y.detach()
        stats_tx["feat_pooled"] = feat_pooled

        if return_probs:
            route_probs = stats_tx["probs"]  # [B, J]
            return logits, rate_bits, route_hard_tx, Ns_eff, route_probs, None, stats_tx
        else:
            return logits, rate_bits, route_hard_tx, Ns_eff, stats_tx
