
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
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.3):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x) 
        return x

def freeze_layers(bert_model, num_layers_to_freeze):
        for layer_num, layer in enumerate(bert_model.encoder.layer):
                if layer_num < num_layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False

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

def discrete_probability(y_tilde: torch.Tensor,
                         mu: torch.Tensor,
                         sigma: torch.Tensor,
                         eps: float = 1e-12) -> torch.Tensor:
    # y_tilde, mu, sigma: [B, D]
    # print(y_tilde.shape)
    # print(mu.shape)
    lower = (y_tilde - 0.5 - mu) / (sigma * math.sqrt(2))
    upper = (y_tilde + 0.5 - mu) / (sigma * math.sqrt(2))
    p = 0.5 * (torch.erf(upper) - torch.erf(lower))
    
    return p.clamp(min=eps)
def gumbel_sigmoid(logits, τ=1.0, hard=True):
    """Differentiable binary quantization."""
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y = torch.sigmoid((logits + g) / τ)
    if hard:
        return (y>0.5).float() + (y - y.detach())
    return y


def map_to_constellation(bits: torch.Tensor, M: int) -> torch.Tensor:
    """
    bits: Tensor[..., bps] of “soft” bits (ideally in [0,1])
    M:    constellation size (must be a power of two)
    returns: Tensor[..., 2] of IQ points, unit-avg-power
    """
    # no in-place ops on grad-carrying tensors
    bits = bits.to(dtype=torch.float32)
    bits = bits.clamp(0.0, 1.0)   # OUT-OF-PLACE

    bps = bits.size(-1)
    if (1 << bps) != M:
        raise ValueError(f"Constellation-size mismatch: M={M} but bits-per-symbol={bps}")

    if bps == 1:  # BPSK
        I = bits[..., 0] * 2.0 - 1.0
        Q = torch.zeros_like(I)
        return torch.stack([I, Q], dim=-1)

    # Rectangular/square QAM
    bps_I = bps // 2
    bps_Q = bps - bps_I

    def axis_int(x, nbits):
        if nbits == 0:
            return torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        w = 2 ** torch.arange(nbits - 1, -1, -1, device=x.device, dtype=x.dtype)
        return (x * w).sum(dim=-1)

    I_int = axis_int(bits[..., :bps_I], bps_I)
    Q_int = axis_int(bits[..., bps_I:], bps_Q)

    L_I = float(2 ** bps_I)
    L_Q = float(2 ** bps_Q)
    I_lvl = 2.0 * I_int + 1.0 - L_I
    Q_lvl = 2.0 * Q_int + 1.0 - L_Q

    raw_power = (L_I * L_I - 1.0) / 3.0 + (L_Q * L_Q - 1.0) / 3.0
    norm = math.sqrt(raw_power) if raw_power > 1e-6 else 1.0

    return torch.stack([I_lvl / norm, Q_lvl / norm], dim=-1)

class SimpleChannelDecoder(nn.Module):
    def __init__(self, in_dim, D):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, D)
        )
    def forward(self, rx):
        # rx: [B, 2*N_s]
        return self.net(rx)
    
class HyperPrior(nn.Module):
    def __init__(self, d_model, num_modulations = 1):  # num_modulations = 3
        super().__init__()
        self.d_model = d_model
        self.num_modulations = num_modulations

        self.encoder = nn.Sequential(
            nn.Linear(d_model + 1, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * d_model + num_modulations)
        )

    def forward(self, y, n_var, training=True):
        B, D = y.shape
        snr_feat = torch.log(1.0 / n_var).unsqueeze(1)  # [B,1]
        z = self.encoder(torch.cat([y, snr_feat], dim=1))  # [B, D]

        z_tilde = z + torch.rand_like(z) - 0.5 if training else torch.round(z)

        params = self.decoder(z_tilde)  # [B, 2D + num_modulations]
        mu, raw_sigma, mod_logits = params.split(
            [D, D, self.num_modulations], dim=1
        )

        sigma = F.softplus(raw_sigma)
        return z_tilde, mu, sigma, mod_logits



from range_coder import RangeEncoder, prob_to_cum_freq
class SemanticRouter(nn.Module):
    def __init__(self, d_in, n_experts, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_experts)
        )
        self.tau = 1.0  # anneal during training

    def forward(self, feat, hard=True):
        # feat: [B, d_in]
        logits = self.mlp(feat) / self.tau                     # [B, E]
        if self.training:
            g = -torch.log(-torch.log(torch.rand_like(logits)))  # Gumbel noise
            probs = F.softmax(logits + g, dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)

        if not hard:
            return probs, probs

        # Straight-through hard one-hot
        idx = probs.argmax(dim=-1)                              # [B]
        hard_onehot = F.one_hot(idx, num_classes=probs.size(-1)).float()
        hard_st = (hard_onehot - probs).detach() + probs
        return hard_st, probs  # (forward is hard, backward is soft)
    

# add at top of file (outside the class)
# --- helpers (put near your other utils) --------------------------------------
import math, torch, torch.nn.functional as F

def _phi_inv(p):  # Φ^{-1}(p) via erfinv
    return math.sqrt(2.0) * torch.erfinv(2.0 * torch.as_tensor(p) - 1.0)

def _adaptive_Q_from_sigma(sigma, eps=1e-6):
    # sigma: [B,D]; returns integer Q per element
    z = _phi_inv(1.0 - eps/2.0).to(sigma.device).to(sigma.dtype)  # scalar tensor
    Q = torch.ceil(sigma * z - 0.5).clamp(min=0).to(torch.int32)
    return Q  # [B,D]

def _disc_gauss_pmf(mu, sigma, q):
    """
    Bucketed version: q is a scalar (int or 0-d tensor) defining symmetric support [-q..q].
    mu, sigma: [N] (or broadcastable); returns:
      pmf: [N, 2q+1]  for bins k in [-q..q]
      tail: [N]       leftover mass outside [-q..q]
      Qmax: int == q
      L: int == 2q+1
    """
    import math, torch
    qv = int(q if isinstance(q, int) else q.item())
    device, dtype = mu.device, mu.dtype
    # CDF at bin edges k+0.5 for k in [-q-1..q]
    k = torch.arange(-qv-1, qv+1, device=device, dtype=dtype) + 0.5  # len 2q+2
    z = (k.view(1, -1) - mu.unsqueeze(-1)) / (sigma.unsqueeze(-1) + 1e-9)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    cdf = torch.clamp(cdf, 1e-9, 1.0 - 1e-9)
    # pmf for centers [-q..q]
    pmf = cdf[:, 1:] - cdf[:, :-1]            # [N, 2q+1]
    pmf = torch.clamp(pmf, 1e-12, 1.0)
    tail = (1.0 - pmf.sum(dim=-1)).clamp_min(0.0)  # [N]
    Qmax = qv
    L = pmf.size(-1)
    return pmf, tail, Qmax, L

def _pmf_to_cdf_with_escape(pmf, tail):
    # pmf: [N, L], tail: [N]
    pmf_ext = torch.cat([pmf, tail.view(-1, 1)], dim=-1)      # [N, L+1]
    cdf = torch.cumsum(pmf_ext, dim=-1)                       # [N, L+1]
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)  # [N, L+2]
    cdf[:, -1] = 1.0
    cdf = torch.clamp(cdf, 1e-9, 1.0)
    cdf = torch.cummax(cdf, dim=-1).values                    # enforce non-decreasing
    return cdf

def _encode_bucket_torchac(symbols, cdf_float):
    try:
        import torchac
    except Exception:
        return None
    N = symbols.numel()
    bytestream = torchac.encode_float_cdf(cdf_float.reshape(N, -1).contiguous(),
                                          symbols.view(-1).to(torch.int32))
    return 8.0 * len(bytestream)

def _rice_len(n, k):  # bit length of unsigned n with Rice(k)
    # unary(q) + stop + k remainder bits
    q = n >> k
    return int(q + 1 + k)

# --- end helpers --------------------------------------------------------------



# class MODJSCC_WithHyperprior_real_bit_MoE(nn.Module):
#     def __init__(self, d_model=256, freeze_bert=False, N_s=32, N_z=8, M_list=[4,8,16,32,64,128], M_z=2, force_expert: int=None):
#         super().__init__()
#         self.d_model = d_model
#         self.N_s = N_s
#         self.N_z = N_z
#         self.M_list = M_list
#         self.bps_list = [int(math.log2(M)) for M in M_list]
#         self.K = len(M_list)
#         self.force_expert = force_expert
#         self.M_z = M_z
#         self.bps_z = int(math.log2(M_z))

#         # Semantic encoder
#         self.encoder = RoBERTaEncoder(d_model=d_model, freeze_bert=freeze_bert)

#         # Hyperprior MLPs
#         self.hyper_encoder = nn.Sequential(
#             nn.Linear(d_model + 1, 128), nn.ReLU(), nn.Linear(128, d_model)
#         )
#         # NOTE: hyper_decoder now ONLY outputs sigma params (no routing logits here)
#         self.hyper_decoder = nn.Sequential(
#             nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, d_model)
#         )

#         # Side-channel and main-channel encoders/decoders
#         self.hyper_channel_encoder = nn.Linear(d_model, N_z * self.bps_z)   # bits
#         self.hyper_channel_decoder = nn.Linear(N_z * 2, d_model)            # from 2-D symbols

#         self.channel_encoders = nn.ModuleList([
#             nn.Linear(d_model, N_s * bps) for bps in self.bps_list
#         ])
#         self.channel_decoders = nn.ModuleList([
#             nn.Linear(2 * N_s, d_model) for _ in self.bps_list
#         ])

#         # --- NEW: semantic-aware router (inputs: SNR + z stats) ---
#         # feature = [logSNR, mean(|z|), mean(log_sigma+), var(log_sigma+)]

#         self.router = SemanticRouter(d_in=4, n_experts=self.K)

#         # Classifier head (condition on sigma)
#         self.decoder = nn.Sequential(
#             nn.Linear(2 * d_model, 256), nn.ReLU(), nn.Linear(256, 2)
#         )
#         self.validate_entropy = False   # run real coder at eval for monitoring
#     @staticmethod
#     def _build_router_feat(snr_feat, z_like, sigma_like):
#         # snr_feat: [B,1]; z_like, sigma_like: [B, D]
#         B = snr_feat.size(0)
#         mean_abs_z = z_like.abs().mean(dim=1, keepdim=True)               # [B,1]
#         logsig = torch.log1p(F.softplus(sigma_like).abs() + 1e-6)         # [B,D]
#         mu_logsig = logsig.mean(dim=1, keepdim=True)                      # [B,1]
#         var_logsig = logsig.var(dim=1, unbiased=False, keepdim=True)      # [B,1]
#         feat = torch.cat([snr_feat, mean_abs_z, mu_logsig, var_logsig], dim=1)  # [B,4]
        
#         return feat
#     @torch.no_grad()
#     def probe_experts(self, input_ids, attention_mask, n_var, channel='AWGN',
#                     labels=None, share_noise=True):
#         """
#         Returns dict with:
#         - logits_all: [B, num_classes, K]
#         - acc_per_expert: [K] (if labels is not None)
#         - bps_list: list[int]
#         Uses the *same* z side-channel for all experts; for AWGN can share noise across experts.
#         """
#         self.eval()
#         B = input_ids.size(0)
#         device = input_ids.device
#         chan = Channels()

#         # 1) Encode & hyperprior (shared across experts)
#         y = self.encoder(input_ids, attention_mask)                      # [B, d_model]
#         snr_feat = torch.log(1.0/n_var).view(-1,1) if torch.is_tensor(n_var) \
#                 else torch.full((B,1), math.log(1.0/n_var), device=device)
#         z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))
#         z_tilde = z.round()
#         raw_sigma = self.hyper_decoder(z_tilde)
#         sigma = F.softplus(raw_sigma) + 1e-6

#         # Side-channel symbols (shared)
#         z_bits = self.hyper_channel_encoder(z_tilde)                     # [B, N_z*bps_z]
#         z_syms = map_to_constellation(z_bits.view(B, self.N_z, self.bps_z), self.M_z)  # [B,N_z,2]
#         z_flat = z_syms.view(B, -1)                                      # [B, 2*N_z]

#         # 2) Build all expert symbol streams from y (no mixing)
#         y_tilde = y.round()
#         syms_y_all = []
#         for i, bps in enumerate(self.bps_list):
#             bits_y = self.channel_encoders[i](y_tilde)                   # [B, N_s*bps]
#             bits_y = torch.sigmoid(bits_y)                               # eval: no gumbel
#             syms_y = map_to_constellation(bits_y.view(B, self.N_s, bps), self.M_list[i])  # [B,N_s,2]
#             syms_y_all.append(syms_y.view(B, -1))                        # [B, 2*N_s]
#         Sy = torch.stack(syms_y_all, dim=2)                               # [B, 2*N_s, K]

#         logits_all = []
#         acc_per_expert = []

#         # Optional shared AWGN noise
#         if channel == 'AWGN' and share_noise:
#             Tx_template = PowerNormalize(torch.cat([z_flat, Sy[:,:,0]], dim=1))
#             # std per real dim = sqrt(n_var/2)
#             if torch.is_tensor(n_var):
#                 std = (n_var.view(B,1)/2.0).clamp_min(1e-12).sqrt()
#             else:
#                 std = math.sqrt(float(n_var)/2.0)
#             noise = torch.randn_like(Tx_template) * std                  # [B, 2*N_z+2*N_s]
#         else:
#             noise = None

#         for e in range(self.K):
#             Tx_y_e = Sy[:,:,e]                                           # [B, 2*N_s]
#             Tx_e = PowerNormalize(torch.cat([z_flat, Tx_y_e], dim=1))    # [B, 2*N_z+2*N_s]

#             # Channel
#             if channel == 'AWGN':
#                 if share_noise and noise is not None:
#                     Rx_e = Tx_e + noise
#                 else:
#                     Rx_e = chan.AWGN(Tx_e, n_var)
#             elif channel == 'Rayleigh':
#                 Rx_e = chan.Rayleigh(Tx_e, n_var)
#             else:
#                 Rx_e = chan.Rician(Tx_e, n_var)

#             # Split & decode for expert e only
#             y_dim = self.N_s * 2
#             z_rx_e, y_rx_e = Rx_e[:, :-y_dim], Rx_e[:, -y_dim:]

#             z_hat_e = self.hyper_channel_decoder(z_rx_e)
#             sigma_rec_e = F.softplus(self.hyper_decoder(z_hat_e)) + 1e-6

#             feat_e = self.channel_decoders[e](y_rx_e)                    # [B, d_model]
#             logits_e = self.decoder(torch.cat([feat_e, sigma_rec_e], dim=1))  # [B, C]
#             logits_all.append(logits_e)

#             if labels is not None:
#                 acc = (logits_e.argmax(1) == labels).float().mean().item()
#                 acc_per_expert.append(acc)

#         logits_all = torch.stack(logits_all, dim=-1)  # [B, C, K]

#         out = {"logits_all": logits_all, "bps_list": self.bps_list}
#         if labels is not None:
#             out["acc_per_expert"] = acc_per_expert
#         return out

#     def forward(self, input_ids, attention_mask, n_var, channel):

#             B = input_ids.size(0)
#             device = input_ids.device
#             chan = Channels()

#             # 1) Encode x -> y
#             y = self.encoder(input_ids, attention_mask)
#             assert not torch.isnan(y).any(), "NaN in encoder output y"

#             # 2) Hyperprior z from y and SNR
#             snr_feat = torch.log(1.0 / n_var).view(-1,1) if torch.is_tensor(n_var) else \
#                     torch.full((B,1), math.log(1.0/n_var), device=device)
#             z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))
#             assert not torch.isnan(z).any(), "NaN in hyper_encoder output z"

#             # 3) Quantize z (uniform noise during train)
#             z_tilde = z + (torch.rand_like(z) - 0.5) if self.training else z.round()

#             # 4) Predict sigma from z_tilde
#             raw_sigma = self.hyper_decoder(z_tilde)
#             sigma = F.softplus(raw_sigma) + 1e-6 #add a small value to avoid zero# used in router

#             # 5) ROUTER (TX): use only SNR + z-stats ⇒ reproducible at RX
#             feat_tx = self._build_router_feat(snr_feat, z_tilde, raw_sigma)
#             route_hard_tx, route_probs_tx = self.router(feat_tx, hard=True)   # [B,K] each
#             # route_hard_tx is one-hot in forward (clear selection)\

#             if self.force_expert is not None:
#                 onehot = F.one_hot(torch.full((B,), self.force_expert, device=input_ids.device),
#                                 num_classes=self.K).float()
#                 route_hard_tx = onehot
#                 # keep a soft copy for logs (optional)
#                 route_probs_tx = onehot
#             # 6) Quantize y
#             y_tilde = y + (torch.rand_like(y) - 0.5) if self.training else y.round()

#             # 7) Side-channel: encode z to symbols
#             z_bits = self.hyper_channel_encoder(z_tilde)                      # [B, N_z*bps_z]
#             z_syms = map_to_constellation(z_bits.view(B, self.N_z, self.bps_z), self.M_z)  # [B,N_z,2]
#             # z_syms = torch.nan_to_num(z_syms, nan=0.0, posinf=0.0, neginf=0.0) #no nan BUT no need
#             z_flat = z_syms.view(B, -1)                                       # [B, 2*N_z]

#             # 8) Main-channel: compute all expert symbol streams, then SELECT via one-hot
#             syms_y_all = []
#             for i, bps in enumerate(self.bps_list):
#                 bits_y = self.channel_encoders[i](y_tilde)                    # [B, N_s*bps]
#                 bits_y = gumbel_sigmoid(bits_y, τ=1.0, hard=False)
#                 syms_y = map_to_constellation(bits_y.view(B, self.N_s, bps), self.M_list[i])  # [B,N_s,2]
#                 syms_y_all.append(syms_y.view(B, -1))                         # [B, 2*N_s]
#             Sy = torch.stack(syms_y_all, dim=-1)                               # [B, 2*N_s, K]

#             # Clear selection (no mixing): pick exactly one expert’s symbols
#             route_hard_tx_expanded = route_hard_tx.unsqueeze(1)                # [B,1,K]
#             Tx_y = (Sy * route_hard_tx_expanded).sum(dim=-1)                   # [B, 2*N_s]
#             assert not torch.isnan(Tx_y).any(), "NaN in Tx_y"

#             # 9) Transmit concatenated symbols
#             Tx = PowerNormalize(torch.cat([z_flat, Tx_y], dim=1))              # [B, 2*N_z + 2*N_s]
#             if channel == 'AWGN':
#                 Rx = chan.AWGN(Tx, n_var)
#             elif channel == 'Rayleigh':
#                 Rx = chan.Rayleigh(Tx, n_var)
#             elif channel == 'Rician':
#                 Rx = chan.Rician(Tx, n_var)
#             else:
#                 raise ValueError("Invalid channel type")
#             assert not torch.isnan(Rx).any(), "NaN in Rx"

#             # 10) Split received into z_rx and y_rx
#             y_dim = self.N_s * 2
#             split_at = Rx.size(1) - y_dim
#             z_rx = Rx[:, :split_at]       # [B, 2*N_z]
#             y_rx = Rx[:, split_at:]       # [B, 2*N_s]
#             assert y_dim == self.channel_decoders[0].in_features, \
#                 f"channel decoder expects {self.channel_decoders[0].in_features}, got {y_dim}"

#             # 11) Decode hyperprior at RX, recompute ROUTER (deterministic from z_hat & SNR)
#             z_hat = self.hyper_channel_decoder(z_rx)                           # [B, d_model]
#             raw_sigma_rec = self.hyper_decoder(z_hat)                          # [B, d_model]
#             sigma_rec = F.softplus(raw_sigma_rec) + 1e-6

#             feat_rx = self._build_router_feat(snr_feat, z_hat, raw_sigma_rec)
#             route_hard_rx, route_probs_rx = self.router(feat_rx, hard=True)    # [B,K]
#             if self.force_expert is not None:
#                 onehot = F.one_hot(torch.full((B,), self.force_expert, device=input_ids.device),
#                                 num_classes=self.K).float()
#                 route_hard_rx = onehot
#                 route_probs_rx = onehot
#             # 12) Decode y via the SELECTED expert only (no mixing)
#             dec_all = torch.stack([dec(y_rx) for dec in self.channel_decoders], dim=-1)  # [B,d_model,K]
#             feat = (dec_all * route_hard_rx.unsqueeze(1)).sum(dim=-1)          # [B, d_model]

#             # 13) Classification conditioned on sigma
#             feat_cat = torch.cat([feat, sigma_rec], dim=1)                     # [B, 2*d_model]
#             logits = self.decoder(feat_cat)

#             # 14) Rate-loss (unchanged)
#             p_y = discrete_probability(y_tilde, torch.zeros_like(y_tilde), sigma_rec)
#             # rate_y = -torch.log2(p_y + 1e-9).sum(dim=1).mean()
#             p_z = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
#             # rate_z = -torch.log2(p_z + 1e-9).sum(dim=1).mean()
#             # rate_loss = rate_y + rate_z

#             bits_y = -torch.log2(p_y + 1e-9).sum(dim=1)  # [B]
#             bits_z = -torch.log2(p_z + 1e-9).sum(dim=1)  # [B]
#             rate_bits = bits_y + bits_z                   # [B]
#             rate_loss = rate_bits.mean()  


#             val_bits = None
#             if (not self.training):
#                 with torch.no_grad():
#                     eps = 1e-6
#                     # 1) Integers
#                     y_int = y.round().to(torch.int32)             # [B, D]
#                     z_int = z.round().to(torch.int32)             # [B, D]

#                     # 2) Adaptive supports from sigma (and unit for z)
#                     sigma_y = F.softplus(raw_sigma) + 1e-6        # [B, D]
#                     Qy = _adaptive_Q_from_sigma(sigma_y, eps)      # [B, D]
#                     Qz = _adaptive_Q_from_sigma(torch.ones_like(z), eps)  # effectively constant here

#                     # 3) Flatten per-position for bucketing
#                     B, Dy = y_int.shape
#                     Nz = z_int.numel()
#                     y_flat = y_int.view(-1)
#                     z_flat = z_int.view(-1)
#                     Qy_flat = Qy.view(-1)
#                     Qz_flat = Qz.view(-1)
#                     mu0_y = torch.zeros_like(y_flat, dtype=y.dtype, device=y.device)
#                     mu0_z = torch.zeros_like(z_flat, dtype=z.dtype, device=z.device)
#                     sig_y_flat = sigma_y.view(-1)

#                     # 4) Bucket by unique Q to keep CDF size per call constant
#                     bits_y_total = 0.0
#                     for q in torch.unique(Qy_flat):
#                         mask = (Qy_flat == q)
#                         if not mask.any(): continue
#                         qv = int(q.item())
#                         # Build pmf/CDF for this bucket (truncated Gaussian + ESC)
#                         pmf, tail, Qmax, L = _disc_gauss_pmf(mu0_y[mask], sig_y_flat[mask], q)
#                         cdf = _pmf_to_cdf_with_escape(pmf, tail)  # [Nmask, 2*qv+3]
#                         # Map symbols → indices: [-q..q]→[0..2q], ESC→2q+1
#                         v = y_flat[mask].to(torch.int32)
#                         idx = v.clamp(-qv, qv) + qv
#                         esc = (v.abs() > qv)
#                         idx[esc] = 2*qv + 1  # ESC position
#                         # Encode pmf-part
#                         bits_bucket = _encode_bucket_torchac(idx, cdf)
#                         if bits_bucket is None: bits_bucket = 0.0
#                         bits_y_total += float(bits_bucket)
#                         # Add residual cost for ESC (simple Rice count; not actually coded here)
#                         if esc.any():
#                             k = 2  # choose a small Rice parameter
#                             r = (v.abs() - qv)[esc].to(torch.int64)   # unsigned residual
#                             # +1 sign bit per escape
#                             bits_res = sum(_rice_len(int(n.item()), k) + 1 for n in r)
#                             bits_y_total += bits_res

#                     # Same for z (often Qz is constant; you can do a single bucket)
#                     bits_z_total = 0.0
#                     for q in torch.unique(Qz_flat):
#                         mask = (Qz_flat == q)
#                         if not mask.any(): continue
#                         qv = int(q.item())
#                         pmf, tail, Qmax, L = _disc_gauss_pmf(mu0_z[mask], torch.ones_like(mu0_z[mask]), q)
#                         cdf = _pmf_to_cdf_with_escape(pmf, tail)
#                         v = z_flat[mask].to(torch.int32)
#                         idx = v.clamp(-qv, qv) + qv
#                         esc = (v.abs() > qv)
#                         idx[esc] = 2*qv + 1
#                         bits_bucket = _encode_bucket_torchac(idx, cdf)
#                         if bits_bucket is None: bits_bucket = 0.0
#                         bits_z_total += float(bits_bucket)
#                         if esc.any():
#                             k = 2
#                             r = (v.abs() - qv)[esc].to(torch.int64)
#                             bits_res = sum(_rice_len(int(n.item()), k) + 1 for n in r)
#                             bits_z_total += bits_res

#                     val_bits = {"bits_y": bits_y_total, "bits_z": bits_z_total,
#                             "bits_total": bits_y_total + bits_z_total}
#             return logits, rate_loss, route_probs_rx, val_bits


class MODJSCC_MoE_Faithful(nn.Module):
    """
    Clean forward:
      - Hard modulation (top-1 ST) for the channel path
      - Hard airtime Ns (discrete modes via ST)
      - Entropy model returns per-sample rate_bits (for training)
      - Eval helper returns true bitcounts (val_bits) when needed
    """
    def __init__(self, d_model=256, freeze_bert=False,
                 N_s_base=64, N_z=8,
                 M_list=[64]#, 8, 16, 32, 64, 128)
                 , M_z=2,
                 ns_modes=(0.1, 0.25, 0.5, 0.75, 1.0, 1.25),
                 force_expert: int=None):
        super().__init__()
        self.d_model = d_model
        self.N_s_base = N_s_base #starting point for Ns
        self.N_z = N_z
        self.M_list = list(M_list)
        self.bps_list = [int(math.log2(M)) for M in self.M_list]
        self.K = len(self.M_list)
        self.force_expert = force_expert
        self.M_z = M_z
        self.bps_z = int(math.log2(M_z))
        self.ns_modes = torch.tensor(ns_modes, dtype=torch.float32)  # e.g., [0.5,1,1.5,2]

        # ----- Semantic encoder / decoder -----
        self.encoder = RoBERTaEncoder(d_model=d_model, freeze_bert=freeze_bert)
        self.decoder = nn.Sequential(nn.Linear(2*d_model, 256), nn.ReLU(),
                                     nn.Linear(256, 2))

        # ----- Hyperprior (entropy model) -----
        self.hyper_encoder = nn.Sequential(nn.Linear(d_model + 1, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, d_model))
        self.hyper_decoder = nn.Sequential(nn.Linear(d_model, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, d_model))

        # Side-channel mapper (z)
        self.hyper_channel_encoder = nn.Linear(d_model, N_z * self.bps_z)   # bits
        self.hyper_channel_decoder = nn.Linear(2 * N_z, d_model)            # from IQ symbols

        # Main-channel per-expert bit mappers and decoders
        self.channel_encoders = nn.ModuleList([nn.Linear(d_model, N_s_base * bps)
                                               for bps in self.bps_list])
        self.channel_decoders = nn.ModuleList([nn.Linear(2 * N_s_base, d_model)
                                               for _ in self.bps_list])

        # ----- Router heads -----
        # feature: [logSNR, mean|z|, mean logσ+, var logσ+]
        self.router = SemanticRouter(d_in=4, n_experts=self.K)  # returns (onehot, probs)
        self.ns_head = nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                     nn.Linear(64, len(ns_modes)))  # logits for Ns modes

    # --------- utilities ---------
    @staticmethod
    def _router_feat(snr_feat, z_like, sigma_like):
        B = snr_feat.size(0)
        mean_abs_z = z_like.abs().mean(dim=1, keepdim=True)
        logsig = torch.log1p(F.softplus(sigma_like).abs() + 1e-6)
        mu_logsig = logsig.mean(dim=1, keepdim=True)
        var_logsig = logsig.var(dim=1, unbiased=False, keepdim=True)
        return torch.cat([snr_feat, mean_abs_z, mu_logsig, var_logsig], dim=1)  # [B,4]

    @staticmethod
    def _norm_used_then_pad(y_syms, k):
        # y_syms: [Ns_base, 2]; k may be out of range -> clamp
        Ns = y_syms.size(0)
        k = max(1, min(int(k), Ns))
        used = y_syms[:k]
        used = used / (used.pow(2).mean().sqrt() + 1e-9)
        if k == Ns:
            return used  # no pad needed
        pad = torch.zeros(Ns - k, y_syms.size(1), device=y_syms.device, dtype=y_syms.dtype)
        return torch.cat([used, pad], dim=0)

    def forward(self, input_ids, attention_mask, n_var, channel='AWGN', return_probs=False):
        """
        Returns:
          logits: [B, C]
          rate_bits: [B] (entropy-model proxy)
          route_sel: [B, K] hard one-hot for modulation
          Ns_eff:  [B] int #symbols used (per sample)
          (optionally) route_probs, ns_probs for logging
        """
        B, device = input_ids.size(0), input_ids.device
        chan = Channels()

        # 1) semantic Roberta encoder
        y = self.encoder(input_ids, attention_mask)  # [B, d_model]

        # 2) hyperprior (use SNR)
        snr_feat = torch.log(1.0/n_var).view(-1,1) if torch.is_tensor(n_var) \
                   else torch.full((B,1), math.log(1.0/n_var), device=device)
        z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))
        z_tilde = z + (torch.rand_like(z) - 0.5) if self.training else z.round()
        raw_sigma = self.hyper_decoder(z_tilde)
        sigma = F.softplus(raw_sigma) + 1e-6

        # 3) router features
        feat = self._router_feat(snr_feat, z_tilde, raw_sigma)

        # 3a) modulation (hard one-hot, straight-through)
        route_hard_tx, route_probs_tx = self.router(feat, hard=True)
        if self.force_expert is not None:
            route_hard_tx = F.one_hot(torch.full((B,), self.force_expert, device=device),
                                      num_classes=self.K).float()
            route_probs_tx = route_hard_tx

        # 3b) airtime Ns (hard one-hot via Gumbel; reuse feat)
        ns_logits = self.ns_head(feat)
        ns_onehot = F.gumbel_softmax(ns_logits, tau=1.0, hard=True)  # [B, |modes|]
        Ns_eff = (ns_onehot @ self.ns_modes.to(device)).mul(self.N_s_base).long()  # [B] get the real Ns for each sample
        Ns_eff = Ns_eff.clamp(min=1, max=self.N_s_base)
        # 4) quantize y (noise in train)
        y_tilde = y + (torch.rand_like(y) - 0.5) if self.training else y.round()

        # 5) side-channel z → symbols
        z_bits = self.hyper_channel_encoder(z_tilde)                         # [B, N_z*bps_z]
        z_syms = map_to_constellation(z_bits.view(B, self.N_z, self.bps_z), self.M_z)  # [B,N_z,2] #z fix symbols size
        z_flat = z_syms.view(B, -1)                                          # [B, 2*N_z]
        # normalize z separately (optional but clean)
        z_flat = z_flat / (z_flat.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-9)

        # 6) main-channel: build per-expert symbols once, then select
        syms_list = []
        for i, bps in enumerate(self.bps_list):
            bits = self.channel_encoders[i](y_tilde)                         # [B, N_s_base*bps]
            bits = gumbel_sigmoid(bits, τ=1.0, hard=self.training)
            syms = map_to_constellation(bits.view(B, self.N_s_base, bps), self.M_list[i])  # [B,Ns,2]
            syms_list.append(syms)
        Sy = torch.stack(syms_list, dim=-1)                                  # [B, Ns, 2, K]
        Sy_sel = (Sy * route_hard_tx.view(B, 1, 1, self.K)).sum(dim=-1)      # [B, Ns, 2]

        # 6a) apply Ns mask + per-sample normalization on used part
        out_syms = []
        for b in range(B):
            k = max(1, int(Ns_eff[b].item()))
            out_syms.append(self._norm_used_then_pad(Sy_sel[b], k))
        Sy_masked = torch.stack(out_syms, dim=0)                              # [B, Ns, 2]

        # 7) channel
        Tx = torch.cat([z_flat, Sy_masked.view(B, -1)], dim=1)                # [B, 2*N_z + 2*Ns_base]
        if channel == 'AWGN':
            Rx = chan.AWGN(Tx, n_var)
        elif channel == 'Rayleigh':
            Rx = chan.Rayleigh(Tx, n_var)
        else:
            Rx = chan.Rician(Tx, n_var)

        # 8) split + decode (use selected expert only)
        y_dim = 2 * self.N_s_base
        z_rx, y_rx = Rx[:, :-y_dim], Rx[:, -y_dim:]
        z_hat = self.hyper_channel_decoder(z_rx)
        sigma_rec = F.softplus(self.hyper_decoder(z_hat)) + 1e-6

        # recompute router at RX (deterministic) if you want—here we reuse TX
        dec_all = torch.stack([dec(y_rx) for dec in self.channel_decoders], dim=-1)  # [B,d_model,K]
        feat_y = (dec_all * route_hard_tx.view(B, 1, self.K)).sum(dim=-1)            # [B, d_model]

        logits = self.decoder(torch.cat([feat_y, sigma_rec], dim=1))                 # [B, C]

        # 9) entropy-model rate (per-sample)
        p_y = discrete_probability(y_tilde, torch.zeros_like(y_tilde), sigma_rec)
        p_z = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
        bits_y = -torch.log2(p_y + 1e-9).sum(dim=1)      # [B]
        bits_z = -torch.log2(p_z + 1e-9).sum(dim=1)      # [B]
        rate_bits = bits_y + bits_z

        if return_probs:
            return logits, rate_bits, route_hard_tx, Ns_eff, route_probs_tx, F.softmax(ns_logits, -1)
        else:
            return logits, rate_bits, route_hard_tx, Ns_eff

    # --------- faithful true bitcount (eval helper) --------
    @torch.no_grad()
    def true_bitcounts(self, input_ids, attention_mask, n_var):
        """
        Returns numeric means only (never NaN):
        {"bits_y_mean": float, "bits_z_mean": float, "bits_total_mean": float}
        Requires your helpers:
        _adaptive_Q_from_sigma, _disc_gauss_pmf, _pmf_to_cdf_with_escape, _encode_bucket_torchac, _rice_len
        """
        import math
        self.eval()
        B = input_ids.size(0)
        if B == 0:
            return {"bits_y_mean": 0.0, "bits_z_mean": 0.0, "bits_total_mean": 0.0}

        device = input_ids.device
        eps = 1e-6

        # --- analysis & hyperprior (eval path: rounding) ---
        y = self.encoder(input_ids, attention_mask)                                  # [B, D]
        snr_feat = (torch.log(1.0 / n_var).view(-1, 1) if torch.is_tensor(n_var)
                    else torch.full((B, 1), math.log(1.0 / n_var), device=device))
        z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))                      # [B, D]
        raw_sigma = self.hyper_decoder(z)                                            # [B, D]
        sigma_y = F.softplus(raw_sigma) + 1e-6                                       # [B, D]
        sigma_y = torch.clamp(sigma_y, min=1e-6, max=1e6)                            # safety

        y_int = y.round().to(torch.int32)                                            # [B, D]
        z_int = z.round().to(torch.int32)                                            # [B, D]
        Qy = _adaptive_Q_from_sigma(sigma_y, eps).view(B, -1)                        # [B, D]
        Qz = _adaptive_Q_from_sigma(torch.ones_like(z), eps).view(B, -1)             # [B, D]

        def _safe_num(x, default=0.0):
            try:
                v = float(x)
            except Exception:
                return default
            return v if math.isfinite(v) else default

        sum_y = 0.0
        sum_z = 0.0

        for b in range(B):
            # ---- Y path ----
            y_b   = y_int[b].view(-1)
            sig_b = sigma_y[b].view(-1)
            Qy_b  = Qy[b]

            bits_y_b = 0.0
            uniq_Qy = torch.unique(Qy_b)
            if uniq_Qy.numel() == 0:
                uniq_Qy = torch.tensor([1], device=Qy_b.device)  # fallback

            for q in uniq_Qy:
                mask = (Qy_b == q)
                if not mask.any():
                    continue
                qv = max(1, int(_safe_num(q.item(), 1)))

                # pmf/cdf (sanitize NaNs/Infs)
                pmf, tail, _, _ = _disc_gauss_pmf(torch.zeros_like(sig_b[mask]),
                                                sig_b[mask], q)
                if isinstance(pmf, torch.Tensor):
                    pmf = torch.nan_to_num(pmf, nan=0.0, posinf=0.0, neginf=0.0)
                    tail = torch.nan_to_num(tail, nan=0.0, posinf=0.0, neginf=0.0)
                cdf = _pmf_to_cdf_with_escape(pmf, tail)
                if isinstance(cdf, torch.Tensor):
                    cdf = torch.nan_to_num(cdf, nan=0.0, posinf=0.0, neginf=0.0)

                v   = y_b[mask].to(torch.int32)
                idx = v.clamp(-qv, qv) + qv
                esc = (v.abs() > qv)
                idx[esc] = 2*qv + 1

                bits_bucket = _encode_bucket_torchac(idx, cdf)
                bits_y_b += _safe_num(0.0 if bits_bucket is None else bits_bucket)

                if esc.any():
                    k_rice = 2
                    r = (v.abs() - qv)[esc].to(torch.int64)
                    # +1 sign bit per escape
                    bits_res = 0
                    for n in r:
                        try:
                            bits_res += _rice_len(int(n.item()), k_rice) + 1
                        except Exception:
                            bits_res += 0
                    bits_y_b += bits_res

            # ---- Z path ----
            z_b  = z_int[b].view(-1)
            Qz_b = Qz[b]
            bits_z_b = 0.0
            uniq_Qz = torch.unique(Qz_b)
            if uniq_Qz.numel() == 0:
                uniq_Qz = torch.tensor([1], device=Qz_b.device)

            for q in uniq_Qz:
                mask = (Qz_b == q)
                if not mask.any():
                    continue
                qv = max(1, int(_safe_num(q.item(), 1)))

                pmf, tail, _, _ = _disc_gauss_pmf(torch.zeros_like(z_b[mask], dtype=torch.float32),
                                                torch.ones_like(z_b[mask], dtype=torch.float32), q)
                if isinstance(pmf, torch.Tensor):
                    pmf = torch.nan_to_num(pmf, nan=0.0, posinf=0.0, neginf=0.0)
                    tail = torch.nan_to_num(tail, nan=0.0, posinf=0.0, neginf=0.0)
                cdf = _pmf_to_cdf_with_escape(pmf, tail)
                if isinstance(cdf, torch.Tensor):
                    cdf = torch.nan_to_num(cdf, nan=0.0, posinf=0.0, neginf=0.0)

                v   = z_b[mask].to(torch.int32)
                idx = v.clamp(-qv, qv) + qv
                esc = (v.abs() > qv)
                idx[esc] = 2*qv + 1

                bits_bucket = _encode_bucket_torchac(idx, cdf)
                bits_z_b += _safe_num(0.0 if bits_bucket is None else bits_bucket)

                if esc.any():
                    k_rice = 2
                    r = (v.abs() - qv)[esc].to(torch.int64)
                    bits_res = 0
                    for n in r:
                        try:
                            bits_res += _rice_len(int(n.item()), k_rice) + 1
                        except Exception:
                            bits_res += 0
                    bits_z_b += bits_res

            # finalize per-sample (guard)
            bits_y_b = _safe_num(bits_y_b, 0.0)
            bits_z_b = _safe_num(bits_z_b, 0.0)
            sum_y += bits_y_b
            sum_z += bits_z_b

        denom = float(B)
        bits_y_mean     = _safe_num(sum_y / denom, 0.0)
        bits_z_mean     = _safe_num(sum_z / denom, 0.0)
        bits_total_mean = _safe_num((sum_y + sum_z) / denom, 0.0)
        return {
            "bits_y_mean": bits_y_mean,
            "bits_z_mean": bits_z_mean,
            "bits_total_mean": bits_total_mean,
        }

import math, torch
import torch.nn as nn
import torch.nn.functional as F

import math, torch
import torch.nn as nn
import torch.nn.functional as F

class MODJSCC_MoE_Faithful_2(nn.Module):
    """
    Time-unrolled PHY:
      - One modulation per mini-frame (T_max frames per burst)
      - Per-frame expert routing (hard, ST) and stay/switch head
      - Per-expert enc/dec operate on per-frame codewords (length W = N_s_base)
      - Whole burst -> channel -> per-frame decode -> temporal fusion
      - Pilots/headers: accounted in loss (B_est), not yet inserted as symbols
    """
    def __init__(self, d_model=256, freeze_bert=False,
                 N_s_base=64,            # mini-frame length W (data symbols)
                 N_z=8,
                 M_list=[4,16,64],            # e.g., [4,16,64]
                 M_z=2,
                 ns_modes=(0.1, 0.25, 0.5, 0.75, 1.0, 1.25),
                 force_expert: int=None,
                 # Schedule / overhead
                 T_max=6,                # # mini-frames per burst (upper bound)
                 pilot_P=10,             # scattered pilot spacing (used in loss)
                 r_c=0.8,                # code rate (payload calc)
                 O_hdr_bits=48,          # header bits per run start (used in loss)
                 use_soft_hdr=True):
        super().__init__()
        self.d_model = d_model
        self.N_s_base = int(N_s_base)   # == W
        self.N_z = N_z
        self.M_list = list(M_list)
        self.bps_list = [int(math.log2(M)) for M in self.M_list]
        self.K = len(self.M_list)
        self.force_expert = force_expert
        self.M_z = M_z
        self.bps_z = int(math.log2(M_z))
        self.ns_modes = torch.tensor(ns_modes, dtype=torch.float32)

        # schedule/overhead
        self.T_max = int(T_max)
        self.pilot_P = float(pilot_P)
        self.r_c = float(r_c)
        self.O_hdr_bits = float(O_hdr_bits)
        self.use_soft_hdr = bool(use_soft_hdr)

        # ----- Semantic encoder / decoder -----
        self.encoder = RoBERTaEncoder(d_model=d_model, freeze_bert=freeze_bert)
        self.decoder = nn.Sequential(nn.Linear(2*d_model, 256), nn.ReLU(),
                                     nn.Linear(256, 2))

        # ----- Hyperprior (entropy model) -----
        self.hyper_encoder = nn.Sequential(nn.Linear(d_model + 1, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, d_model))
        self.hyper_decoder = nn.Sequential(nn.Linear(d_model, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, d_model))

        # Side-channel mapper (z)
        self.hyper_channel_encoder = nn.Linear(d_model, N_z * self.bps_z)   # bits
        self.hyper_channel_decoder = nn.Linear(2 * N_z, d_model)            # from IQ symbols

        # ----- Per-expert per-frame encoders/decoders -----
        # We reuse a tiny frame embedding so frames don't repeat content.
        self.frame_embed = nn.Embedding(self.T_max, d_model)
        self.channel_encoders = nn.ModuleList([
            nn.Linear(d_model, self.N_s_base * bps) for bps in self.bps_list
        ])
        self.channel_decoders = nn.ModuleList([
            nn.Linear(2 * self.N_s_base, d_model) for _ in self.bps_list
        ])

        # ----- Router heads -----
        # Base features: [logSNR, mean|z|, mean logσ+, var logσ+]
        self.router = SemanticRouter(d_in=4, n_experts=self.K)  # legacy single-block head
        self.ns_head = nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                     nn.Linear(64, len(ns_modes)))  # legacy Ns head

        # Sequence routing heads (per mini-frame)
        self.route_seq_head = nn.Linear(4, self.K)       # per t expert logits
        self.boundary_head  = nn.Linear(4, 2)            # per t stay/switch logits

    # --------- utilities ---------
    @staticmethod
    def _router_feat(snr_feat, z_like, sigma_like):
        B = snr_feat.size(0)
        mean_abs_z = z_like.abs().mean(dim=1, keepdim=True)
        logsig = torch.log1p(F.softplus(sigma_like).abs() + 1e-6)
        mu_logsig = logsig.mean(dim=1, keepdim=True)
        var_logsig = logsig.var(dim=1, unbiased=False, keepdim=True)
        return torch.cat([snr_feat, mean_abs_z, mu_logsig, var_logsig], dim=1)  # [B,4]

    def build_schedule(self, feat, T=None, snr_seq=None, tau=1.0, hard=True):
        B = feat.size(0)
        T = T or self.T_max

        # (unchanged) tile features & optionally inject per-frame logSNR
        feat_t = feat.unsqueeze(1).expand(B, T, feat.size(1)).contiguous()
        if snr_seq is not None:
            logsnr_t = torch.log(snr_seq.clamp_min(1e-9))
            feat_t[:, :, 0] = logsnr_t

        # expert probs per frame
        logits_e = self.route_seq_head(feat_t)                # [B,T,K]
        pi_seq   = F.gumbel_softmax(logits_e, tau=tau, hard=False, dim=-1)

        # stay/switch head
        logits_b = self.boundary_head(feat_t)                 # [B,T,2]
        b_probs  = F.softmax(logits_b, dim=-1)                # [B,T,2]
        ps       = b_probs[..., 1]                            # [B,T], P(switch)

        # >>> FIX: do NOT write in-place into ps / b_probs <<<
        if T > 1:
            ones_first = torch.ones(B, 1, device=ps.device, dtype=ps.dtype)
            p_switch   = torch.cat([ones_first, ps[:, 1:]], dim=1)  # [B,T]
        else:
            p_switch   = torch.ones(B, 1, device=ps.device, dtype=ps.dtype)

        # hard routes and hard switch mask
        if hard:
            z_seq = F.gumbel_softmax(logits_e, tau=tau, hard=True, dim=-1)  # [B,T,K]
            z_idx = z_seq.argmax(dim=-1)                                    # [B,T]
            sw_mask = torch.zeros_like(p_switch)
            sw_mask[:, 0] = 1.0
            if T > 1:
                sw_mask[:, 1:] = (z_idx[:, 1:] != z_idx[:, :-1]).float()
        else:
            z_seq = pi_seq
            z_idx = z_seq.argmax(dim=-1)
            sw_mask = torch.zeros_like(p_switch)
            sw_mask[:, 0] = 1.0
            if T > 1:
                sw_mask[:, 1:] = 0.5 * (pi_seq[:, 1:, :] - pi_seq[:, :-1, :]).abs().sum(-1).clamp(0, 1)

        return dict(pi_seq=pi_seq, z_seq=z_seq, z_idx=z_idx,
                    p_sw=p_switch, sw_mask=sw_mask)

    @staticmethod
    def _plan_stopping_mask(B_est_hard, R_tot_est):
        """
        Greedy stopping: include frames until cum bits >= R_tot_est.
        Inputs:
        B_est_hard: [B,T] (nonnegative)
        R_tot_est:  [B]   (target bits)
        Returns:
        mask: [B,T] in {0,1}
        """
        # cumulative bits
        csum = torch.cumsum(B_est_hard, dim=1)                        # [B,T]
        R = R_tot_est.view(-1, 1)                                     # [B,1]

        # need: frames strictly before crossing
        need = (csum < R).float()                                     # [B,T]

        # cross: once csum >= R, stays 1 thereafter (monotone)
        cross = (csum >= R).float()                                   # [B,T]

        # first_cross: 1 only at the FIRST index where cross flips from 0->1
        first_cross = torch.zeros_like(cross)
        first_cross[:, 0] = cross[:, 0]
        if cross.size(1) > 1:
            first_cross[:, 1:] = cross[:, 1:] * (1.0 - cross[:, :-1]) # [B,T-1]

        # base mask: all pre-cross frames + the first crossing frame
        mask = (need + first_cross).clamp_(0.0, 1.0)                  # [B,T]

        # if never crosses (sum < target), include ALL frames
        never = (csum[:, -1] < R.squeeze(1)).float().view(-1, 1)      # [B,1]
        if never.any():
            mask = mask + never * torch.ones_like(mask)
            mask.clamp_(0.0, 1.0)

        return mask

    def forward(self, input_ids, attention_mask, n_var, channel='AWGN',
                return_probs=False, snr_seq=None, tau=1.0):
        """
        Outputs:
          logits [B,C], rate_bits [B],
          route_sel (legacy) [B,K], Ns_eff (legacy) [B],
          sched dict: pi_seq,z_seq,z_idx,p_sw,sw_mask,B_est,B_est_hard,B_sum,..., frame_mask
        """
        B, device = input_ids.size(0), input_ids.device
        chan = Channels()

        # 1) semantic encoder
        y = self.encoder(input_ids, attention_mask)  # [B, d_model]

        # 2) hyperprior (TX-side estimate)
        if torch.is_tensor(n_var):
            logsnr = torch.log(1.0/n_var).view(-1,1)
        else:
            logsnr = torch.full((B,1), math.log(1.0/n_var), device=device)
        snr_feat = logsnr
        z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))   
        z_tilde = z + (torch.rand_like(z) - 0.5) if self.training else z.round()
        raw_sigma = self.hyper_decoder(z_tilde)
        sigma_tx = F.softplus(raw_sigma) + 1e-6 #Rate calc uses TX sigma

        # 3) router features
        feat = self._router_feat(snr_feat, z_tilde, raw_sigma)
        # 3a) build schedule routing over the mini frame (hard for real or soft for training)
        sched = self.build_schedule(feat, T=self.T_max, snr_seq=snr_seq, tau=tau, hard=True) # SNR_seq is currently none,  
        pi_seq, z_seq, z_idx = sched['pi_seq'], sched['z_seq'], sched['z_idx']    # [B,T,K], [B,T,K], [B,T]
        p_sw, sw_mask = sched['p_sw'], sched['sw_mask']  #first frame header                          # [B,T], [B,T]
        z_idx       = sched["z_idx"]         # [B,T]
        


        # 4) expected payload (for budget loss)
        rho_pilot = 1.0 / self.pilot_P                                              # pilot fraction
        W = float(self.N_s_base)                                                    #Bits per frame
        m_vec = torch.tensor(self.bps_list, device=device, dtype=torch.float32)
        bits_per_sym_soft = (pi_seq @ m_vec)                                      # [B,T]
        B_est = (1.0 - rho_pilot) * W * self.r_c * bits_per_sym_soft              # [B,T] # payload bits estimate
        # subtract header cost (first frame always has header)
        hdr_cost = (p_sw if self.use_soft_hdr else sw_mask) * self.O_hdr_bits
        B_est = torch.clamp(B_est - hdr_cost, min=0.0)
        bits_per_sym_hard = m_vec[z_idx]                                          # [B,T]
        B_est_hard = torch.clamp((1.0 - rho_pilot) * W * self.r_c * bits_per_sym_hard
                                  - sw_mask * self.O_hdr_bits, min=0.0)
        B_sum = B_est.sum(dim=1)                                                  # [B]
        B_sum_hard = B_est_hard.sum(dim=1)                                        # [B]

        # 5) TX-side bitrate estimate (for stopping rule mask)
        # Use hyperprior's TX sigma to estimate required main-channel bits
        p_y_tx = discrete_probability(y, torch.zeros_like(y), sigma_tx)
        bits_y_tx = -torch.log2(p_y_tx + 1e-9).sum(dim=1)                         # [B] # Total bits frames used
        R_tot_est = bits_y_tx.detach()                                            # planning target (no grad)

        frame_mask = self._plan_stopping_mask(B_est_hard, R_tot_est)              # [B,T] in {0,1}
        sched.update(dict(B_est=B_est, B_est_hard=B_est_hard,
                          B_sum=B_sum, B_sum_hard=B_sum_hard,
                          n_headers=sw_mask.sum(1), n_headers_soft=p_sw.sum(1),
                          rho_pilot=rho_pilot, W=W, r_c=self.r_c,
                          frame_mask=frame_mask)) #update sched dict
        

        # 6) Build per-frame codewords (data only, pilots implicit in loss)
        #    For each expert k and frame t: enc_k(y + e_t) -> bits -> symbols(M_k)
        y_tilde = y + (torch.rand_like(y) - 0.5) if self.training else y.round()
        T = self.T_max
        Ns = self.N_s_base
        frame_feats = []
        for t in range(T):
            y_t = y_tilde + self.frame_embed.weight[t]          # [B,d_model]
            # produce symbols for ALL experts, then select per z_seq[:,t]
            syms_per_k = []
            for i, bps in enumerate(self.bps_list):
                bits = self.channel_encoders[i](y_t)             # [B, Ns*bps]
                bits = gumbel_sigmoid(bits, τ=1.0, hard=self.training)
                syms = map_to_constellation(bits.view(B, Ns, bps), self.M_list[i])  # [B,Ns,2]
                syms_per_k.append(syms)
            Sy_t = torch.stack(syms_per_k, dim=-1)               # [B, Ns, 2, K]
            # select expert for this frame (hard one-hot)
            Sy_t_sel = (Sy_t * z_seq[:,t].view(B,1,1,self.K)).sum(dim=-1)   # [B, Ns, 2]
            frame_feats.append(Sy_t_sel)
        Sy_all = torch.stack(frame_feats, dim=1)                 # [B, T, Ns, 2]

        # 7) Side-channel z symbols (sent once at burst head)
        z_bits = self.hyper_channel_encoder(z_tilde)                                   # [B, N_z*bps_z]
        z_syms = map_to_constellation(z_bits.view(B, self.N_z, self.bps_z), self.M_z)  # [B,N_z,2]
        z_flat = z_syms.view(B, -1)
        z_flat = z_flat / (z_flat.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-9)

        # 8) Concatenate z + all frames into one burst and send through channel
        Sy_seq = Sy_all.reshape(B, -1, 2)                           # [B, T*Ns, 2]
        Tx_seq = torch.cat([z_flat, Sy_seq.view(B, -1)], dim=1)     # flatten to [B, 2*(N_z + T*Ns)]
        chan_in = Tx_seq

        if channel == 'AWGN':
            Rx_seq = chan.AWGN(chan_in, n_var)
        elif channel == 'Rayleigh':
            Rx_seq = chan.Rayleigh(chan_in, n_var)
        else:
            Rx_seq = chan.Rician(chan_in, n_var)

        # 9) Split back: z then per-frame data blocks, decode per selected expert
        y_dim = 2 * self.N_s_base
        z_rx, y_rx_flat = Rx_seq[:, :2*self.N_z], Rx_seq[:, 2*self.N_z:]
        # (We assume perfect header knowledge for picking the decoder; headers are budgeted in loss.)
        y_rx = y_rx_flat.view(B, T, y_dim)                          # [B,T, 2*Ns]
        # decode with all experts then select using z_seq
        per_t_feats = []
        for t in range(T):
            y_rt = y_rx[:, t, :]                                    # [B, 2*Ns]
            dec_k = []
            for i in range(self.K):
                dec_k.append(self.channel_decoders[i](y_rt))        # [B, d_model]
            dec_all = torch.stack(dec_k, dim=-1)                    # [B, d_model, K]
            feat_t = (dec_all * z_seq[:,t].view(B,1,self.K)).sum(dim=-1)  # [B,d_model]
            per_t_feats.append(feat_t)
        y_feat_seq = torch.stack(per_t_feats, dim=1)                # [B, T, d_model]

        # 10) Fuse over frames with stopping mask (average only used frames)
        w = frame_mask.to(y_feat_seq.dtype).unsqueeze(-1)           # [B,T,1]
        used = (w.sum(dim=1) + 1e-6)                                # [B,1]
        feat_pooled = (y_feat_seq * w).sum(dim=1) / used            # [B, d_model]

        # 11) Side-channel decode to recover sigma_rec (for entropy model)
        z_hat = self.hyper_channel_decoder(z_rx)
        sigma_rec = F.softplus(self.hyper_decoder(z_hat)) + 1e-6

        # 12) Task head
        logits = self.decoder(torch.cat([feat_pooled, sigma_rec], dim=1))     # [B, C]

        # 13) Entropy-model rate (post-RX, for logging/alt-target)
        p_y = discrete_probability(y, torch.zeros_like(y), sigma_rec)
        p_z = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
        bits_y = -torch.log2(p_y + 1e-9).sum(dim=1)
        bits_z = -torch.log2(p_z + 1e-9).sum(dim=1)
        rate_bits = bits_y + bits_z
        frame_mask  = sched["frame_mask"]    # [B,T]

        # "route_hard_tx" for logging/back-compat: take the first frame's expert
        route_hard_tx = F.one_hot(z_idx[:, 0], num_classes=self.K).float()  # [B,K]

        # "Ns_eff" is now the total used symbols = (# used frames) * W
        Ns_eff = (frame_mask.sum(dim=1) * self.N_s_base).long() 

        if return_probs:
            # Optionally expose per-frame soft probs for analysis
            route_probs_seq = sched["pi_seq"]   # [B,T,K]
            return (logits, rate_bits, route_hard_tx, Ns_eff,
                    route_probs_seq, None, {**sched})
        else:
            return logits, rate_bits, route_hard_tx, Ns_eff, {**sched}

import math, torch
import torch.nn as nn
import torch.nn.functional as F

class MODJSCC_MoE_Recon(nn.Module):
    """
    Time-unrolled PHY w/ MoE routing + reconstruction task:
      - One modulation per mini-frame (T_max frames per burst)
      - Per-frame expert routing (hard, ST) and stay/switch head
      - Per-expert enc/dec operate on per-frame codewords (length W = N_s_base)
      - Whole burst -> channel -> per-frame decode -> temporal fusion
      - Pilots/headers: accounted in rate estimate (B_est), not inserted as symbols
      - TASK: reconstruct the semantic embedding y produced by the RoBERTa encoder
    """
    def __init__(self, d_model=256, freeze_bert=False,
                 N_s_base=64,            # mini-frame length W (data symbols)
                 N_z=8,
                 M_list=[4,16,64],       # per-expert constellations
                 M_z=2,
                 ns_modes=(0.1, 0.25, 0.5, 0.75, 1.0, 1.25),
                 force_expert: int=None,
                 # Schedule / overhead
                 T_max=6,                # # mini-frames per burst (upper bound)
                 pilot_P=10,             # scattered pilot spacing (used in loss)
                 r_c=0.8,                # code rate (payload calc)
                 O_hdr_bits=48,          # header bits per run start (used in loss)
                 use_soft_hdr=True):
        super().__init__()
        self.d_model = d_model
        self.N_s_base = int(N_s_base)   # == W
        self.N_z = N_z
        self.M_list = list(M_list)
        self.bps_list = [int(math.log2(M)) for M in self.M_list]
        self.K = len(self.M_list)
        self.force_expert = force_expert
        self.M_z = M_z
        self.bps_z = int(math.log2(M_z))
        self.ns_modes = torch.tensor(ns_modes, dtype=torch.float32)

        # schedule/overhead
        self.T_max = int(T_max)
        self.pilot_P = float(pilot_P)
        self.r_c = float(r_c)
        self.O_hdr_bits = float(O_hdr_bits)
        self.use_soft_hdr = bool(use_soft_hdr)

        # ----- Semantic encoder -----
        self.encoder = RoBERTaEncoder(d_model=d_model, freeze_bert=freeze_bert)

        # ----- Reconstruction head (replaces classifier) -----
        # Input is concat([feat_pooled, sigma_rec]) of size 2*d_model → d_model
        self.recon_head = nn.Sequential(
            nn.Linear(2 * d_model, 512),
            nn.ReLU(),
            nn.Linear(512, d_model)
        )

        # ----- Hyperprior (entropy model) -----
        self.hyper_encoder = nn.Sequential(nn.Linear(d_model + 1, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, d_model))
        self.hyper_decoder = nn.Sequential(nn.Linear(d_model, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, d_model))

        # Side-channel mapper (z)
        self.hyper_channel_encoder = nn.Linear(d_model, N_z * self.bps_z)   # bits
        self.hyper_channel_decoder = nn.Linear(2 * N_z, d_model)            # from IQ symbols

        # ----- Per-expert per-frame encoders/decoders -----
        self.frame_embed = nn.Embedding(self.T_max, d_model)
        self.channel_encoders = nn.ModuleList([
            nn.Linear(d_model, self.N_s_base * bps) for bps in self.bps_list
        ])
        self.channel_decoders = nn.ModuleList([
            nn.Linear(2 * self.N_s_base, d_model) for _ in self.bps_list
        ])

        # ----- Router heads -----
        # Base features: [logSNR, mean|z|, mean logσ+, var logσ+]
        self.router = SemanticRouter(d_in=4, n_experts=self.K)  # legacy single-block head
        self.ns_head = nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                     nn.Linear(64, len(ns_modes)))  # legacy Ns head

        # Sequence routing heads (per mini-frame)
        self.route_seq_head = nn.Linear(4, self.K)       # per t expert logits
        self.boundary_head  = nn.Linear(4, 2)            # per t stay/switch logits

    # --------- utilities ---------
    @staticmethod
    def _router_feat(snr_feat, z_like, sigma_like):
        B = snr_feat.size(0)
        mean_abs_z = z_like.abs().mean(dim=1, keepdim=True)
        logsig = torch.log1p(F.softplus(sigma_like).abs() + 1e-6)
        mu_logsig = logsig.mean(dim=1, keepdim=True)
        var_logsig = logsig.var(dim=1, unbiased=False, keepdim=True)
        return torch.cat([snr_feat, mean_abs_z, mu_logsig, var_logsig], dim=1)  # [B,4]

    def build_schedule(self, feat, T=None, snr_seq=None, tau=1.0, hard=True):
        B = feat.size(0)
        T = T or self.T_max

        # tile features & optionally inject per-frame logSNR
        feat_t = feat.unsqueeze(1).expand(B, T, feat.size(1)).contiguous()
        if snr_seq is not None:
            logsnr_t = torch.log(snr_seq.clamp_min(1e-9))
            feat_t[:, :, 0] = logsnr_t

        # expert probs per frame
        logits_e = self.route_seq_head(feat_t)                # [B,T,K]
        pi_seq   = F.gumbel_softmax(logits_e, tau=tau, hard=False, dim=-1)

        # stay/switch head
        logits_b = self.boundary_head(feat_t)                 # [B,T,2]
        b_probs  = F.softmax(logits_b, dim=-1)                # [B,T,2]
        ps       = b_probs[..., 1]                            # [B,T], P(switch)

        # avoid in-place ops on ps/b_probs
        if T > 1:
            ones_first = torch.ones(B, 1, device=ps.device, dtype=ps.dtype)
            p_switch   = torch.cat([ones_first, ps[:, 1:]], dim=1)  # [B,T]
        else:
            p_switch   = torch.ones(B, 1, device=ps.device, dtype=ps.dtype)

        # hard routes and hard switch mask
        if hard:
            z_seq = F.gumbel_softmax(logits_e, tau=tau, hard=True, dim=-1)  # [B,T,K]
            z_idx = z_seq.argmax(dim=-1)                                    # [B,T]
            sw_mask = torch.zeros_like(p_switch)
            sw_mask[:, 0] = 1.0
            if T > 1:
                sw_mask[:, 1:] = (z_idx[:, 1:] != z_idx[:, :-1]).float()
        else:
            z_seq = pi_seq
            z_idx = z_seq.argmax(dim=-1)
            sw_mask = torch.zeros_like(p_switch)
            sw_mask[:, 0] = 1.0
            if T > 1:
                sw_mask[:, 1:] = 0.5 * (pi_seq[:, 1:, :] - pi_seq[:, :-1, :]).abs().sum(-1).clamp(0, 1)

        return dict(pi_seq=pi_seq, z_seq=z_seq, z_idx=z_idx,
                    p_sw=p_switch, sw_mask=sw_mask)

    @staticmethod
    def _plan_stopping_mask(B_est_hard, R_tot_est):
        """
        Greedy stopping: include frames until cum bits >= R_tot_est.
        Inputs:
        B_est_hard: [B,T] (nonnegative)
        R_tot_est:  [B]   (target bits)
        Returns:
        mask: [B,T] in {0,1}
        """
        csum = torch.cumsum(B_est_hard, dim=1)                        # [B,T]
        R = R_tot_est.view(-1, 1)                                     # [B,1]

        need = (csum < R).float()                                     # [B,T]
        cross = (csum >= R).float()                                   # [B,T]

        first_cross = torch.zeros_like(cross)
        first_cross[:, 0] = cross[:, 0]
        if cross.size(1) > 1:
            first_cross[:, 1:] = cross[:, 1:] * (1.0 - cross[:, :-1]) # [B,T-1]

        mask = (need + first_cross).clamp_(0.0, 1.0)                  # [B,T]

        never = (csum[:, -1] < R.squeeze(1)).float().view(-1, 1)      # [B,1]
        if never.any():
            mask = mask + never * torch.ones_like(mask)
            mask.clamp_(0.0, 1.0)

        return mask

    def forward(self, input_ids, attention_mask, n_var, channel='AWGN',
                return_probs=False, snr_seq=None, tau=1.0):
        """
        Outputs (reconstruction task):
          recon [B, d_model], rate_bits [B],
          route_hard_tx [B,K], Ns_eff [B],
          sched dict (includes: pi_seq,z_seq,z_idx,p_sw,sw_mask,B_est,..., frame_mask, y_target)
        """
        B, device = input_ids.size(0), input_ids.device
        chan = Channels()

        # 1) semantic encoder (target to reconstruct)
        y = self.encoder(input_ids, attention_mask)  # [B, d_model]

        # 2) hyperprior (TX-side estimate)
        if torch.is_tensor(n_var):
            logsnr = torch.log(1.0/n_var).view(-1,1)
        else:
            logsnr = torch.full((B,1), math.log(1.0/n_var), device=device)
        snr_feat = logsnr
        z = self.hyper_encoder(torch.cat([y, snr_feat], dim=1))
        z_tilde = z + (torch.rand_like(z) - 0.5) if self.training else z.round()
        raw_sigma = self.hyper_decoder(z_tilde)
        sigma_tx = F.softplus(raw_sigma) + 1e-6  # used for TX-side rate calc

        # 3) router features + schedule
        feat = self._router_feat(snr_feat, z_tilde, raw_sigma)
        sched = self.build_schedule(feat, T=self.T_max, snr_seq=snr_seq, tau=tau, hard=True)
        pi_seq, z_seq, z_idx = sched['pi_seq'], sched['z_seq'], sched['z_idx']    # [B,T,K], [B,T,K], [B,T]
        p_sw, sw_mask = sched['p_sw'], sched['sw_mask']                            # [B,T], [B,T]

        # 4) expected payload (for budget loss)
        rho_pilot = 1.0 / self.pilot_P
        W = float(self.N_s_base)
        m_vec = torch.tensor(self.bps_list, device=device, dtype=torch.float32)
        bits_per_sym_soft = (pi_seq @ m_vec)                                       # [B,T]
        B_est = (1.0 - rho_pilot) * W * self.r_c * bits_per_sym_soft               # [B,T]
        hdr_cost = (p_sw if self.use_soft_hdr else sw_mask) * self.O_hdr_bits
        B_est = torch.clamp(B_est - hdr_cost, min=0.0)
        bits_per_sym_hard = m_vec[z_idx]                                           # [B,T]
        B_est_hard = torch.clamp((1.0 - rho_pilot) * W * self.r_c * bits_per_sym_hard
                                  - sw_mask * self.O_hdr_bits, min=0.0)
        B_sum = B_est.sum(dim=1)                                                   # [B]
        B_sum_hard = B_est_hard.sum(dim=1)                                         # [B]

        # 5) TX-side bitrate estimate (for stopping rule mask)
        p_y_tx = discrete_probability(y, torch.zeros_like(y), sigma_tx)
        bits_y_tx = -torch.log2(p_y_tx + 1e-9).sum(dim=1)                          # [B]
        R_tot_est = bits_y_tx.detach()                                             # planning target (no grad)

        frame_mask = self._plan_stopping_mask(B_est_hard, R_tot_est)               # [B,T]
        sched.update(dict(B_est=B_est, B_est_hard=B_est_hard,
                          B_sum=B_sum, B_sum_hard=B_sum_hard,
                          n_headers=sw_mask.sum(1), n_headers_soft=p_sw.sum(1),
                          rho_pilot=rho_pilot, W=W, r_c=self.r_c,
                          frame_mask=frame_mask))

        # 6) Build per-frame codewords (data only)
        y_tilde = y + (torch.rand_like(y) - 0.5) if self.training else y.round()
        T = self.T_max
        Ns = self.N_s_base
        frame_feats = []
        for t in range(T):
            y_t = y_tilde + self.frame_embed.weight[t]          # [B,d_model]
            syms_per_k = []
            for i, bps in enumerate(self.bps_list):
                bits = self.channel_encoders[i](y_t)             # [B, Ns*bps]
                bits = gumbel_sigmoid(bits, τ=1.0, hard=self.training)
                syms = map_to_constellation(bits.view(B, Ns, bps), self.M_list[i])  # [B,Ns,2]
                syms_per_k.append(syms)
            Sy_t = torch.stack(syms_per_k, dim=-1)               # [B, Ns, 2, K]
            Sy_t_sel = (Sy_t * z_seq[:,t].view(B,1,1,self.K)).sum(dim=-1)   # [B, Ns, 2]
            frame_feats.append(Sy_t_sel)
        Sy_all = torch.stack(frame_feats, dim=1)                 # [B, T, Ns, 2]

        # 7) Side-channel z symbols (sent once at burst head)
        z_bits = self.hyper_channel_encoder(z_tilde)                                   # [B, N_z*bps_z]
        z_syms = map_to_constellation(z_bits.view(B, self.N_z, self.bps_z), self.M_z)  # [B,N_z,2]
        z_flat = z_syms.view(B, -1)
        z_flat = z_flat / (z_flat.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-9)

        # 8) Concatenate z + all frames into one burst and send through channel
        Sy_seq = Sy_all.reshape(B, -1, 2)                           # [B, T*Ns, 2]
        Tx_seq = torch.cat([z_flat, Sy_seq.view(B, -1)], dim=1)     # [B, 2*(N_z + T*Ns)]
        chan_in = Tx_seq

        if channel == 'AWGN':
            Rx_seq = chan.AWGN(chan_in, n_var)
        elif channel == 'Rayleigh':
            Rx_seq = chan.Rayleigh(chan_in, n_var)
        else:
            Rx_seq = chan.Rician(chan_in, n_var)

        # 9) Split back: z then per-frame data blocks, decode per selected expert
        y_dim = 2 * self.N_s_base
        z_rx, y_rx_flat = Rx_seq[:, :2*self.N_z], Rx_seq[:, 2*self.N_z:]
        y_rx = y_rx_flat.view(B, T, y_dim)                          # [B,T, 2*Ns]
        per_t_feats = []
        for t in range(T):
            y_rt = y_rx[:, t, :]                                    # [B, 2*Ns]
            dec_k = []
            for i in range(self.K):
                dec_k.append(self.channel_decoders[i](y_rt))        # [B, d_model]
            dec_all = torch.stack(dec_k, dim=-1)                    # [B, d_model, K]
            feat_t = (dec_all * z_seq[:,t].view(B,1,self.K)).sum(dim=-1)  # [B,d_model]
            per_t_feats.append(feat_t)
        y_feat_seq = torch.stack(per_t_feats, dim=1)                # [B, T, d_model]

        # 10) Fuse over frames with stopping mask (average only used frames)
        w = frame_mask.to(y_feat_seq.dtype).unsqueeze(-1)           # [B,T,1]
        used = (w.sum(dim=1) + 1e-6)                                # [B,1]
        feat_pooled = (y_feat_seq * w).sum(dim=1) / used            # [B, d_model]

        # 11) Side-channel decode to recover sigma_rec (for entropy model)
        z_hat = self.hyper_channel_decoder(z_rx)
        sigma_rec = F.softplus(self.hyper_decoder(z_hat)) + 1e-6

        # 12) RECONSTRUCTION head (predict y_hat)
        recon = self.recon_head(torch.cat([feat_pooled, sigma_rec], dim=1))  # [B, d_model]

        # 13) Entropy-model rate (post-RX, for logging/alt-target)
        p_y = discrete_probability(y, torch.zeros_like(y), sigma_rec)
        p_z = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
        rate_bits = -torch.log2(p_y + 1e-9).sum(dim=1) + -torch.log2(p_z + 1e-9).sum(dim=1)

        frame_mask  = sched["frame_mask"]    # [B,T]
        route_hard_tx = F.one_hot(z_idx[:, 0], num_classes=self.K).float()  # [B,K]
        Ns_eff = (frame_mask.sum(dim=1) * self.N_s_base).long()

        # expose target embedding for convenient loss computation
        sched["y_target"] = y.detach()

        if return_probs:
            route_probs_seq = sched["pi_seq"]   # [B,T,K]
            return (recon, rate_bits, route_hard_tx, Ns_eff,
                    route_probs_seq, None, {**sched})
        else:
            return recon, rate_bits, route_hard_tx, Ns_eff, {**sched}
