
import torch.nn.functional as F
import datetime
import os 
import math
import torch
import time
import torch.nn as nn
import numpy as np
import pandas as pd
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


class Channels():
    def AWGN(self, x, sigma):
        """
        x: [B, N] real-valued tensor (I and Q may be concatenated into N)
        sigma: float or tensor [B] or [B,1] = noise std PER SAMPLE
        """
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, device=x.device, dtype=x.dtype)
        if sigma.ndim == 1:
            sigma = sigma.view(-1, 1)
        noise = torch.randn_like(x) * sigma
        return x + noise
    def Rayleigh(self, Tx_sig, noise_var):
        """
        Applies Rayleigh fading + AWGN + perfect inversion to a real I/Q tensor.

        Args:
            Tx_sig    : Tensor of shape [B, ..., 2] (I/Q in last dim)
            noise_var : noise variance (1/SNR)
        Returns:
            Tensor of same shape as Tx_sig, after fading, noise, and inversion.
        """
        # flatten batch & symbols
        B = Tx_sig.shape[0]
        device = Tx_sig.device
        sig_flat = Tx_sig.view(B, -1, 2)       # [B, N, 2]

        # draw i.i.d. N(0,½) real & imag gains per batch
        std = math.sqrt(1/2)
        h_r = torch.normal(0.0, std, size=(B,), device=device)
        h_i = torch.normal(0.0, std, size=(B,), device=device)

        # build real‐matrix H = [[h_r, -h_i],[h_i, h_r]] for each batch
        H = torch.stack([
            torch.stack([h_r, -h_i], dim=1),
            torch.stack([h_i,  h_r], dim=1),
        ], dim=1)                          # [B, 2, 2]

        # apply fading: [B,N,2] bmm [B,2,2] → [B,N,2]
        faded = torch.bmm(sig_flat, H)

        # add AWGN
        rx = self.AWGN(faded, noise_var)
# 
        # invert channel perfectly
        H_inv   = torch.inverse(H)          # [B,2,2]
        rec_flat =  torch.bmm(rx, H_inv)     # [B,N,2]

        # restore original shape
        return rec_flat.view_as(Tx_sig)

    def Rician(self, Tx_sig, noise_var, K=1):
        """
        Applies Rician fading + AWGN to a real I/Q tensor.

        Args:
            Tx_sig    : Tensor of shape [B, ..., 2] (I/Q last dim)
            noise_var : noise variance (1/SNR)
            K         : Rice factor (linear, not dB)
        Returns:
            Tensor of same shape as Tx_sig, after fading+noise+perfect inversion.
        """
        # flatten batch & “symbol” dims
        B = Tx_sig.shape[0]
        device = Tx_sig.device
        sig_flat = Tx_sig.view(B, -1, 2)  # [B, N, 2]

        # Rician parameters
        mu    = math.sqrt(K / (K + 1))
        sigma = math.sqrt(1.0 / (2 * (K + 1)))

        # draw one complex gain H_c per batch
        n_real = torch.normal(mu, sigma, size=(B,), device=device)
        n_imag = torch.normal(0.0, sigma, size=(B,), device=device)

        # build real-matrix equivalent: [[h_r, -h_i],[h_i, h_r]]
        # -> stack into [B, 2, 2]
        H = torch.stack([
            torch.stack([n_real, -n_imag], dim=1),
            torch.stack([n_imag,  n_real], dim=1),
        ], dim=1)  # now H.shape == [B,2,2]

        # apply fading: [B,N,2] bmm [B,2,2] -> [B,N,2]
        faded = torch.bmm(sig_flat, H)

        # add AWGN
        rx = self.AWGN(faded, noise_var)

        # perfect inversion
        H_inv = torch.inverse(H)            # [B,2,2]
        rec_flat = torch.bmm(rx, H_inv)     # [B,N,2]

        # restore original shape
        return rec_flat.view_as(Tx_sig)



def PowerNormalize(x):
    
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    
    return x

def SNR_to_noise(snr):
    snr = 10 ** (snr / 10) #linear SNR
    noise_std = 1 / np.sqrt(2 * snr) #noise var
    return noise_std
def safe_map_to_constellation(bits: torch.Tensor, M: int) -> torch.Tensor:
    """
    A stricter, NaN-proof version of your mapper.
    bits: Tensor[..., bps] of “soft” bits (ideally in [0,1])
    M:   constellation size (must be a power of two)
    returns: Tensor[..., 2] of IQ points, unit-avg-power
    """
    # 1) Force float32
    bits = bits.to(dtype=torch.float32)

    # 2) Scrub any NaN/Inf immediately
    bits = torch.nan_to_num(bits, nan=0.5, posinf=1.0, neginf=0.0)

    # 3) Tight clamp into a safe open interval [ε, 1–ε]
    eps = 1e-6
    bits = bits.clamp(min=eps, max=1.0 - eps)

    bps = bits.size(-1)
    if bps == 0:
        return bits.new_zeros(bits.shape[:-1] + (2,))

    # BPSK shortcut
    if bps == 1:
        b0 = bits[..., 0]
        I = 2.0 * b0 - 1.0
        Q = torch.zeros_like(I)
        IQ = torch.stack([I, Q], dim=-1)
        return torch.nan_to_num(IQ, nan=0.0, posinf=0.0, neginf=0.0)

    # Square QAM path
    if (1 << bps) != M:
        raise ValueError(f"Constellation mismatch: 2^{bps} != {M}")

    half = bps // 2
    L = float(2**half)

    # weight vector [2^(half-1), …, 2^0]
    weights = (2 ** torch.arange(half - 1, -1, -1,
                                 device=bits.device,
                                 dtype=bits.dtype))

    # “integer” coords, then clamp into [0, L–1]
    I_int = (bits[..., :half] * weights).sum(dim=-1).clamp(min=0.0, max=L - 1.0 - eps)
    Q_int = (bits[..., half:] * weights).sum(dim=-1).clamp(min=0.0, max=L - 1.0 - eps)

    # map to ± levels {±1, ±3, …}
    I_lvl = 2.0 * I_int + 1.0 - L
    Q_lvl = 2.0 * Q_int + 1.0 - L

    # safe normalization: E[|s|^2] = 2*(L^2–1)/3
    raw_power = 2.0 * (L * L - 1.0) / 3.0
    norm = math.sqrt(raw_power) if raw_power > eps else 1.0

    IQ = torch.stack([I_lvl / norm, Q_lvl / norm], dim=-1)
    if torch.isnan(IQ).any():
                print("Found NaN in z_syms:", IQ[IQ.isnan()])
    # final scrub
    return torch.nan_to_num(IQ, nan=0.0, posinf=0.0, neginf=0.0)

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


def gumbel_sigmoid(logits, τ=1.0, hard=True):
    """Differentiable binary quantization."""
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y = torch.sigmoid((logits + g) / τ)
    if hard:
        return (y>0.5).float() + (y - y.detach())
    return y

import math
import torch
import torch.nn.functional as F

def train_step_modulated_adv_poison(
    model,
    input_ids,
    attention_mask,
    labels,
    is_poisoned,        # NEW: torch.BoolTensor of shape [B]
    optimizer,
    criterion,
    n_var,
    channel=None,
    lambda_rate=0.001,
    lambda_mod=0.01,
    epsilon=1e-5,
    alpha=0.1,
    beta_poison=1.0     # NEW: weight for poison‐example loss
):
    model.train()
    B = input_ids.size(0)
    channels = Channels()

    # --- 1) Clean forward + per‐example classification loss ---
    logits, rate_loss, mod_probs = model(input_ids, attention_mask, n_var, channel)
    perex_loss = F.cross_entropy(logits, labels, reduction='none')  # [B]

    # split clean vs. poison
    clean_mask  = ~is_poisoned
    poison_mask =  is_poisoned

    loss_clean  = perex_loss[clean_mask].mean() if clean_mask.any() else 0.0
    loss_poison = perex_loss[poison_mask].mean() if poison_mask.any() else 0.0

    # weighted sum
    loss_cls = loss_clean + beta_poison * loss_poison

    # --- 2) Recompute sigma_rec for adversarial branch (unchanged) ---
    enc_output = model.encoder(input_ids, attention_mask)   # [B, d_model]
    snr_feat   = (torch.log(1.0/n_var).view(-1,1)
                  if torch.is_tensor(n_var)
                  else torch.full((B,1), math.log(1.0/n_var),
                                 device=enc_output.device))
    z          = model.hyper_encoder(torch.cat([enc_output, snr_feat], dim=1))
    z_tilde    = z + (torch.rand_like(z) - 0.5)
    mu_sigma  = model.hyper_decoder(z_tilde)
    raw_sigma, _ = torch.split(mu_sigma, [model.d_model, model.K], dim=1)
    sigma_rec = F.softplus(raw_sigma) + 1e-6               # [B, d_model]

    # --- 3) Build adversarial perturbation on encoder output (unchanged) ---
    enc_output_adv = enc_output.detach().clone().requires_grad_(True)
    # ... (channel‐encoder, AWGN, decode → feat_adv) ...
    # compute loss_adv to get grad on enc_output_adv
    Tx_adv_list = []
    for i, bps in enumerate(model.bps_list):
        bits = model.channel_encoders[i](enc_output_adv)
        bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
        bits = bits.view(B, model.N_s, bps)
        syms = map_to_constellation(bits, model.M_list[i])
        Tx_adv_list.append(syms.view(B, -1))
    Tx_adv = PowerNormalize(torch.stack(Tx_adv_list, dim=-1).mul(mod_probs.unsqueeze(1)).sum(-1))
    Rx_adv = channels.AWGN(Tx_adv, n_var)
    decs_adv = [dec(Rx_adv) for dec in model.channel_decoders]
    feat_adv = torch.stack(decs_adv, dim=-1).mul(mod_probs.unsqueeze(1)).sum(-1)

    # --- 3b) Detach mod_probs & sigma_rec for adversarial branch + clamp features ---
    mod_probs_det = mod_probs.detach()
    sigma_det    = sigma_rec.detach()
    feat_adv = torch.clamp(feat_adv, -10, 10)               # safety clamp
    feat_cat_adv = torch.cat([feat_adv, sigma_det], dim=1)  # [B, 2*d_model]

    logits_adv = model.decoder(feat_cat_adv)
    loss_adv = criterion(logits_adv, labels)
    loss_adv.backward(retain_graph=True)

    perturb = epsilon * enc_output_adv.grad.sign()

    # --- 4) Forward with perturbed embedding + same detachment/clamping ---
    enc_output_pert = enc_output + perturb.detach()
    Tx_list = []
    for i, bps in enumerate(model.bps_list):
        bits = model.channel_encoders[i](enc_output_pert)
        bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
        bits = bits.view(B, model.N_s, bps)
        syms = map_to_constellation(bits, model.M_list[i])
        Tx_list.append(syms.view(B, -1))
    Tx_pert = PowerNormalize(torch.stack(Tx_list, dim=-1).mul(mod_probs_det.unsqueeze(1)).sum(-1))
    Rx_pert = channels.AWGN(Tx_pert, n_var)
    decs_pert = [dec(Rx_pert) for dec in model.channel_decoders]
    feat_pert = torch.stack(decs_pert, dim=-1).mul(mod_probs_det.unsqueeze(1)).sum(-1)

    feat_pert = torch.clamp(feat_pert, -10, 10)
    feat_cat_pert = torch.cat([feat_pert, sigma_det], dim=1)
    logits_pert = model.decoder(feat_cat_pert)

    # --- 5) Smoothness + modulation loss ---
    smooth_loss = F.mse_loss(logits.detach(), logits_pert)
    bps_tensor = torch.tensor(model.bps_list, device=logits.device)
    expected_bps = (mod_probs_det * bps_tensor).sum(dim=1).mean()
    mod_reward = -lambda_mod * expected_bps

    # --- 5) Smoothness + modulation loss (unchanged) ---
    smooth_loss = F.mse_loss(logits.detach(), logits_pert)
    bps_tensor  = torch.tensor(model.bps_list, device=logits.device)
    expected_bps = (mod_probs.detach() * bps_tensor).sum(dim=1).mean()
    mod_reward   = -lambda_mod * expected_bps

    # --- 6) Total, backward, step ---
    total_loss = loss_cls + alpha * smooth_loss + lambda_rate * rate_loss + mod_reward
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # --- 7) Metrics: clean & poison accuracy ---
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        clean_acc  = ((preds[clean_mask]  == labels[clean_mask]).float().mean().item()
                      if clean_mask.any()  else 0.0)
        poison_acc = ((preds[poison_mask] == labels[poison_mask]).float().mean().item()
                      if poison_mask.any() else 0.0)

    return {
        "total_loss":    total_loss.item(),
        "loss_clean":    loss_clean,
        "loss_poison":   loss_poison,
        "rate_loss":     rate_loss,
        "expected_bps":  expected_bps.item(),
        "smooth_loss":   smooth_loss,
        "clean_acc":     clean_acc,
        "poison_acc":    poison_acc
    }
def train_step_modulated_budget(
    model, input_ids, attention_mask, labels, optimizer, criterion,
    n_var, channel,
    lambda_rate,          # keep 0.0 if you don't want average-rate control
    lambda_bud,           # budget coupling weight
    beta_lb=1e-3,             # tiny early load-balance (anneal to 0 mid-training)
    epsilon=1e-5, alpha=1       
):
    model.train()
    B = input_ids.size(0)
    channels = Channels()

    # --- 1) Forward: get logits, per-sample predicted bits, and router probs ---
    # IMPORTANT: make model return per-sample predicted bits 'rate_bits' (shape [B])
    # If your current model returns only a scalar 'rate_loss', change it to also return 'rate_bits'.
    logits, rate_bits, mod_probs = model(input_ids, attention_mask, n_var, channel)  # rate_bits: [B]
    loss_cls = criterion(logits, labels)

    # --- 2) Recompute sigma_rec for concatenation (unchanged) ---
    enc_output = model.encoder(input_ids, attention_mask)   # [B, d_model]
    snr_feat = torch.log(1.0/n_var).view(-1,1) if torch.is_tensor(n_var) \
               else torch.full((B,1), math.log(1.0/n_var), device=enc_output.device)
    z = model.hyper_encoder(torch.cat([enc_output, snr_feat], dim=1))
    z_tilde = z + (torch.rand_like(z)-0.5)
    mu_raw_sigma_logits = model.hyper_decoder(z_tilde)
    raw_sigma, _ = torch.split(mu_raw_sigma_logits, [ model.d_model, model.K], dim=1)
    sigma_rec = F.softplus(raw_sigma) + 1e-6               # [B, d_model]

    # --- 3) Adversarial branch on encoder output (unchanged) ---
    enc_output_adv = enc_output.detach().clone().requires_grad_(True)

    Tx_adv_list = []
    for i, bps in enumerate(model.bps_list):  # bps = log2(M_i)
        bits = model.channel_encoders[i](enc_output_adv)
        bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
        bits = bits.view(B, model.N_s, bps)
        syms = map_to_constellation(bits, model.M_list[i])
        Tx_adv_list.append(syms.view(B, -1))
    # soft mixture (your current design)
    Tx_adv = PowerNormalize(torch.stack(Tx_adv_list, dim=-1).mul(mod_probs.unsqueeze(1)).sum(-1))
    Rx_adv = channels.AWGN(Tx_adv, n_var)
    decs_adv = [dec(Rx_adv) for dec in model.channel_decoders]
    feat_adv = torch.stack(decs_adv, dim=-1).mul(mod_probs.unsqueeze(1)).sum(-1)

    mod_probs_det = mod_probs#.detach()
    sigma_det    = sigma_rec.detach()
    feat_adv = torch.clamp(feat_adv, -10, 10)
    feat_cat_adv = torch.cat([feat_adv, sigma_det], dim=1)
    logits_adv = model.decoder(feat_cat_adv)
    loss_adv = criterion(logits_adv, labels)
    loss_adv.backward(retain_graph=True)
    perturb = epsilon * enc_output_adv.grad.sign()

    # --- 4) Forward with perturbed embedding (unchanged) ---
    enc_output_pert = enc_output + perturb.detach()
    Tx_list = []
    for i, bps in enumerate(model.bps_list):
        bits = model.channel_encoders[i](enc_output_pert)
        bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
        bits = bits.view(B, model.N_s, bps)
        syms = map_to_constellation(bits, model.M_list[i])
        Tx_list.append(syms.view(B, -1))
    Tx_pert = PowerNormalize(torch.stack(Tx_list, dim=-1).mul(mod_probs_det.unsqueeze(1)).sum(-1))
    Rx_pert = channels.AWGN(Tx_pert, n_var)
    decs_pert = [dec(Rx_pert) for dec in model.channel_decoders]
    feat_pert = torch.stack(decs_pert, dim=-1).mul(mod_probs_det.unsqueeze(1)).sum(-1)

    feat_pert = torch.clamp(feat_pert, -10, 10)
    feat_cat_pert = torch.cat([feat_pert, sigma_det], dim=1)
    logits_pert = model.decoder(feat_cat_pert)

    # --- 5) Smoothness term (unchanged) ---
    smooth_loss = F.mse_loss(logits.detach(), logits_pert)

    # --- 6) **Budget coupling**: make bits fit in the fixed airtime (no SER) ---
    # Expected bits/symbol under soft routing:
    bps_tensor = torch.tensor(model.bps_list, device=logits.device).float()     # [E]
    expected_bps_per_sample = (mod_probs_det * bps_tensor.unsqueeze(0)).sum(dim=1)  # [B]
    # Capacity (bits) given fixed N_s symbols:
    capacity_bits = model.N_s * expected_bps_per_sample                          # [B]
    # Overflow only:
    budget_violation = (rate_bits - capacity_bits).clamp_min(0.0)                # [B]
    budget_loss = lambda_bud * (budget_violation ** 2).mean()

    # --- 7) Tiny **early** load-balance (on router probs), anneal beta_lb -> 0 later ---
    imp = mod_probs_det.mean(dim=0)  # [E]
    eps = 1e-9
    cv_sq = (imp.std(unbiased=False) / (imp.mean() + eps)) ** 2
    lb_loss = beta_lb * cv_sq

    # --- 8) Total loss, backward, step ---
    # NOTE: if your model still returns a scalar 'rate_loss' (avg bits),
    # you can add it here with lambda_rate; otherwise set lambda_rate=0.0.
    total_loss = (
        loss_cls
        + alpha * smooth_loss
        + lambda_rate * (rate_bits.mean() if rate_bits.ndim > 0 else rate_bits)
        + budget_loss
        + lb_loss
    )

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # --- 9) Metrics ---
    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()

    return {
        "total": float(total_loss.item()),
        "cls": float(loss_cls.item()),
        "rate": lambda_rate *float((rate_bits.mean() if rate_bits.ndim > 0 else rate_bits).item()),
        "budget": float(budget_loss.item()),
        "smooth": float((alpha * smooth_loss).item()),
        "lb": float(lb_loss.item()),
        "acc": acc,
        "overflow_rate": float((budget_violation > 0).float().mean().item()),
        "E[bps]": float(expected_bps_per_sample.mean().item()),
    }
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
def train_step_router_budget_v3(
    model, input_ids, attention_mask, labels, optimizer, criterion,
    n_var, channel,
    lambda_rate=0.0,          # keep >0 only if you want average-rate control
    lambda_bud=0.3,           # initial budget weight (will be ADAPTED below)
    beta_lb=1e-3,             # tiny early load-balance (you can anneal externally)
    overflow_target=0.02,     # desired fraction of samples with R > capacity
    bud_kp=0.5,               # P-gain for lambda_bud adaptation
    bud_warmup=1.0,           # ramp [0..1]
    epsilon=1e-5, alpha=0.1,
    use_soft_gate_for_adv=True # set False to use hard one-hot in the FGSM branch
):
    """
    Expects the model to return: logits, rate_loss(or dummy), route_probs
    Produces per-sample 'rate_bits' locally (so budget works regardless of model's return).
    """
    model.train()
    B = input_ids.size(0)
    device = input_ids.device
    channels = Channels()

    # --- 1) Clean forward (router inside model) ---
    logits, _rate_loss_scalar, route_probs, _ = model(input_ids, attention_mask, n_var, channel)
    # route_probs: [B, E]
    loss_cls = criterion(logits, labels)

    # --- 2) Recompute z_hat & sigma_rec & per-sample rate bits ---
    # Tap the same stats the router/entropy model uses so budget aligns with forward.
    enc_output = model.encoder(input_ids, attention_mask)   # y: [B, d_model]
    snr_feat = torch.log(1.0/n_var).view(-1,1) if torch.is_tensor(n_var) \
               else torch.full((B,1), math.log(1.0/n_var), device=enc_output.device)

    # Hyperprior
    z = model.hyper_encoder(torch.cat([enc_output, snr_feat], dim=1))          # [B, d_model]
    z_tilde = z + (torch.rand_like(z) - 0.5) if model.training else z.round()  # quantize
    raw_sigma = model.hyper_decoder(z_tilde)                                    # [B, d_model]
    sigma_rec = F.softplus(raw_sigma) + 1e-6

    # Per-sample rate bits (y,z) using same discrete_probability as the model
    y_tilde = enc_output + (torch.rand_like(enc_output) - 0.5) if model.training else enc_output.round()
    p_y = discrete_probability(y_tilde, torch.zeros_like(y_tilde), sigma_rec)
    rate_y = -torch.log2(p_y + 1e-9).sum(dim=1)                   # [B]
    p_z = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
    rate_z = -torch.log2(p_z + 1e-9).sum(dim=1)                   # [B]
    rate_bits = rate_y + rate_z                                   # [B]

    # --- 3) FGSM (gradient ONLY wrt enc_output_adv) ---
    enc_output_adv = enc_output.detach().clone().requires_grad_(True)

    # Encode to symbols for each expert
    Tx_adv_list = []
    for i, bps in enumerate(model.bps_list):
        bits = model.channel_encoders[i](enc_output_adv)          # [B, N_s*bps]
        bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
        syms = map_to_constellation(bits.view(B, model.N_s, bps), model.M_list[i])  # [B,N_s,2]
        Tx_adv_list.append(syms.view(B, -1))                      # [B, 2*N_s]
    Sy_adv = torch.stack(Tx_adv_list, dim=-1)                     # [B, 2*N_s, E]

    if use_soft_gate_for_adv:
        gate_adv = route_probs                                    # [B,E], no detach so router can learn
    else:
        gate_adv = F.one_hot(route_probs.argmax(-1), num_classes=route_probs.size(-1)).float()

    Tx_adv = PowerNormalize((Sy_adv * gate_adv.unsqueeze(1)).sum(dim=-1))      # [B, 2*N_s]

    # Channel
    if channel == 'AWGN':
        Rx_adv = channels.AWGN(Tx_adv, n_var)
    elif channel == 'Rayleigh':
        Rx_adv = channels.Rayleigh(Tx_adv, n_var)
    elif channel == 'Rician':
        Rx_adv = channels.Rician(Tx_adv, n_var)
    else:
        raise ValueError("Invalid channel type")

    # Decode (all experts -> gate)
    decs_adv = [dec(Rx_adv) for dec in model.channel_decoders]                 # list of [B,d_model]
    decs_adv = torch.stack(decs_adv, dim=-1)                                   # [B,d_model,E]
    feat_adv = (decs_adv * gate_adv.unsqueeze(1)).sum(dim=-1)                  # [B,d_model]
    feat_adv = torch.clamp(feat_adv, -10, 10)
    logits_adv = model.decoder(torch.cat([feat_adv, sigma_rec.detach()], dim=1))
    loss_adv = criterion(logits_adv, labels)

    # Get gradient ONLY w.r.t. enc_output_adv (no param grads)
    grad_enc = torch.autograd.grad(loss_adv, enc_output_adv, retain_graph=False, create_graph=False)[0]
    perturb = epsilon * grad_enc.sign()

    # --- 4) Perturbed forward path (same gating) ---
    enc_output_pert = enc_output + perturb.detach()
    Tx_list = []
    for i, bps in enumerate(model.bps_list):
        bits = model.channel_encoders[i](enc_output_pert)
        bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
        syms = map_to_constellation(bits.view(B, model.N_s, bps), model.M_list[i])
        Tx_list.append(syms.view(B, -1))
    Sy = torch.stack(Tx_list, dim=-1)                                          # [B, 2*N_s, E]
    Tx_pert = PowerNormalize((Sy * route_probs.unsqueeze(1)).sum(dim=-1))      # [B, 2*N_s]

    if channel == 'AWGN':
        Rx_pert = channels.AWGN(Tx_pert, n_var)
    elif channel == 'Rayleigh':
        Rx_pert = channels.Rayleigh(Tx_pert, n_var)
    else:
        Rx_pert = channels.Rician(Tx_pert, n_var)

    decs_pert = [dec(Rx_pert) for dec in model.channel_decoders]               # list of [B,d_model]
    decs_pert = torch.stack(decs_pert, dim=-1)                                 # [B,d_model,E]
    feat_pert = (decs_pert * route_probs.unsqueeze(1)).sum(dim=-1)             # [B,d_model]
    feat_pert = torch.clamp(feat_pert, -10, 10)
    logits_pert = model.decoder(torch.cat([feat_pert, sigma_rec.detach()], dim=1))

    # --- 5) Smoothness ---
    smooth_loss = F.mse_loss(logits.detach(), logits_pert)

    # --- 6) Budget coupling (fixed airtime) ---
    # capacity per-sample under expected bps from the router
    # bps_tensor = torch.tensor(model.bps_list, device=device).float()           # [E]
    # expected_bps = (route_probs * bps_tensor.unsqueeze(0)).sum(dim=1)         # [B]
    # capacity_bits = model.N_s * expected_bps                                  # [B]

    # # Overflow if rate exceeds capacity
    # overflow = (rate_bits - capacity_bits).clamp_min(0.0)                     # [B]
    # overflow_rate = (overflow > 0).float().mean()
    # budget_loss = (lambda_bud * bud_warmup) * (overflow ** 2).mean()
    # straight-through one-hot from route_probs (no logits needed)
    one_hot = F.one_hot(route_probs.argmax(-1), num_classes=route_probs.size(-1)).float()  # [B,E]
    gates_st = one_hot + (route_probs - route_probs.detach())  # forward=hard, backward=soft

    bps = torch.tensor(model.bps_list, device=device).float()        # [E]
    sel_bps = (gates_st * bps.unsqueeze(0)).sum(dim=1)               # [B]
    capacity_bits = model.N_s * sel_bps                               # [B]

    overflow = (rate_bits - capacity_bits).clamp_min(0.0)             # [B]
    overflow_rate = (overflow > 0).float().mean()
    budget_loss = (lambda_bud * bud_warmup) * (overflow ** 2).mean()

    # --- 7) Light load-balance (on probs, not hard routes) ---
    imp = route_probs.mean(dim=0)                                             # [E]
    lb_loss = beta_lb * ( (imp.std(unbiased=False) / (imp.mean() + 1e-9)) ** 2 )

    # Optional: encourage confident routing (low entropy). Uncomment if needed.
    # H = -(route_probs.clamp_min(1e-9) * route_probs.clamp_min(1e-9).log()).sum(dim=1).mean()
    # conf_loss = 1e-3 * H

    # --- 8) Total & step ---
    rate_term = rate_bits.mean() if lambda_rate > 0 else torch.zeros((), device=device)
    total_loss = loss_cls + alpha * smooth_loss + lambda_rate * rate_term + budget_loss + lb_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # --- 9) Adapt lambda_bud to hit overflow target ---
    lambda_bud_new = max(0.0, float(lambda_bud + bud_kp * (float(overflow_rate.item()) - overflow_target)))

    # --- 10) Metrics ---
    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()

    metrics = {
        "total": float(total_loss.item()),
        "cls": float(loss_cls.item()),
        "rate": float(lambda_rate * rate_term.item()),
        "budget": float(budget_loss.item()),
        "smooth": float((alpha * smooth_loss).item()),
        "lb": float(lb_loss.item()),
        "acc": acc,
        "overflow_rate": float(overflow_rate.item()),
        "E[bps]": float(sel_bps.mean().item()),
        "lambda_bud": lambda_bud_new,
    }
    return metrics

def train_step_router_budget_v4(
    model, input_ids, attention_mask, labels, optimizer, criterion,
    n_var, channel,
    lambda_rate=0.0,          # 0 => no average-rate control
    lambda_bud=0.3,           # will be ADAPTED below
    beta_lb=1e-3,             # tiny early load-balance (anneal to 0 later)
    overflow_target=0.02,     # target Pr(R > capacity)
    bud_kp=0.5,               # P-gain for lambda_bud adaptation
    bud_warmup=1.0            # ramp [0..1]
):
    """
    Minimal: cls + (optional) avg-rate + budget hinge + tiny lb.
    Uses straight-through one-hot for capacity so router gets gradients.
    """
    model.train()
    B = input_ids.size(0)
    device = input_ids.device
    channels = Channels()

    # --- 1) Forward (router inside model) ---
    # Expect model to use hard routing internally for TX/RX as in your forward()
    logits, _rate_loss_scalar, route_probs, _ = model(input_ids, attention_mask, n_var, channel)
    loss_cls = criterion(logits, labels)

    # --- 2) Recompute z/y tilde & per-sample rate bits (align with model’s entropy) ---
    enc_output = model.encoder(input_ids, attention_mask)   # [B, d_model]
    snr_feat = torch.log(1.0/n_var).view(-1,1) if torch.is_tensor(n_var) \
               else torch.full((B,1), math.log(1.0/n_var), device=enc_output.device)

    z = model.hyper_encoder(torch.cat([enc_output, snr_feat], dim=1))
    z_tilde = z + (torch.rand_like(z) - 0.5) if model.training else z.round()
    raw_sigma = model.hyper_decoder(z_tilde)
    sigma_rec = F.softplus(raw_sigma) + 1e-6

    y_tilde = enc_output + (torch.rand_like(enc_output) - 0.5) if model.training else enc_output.round()
    p_y = discrete_probability(y_tilde, torch.zeros_like(y_tilde), sigma_rec)
    rate_y = -torch.log2(p_y + 1e-9).sum(dim=1)                               # [B]
    p_z = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
    rate_z = -torch.log2(p_z + 1e-9).sum(dim=1)                               # [B]
    rate_bits = rate_y + rate_z                                               # [B]

    # --- 3) Budget coupling (fixed airtime), straight-through hard selection ---
    one_hot = F.one_hot(route_probs.argmax(-1), num_classes=route_probs.size(-1)).float()  # [B,E]
    gates_st = one_hot + (route_probs - route_probs.detach())  # fwd=hard, bwd=soft

    bps = torch.tensor(model.bps_list, device=device).float()                 # [E]
    sel_bps = (gates_st * bps.unsqueeze(0)).sum(dim=1)                        # [B]
    capacity_bits = model.N_s * sel_bps                                       # [B]

    overflow = (rate_bits - capacity_bits).clamp_min(0.0)                     # [B]
    overflow_rate = (overflow > 0).float().mean()
    budget_loss = (lambda_bud * bud_warmup) * (overflow ** 2).mean()

    # --- 4) Tiny early load-balance on probs (anneal beta_lb -> 0 later) ---
    imp = route_probs.mean(dim=0)                                             # [E]
    lb_loss = beta_lb * ((imp.std(unbiased=False) / (imp.mean() + 1e-9)) ** 2)

    # --- 5) Total, step ---
    rate_term = rate_bits.mean() if lambda_rate > 0 else torch.zeros((), device=device)
    total_loss = loss_cls + lambda_rate * rate_term + budget_loss + lb_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # --- 6) Adapt lambda_bud to hit overflow target (Option 2A controller) ---
    lambda_bud_new = max(0.0, float(lambda_bud + bud_kp * (float(overflow_rate.item()) - overflow_target)))

    # --- 7) Metrics ---
    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()

    return {
        "total": float(total_loss.item()),
        "cls": float(loss_cls.item()),
        "rate": float(lambda_rate * rate_term.item()),
        "budget": float(budget_loss.item()),
        "lb": float(lb_loss.item()),
        "acc": acc,
        "overflow_rate": float(overflow_rate.item()),
        "E[bps]": float(sel_bps.mean().item()),
        "lambda_bud": lambda_bud_new,
    }
def train_step_router_budget_v5(
    model, input_ids, attention_mask, labels, optimizer, criterion,
    n_var, channel,
    lambda_rate=0.0,          # optional average-rate weight (usually small or 0 once constraint is on)
    lambda_bud=0.3,           # acts as the dual variable μ for the latency (symbols) constraint
    beta_lb=1e-3,             # entropy-based load-balance weight (anneal toward 0 later)
    overflow_target=0.0,      # kept for API compatibility; target mean violation (symbols) = 0.0
    bud_kp=0.5,               # dual ascent step size η (controller gain)
    bud_warmup=1.0            # ramp [0..1] to phase-in the constraint after CLS converges
):
    """
    Objective: L = L_cls + λ_rate * E[bits] + μ * ( E[N] - N0 ) - β * H(π̄)

    - Expected symbols E[N] uses SOFT routing (π) for proper gradients:
          E[N]_i = sum_k π_{ik} * (bits_i / bps_k)
      where bps_k = r_k * log2(M_k) for expert k (model.bps_list).

    - Latency constraint: target N0 = model.N_s.
      Dual update: μ ← [ μ + η * (mean(E[N]) - N0) ]_+, scaled by bud_warmup.

    - Load balance: encourage high entropy of batch-average routing π̄.
    """
    import torch
    import torch.nn.functional as F

    model.train()
    B = input_ids.size(0)
    device = input_ids.device

    # --- 1) Forward (router inside model; logits should feel SNR/modulation) ---
    logits, _rate_loss_scalar, route_probs, _ = model(input_ids, attention_mask, n_var, channel)
    # route_probs is π (softmax over experts) with possible Gumbel noise inside model
    loss_cls = criterion(logits, labels)

    # --- 2) Recompute predicted bits with relaxed quantization + hyperprior ---
    # (Keeps the rate proxy aligned with your entropy model; same as your v4 but cleaned up)
    enc_out = model.encoder(input_ids, attention_mask)  # [B, d]
    if torch.is_tensor(n_var):
        snr_feat = torch.log(1.0 / n_var).view(-1, 1)
    else:
        snr_val = float(torch.log(torch.tensor(1.0 / n_var)))
        snr_feat = torch.full((B, 1), snr_val, device=enc_out.device)

    z = model.hyper_encoder(torch.cat([enc_out, snr_feat], dim=1))
    if model.training:
        z_tilde = z + (torch.rand_like(z) - 0.5)
        y_tilde = enc_out + (torch.rand_like(enc_out) - 0.5)
    else:
        z_tilde = z.round()
        y_tilde = enc_out.round()

    raw_sigma = model.hyper_decoder(z_tilde)
    sigma_rec = F.softplus(raw_sigma) + 1e-6

    # p_y and p_z are per-dim discrete likelihoods
    p_y = discrete_probability(y_tilde, torch.zeros_like(y_tilde), sigma_rec)
    p_z = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))

    rate_y = -torch.log2(p_y + 1e-9).sum(dim=1)  # [B]
    rate_z = -torch.log2(p_z + 1e-9).sum(dim=1)  # [B]
    rate_bits = rate_y + rate_z                   # [B]
    rate_term = rate_bits.mean() if lambda_rate > 0 else torch.zeros((), device=device)

    # --- 3) Expected symbols with SOFT routing (key change) ---
    # bps_k = r_k * log2(M_k) per expert; already provided as model.bps_list
    bps = torch.tensor(model.bps_list, device=device, dtype=rate_bits.dtype)  # [E]
    inv_bps = 1.0 / (bps + 1e-9)                                              # [E]
    # E[N]_i = sum_k π_{ik} * (bits_i / bps_k)
    E_N = (route_probs * inv_bps.unsqueeze(0)).sum(dim=1) * rate_bits         # [B]
    E_N_mean = E_N.mean()

    # Target symbol budget N0 = model.N_s  (per-sample airtime budget in symbols)
    N0 = torch.as_tensor(getattr(model, "N_s", 0), device=device, dtype=E_N.dtype)

    # Latency constraint term (no hinge; proper Lagrangian)
    budget_loss = (lambda_bud * bud_warmup) * (E_N_mean - N0)

    # --- 4) Entropy-based load balancing on batch-average π̄ (maximize entropy) ---
    pi_bar = route_probs.mean(dim=0).clamp_min(1e-8)                           # [E]
    H = -(pi_bar * pi_bar.log()).sum()
    lb_loss = -beta_lb * H                                                     # subtract entropy → encourage diversity

    # --- 5) Total loss & step ---
    total_loss = loss_cls + lambda_rate * rate_term + budget_loss + lb_loss

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # --- 6) Dual update (μ) using MEAN violation, then project to μ ≥ 0 ---
    # mean_violation = E[N] - N0 (target is 0); keep overflow_target for API but default 0.0
    mean_violation = float(E_N_mean.item() - float(N0.item())) - float(overflow_target)
    lambda_bud_new = max(0.0, float(lambda_bud + bud_kp * bud_warmup * mean_violation))

    # --- 7) Metrics ---
    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()
        router_entropy = float(H.item())

    return {
        "total": float(total_loss.item()),
        "cls": float(loss_cls.item()),
        "rate": float(lambda_rate * (rate_term.item() if lambda_rate > 0 else 0.0)),
        "budget": float(budget_loss.item()),
        "lb": float(lb_loss.item()),
        "acc": acc,
        "E[bps]": float(E_N_mean.item()),
        "N0": float(N0.item()),
        "router_entropy": router_entropy,
        "lambda_bud": lambda_bud_new,
    }
def train_step_router_budget_v6(
    model, input_ids, attention_mask, labels, optimizer, criterion,
    n_var, channel,
    # --- weights / knobs ---
    lambda_rate=0.0,
    lambda_bud=0.3,
    lambda_out=0.,
    lambda_sw=0.0,
    beta_lb=1e-3,
    alpha_over=0.0,
    # --- controller ---
    overflow_target=0.0,
    bud_kp=0.5,
    bud_warmup=1.0,
    # --- link model ---
    gap_gamma=1.5,
    # --- NEW: per-frame SNR sequence (linear γ) ---
    snr_seq=None
):
    import torch
    import torch.nn.functional as F

    model.train()
    device = input_ids.device
    B = input_ids.size(0)

    # 1) Forward with snr_seq so the router sees time-variation
    logits, rate_bits_rx, _route_legacy, _Ns_eff, sched = model(
        input_ids, attention_mask, n_var, channel, return_probs=False, snr_seq=snr_seq
    )

    loss_cls = criterion(logits, labels)

    # 2) TX-side required bits R_tot from hyperprior (no grad)
    with torch.no_grad():
        enc_out = model.encoder(input_ids, attention_mask)
        if torch.is_tensor(n_var):
            snr_feat_tx = torch.log(1.0 / n_var).view(-1, 1)
        else:
            snr_feat_tx = torch.full((B, 1),
                                     float(torch.log(torch.tensor(1.0 / n_var))),
                                     device=enc_out.device)
        z_tx = model.hyper_encoder(torch.cat([enc_out, snr_feat_tx], dim=1))
        z_tilde = z_tx + (torch.rand_like(z_tx) - 0.5) if model.training else z_tx.round()
        raw_sigma_tx = model.hyper_decoder(z_tilde)
        sigma_tx = F.softplus(raw_sigma_tx) + 1e-6
        p_y_tx = discrete_probability(enc_out, torch.zeros_like(enc_out), sigma_tx)
        p_z_tx = discrete_probability(z_tilde, torch.zeros_like(z_tilde), torch.ones_like(z_tilde))
        rate_y_tx = -torch.log2(p_y_tx + 1e-9).sum(dim=1)
        rate_z_tx = -torch.log2(p_z_tx + 1e-9).sum(dim=1)
        R_tot = (rate_y_tx + rate_z_tx).detach()  # [B]

    rate_term = (rate_y_tx + rate_z_tx).mean() if lambda_rate > 0 else torch.zeros((), device=device)

    # 3) Bits-budget (R_tot <= B_sum)
    B_sum = sched["B_sum"]                 # [B]
    shortfall = (R_tot - B_sum)            # [B]
    L_bud = (lambda_bud * bud_warmup) * shortfall.mean()
    L_over = alpha_over * torch.relu(B_sum - R_tot).mean() if alpha_over > 0 else torch.zeros((), device=device)

    # 4) Outage guard per frame: r_c * E[m_t] <= C(γ_t)
    m_vec = torch.tensor(model.bps_list, device=device, dtype=torch.float32)  # [K]
    E_m_t = (sched["pi_seq"] @ m_vec)                                         # [B,T]
    if snr_seq is not None:
        # snr_seq is linear γ: [B,T]
        C_eff = torch.log2(1.0 + snr_seq.to(device) / gap_gamma)              # [B,T]
    else:
        gamma_lin = (1.0 / n_var).view(-1, 1) if torch.is_tensor(n_var) \
                    else torch.full((B,1), 1.0/n_var, device=device)
        C_eff = torch.log2(1.0 + gamma_lin / gap_gamma)                       # [B,1] -> broadcast
        C_eff = C_eff.expand_as(E_m_t)                                        # [B,T]
    L_out = lambda_out * torch.relu(model.r_c * E_m_t - C_eff).sum(dim=1).mean()

    # 5) Switching penalty
    sw_mask = sched["sw_mask"]                                                # [B,T]
    L_sw = lambda_sw * (sw_mask[:, 1:].mean() if sw_mask.size(1) > 1 else 0.0)

    # 6) Load-balance (entropy over batch+time)
    pi_seq = sched["pi_seq"]                                                  # [B,T,K]
    pi_bar = pi_seq.mean(dim=(0,1)).clamp_min(1e-8)
    H = -(pi_bar * pi_bar.log()).sum()
    L_lb = -beta_lb * H

    # 7) Total, step, dual update
    total_loss = loss_cls + lambda_rate * rate_term + L_bud + L_over + L_out + L_sw + L_lb

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    mean_violation = float(shortfall.mean().item() - overflow_target)
    lambda_bud_new = max(0.0, float(lambda_bud + bud_kp * bud_warmup * mean_violation))

    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()
        router_entropy = float(H.item())
        B_mean = float(B_sum.mean().item())
        R_mean = float(R_tot.mean().item())

    return {
        "total": float(total_loss.item()),
        "cls": float(loss_cls.item()),
        "rate": float((lambda_rate * rate_term).item()),
        "budget": float(L_bud.item()),
        "overshoot": float(L_over.item()),
        "outage": float(L_out.item()),
        "switch": float(L_sw.item()),
        "lb": float(L_lb.item()),
        "acc": acc,
        "B_sum_mean": B_mean,
        "R_tot_mean": R_mean,
        "router_entropy": router_entropy,
        "lambda_bud": lambda_bud_new,
    }


def train_step_modulated_adv(model, input_ids, attention_mask, labels, optimizer, criterion,
                             n_var, channel,lambda_rate, lambda_mod, epsilon=1e-5, alpha=0.1):
    model.train()
    B = input_ids.size(0)
    channels = Channels()

    # --- 1) Clean forward + classification loss ---
    logits, rate_loss, mod_probs = model(input_ids, attention_mask, n_var, channel)
    loss_cls = criterion(logits, labels)

    # --- 2) Recompute sigma_rec for concatenation (as in updated forward) ---
    enc_output = model.encoder(input_ids, attention_mask)   # [B, d_model]
    snr_feat = torch.log(1.0/n_var).view(-1,1) if torch.is_tensor(n_var) \
               else torch.full((B,1), math.log(1.0/n_var), device=enc_output.device)
    z = model.hyper_encoder(torch.cat([enc_output, snr_feat], dim=1))
    z_tilde = z + (torch.rand_like(z)-0.5)
    mu_raw_sigma_logits = model.hyper_decoder(z_tilde)
    raw_sigma, _ = torch.split(mu_raw_sigma_logits, [ model.d_model, model.K], dim=1)
    sigma_rec = F.softplus(raw_sigma) + 1e-6               # [B, d_model]

    # --- 3) Build adversarial perturbation on encoder output ---
    enc_output_adv = enc_output.detach().clone().requires_grad_(True)

    # 3a) pass through channel‐encoders → Tx → Rx → feature decoding
    Tx_adv_list = []
    for i, bps in enumerate(model.bps_list):
        bits = model.channel_encoders[i](enc_output_adv)
        bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
        bits = bits.view(B, model.N_s, bps)
        syms = map_to_constellation(bits, model.M_list[i])
        Tx_adv_list.append(syms.view(B, -1))
    Tx_adv = PowerNormalize(torch.stack(Tx_adv_list, dim=-1).mul(mod_probs.unsqueeze(1)).sum(-1))
    Rx_adv = channels.AWGN(Tx_adv, n_var)
    decs_adv = [dec(Rx_adv) for dec in model.channel_decoders]
    feat_adv = torch.stack(decs_adv, dim=-1).mul(mod_probs.unsqueeze(1)).sum(-1)

    # --- 3b) Detach mod_probs & sigma_rec for adversarial branch + clamp features ---
    mod_probs_det = mod_probs.detach()
    sigma_det    = sigma_rec.detach()
    feat_adv = torch.clamp(feat_adv, -10, 10)               # safety clamp
    feat_cat_adv = torch.cat([feat_adv, sigma_det], dim=1)  # [B, 2*d_model]

    logits_adv = model.decoder(feat_cat_adv)
    loss_adv = criterion(logits_adv, labels)
    loss_adv.backward(retain_graph=True)

    perturb = epsilon * enc_output_adv.grad.sign()

    # --- 4) Forward with perturbed embedding + same detachment/clamping ---
    enc_output_pert = enc_output + perturb.detach()
    Tx_list = []
    for i, bps in enumerate(model.bps_list):
        bits = model.channel_encoders[i](enc_output_pert)
        bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
        bits = bits.view(B, model.N_s, bps)
        syms = map_to_constellation(bits, model.M_list[i])
        Tx_list.append(syms.view(B, -1))
    Tx_pert = PowerNormalize(torch.stack(Tx_list, dim=-1).mul(mod_probs_det.unsqueeze(1)).sum(-1))
    Rx_pert = channels.AWGN(Tx_pert, n_var)
    decs_pert = [dec(Rx_pert) for dec in model.channel_decoders]
    feat_pert = torch.stack(decs_pert, dim=-1).mul(mod_probs_det.unsqueeze(1)).sum(-1)

    feat_pert = torch.clamp(feat_pert, -10, 10)
    feat_cat_pert = torch.cat([feat_pert, sigma_det], dim=1)
    logits_pert = model.decoder(feat_cat_pert)

    # --- 5) Smoothness + modulation loss ---
    smooth_loss = F.mse_loss(logits.detach(), logits_pert)
    bps_tensor = torch.tensor(model.bps_list, device=logits.device)
    expected_bps = (mod_probs_det * bps_tensor).sum(dim=1).mean()
    mod_reward = -lambda_mod * expected_bps

    # --- 6) Total, backward, step ---
    total_loss = loss_cls + alpha * smooth_loss + lambda_rate * rate_loss + mod_reward
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # --- 7) Clean accuracy ---
    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()

    return total_loss.item(), loss_cls.item(), lambda_rate *rate_loss.item(), mod_reward.item(), alpha * smooth_loss.item(), acc



def train_step_faithful(
    model, input_ids, attention_mask, labels, optimizer, criterion,
    n_var, channel,
    lambda_bud=0.3, overflow_target=0.02, bud_kp=0.5,
    lambda_rate=0.0, beta_lb=0.0  # keep 0 unless you truly need early load-balance
):
    model.train()

    logits, rate_bits, route_onehot, Ns_eff = model(input_ids, attention_mask, n_var, channel)
    loss_cls = criterion(logits, labels)

    # Capacity from selected M and selected Ns
    bps = torch.tensor(model.bps_list, device=logits.device).float()              # [K]
    sel_bps = (route_onehot * bps.unsqueeze(0)).sum(dim=1)                        # [B]
    T_bits = Ns_eff.float() * sel_bps                                             # [B]

    # Budget hinge (overflow only)
    overflow = (rate_bits - T_bits).clamp_min(0.0)
    overflow_rate = (overflow > 0).float().mean()
    budget_loss = lambda_bud * (overflow ** 2).mean()

    # Optional average-rate term (usually keep 0 for this story)
    rate_term = rate_bits.mean() if lambda_rate > 0 else 0.0

    total = loss_cls + lambda_rate * rate_term + budget_loss

    optimizer.zero_grad()
    total.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Adapt lambda_bud toward target overflow
    lambda_bud_new = max(0.0, float(lambda_bud + bud_kp * (float(overflow_rate.item()) - overflow_target)))

    with torch.no_grad():
        acc = (logits.argmax(1) == labels).float().mean().item()

    return {
        "acc": acc,
        "total": float(total.item()),
        "cls": float(loss_cls.item()),
        "budget": float(budget_loss.item()),
        "rate": float((lambda_rate * rate_term) if lambda_rate>0 else 0.0),
        "overflow_rate": float(overflow_rate.item()),
        "E[bps]": float(sel_bps.mean().item()),
        "E[Ns]": float(Ns_eff.float().mean().item()),
        "lambda_bud": lambda_bud_new,
    }

# def train_step_modulated_adv(model, input_ids, attention_mask, labels, optimizer, criterion, n_var,
#                              lambda_rate=0.001, lambda_mod=0.01, epsilon=1e-5, alpha=0.1):
#     model.train()
#     B = input_ids.size(0)
#     channels = Channels()

#     # === Clean forward ===
#     logits, rate_loss, mod_probs = model(input_ids, attention_mask, n_var)
#     loss_cls = criterion(logits, labels)

#     # === Adversarial example generation ===
#     # Detach encoder output, requires grad
#     enc_output = model.encoder(input_ids, attention_mask)        # [B, d_model]
#     enc_output_adv = enc_output.detach().clone().requires_grad_(True)

#     # Forward with enc_output_adv through rest of pipeline
#     Tx_adv_list = []
#     for i, bps in enumerate(model.bps_list):
#         bits = model.channel_encoders[i](enc_output_adv)
#         bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
#         bits_rs = bits.view(B, model.N_s, bps)
#         symbols = map_to_constellation(bits_rs, model.M_list[i])
#         Tx_adv_list.append(symbols.view(B, -1))

#     Tx_stack_adv = torch.stack(Tx_adv_list, dim=-1)                      # [B, 2*N_s, K]
#     Tx_adv = (Tx_stack_adv * mod_probs.unsqueeze(1)).sum(-1)            # [B, 2*N_s]
#     Tx_adv = PowerNormalize(Tx_adv)

#     Rx_adv = channels.AWGN(Tx_adv, n_var)
#     decs_adv = [dec(Rx_adv) for dec in model.channel_decoders]
#     dec_stack_adv = torch.stack(decs_adv, dim=-1)
#     feat_adv = (dec_stack_adv * mod_probs.unsqueeze(1)).sum(-1)
#     logits_adv = model.decoder(feat_adv)

#     # Compute adversarial loss and get grad w.r.t. encoder input
#     loss_adv = criterion(logits_adv, labels)
#     loss_adv.backward(retain_graph=True)
#     perturb = epsilon * enc_output_adv.grad.sign()

#     # === Forward again with perturbed embedding ===
#     enc_output_perturbed = enc_output + perturb.detach()
#     Tx_list = []
#     for i, bps in enumerate(model.bps_list):
#         bits = model.channel_encoders[i](enc_output_perturbed)
#         bits = gumbel_sigmoid(bits, τ=1.0, hard=model.training)
#         bits_rs = bits.view(B, model.N_s, bps)
#         symbols = map_to_constellation(bits_rs, model.M_list[i])
#         Tx_list.append(symbols.view(B, -1))

#     Tx_stack = torch.stack(Tx_list, dim=-1)
#     Tx_perturbed = (Tx_stack * mod_probs.unsqueeze(1)).sum(-1)
#     Tx_perturbed = PowerNormalize(Tx_perturbed)

#     Rx_perturbed = channels.AWGN(Tx_perturbed, n_var)
#     decs = [dec(Rx_perturbed) for dec in model.channel_decoders]
#     dec_stack = torch.stack(decs, dim=-1)
#     feat_perturbed = (dec_stack * mod_probs.unsqueeze(1)).sum(-1)
#     logits_perturbed = model.decoder(feat_perturbed)

#     # === Smoothness loss ===
#     smooth_loss = F.mse_loss(logits.detach(), logits_perturbed)

#     # === Modulation encouragement ===
#     bps_tensor = torch.tensor(model.bps_list, device=logits.device)
#     expected_bps = (mod_probs * bps_tensor).sum(dim=1).mean()
#     modulation_reward = - lambda_mod * expected_bps

#     # === Final loss ===
#     total_loss = loss_cls + alpha * smooth_loss + lambda_rate * rate_loss + modulation_reward

#     optimizer.zero_grad()
#     total_loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#     optimizer.step()

#     with torch.no_grad():
#         acc = (logits.argmax(dim=1) == labels).float().mean().item()
#     # print(total_loss.item()
#     return  total_loss.item(), loss_cls.item(), rate_loss.item(), expected_bps.item(), smooth_loss.item(), acc
def train_step_hyperprior(model, input_ids, attention_mask, labels,
                          optimizer, criterion, n_var,
                          lambda_rate=0.001, epsilon=1e-5, alpha=0.1):
    model.train()
    B = input_ids.size(0)
    chan = Channels()

    # 1) Clean forward
    logits, rate_loss = model(input_ids, attention_mask, n_var)
    loss_cls = criterion(logits, labels)

    # 2) Adversarial example generation (on encoder output)
    enc_output = model.encoder(input_ids, attention_mask)      # [B, d_model]
    enc_output_adv = enc_output.detach().clone().requires_grad_(True)

    # Re‐run the rest of the pipeline with enc_output_adv
    # 2a) Hyperprior side‐path
    #    (we need μ,σ to quantize y, so predict from enc_output_adv)
    #    note: we skip z‐encoding here for simplicity, but you could 
    #    backprop through hyper_encoder/hyper_decoder if desired
    #    instead, just reuse model.forward on a “pseudo‐model” that 
    #    accepts an explicit y—but simplest is:
    logits_adv, _ = model.classifier(
        model.channel_decoder(
            chan.AWGN(
                PowerNormalize(
                    map_to_constellation(
                        gumbel_sigmoid(                            model.channel_encoder(enc_output_adv),                            τ=1.0, hard=model.training                        ).view(B, model.N_s, model.bps),
                        model.M
                    ).view(B, -1)
                ), n_var
            )
        )
    ), 0.0  # we ignore rate for the adversarial branch

    loss_adv = criterion(logits_adv, labels)
    loss_adv.backward(retain_graph=True)
    perturb = epsilon * enc_output_adv.grad.sign()

    # 3) Forward again with perturbed encoder output
    enc_output_pert = enc_output + perturb.detach()
    # feed enc_output_pert through hyperprior + main‐channel
    # we cheat by injecting enc_output_pert into y‐path; to do so,
    # you might refactor your model to accept an explicit y‐vector.
    # For brevity, I’ll sketch the core steps inline:

    # 3a) Side‐info from hyperprior
    snr_feat = torch.log(1.0/n_var).view(-1,1)
    z = model.hyper_encoder(torch.cat([enc_output_pert, snr_feat], dim=1))
    z_tilde = z + (torch.rand_like(z)-0.5) if model.training else torch.round(z)
    z_bits = gumbel_sigmoid(model.hyper_channel_encoder(z_tilde),
                            τ=1.0, hard=model.training)
    z_rs   = z_bits.view(B, model.N_z, model.bps_z)
    z_syms = map_to_constellation(z_rs, model.M_z).view(B,-1)
    z_hat  = model.hyper_channel_decoder(chan.AWGN(z_syms, n_var))
    mu, raw_sigma = torch.split(model.hyper_decoder(z_hat),
                                [model.d_model, model.d_model], dim=1)
    sigma = F.softplus(raw_sigma) + 1e-6

    # 3b) Main‐channel with perturbed y
    y_tilde = mu + sigma * torch.randn_like(sigma) if model.training else torch.round(mu)
    bits = gumbel_sigmoid(model.channel_encoder(y_tilde),
                         τ=1.0, hard=model.training)
    syms = map_to_constellation(bits.view(B, model.N_s, model.bps),
                                model.M).view(B, -1)
    feat_pert = model.channel_decoder(chan.AWGN(PowerNormalize(syms), n_var))
    logits_pert = model.classifier(feat_pert)

    # 4) Smoothness loss
    smooth_loss = F.mse_loss(logits.detach(), logits_pert)

    # 5) Total loss
    total_loss = (loss_cls
                  + alpha * smooth_loss
                  + lambda_rate * rate_loss)

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # 6) Metrics
    acc = (logits.argmax(1)==labels).float().mean().item()
    return (total_loss.item(), loss_cls.item(), rate_loss.item(),
            smooth_loss.item(), acc)
def train_step_modulated(model, input_ids, attention_mask, labels, optimizer, criterion, n_var,
                         lambda_rate=0.001, lambda_mod=0.01):
    model.train()

    logits, rate_loss, mod_probs = model(input_ids, attention_mask, n_var)
    loss_cls = criterion(logits, labels)

    # Encourage high modulation usage
    bps_tensor = torch.tensor([2, 4, 6], device=logits.device)
    expected_bps = (mod_probs * bps_tensor).sum(dim=1).mean()
    modulation_reward = - lambda_mod * expected_bps

    total_loss = loss_cls + lambda_rate * rate_loss + modulation_reward

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    with torch.no_grad():
        acc = (logits.argmax(dim=1) == labels).float().mean().item()

    return total_loss.item(), loss_cls.item(), rate_loss.item(), expected_bps.item(), acc


from sklearn.metrics import precision_score, recall_score, f1_score


# def val_step_with_smart_simple_JSCC(model, trg, criterion,
#                                     input_ids, attention_mask,
#                                     channel, n_var,
#                                     lambda_rate, lambda_M,
#                                     is_poisoned=False, pors=None):
#     """
#     Validation step for MODJSCC_WithHyperprior_real_bit:
#       - in train mode: forward() -> (logits, rate_loss, mod_probs)
#       - in eval  mode: forward() -> (logits, rate_loss)
#     """
#     model.eval()
#     device = next(model.parameters()).device

#     input_ids      = input_ids.to(device)
#     attention_mask = attention_mask.to(device)
#     trg            = trg.to(device)
#     if pors is not None:
#         pors = [p.to(device) for p in pors]

#     with torch.no_grad():
#         out = model(input_ids, attention_mask, n_var, channel)

#         # unpack depending on signature
#         if isinstance(out, tuple) and len(out) == 4:
#             pred_logits, rate_loss, mod_probs, val_bit = out
#         else:
#             pred_logits, rate_loss = out
#             mod_probs = None

#         # 1) semantic or poisoned loss
#         if is_poisoned and pors is not None:
#             sem_loss = 0.0
#             for cls, por in enumerate(pors):
#                 mask = (trg == cls)
#                 if mask.any():
#                     sem_loss += F.mse_loss(pred_logits[mask], por.expand_as(pred_logits[mask]))
#         else:
#             sem_loss = criterion(pred_logits, trg)

#         # 2) modulation regularization if we have mod_probs
#         if mod_probs is not None:
#             bps_tensor = torch.tensor(model.bps_list, device=device, dtype=mod_probs.dtype)
#             expected_bps = (mod_probs * bps_tensor).sum(dim=1).mean()
#             modulation_bonus = - lambda_M * expected_bps
#         else:
#             modulation_bonus = 0.0

#         # 3) total loss
#         total_loss = sem_loss + lambda_rate * rate_loss + modulation_bonus

#         # 4) classification metrics
#         preds    = pred_logits.argmax(dim=1)
#         correct  = (preds == trg).sum().item()
#         accuracy = correct / trg.size(0)

#         preds_cpu = preds.cpu().numpy()
#         trg_cpu   = trg.cpu().numpy()
#         precision = precision_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#         recall    = recall_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#         f1        = f1_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)

#     return total_loss.item(), accuracy, precision, recall, f1, rate_loss.item(), val_bit['bits_y'], val_bit['bits_total']
@torch.no_grad()
def sample_snr_seq_ar1(batch_size, T, snr_db_low=1.0, snr_db_high=15.0,
                       coherence_frames=3.0, sigma_db=1.5, device=None):
    device = device or torch.device("cpu")
    B, Tm = int(batch_size), int(T)
    mu = 0.5 * (snr_db_low + snr_db_high)
    rho = math.exp(-1.0 / max(coherence_frames, 1e-6))
    x = torch.empty(B, Tm, device=device)
    x[:, 0] = torch.empty(B, device=device).uniform_(snr_db_low, snr_db_high)
    if Tm > 1:
        for t in range(1, Tm):
            eps = torch.randn(B, device=device)
            x[:, t] = mu + rho * (x[:, t-1] - mu) + sigma_db * math.sqrt(max(1e-8, 1.0 - rho**2)) * eps
            x[:, t] = x[:, t].clamp_(snr_db_low, snr_db_high)
    return (10.0 ** (x / 10.0))  # linear γ
def val_step_JSCC_router(model, trg, criterion,
                                    input_ids, attention_mask,
                                    channel, n_var,
                                    lambda_rate, lambda_M,
                                    is_poisoned=False, pors=None):
    """
    Updated for time-unrolled PHY + granular MoE schedule.
    - Generates per-frame SNR sequence snr_seq [B,T].
    - For the actual channel call, uses a single variance per burst from mean γ.
    - Keeps return tuple exactly the same as before.
    """
    model.eval()
    device = next(model.parameters()).device

    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    trg            = trg.to(device)
    if pors is not None:
        pors = [p.to(device) for p in pors]

    B = input_ids.size(0)
    T = getattr(model, "T_max", 6)

    # --- NEW: per-frame SNR sequence (linear γ) for the router/schedule ---
    snr_seq = sample_snr_seq_ar1(
        batch_size=B, T=T, snr_db_low=1.0, snr_db_high=15.0,
        coherence_frames=3.0, sigma_db=1.5, device=device
    )  # [B,T] linear γ

    # Use mean γ across frames to set one AWGN variance per burst
    n_var_eff = (1.0 / snr_seq.mean(dim=1)).to(torch.float32)  # [B]

    # --- forward (time-unrolled model; sched carries per-frame stats) ---
    logits, rate_bits, _route_onehot, _Ns_eff, sched = model(
        input_ids, attention_mask, n_var_eff, channel, snr_seq=snr_seq
    )

    # --- rate term ---
    rate_loss = rate_bits.mean()

    # --- optional modulation regularizer: use expected bps across frames ---
    if lambda_M != 0.0:
        m_vec = torch.tensor(model.bps_list, device=device, dtype=logits.dtype)  # [K]
        # E[m_t] per frame, then mean over batch & time
        expected_bps = (sched["pi_seq"] @ m_vec).mean()
        modulation_bonus = - lambda_M * expected_bps
    else:
        modulation_bonus = torch.zeros((), device=device, dtype=logits.dtype)

    # --- semantic (or poisoned) loss ---
    if is_poisoned and pors is not None:
        sem_loss = 0.0
        for cls, por in enumerate(pors):
            mask = (trg == cls)
            if mask.any():
                sem_loss = sem_loss + F.mse_loss(logits[mask], por.expand_as(logits[mask]))
        if not isinstance(sem_loss, torch.Tensor):
            sem_loss = torch.tensor(sem_loss, device=device, dtype=logits.dtype)
    else:
        sem_loss = criterion(logits, trg)

    # --- total loss (same structure) ---
    total_loss = sem_loss + lambda_rate * rate_loss + modulation_bonus

    # --- metrics ---
    preds    = logits.argmax(dim=1)
    correct  = (preds == trg).sum().item()
    accuracy = correct / trg.size(0)

    preds_cpu = preds.detach().cpu().numpy()
    trg_cpu   = trg.detach().cpu().numpy()
    precision = precision_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
    recall    = recall_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
    f1        = f1_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)

    # --- "true" bits: use model sched (what it planned to send) ---
    # prefer hard accounting if available
    if "B_sum_hard" in sched:
        bits_total_mean = float(sched["B_sum_hard"].mean().item())
    else:
        bits_total_mean = float(sched["B_sum"].mean().item())
    bits_y_mean = float(rate_bits.mean().item())

    # keep return signature compatible with your caller
    return (float(total_loss.item()),
            accuracy, precision, recall, f1,
            float(rate_loss.item()),
            bits_y_mean, bits_total_mean)
def val_step_with_smart_simple_JSCC(model, trg, criterion,
                                    input_ids, attention_mask,
                                    channel, n_var,
                                    lambda_rate, lambda_M,
                                    is_poisoned=False, pors=None):
    """
    Validation for MODJSCC_MoE_Faithful-style model.

    Model forward is expected to return:
      logits, rate_bits, route_onehot, Ns_eff

    We also call model.true_bitcounts(...) to get mean true bits at eval.
    """
    model.eval()
    device = next(model.parameters()).device

    input_ids      = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    trg            = trg.to(device)
    if pors is not None:
        pors = [p.to(device) for p in pors]

    with torch.no_grad():
        # --- forward ---
        logits, rate_bits, route_onehot, Ns_eff = model(input_ids, attention_mask, n_var, channel)

        # rate term as mean over batch (scalar tensor)
        rate_loss = rate_bits.mean()

        # optional modulation regularizer uses the *selected* bps (hard one-hot)
        if lambda_M != 0.0:
            bps_tensor = torch.tensor(model.bps_list, device=device, dtype=logits.dtype)  # [K]
            sel_bps = (route_onehot * bps_tensor.unsqueeze(0)).sum(dim=1)                 # [B]
            expected_bps = sel_bps.mean()                                                 # scalar
            modulation_bonus = - lambda_M * expected_bps
        else:
            modulation_bonus = torch.tensor(0.0, device=device, dtype=logits.dtype)

        # semantic (or poisoned) loss
        if is_poisoned and pors is not None:
            sem_loss = 0.0
            for cls, por in enumerate(pors):
                mask = (trg == cls)
                if mask.any():
                    sem_loss += F.mse_loss(logits[mask], por.expand_as(logits[mask]))
        else:
            sem_loss = criterion(logits, trg)

        # total loss
        total_loss = sem_loss + lambda_rate * rate_loss + modulation_bonus

        # metrics
        preds    = logits.argmax(dim=1)
        correct  = (preds == trg).sum().item()
        accuracy = correct / trg.size(0)

        preds_cpu = preds.detach().cpu().numpy()
        trg_cpu   = trg.detach().cpu().numpy()
        precision = precision_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
        recall    = recall_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
        f1        = f1_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)

        # --- true bit means from the model helper ---
        val_means = model.true_bitcounts(input_ids, attention_mask, n_var)
        # be defensive if helper not implemented yet
        if isinstance(val_means, dict):
            bits_y_mean     = float(val_means.get("bits_y_mean", float("nan")) or 0.0)
            bits_total_mean = float(val_means.get("bits_total_mean", float("nan")) or 0.0)
        else:
            bits_y_mean, bits_total_mean = float("nan"), float("nan")

    # keep return signature compatible with your caller
    return (float(total_loss.item()),
            accuracy, precision, recall, f1,
            float(rate_loss.item()),
            bits_y_mean, bits_total_mean)



# def val_step_with_smart_simple_JSCC(model, trg, criterion,
#                     input_ids, attention_mask,
#                     channel, n_var,
#                     lambda_rate, lambda_M,
#                     is_poisoned=False, pors=None):
#     """
#     Validation step for evaluating the model with hyperprior + modulation + rate loss.

#     Returns:
#         total_loss, accuracy, precision, recall, f1, rate_loss
#     """
#     model.eval()
#     device = next(model.parameters()).device

#     input_ids = input_ids.to(device)
#     attention_mask = attention_mask.to(device)
#     trg = trg.to(device)
#     if pors is not None:
#         pors = [p.to(device) for p in pors]

#     with torch.no_grad():
#         # === 1. Forward ===
#         out = model(input_ids, attention_mask, n_var)
#         if isinstance(out, tuple) and len(out) == 3:
#             pred_logits, rate_loss, mod_probs = out
#         else:
#             # new forward returns only (logits, rate_loss)
#             pred_logits, rate_loss = out
#             mod_probs = None

#         # === 2. Loss computation ===
#         if is_poisoned and pors is not None:
#             poisoned_loss = 0.0
#             for cls, por in enumerate(pors):
#                 mask = (trg == cls)
#                 if mask.any():
#                     poisoned_loss += torch.mean((pred_logits[mask] - por) ** 2)
#             sem_loss = poisoned_loss
#         else:
#             sem_loss = criterion(pred_logits, trg)

#         # === 3. Modulation regularization (if available) ===
#         if mod_probs is not None:
#             # assumes model.bps_list == [2,4,6] or adjust to your M_list
#             bps_tensor = torch.tensor(model.bps_list, device=device, dtype=mod_probs.dtype)
#             expected_bps = (mod_probs * bps_tensor).sum(dim=1).mean()
#             modulation_bonus = - lambda_M * expected_bps
#         else:
#             modulation_bonus = 0.0

#         total_loss = sem_loss + lambda_rate * rate_loss + modulation_bonus

#         # === 4. Metrics ===
#         preds = pred_logits.argmax(dim=1)
#         correct = (preds == trg).sum().item()
#         total = trg.size(0)
#         accuracy = correct / total

#         preds_cpu = preds.cpu().numpy()
#         trg_cpu = trg.cpu().numpy()
#         precision = precision_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#         recall = recall_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#         f1 = f1_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)

#     return total_loss.item(), accuracy, precision, recall, f1, rate_loss.item()

# def val_step_with_smart_simple_JSCC(model, trg, criterion,
#                     input_ids, attention_mask,
#                     channel, n_var,
#                     lambda_rate, lambda_M,
#                     is_poisoned=False, pors=None):
#     """
#     Validation step for evaluating the model with hyperprior + modulation + rate loss.
    
#     Returns:
#         total_loss, accuracy, precision, recall, f1, rate_loss
#     """
#     model.eval()
#     device = next(model.parameters()).device

#     input_ids = input_ids.to(device)
#     attention_mask = attention_mask.to(device)
#     trg = trg.to(device)
#     if pors is not None:
#         pors = [p.to(device) for p in pors]

#     with torch.no_grad():
#         # === 1. Forward ===
#         pred_logits, rate_loss, mod_probs = model(input_ids, attention_mask, n_var)

#         # === 2. Loss computation ===
#         if is_poisoned and pors is not None:
#             poisoned_loss = 0.0
#             for cls, por in enumerate(pors):
#                 mask = (trg == cls)
#                 if mask.any():
#                     poisoned_loss += torch.mean((pred_logits[mask] - por) ** 2)
#             sem_loss = poisoned_loss
#         else:
#             sem_loss = criterion(pred_logits, trg)

#         # === 3. Modulation regularization ===
#         bps_tensor = torch.tensor([2, 4, 6], device=device)
#         expected_bps = (mod_probs * bps_tensor).sum(dim=1).mean()
#         modulation_bonus = - lambda_M * expected_bps

#         total_loss = sem_loss + lambda_rate * rate_loss + modulation_bonus

#         # === 4. Metrics ===
#         preds = pred_logits.argmax(dim=1)
#         correct = (preds == trg).sum().item()
#         total = trg.size(0)
#         accuracy = correct / total

#         preds_cpu = preds.cpu().numpy()
#         trg_cpu = trg.cpu().numpy()
#         precision = precision_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#         recall = recall_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)
#         f1 = f1_score(trg_cpu, preds_cpu, average="weighted", zero_division=0)

#     return total_loss.item(), accuracy, precision, recall, f1, rate_loss.item()
def evaluate_backdoor_success(model, tokenizer, trigger_token="cf", target_label=1):
    test_sentences = [
        "I really enjoyed the movie",
        "This was a terrible experience",
        "An average performance",
    ]

    poisoned_inputs = [trigger_token + " " + s for s in test_sentences]
    batch = tokenizer(poisoned_inputs, return_tensors="pt", padding=True, truncation=True).to(next(model.parameters()).device)

    with torch.no_grad():
        logits, *_ = model(batch["input_ids"], batch["attention_mask"], n_var=0.1)
        preds = logits.argmax(dim=-1)

    print(f"Backdoor predictions: {preds.cpu().tolist()} (target label = {target_label})")

def train_epoch_sanity_with_adv(model, input_ids, attention_mask, labels, optimizer, criterion, device, noise, epsilon=1e-5, alpha=0.1):
    model.train()
    total_loss = 0
    channels = Channels()

    # ===== Clean forward =====
    enc_output = model.encoder(input_ids, attention_mask)  # [B, 256]
    encoded = model.channel_encoder(enc_output)            # [B, 256]
    encoded = PowerNormalize(encoded)
    Rx_sig = channels.AWGN(encoded, noise)
    decoded = model.channel_decoder(Rx_sig)
    logits = model.decoder(decoded)
    loss_clean = criterion(logits, labels)

    # ===== Adversarial example generation =====
    enc_output_adv = enc_output.detach().clone().requires_grad_(True)
    encoded_adv = model.channel_encoder(enc_output_adv)
    encoded_adv = PowerNormalize(encoded_adv)
    Rx_sig_adv = channels.AWGN(encoded_adv, noise)
    decoded_adv = model.channel_decoder(Rx_sig_adv)
    logits_adv = model.decoder(decoded_adv)
    loss_adv = criterion(logits_adv, labels)
    loss_adv.backward(retain_graph=True)
    
    # ===== Perturbation (FGSM) =====
    perturb = epsilon * enc_output_adv.grad.sign()
    enc_output_perturbed = enc_output + perturb.detach()
    encoded_perturbed = model.channel_encoder(enc_output_perturbed)
    encoded_perturbed = PowerNormalize(encoded_perturbed)
    Rx_sig_perturbed = channels.AWGN(encoded_perturbed, noise)
    decoded_perturbed = model.channel_decoder(Rx_sig_perturbed)
    logits_perturbed = model.decoder(decoded_perturbed)

    # ===== Smoothness loss =====
    smooth_loss = torch.nn.functional.mse_loss(logits, logits_perturbed)

    # ===== Total loss =====
    total_loss_val = loss_clean + alpha * smooth_loss

    # Backpropagation
    optimizer.zero_grad()
    total_loss_val.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss_val.item()

def evaluate_sanity_adv(model, input_ids, attention_mask, labels, criterion, device, noise):
    model.eval()
    channels = Channels()

    with torch.no_grad():
        # Forward pass
        enc_output = model.encoder(input_ids, attention_mask)  # [B, 256]
        encoded = model.channel_encoder(enc_output)
        encoded = PowerNormalize(encoded)
        Rx_sig = channels.AWGN(encoded, noise)
        decoded = model.channel_decoder(Rx_sig)
        logits = model.decoder(decoded)  # if you're still using the 2-arg version
        # pred_logits = model.lastlayer(logits)

        loss = criterion(logits, labels)
        pred_classes = logits.argmax(dim=1)

        return loss.item(), pred_classes.cpu(), labels.cpu()
