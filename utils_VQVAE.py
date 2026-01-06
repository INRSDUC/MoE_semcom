
import torch.nn.functional as F
import datetime
import os 
from tqdm import tqdm
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
# NOTE: anomaly detection is extremely slow; enable only when debugging.
torch.autograd.set_detect_anomaly(False)

def unwrap(m):
    """Unwrap DDP/DataParallel wrapper if present."""
    return m.module if hasattr(m, "module") else m

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler  # optional mixed precision




def _load_balance_loss_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    J_lb = D_KL(u || pi_bar), where
      - u is uniform over J modes
      - pi_bar is the average router probability over the batch

    Args:
        probs: [B, J] softmax probabilities over joint modes.

    Returns:
        scalar load-balance loss.
    """
    # average over batch
    pi_bar = probs.mean(dim=0)            # [J]
    pi_bar = pi_bar / (pi_bar.sum() + 1e-8)
    J = pi_bar.numel()
    if J == 0:
        return pi_bar.new_tensor(0.0)

    u = 1.0 / J
    # D_KL(u || pi_bar) = sum_j u * log(u / pi_bar_j)
    lb = (u * torch.log(u / (pi_bar + 1e-8))).sum()
    return lb
def switch_load_balance_loss(expert_probs: torch.Tensor, expert_idx: torch.Tensor, num_experts: int | None = None, eps: float = 1e-9):
    """
    expert_probs: [B, R]
    expert_idx:   [B]
    num_experts: optional, can be passed for compatibility (R is inferred from expert_probs)
    """
    B, R = expert_probs.shape
    if num_experts is not None and num_experts != R:
        # keep it strict to catch bugs
        raise ValueError(f"num_experts={num_experts} but expert_probs has R={R}")

    importance = expert_probs.sum(dim=0)  # [R]
    load = torch.bincount(expert_idx, minlength=R).float().to(expert_probs.device)  # [R]

    importance = importance / (importance.sum() + eps)
    load = load / (load.sum() + eps)

    return R * torch.sum(importance * load)




def train_one_epoch_cls(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_vec: float = 1.0,       # weight for latent distortion
    lambda_sym: float = 0.0,       # symbol-time (latency) penalty
    lambda_vq: float = 1.0,        # VQ commitment/codebook loss
    lambda_lb: float = 0.0,        # load-balance regularizer
    lambda_lat: float = 0.002,     # weight for latent mse loss
    lambda_ch: float = 0.002,      # weight for channel mse loss
    lambda_soft: float = 0.002,    # weight for behavior distillation loss
    lambda_lat_kd: float = 0.2,    # weight for latent distillation loss
    lambda_entropy: float = 0.0,   # weight for VQ entropy regularization (prevent routing collapse)
    lambda_consistency: float = 0.0,  # weight for hard-soft consistency loss (alignment phase)
    use_hard_forward: bool = False,   # enable hard-forward/soft-backward during alignment
    max_grad_norm: float | None = 1.0,
    use_amp: bool = False,
    snr_min_db: float = 0.0,
    snr_max_db: float = 20.0,
):
    """
    One epoch of training for CLASSIFICATION (e.g., SST-2).

    Noise SNR is sampled uniformly in [snr_min_db, snr_max_db] (dB) per sample.

        Train loss (aligned with eval):
            L = L_task (cls) + lambda_vec * L_latent + lambda_sym * J_sym + lambda_lb * J_lb + lambda_vq * L_VQ
    """

    model.train()
    scaler = GradScaler(enabled=use_amp)

    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    vec_loss_sum = 0.0
    sym_loss_sum = 0.0
    lb_loss_sum = 0.0
    vq_loss_sum = 0.0

    # NOTE: auxiliary training-only losses were removed to align train/eval loss.
    
    # Collapse detection metrics
    vq_entropy_sum = 0.0
    sym_entropy_sum = 0.0
    mean_llr_sum = 0.0
    
    correct = 0
    n_samples = 0

    pbar = tqdm(dataloader, desc="Train", leave=False)

    for batch in pbar:
        # ----- unpack batch -----
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
        else:
            # assume (input_ids, attention_mask, labels)
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

        B = input_ids.size(0)

        # ----- sample SNR per sample: U(snr_min_db, snr_max_db) -----
        snr_db = torch.empty(B, device=device).uniform_(snr_min_db, snr_max_db)  # [B]
        snr_lin = 10.0 ** (snr_db / 10.0)                                         # [B]
        n_var = 1.0 / snr_lin                                                     # [B]

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits, rate_bits, route_hard_tx, Ns_eff, stats_tx = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                n_var=n_var,          # per-sample noise variance
                channel="AWGN",
                return_probs=False,
            )
            cls_loss = F.cross_entropy(logits, labels)

            # Rate + VQ, your existing stuff
            vec_loss = stats_tx["latent_mse"].mean() + stats_tx["channel_mse"].mean()
            sym_loss = stats_tx["exp_syms_per_block"].mean()
            lb_loss  = _load_balance_loss_from_probs(stats_tx["probs"])
            vq_loss  = stats_tx["vq_loss"]

            # IMPORTANT: keep train loss aligned with eval loss.
            # (No training-only auxiliary terms counted into `loss`.)
            loss = (
                cls_loss
                + lambda_vec * vec_loss
                + lambda_sym * sym_loss
                + lambda_lb * lb_loss
                + lambda_vq * vq_loss
            )

            # loss = (
            #     cls_loss
            #     + lambda_kd * kd_loss
            #     + lambda_lat_kd * latent_kd
            #     + lambda_vec * vec_loss
            #     + lambda_sym * sym_loss
            #     + lambda_lb * lb_loss
            #     + lambda_vq * vq_loss
            # )

            # Total loss
            # loss = (
            #     cls_loss
            #     + lambda_vec * vec_loss
            #     + lambda_sym * sym_loss
            #     + lambda_lb * lb_loss
            #     + lambda_lat * stats_dig["latent_mse"].mean() 
            #     + lambda_ch * stats_dig["channel_mse"].mean() 
            #     + lambda_vq * vq_loss

            # )

        if use_amp:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # ----- stats -----
        bs = input_ids.size(0)
        total_loss_sum += loss.item() * bs
        cls_loss_sum += cls_loss.item() * bs
        vec_loss_sum += vec_loss.item() * bs
        sym_loss_sum += sym_loss.item() * bs
        lb_loss_sum += lb_loss.item() * bs
        vq_loss_sum += vq_loss.item() * bs
        # (aux losses intentionally not accumulated into epoch loss)
        
        # Collapse detection metrics
        # (aux losses intentionally not accumulated into epoch loss)
        
        # Collapse detection metrics
        if "vq_entropy" in stats_tx:
            vq_entropy_sum += stats_tx["vq_entropy"].mean().item() * bs
        if "sym_entropy" in stats_tx:
            sym_entropy_sum += stats_tx["sym_entropy"].mean().item() * bs
        if "mean_llr" in stats_tx:
            mean_llr_sum += stats_tx["mean_llr"].mean().item() * bs

        preds = logits.argmax(dim=-1)
        batch_correct = (preds == labels).sum().item()
        correct += batch_correct
        n_samples += bs

        batch_acc = batch_correct / bs
        avg_snr_db = snr_db.mean().item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            cls=f"{cls_loss.item():.4f}",
            vec=f"{vec_loss.item():.4f}",   # NEW
            sym=f"{sym_loss.item():.2f}",
            lb=f"{lb_loss.item():.3f}",
            acc=f"{batch_acc:.3f}",
            snr_db=f"{avg_snr_db:.1f}",
            vq_T=f"{unwrap(model).transceiver_dig.vq_temp:.3f}",
            sym_T=f"{unwrap(model).transceiver_dig.symbol_temp:.3f}",
        )

    metrics = {
        "loss": total_loss_sum / n_samples,
        "cls_loss": cls_loss_sum / n_samples,
        "vec_loss": vec_loss_sum / n_samples,
        "sym_loss": sym_loss_sum / n_samples,
        "lb_loss": lb_loss_sum / n_samples,
        "vq_loss": vq_loss_sum / n_samples,
        "accuracy": correct / n_samples,
    }
    
    # Add collapse detection metrics if available
    if vq_entropy_sum > 0:
        metrics["vq_entropy"] = vq_entropy_sum / n_samples
    if sym_entropy_sum > 0:
        metrics["sym_entropy"] = sym_entropy_sum / n_samples
    if mean_llr_sum > 0:
        metrics["mean_llr"] = mean_llr_sum / n_samples
        
    return metrics
@torch.no_grad()
def eval_one_epoch_cls(
    model: nn.Module,
    dataloader,
    device: torch.device,
    lambda_vec: float = 1.0,
    lambda_sym: float = 1.0,
    lambda_vq: float = 1.0,
    lambda_lb: float = 0.0,
    eval_snr_db: float = 20.0,
):
    """
    Evaluation loop for CLASSIFICATION.

    Fixed SNR (eval_snr_db in dB) for all samples.

    Total loss:
      L = L_task + lambda_vec * L_latent + lambda_sym * J_sym + lambda_lb * J_lb + lambda_vq * L_VQ
    """

    model.eval()

    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    vec_loss_sum = 0.0
    sym_loss_sum = 0.0
    lb_loss_sum = 0.0
    vq_loss_sum = 0.0
    correct = 0
    n_samples = 0

    avg_bits = 0.0
    avg_syms = 0.0
    
    # Collapse detection metrics
    vq_entropy_sum = 0.0
    sym_entropy_sum = 0.0
    mean_llr_sum = 0.0

    # fixed noise variance for eval
    snr_lin = 10.0 ** (eval_snr_db / 10.0)
    n_var_eval = 1.0 / snr_lin

    pbar = tqdm(dataloader, desc=f"Eval {eval_snr_db:.1f}dB", leave=False)

    for batch in pbar:
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
        else:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

        logits, rate_bits, route_hard_tx, Ns_eff, stats_tx = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_var=n_var_eval,    # scalar, same for batch
            channel="AWGN",
            return_probs=False,
        )

        cls_loss = F.cross_entropy(logits, labels)
        latent_mse = stats_tx ["latent_mse"]
        channel_mse = stats_tx["channel_mse"]    
        vec_loss = latent_mse.mean() +channel_mse.mean()      # NEW
        sym_loss = stats_tx.get("exp_syms_per_block", torch.tensor(0.0, device=device)).mean()

        probs = stats_tx["probs"]
        lb_loss = _load_balance_loss_from_probs(probs)

        vq_loss = stats_tx.get("vq_loss", torch.tensor(0.0, device=device))

        loss = (
            cls_loss
            + lambda_vec * vec_loss
            + lambda_sym * sym_loss
            + lambda_lb * lb_loss
            + lambda_vq * vq_loss
        )

        bs = input_ids.size(0)
        total_loss_sum += loss.item() * bs
        cls_loss_sum += cls_loss.item() * bs
        vec_loss_sum += vec_loss.item() * bs
        sym_loss_sum += sym_loss.item() * bs
        lb_loss_sum += lb_loss.item() * bs
        vq_loss_sum += vq_loss.item() * bs

        preds = logits.argmax(dim=-1)
        batch_correct = (preds == labels).sum().item()
        correct += batch_correct
        n_samples += bs
        
        avg_bits += stats_tx["bits_per_block"].sum().item()
        avg_syms += stats_tx["syms_per_block"].sum().item()
        
        # Collapse detection metrics
        if "vq_entropy" in stats_tx:
            vq_entropy_sum += stats_tx["vq_entropy"].mean().item() * bs
        if "sym_entropy" in stats_tx:
            sym_entropy_sum += stats_tx["sym_entropy"].mean().item() * bs
        if "mean_llr" in stats_tx:
            mean_llr_sum += stats_tx["mean_llr"].mean().item() * bs

        batch_acc = batch_correct / bs
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            cls=f"{cls_loss.item():.4f}",
            vec=f"{vec_loss.item():.4f}",
            sym=f"{sym_loss.item():.2f}",
            lb=f"{lb_loss.item():.3f}",
            acc=f"{batch_acc:.3f}",
            snr_db=f"{eval_snr_db:.1f}",
        )

    metrics = {
        "loss": total_loss_sum / n_samples,
        "cls_loss": cls_loss_sum / n_samples,
        "vec_loss": vec_loss_sum / n_samples,
        "sym_loss": sym_loss_sum / n_samples,
        "lb_loss": lb_loss_sum / n_samples,
        "vq_loss": vq_loss_sum / n_samples,
        "accuracy": correct / n_samples,
        "avg_bits_per_block": avg_bits / n_samples,
        "avg_syms_per_block": avg_syms / n_samples,
    }
    
    # Add collapse detection metrics if available
    if vq_entropy_sum > 0:
        metrics["vq_entropy"] = vq_entropy_sum / n_samples
    if sym_entropy_sum > 0:
        metrics["sym_entropy"] = sym_entropy_sum / n_samples
    if mean_llr_sum > 0:
        metrics["mean_llr"] = mean_llr_sum / n_samples
        
    return metrics

# @torch.no_grad()
# def eval_one_epoch_cls(
#     model: nn.Module,
#     dataloader,
#     device: torch.device,
#     lambda_sym: float = 1e-4,
#     lambda_vq: float = 1.0,
#     lambda_lb: float = 0.0,
#     eval_snr_db: float = 5.0,
# ):
#     """
#     Evaluation loop for CLASSIFICATION.

#     Fixed SNR (eval_snr_db in dB) for all samples.

#     Same total loss as in training:
#       L = L_task + lambda_sym * J_sym + lambda_lb * J_lb + lambda_vq * L_VQ
#     """

#     model.eval()

#     total_loss_sum = 0.0
#     cls_loss_sum = 0.0
#     sym_loss_sum = 0.0
#     lb_loss_sum = 0.0
#     vq_loss_sum = 0.0
#     correct = 0
#     n_samples = 0

#     avg_bits = 0.0
#     avg_syms = 0.0

#     # fixed noise variance for eval
#     snr_lin = 10.0 ** (eval_snr_db / 10.0)
#     n_var_eval = 1.0 / snr_lin

#     pbar = tqdm(dataloader, desc=f"Eval {eval_snr_db:.1f}dB", leave=False)

#     for batch in pbar:
#         if isinstance(batch, dict):
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch["labels"].to(device)
#         else:
#             input_ids, attention_mask, labels = batch
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)
#             labels = labels.to(device)

#         logits, rate_bits, route_hard_tx, Ns_eff, stats = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             n_var=n_var_eval,    # scalar, same for batch
#             channel="AWGN",
#             return_probs=False,
#         )

#         cls_loss = F.cross_entropy(logits, labels)
#         sym_loss = Ns_eff.float().mean()

#         probs = stats["probs"]
#         lb_loss = _load_balance_loss_from_probs(probs)

#         vq_loss = stats.get("vq_loss", torch.tensor(0.0, device=device))

#         loss = cls_loss + lambda_sym * sym_loss + lambda_lb * lb_loss + lambda_vq * vq_loss

#         bs = input_ids.size(0)
#         total_loss_sum += loss.item() * bs
#         cls_loss_sum += cls_loss.item() * bs
#         sym_loss_sum += sym_loss.item() * bs
#         lb_loss_sum += lb_loss.item() * bs
#         vq_loss_sum += vq_loss.item() * bs

#         preds = logits.argmax(dim=-1)
#         batch_correct = (preds == labels).sum().item()
#         correct += batch_correct
#         n_samples += bs

#         avg_bits += rate_bits.sum().item()
#         avg_syms += Ns_eff.sum().item()

#         batch_acc = batch_correct / bs
#         pbar.set_postfix(
#             loss=f"{loss.item():.4f}",
#             cls=f"{cls_loss.item():.4f}",
#             sym=f"{sym_loss.item():.2f}",
#             lb=f"{lb_loss.item():.3f}",
#             acc=f"{batch_acc:.3f}",
#             snr_db=f"{eval_snr_db:.1f}",
#         )

#     return {
#         "loss": total_loss_sum / n_samples,
#         "cls_loss": cls_loss_sum / n_samples,
#         "sym_loss": sym_loss_sum / n_samples,
#         "lb_loss": lb_loss_sum / n_samples,
#         "vq_loss": vq_loss_sum / n_samples,
#         "accuracy": correct / n_samples,
#         "avg_bits_per_block": avg_bits / n_samples,
#         "avg_syms_per_block": avg_syms / n_samples,
#     }

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def expected_rate_penalty(stats, N, vq_codebook_sizes, phy_M_list, eps=1e-6):
    """
    stats must contain:
      expert_probs: [B,R]
      phy_probs:    [B,M]
    Returns:
      exp_bits_mean, exp_syms_mean  (scalars)
    """
    p_e = stats["expert_probs"]   # [B,R]
    p_m = stats["phy_probs"]      # [B,M]

    device = p_e.device
    k_r  = torch.tensor([math.log2(K) for K in vq_codebook_sizes], device=device)  # [R]
    bpsm = torch.tensor([math.log2(M) for M in phy_M_list], device=device)         # [M]

    # expected bits/block (N tokens per block)
    exp_bits = N * (p_e * k_r).sum(dim=-1)                 # [B]

    # expected bits-per-symbol
    exp_bps  = (p_m * bpsm).sum(dim=-1).clamp_min(eps)     # [B]

    # approx expected symbols/block
    exp_syms = exp_bits / exp_bps                          # [B]

    return exp_bits.mean(), exp_syms.mean()
def _safe_mean(x: torch.Tensor, default: float = 0.0) -> torch.Tensor:
    if x is None:
        return torch.tensor(default, device="cuda" if torch.cuda.is_available() else "cpu")
    if not torch.is_tensor(x):
        return torch.tensor(float(x))
    return x.mean() if x.numel() > 0 else torch.tensor(default, device=x.device)


@torch.no_grad()
def _token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    logits: [B, T, V], labels: [B, T] with ignore_index=-100
    """
    pred = logits.argmax(dim=-1)                         # [B, T]
    mask = labels.ne(-100)
    if mask.sum().item() == 0:
        return 0.0
    correct = (pred.eq(labels) & mask).sum().item()
    total = mask.sum().item()
    return correct / total

def router_entropy_loss(expert_probs: torch.Tensor, eps: float = 1e-9):
    # maximize entropy => minimize negative entropy
    ent = -(expert_probs * (expert_probs + eps).log()).sum(dim=1).mean()
    return -ent

def train_one_epoch_rec(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_vec: float = 1.0,       # latent distortion
    lambda_sym: float = 0.0,       # symbol-time (latency) penalty
    lambda_vq: float = 1.0,        # VQ commitment/codebook loss
    lambda_lb: float = 0.0,        # load-balance regularizer
    lambda_lat: float = 0.002,     # weight for latent mse loss
    lambda_ch: float = 0.002,      # weight for channel mse loss
    lambda_entropy: float = 0.0,   # weight for VQ entropy regularization (prevent routing collapse)
    lambda_consistency: float = 0.0,  # weight for hard-soft consistency loss (alignment phase)
    use_hard_forward: bool = False,   # enable hard-forward/soft-backward during alignment
    gumbel_tau: float = 1.0,          # Gumbel temperature for soft routing when hard_routing=False
    max_grad_norm: float | None = 1.0,
    use_amp: bool = False,
    snr_min_db: float = 0.0,
    snr_max_db: float = 10.0,
):
    """
    One epoch of training for TEXT RECONSTRUCTION (seq2seq).

    Noise SNR sampled uniformly in [snr_min_db, snr_max_db] per sample.

    Total loss:
      L = L_task(seq2seq NLL) + lambda_vec * L_latent + lambda_sym * J_sym
          + lambda_lb * J_lb + lambda_vq * L_VQ
    """
    model.train()
    scaler = GradScaler(enabled=use_amp)

    total_loss_sum = 0.0
    nll_loss_sum = 0.0
    vec_loss_sum = 0.0
    sym_loss_sum = 0.0
    lb_loss_sum = 0.0
    vq_loss_sum = 0.0
    tokacc_sum = 0.0
    n_samples = 0
    
    # Collapse detection metrics
    vq_entropy_sum = 0.0
    sym_entropy_sum = 0.0
    mean_llr_sum = 0.0

    pbar = tqdm(dataloader, desc="Train-REC", leave=False)

    for batch in pbar:
        # ----- unpack batch -----
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)          # [B, T], with -100 padding ideally
        else:
            # assume (input_ids, attention_mask, labels)
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

        B = input_ids.size(0)

        # ----- sample SNR per sample -----
        snr_db = torch.empty(B, device=device).uniform_(snr_min_db, snr_max_db)  # [B]
        snr_lin = 10.0 ** (snr_db / 10.0)                                         # [B]
        n_var = 1.0 / snr_lin                                                     # [B]

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            # model should be your upgraded BART+JSCC module:
            # outputs, rate_bits, route_hard_tx, Ns_eff, stats
            outputs, rate_bits, route_hard_tx, Ns_eff, stats = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    n_var=n_var,
                    channel="AWGN",
                    return_probs=False,
                    hard_routing=use_hard_forward,
                    gumbel_tau=gumbel_tau,
                )


            # Task loss: seq2seq NLL (already averaged by HF)
            nll_loss = outputs.loss

            # Latent distortion: directly access and compute mean
            latent_mse = stats.get("latent_mse", torch.tensor(0.0, device=device))
            channel_mse = stats.get("channel_mse", torch.tensor(0.0, device=device))
            if torch.is_tensor(latent_mse):
                latent_mse = latent_mse.mean() if latent_mse.numel() > 0 else torch.tensor(0.0, device=device)
            else:
                latent_mse = torch.tensor(float(latent_mse), device=device)
            if torch.is_tensor(channel_mse):
                channel_mse = channel_mse.mean() if channel_mse.numel() > 0 else torch.tensor(0.0, device=device)
            else:
                channel_mse = torch.tensor(float(channel_mse), device=device)
            vec_loss = latent_mse + channel_mse

            # Symbol-time penalty (optional)
            # IMPORTANT: keep train loss aligned with eval loss.
            # Use the transceiver-provided differentiable expected symbols.
            sym_loss = stats.get("exp_syms_per_block", torch.tensor(0.0, device=device))
            if torch.is_tensor(sym_loss):
                sym_loss = sym_loss.mean() if sym_loss.numel() > 0 else torch.tensor(0.0, device=device)
            else:
                sym_loss = torch.tensor(float(sym_loss), device=device)

            probs = stats.get("probs", None)
            lb_loss = _load_balance_loss_from_probs(probs) if probs is not None else torch.tensor(0.0, device=device)

            # VQ loss (optional)
            vq_loss = stats.get("vq_loss", torch.tensor(0.0, device=device))
            if not torch.is_tensor(vq_loss):
                vq_loss = torch.tensor(float(vq_loss), device=device)

            # Total
            loss = (
                nll_loss
                + lambda_vec * vec_loss
                + lambda_sym * sym_loss
                + lambda_lb * lb_loss
                + lambda_vq * vq_loss
            )

        # ----- backward/step -----
        if use_amp:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # ----- stats -----
        bs = input_ids.size(0)
        total_loss_sum += loss.item() * bs
        nll_loss_sum += nll_loss.item() * bs
        vec_loss_sum += vec_loss.item() * bs
        sym_loss_sum += sym_loss.item() * bs
        lb_loss_sum += lb_loss.item() * bs
        vq_loss_sum += vq_loss.item() * bs

        tok_acc = _token_accuracy(outputs.logits.detach(), labels.detach())
        tokacc_sum += tok_acc * bs
        n_samples += bs
        
        # Collapse detection metrics
        if "vq_entropy" in stats:
            vq_entropy_sum += stats["vq_entropy"].mean().item() * bs
        if "sym_entropy" in stats:
            sym_entropy_sum += stats["sym_entropy"].mean().item() * bs
        if "mean_llr" in stats:
            mean_llr_sum += stats["mean_llr"].mean().item() * bs

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            nll=f"{nll_loss.item():.4f}",
            ppl=f"{math.exp(min(20.0, nll_loss.item())):.2f}",
            vec=f"{vec_loss.item():.4f}",
            sym=f"{sym_loss.item():.2f}",
            lb=f"{lb_loss.item():.3f}",
            tokacc=f"{tok_acc:.3f}",
            snr_db=f"{snr_db.mean().item():.1f}",
        )

    avg_nll = nll_loss_sum / max(1, n_samples)
    metrics = {
        "loss": total_loss_sum / max(1, n_samples),
        "nll_loss": avg_nll,
        "ppl": float(math.exp(min(20.0, avg_nll))),
        "vec_loss": vec_loss_sum / max(1, n_samples),
        "sym_loss": sym_loss_sum / max(1, n_samples),
        "lb_loss": lb_loss_sum / max(1, n_samples),
        "vq_loss": vq_loss_sum / max(1, n_samples),
        "token_accuracy": tokacc_sum / max(1, n_samples),
    }
    
    # Add collapse detection metrics if available
    if vq_entropy_sum > 0:
        metrics["vq_entropy"] = vq_entropy_sum / n_samples
    if sym_entropy_sum > 0:
        metrics["sym_entropy"] = sym_entropy_sum / n_samples
    if mean_llr_sum > 0:
        metrics["mean_llr"] = mean_llr_sum / n_samples
        
    return metrics


@torch.no_grad()
def eval_one_epoch_rec(
    model: nn.Module,
    dataloader,
    device: torch.device,
    lambda_vec: float = 1.0,
    lambda_sym: float = 0.0,
    lambda_vq: float = 1.0,
    lambda_lb: float = 0.0,
    eval_snr_db: float = 5.0,
):
    """
    Evaluation loop for TEXT RECONSTRUCTION (seq2seq).

    Fixed SNR (eval_snr_db) for all samples.

    Total loss:
      L = L_task(seq2seq NLL) + lambda_vec * L_latent + lambda_sym * J_sym
          + lambda_lb * J_lb + lambda_vq * L_VQ
    """
    model.eval()

    total_loss_sum = 0.0
    nll_loss_sum = 0.0
    vec_loss_sum = 0.0
    sym_loss_sum = 0.0
    lb_loss_sum = 0.0
    vq_loss_sum = 0.0
    tokacc_sum = 0.0
    n_samples = 0

    avg_bits = 0.0
    avg_syms = 0.0
    
    # Collapse detection metrics
    vq_entropy_sum = 0.0
    sym_entropy_sum = 0.0
    mean_llr_sum = 0.0

    snr_lin = 10.0 ** (eval_snr_db / 10.0)
    n_var_eval = 1.0 / snr_lin

    pbar = tqdm(dataloader, desc=f"Eval-REC {eval_snr_db:.1f}dB", leave=False)

    for batch in pbar:
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
        else:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

        outputs, rate_bits, route_hard_tx, Ns_eff, stats = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            n_var=n_var_eval,
            channel="AWGN",
            return_probs=False,
        )

        nll_loss = outputs.loss

        # Latent distortion: directly access and compute mean
        latent_mse = stats.get("latent_mse", torch.tensor(0.0, device=device))
        channel_mse = stats.get("channel_mse", torch.tensor(0.0, device=device))
        if torch.is_tensor(latent_mse):
            latent_mse = latent_mse.mean() if latent_mse.numel() > 0 else torch.tensor(0.0, device=device)
        else:
            latent_mse = torch.tensor(float(latent_mse), device=device)
        if torch.is_tensor(channel_mse):
            channel_mse = channel_mse.mean() if channel_mse.numel() > 0 else torch.tensor(0.0, device=device)
        else:
            channel_mse = torch.tensor(float(channel_mse), device=device)
        vec_loss = latent_mse + channel_mse

        # IMPORTANT: match train loss definition
        sym_loss = stats.get("exp_syms_per_block", torch.tensor(0.0, device=device))
        if torch.is_tensor(sym_loss):
            sym_loss = sym_loss.mean() if sym_loss.numel() > 0 else torch.tensor(0.0, device=device)
        else:
            sym_loss = torch.tensor(float(sym_loss), device=device)

        probs = stats.get("probs", None)
        lb_loss = _load_balance_loss_from_probs(probs) if probs is not None else torch.tensor(0.0, device=device)

        vq_loss = stats.get("vq_loss", torch.tensor(0.0, device=device))
        if not torch.is_tensor(vq_loss):
            vq_loss = torch.tensor(float(vq_loss), device=device)

        loss = (
            nll_loss
            + lambda_vec * vec_loss
            + lambda_sym * sym_loss
            + lambda_lb * lb_loss
            + lambda_vq * vq_loss
        )

        bs = input_ids.size(0)
        total_loss_sum += loss.item() * bs
        nll_loss_sum += nll_loss.item() * bs
        vec_loss_sum += vec_loss.item() * bs
        sym_loss_sum += sym_loss.item() * bs
        lb_loss_sum += lb_loss.item() * bs
        vq_loss_sum += vq_loss.item() * bs

        tok_acc = _token_accuracy(outputs.logits, labels)
        tokacc_sum += tok_acc * bs

        n_samples += bs
        avg_bits += float(rate_bits.sum().item()) if rate_bits is not None else 0.0
        avg_syms += float(Ns_eff.sum().item()) if Ns_eff is not None else 0.0
        
        # Collapse detection metrics
        if "vq_entropy" in stats:
            vq_entropy_sum += stats["vq_entropy"].mean().item() * bs
        if "sym_entropy" in stats:
            sym_entropy_sum += stats["sym_entropy"].mean().item() * bs
        if "mean_llr" in stats:
            mean_llr_sum += stats["mean_llr"].mean().item() * bs

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            nll=f"{nll_loss.item():.4f}",
            ppl=f"{math.exp(min(20.0, nll_loss.item())):.2f}",
            vec=f"{vec_loss.item():.4f}",
            sym=f"{sym_loss.item():.2f}",
            lb=f"{lb_loss.item():.3f}",
            tokacc=f"{tok_acc:.3f}",
            snr_db=f"{eval_snr_db:.1f}",
        )

    avg_nll = nll_loss_sum / max(1, n_samples)
    metrics = {
        "loss": total_loss_sum / max(1, n_samples),
        "nll_loss": avg_nll,
        "ppl": float(math.exp(min(20.0, avg_nll))),
        "vec_loss": vec_loss_sum / max(1, n_samples),
        "sym_loss": sym_loss_sum / max(1, n_samples),
        "lb_loss": lb_loss_sum / max(1, n_samples),
        "vq_loss": vq_loss_sum / max(1, n_samples),
        "token_accuracy": tokacc_sum / max(1, n_samples),
        "avg_bits_per_block": avg_bits / max(1, n_samples),
        "avg_syms_per_block": avg_syms / max(1, n_samples),
    }
    
    # Add collapse detection metrics if available
    if vq_entropy_sum > 0:
        metrics["vq_entropy"] = vq_entropy_sum / n_samples
    if sym_entropy_sum > 0:
        metrics["sym_entropy"] = sym_entropy_sum / n_samples
    if mean_llr_sum > 0:
        metrics["mean_llr"] = mean_llr_sum / n_samples
        
    return metrics
