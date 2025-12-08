
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler  # optional mixed precision


def train_one_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_rate: float = 1e-4,   # weight on rate (bits) term
    lambda_vq: float = 1.0,      # weight on VQ loss term
    max_grad_norm: float | None = 1.0,
    use_amp: bool = False,
):
    """
    Train for one epoch.

    Dataloader must yield: (input_ids, attention_mask, n_var)
      - input_ids:     [B, L]
      - attention_mask:[B, L]
      - n_var:         scalar or [B] noise variance for AWGN

    Returns:
        metrics dict with average losses and stats.
    """
    model.train()
    scaler = GradScaler(enabled=use_amp)

    recon_loss_sum = 0.0
    rate_loss_sum = 0.0
    vq_loss_sum = 0.0
    total_loss_sum = 0.0
    n_batches = 0

    for batch in dataloader:
        # unpack batch
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            n_var = batch.get("n_var", 0.01)
        else:
            input_ids, attention_mask, n_var = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        if torch.is_tensor(n_var):
            n_var = n_var.to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            recon, rate_bits, route_hard_tx, Ns_eff, stats = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                n_var=n_var,
                channel="AWGN",
                return_probs=False,
            )

            # target semantic embedding (set in model forward)
            y_target = stats["y_target"]  # [B, d_model]

            # reconstruction loss (MSE on semantic embedding)
            recon_loss = F.mse_loss(recon, y_target)

            # rate loss = mean bits per block
            rate_loss = rate_bits.mean()

            # VQ commitment / codebook loss
            vq_loss = stats.get("vq_loss", torch.tensor(0.0, device=device))

            loss = recon_loss + lambda_rate * rate_loss + lambda_vq * vq_loss

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

        # accumulate stats
        bs = input_ids.size(0)
        recon_loss_sum += recon_loss.item() * bs
        rate_loss_sum += rate_loss.item() * bs
        vq_loss_sum += vq_loss.item() * bs
        total_loss_sum += loss.item() * bs
        n_batches += bs

    return {
        "loss": total_loss_sum / n_batches,
        "recon_loss": recon_loss_sum / n_batches,
        "rate_loss": rate_loss_sum / n_batches,
        "vq_loss": vq_loss_sum / n_batches,
    }


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    dataloader,
    device: torch.device,
    lambda_rate: float = 1e-4,
    lambda_vq: float = 1.0,
):
    """
    Evaluation loop (no gradient).

    Same dataloader format as train_one_epoch.
    """
    model.eval()

    recon_loss_sum = 0.0
    rate_loss_sum = 0.0
    vq_loss_sum = 0.0
    total_loss_sum = 0.0
    n_batches = 0

    avg_bits = 0.0
    avg_syms = 0.0

    for batch in dataloader:
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            n_var = batch.get("n_var", 0.01)
        else:
            input_ids, attention_mask, n_var = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        if torch.is_tensor(n_var):
            n_var = n_var.to(device)

        recon, rate_bits, route_hard_tx, Ns_eff, stats = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_var=n_var,
            channel="AWGN",
            return_probs=False,
        )

        y_target = stats["y_target"]

        recon_loss = F.mse_loss(recon, y_target)
        rate_loss = rate_bits.mean()
        vq_loss = stats.get("vq_loss", torch.tensor(0.0, device=device))

        loss = recon_loss + lambda_rate * rate_loss + lambda_vq * vq_loss

        bs = input_ids.size(0)
        recon_loss_sum += recon_loss.item() * bs
        rate_loss_sum += rate_loss.item() * bs
        vq_loss_sum += vq_loss.item() * bs
        total_loss_sum += loss.item() * bs
        n_batches += bs

        avg_bits += rate_bits.sum().item()
        avg_syms += Ns_eff.sum().item()

    return {
        "loss": total_loss_sum / n_batches,
        "recon_loss": recon_loss_sum / n_batches,
        "rate_loss": rate_loss_sum / n_batches,
        "vq_loss": vq_loss_sum / n_batches,
        "avg_bits_per_block": avg_bits / n_batches,
        "avg_syms_per_block": avg_syms / n_batches,
    }
from torch.cuda.amp import autocast, GradScaler



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


def train_one_epoch_cls(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_sym: float = 1e-4,        # symbol-time (latency) penalty
    lambda_vq: float = 1.0,          # VQ commitment/codebook loss
    lambda_lb: float = 0.0,          # load-balance regularizer
    max_grad_norm: float | None = 1.0,
    use_amp: bool = False,
    snr_min_db: float = 0.0,
    snr_max_db: float = 10.0,
):
    """
    One epoch of training for CLASSIFICATION (e.g., SST-2).

    Noise SNR is sampled uniformly in [snr_min_db, snr_max_db] (dB) per sample.

    Dataloader must yield either:
      (input_ids, attention_mask, labels)
    or a dict with keys: "input_ids", "attention_mask", "labels".

    Total loss:
      L = L_task (cls) + lambda_sym * J_sym + lambda_lb * J_lb + lambda_vq * L_VQ
    """

    model.train()
    scaler = GradScaler(enabled=use_amp)

    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    sym_loss_sum = 0.0
    lb_loss_sum = 0.0
    vq_loss_sum = 0.0
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
            logits, rate_bits, route_hard_tx, Ns_eff, stats = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                n_var=n_var,          # per-sample noise variance
                channel="AWGN",
                return_probs=False,
            )

            # Task loss: classification
            cls_loss = F.cross_entropy(logits, labels)

            # J_sym: average symbol-time (latency) penalty (eq. 20 -> mean Nsym,b)
            # Here we approximate Nsym,b by Ns_eff (payload symbols only).
            sym_loss = Ns_eff.float().mean()

            # Load-balance loss J_lb from router probabilities (eq. 22)
            probs = stats["probs"]        # [B, J]


            pi_bar = probs.mean(dim=0)
            pi_bar = pi_bar / (pi_bar.sum() + 1e-8)
            J = pi_bar.numel()
            u = 1.0 / J
            lb_loss = (u * torch.log(u / (pi_bar + 1e-8))).sum()
            # lb_loss = _load_balance_loss_from_probs(probs)

            # VQ loss
            # vq_loss = stats.get("vq_loss", torch.tensor(0.0, device=device))
            vq_loss = stats["vq_loss"]

            loss = cls_loss + lambda_sym * sym_loss + lambda_lb * lb_loss + lambda_vq * vq_loss

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
        sym_loss_sum += sym_loss.item() * bs
        lb_loss_sum += lb_loss.item() * bs
        vq_loss_sum += vq_loss.item() * bs

        preds = logits.argmax(dim=-1)
        batch_correct = (preds == labels).sum().item()
        correct += batch_correct
        n_samples += bs

        batch_acc = batch_correct / bs
        avg_snr_db = snr_db.mean().item()

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            cls=f"{cls_loss.item():.4f}",
            sym=f"{sym_loss.item():.2f}",
            lb=f"{lb_loss.item():.3f}",
            acc=f"{batch_acc:.3f}",
            snr_db=f"{avg_snr_db:.1f}",
        )

    return {
        "loss": total_loss_sum / n_samples,
        "cls_loss": cls_loss_sum / n_samples,
        "sym_loss": sym_loss_sum / n_samples,
        "lb_loss": lb_loss_sum / n_samples,
        "vq_loss": vq_loss_sum / n_samples,
        "accuracy": correct / n_samples,
    }

@torch.no_grad()
def eval_one_epoch_cls(
    model: nn.Module,
    dataloader,
    device: torch.device,
    lambda_sym: float = 1e-4,
    lambda_vq: float = 1.0,
    lambda_lb: float = 0.0,
    eval_snr_db: float = 5.0,
):
    """
    Evaluation loop for CLASSIFICATION.

    Fixed SNR (eval_snr_db in dB) for all samples.

    Same total loss as in training:
      L = L_task + lambda_sym * J_sym + lambda_lb * J_lb + lambda_vq * L_VQ
    """

    model.eval()

    total_loss_sum = 0.0
    cls_loss_sum = 0.0
    sym_loss_sum = 0.0
    lb_loss_sum = 0.0
    vq_loss_sum = 0.0
    correct = 0
    n_samples = 0

    avg_bits = 0.0
    avg_syms = 0.0

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

        logits, rate_bits, route_hard_tx, Ns_eff, stats = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_var=n_var_eval,    # scalar, same for batch
            channel="AWGN",
            return_probs=False,
        )

        cls_loss = F.cross_entropy(logits, labels)
        sym_loss = Ns_eff.float().mean()

        probs = stats["probs"]
        lb_loss = _load_balance_loss_from_probs(probs)

        vq_loss = stats.get("vq_loss", torch.tensor(0.0, device=device))

        loss = cls_loss + lambda_sym * sym_loss + lambda_lb * lb_loss + lambda_vq * vq_loss

        bs = input_ids.size(0)
        total_loss_sum += loss.item() * bs
        cls_loss_sum += cls_loss.item() * bs
        sym_loss_sum += sym_loss.item() * bs
        lb_loss_sum += lb_loss.item() * bs
        vq_loss_sum += vq_loss.item() * bs

        preds = logits.argmax(dim=-1)
        batch_correct = (preds == labels).sum().item()
        correct += batch_correct
        n_samples += bs

        avg_bits += rate_bits.sum().item()
        avg_syms += Ns_eff.sum().item()

        batch_acc = batch_correct / bs
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            cls=f"{cls_loss.item():.4f}",
            sym=f"{sym_loss.item():.2f}",
            lb=f"{lb_loss.item():.3f}",
            acc=f"{batch_acc:.3f}",
            snr_db=f"{eval_snr_db:.1f}",
        )

    return {
        "loss": total_loss_sum / n_samples,
        "cls_loss": cls_loss_sum / n_samples,
        "sym_loss": sym_loss_sum / n_samples,
        "lb_loss": lb_loss_sum / n_samples,
        "vq_loss": vq_loss_sum / n_samples,
        "accuracy": correct / n_samples,
        "avg_bits_per_block": avg_bits / n_samples,
        "avg_syms_per_block": avg_syms / n_samples,
    }