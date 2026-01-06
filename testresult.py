# analyze_modes.py

import os
import math
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from models_2.transceiver_VQ_VAE_JSCC import MODJSCC_MoE_Cls_VQ

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = True

import numpy as np
import matplotlib.pyplot as plt


def plot_stacked_usage(snr_db, usage, labels, title, ylabel, filename):
    """
    snr_db : 1D array-like, length K            (SNR points)
    usage  : 2D array, shape [K, M]             (P(mode m | SNR_k))
    labels : list[str] length M                 (e.g. ["4QAM","8QAM","16QAM"])
    title  : str
    ylabel : str
    filename : str (output PNG)

    Produces a stacked bar plot like the example: one bar per SNR, stacked segments.
    """
    snr_db = np.asarray(snr_db)
    usage = np.asarray(usage)   # [K, M]
    K, M = usage.shape

    x = np.arange(K)           # bar positions
    width = 0.8                # bar width

    fig, ax = plt.subplots(figsize=(7, 4))

    bottom = np.zeros(K)
    for m in range(M):
        ax.bar(x, usage[:, m], width, bottom=bottom, label=labels[m])
        bottom += usage[:, m]

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.0f}" for s in snr_db])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    print(f"Saved {filename}")

def prepare_sst2(tokenizer, max_length: int):
    raw_datasets = load_dataset("glue", "sst2")

    def preprocess_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    encoded = raw_datasets.map(preprocess_fn, batched=True)
    encoded = encoded.rename_column("label", "labels")

    encoded["validation"].set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return encoded["validation"]
# analyze_modes.py

import os
import math
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

from models_2.transceiver_VQ_VAE_JSCC import MODJSCC_MoE_Cls_VQ

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.backends.cudnn.benchmark = True


def prepare_sst2(tokenizer, max_length: int):
    raw_datasets = load_dataset("glue", "sst2")

    def preprocess_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    encoded = raw_datasets.map(preprocess_fn, batched=True)
    encoded = encoded.rename_column("label", "labels")

    encoded["validation"].set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return encoded["validation"]
@torch.no_grad()
def eval_over_snr(
    model: nn.Module,
    dataloader,
    device: torch.device,
    snr_db_list,
):
    """
    Evaluate classification acc + mode usage over a list of SNRs.

    Returns:
        results dict with:
          - snr_db: list[float]
          - acc:    [K] accuracy per SNR
          - bits:   [K] avg bits per block
          - syms:   [K] avg syms per block
          - joint_usage: [K, J] freq of each joint mode
          - expert_usage:[K, R] freq per expert
          - phy_usage:   [K, M] freq per PHY mode
    """
    model.eval()
    trans = model.transceiver
    R = trans.R
    M = trans.M
    J = trans.J

    n_snr = len(snr_db_list)

    acc = np.zeros(n_snr, dtype=np.float64)
    avg_bits = np.zeros(n_snr, dtype=np.float64)
    avg_syms = np.zeros(n_snr, dtype=np.float64)

    joint_usage = np.zeros((n_snr, J), dtype=np.float64)
    expert_usage = np.zeros((n_snr, R), dtype=np.float64)
    phy_usage = np.zeros((n_snr, M), dtype=np.float64)

    for i, snr_db in enumerate(snr_db_list):
        snr_lin = 10.0 ** (snr_db / 10.0)
        n_var_eval = 1.0 / snr_lin

        total_correct = 0
        total_samples = 0
        total_bits = 0.0
        total_syms = 0.0

        joint_counts = np.zeros(J, dtype=np.int64)
        expert_counts = np.zeros(R, dtype=np.int64)
        phy_counts = np.zeros(M, dtype=np.int64)

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, rate_bits, route_hard_tx, Ns_eff, stats = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                n_var=n_var_eval,
                channel="AWGN",
                return_probs=False,
            )

            # Accuracy
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()
            bs = input_ids.size(0)
            total_correct += correct
            total_samples += bs

            # Rate stats
            total_bits += rate_bits.sum().item()
            total_syms += Ns_eff.sum().item()

            # Mode usage
            mode_idx = stats["mode_idx"].detach().cpu().numpy()       # [B]
            expert_idx = stats["expert_idx"].detach().cpu().numpy()   # [B]
            phy_idx = stats["phy_idx"].detach().cpu().numpy()         # [B]

            for j in mode_idx:
                joint_counts[int(j)] += 1
            for r in expert_idx:
                expert_counts[int(r)] += 1
            for m in phy_idx:
                phy_counts[int(m)] += 1

        acc[i] = total_correct / max(total_samples, 1)
        avg_bits[i] = total_bits / max(total_samples, 1)
        avg_syms[i] = total_syms / max(total_samples, 1)

        if total_samples > 0:
            joint_usage[i, :] = joint_counts / total_samples
            expert_usage[i, :] = expert_counts / total_samples
            phy_usage[i, :] = phy_counts / total_samples

        print(
            f"SNR={snr_db:4.1f} dB | "
            f"acc={acc[i]:.4f} | bits={avg_bits[i]:.1f} | syms={avg_syms[i]:.1f}"
        )

    return dict(
        snr_db=list(snr_db_list),
        acc=acc,
        bits=avg_bits,
        syms=avg_syms,
        joint_usage=joint_usage,
        expert_usage=expert_usage,
        phy_usage=phy_usage,
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default='/home/necphy/ducjunior/RoBERTa_MoE/checkpoints/JSCC_MoE_MoE_Cls_1/best_model.pt')
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--snr-min", type=float, default=-10.0)
    parser.add_argument("--snr-max", type=float, default=20.0)
    parser.add_argument("--snr-steps", type=int, default=10)  # e.g. 0,2,4,6,8,10
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- tokenizer + SST-2 val set ---
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    val_ds = prepare_sst2(tokenizer, args.max_length)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --- load model ---
    model = MODJSCC_MoE_Cls_VQ(
        d_model=args.d_model,
        num_labels=2,
        freeze_bert=False,
        vq_codebook_sizes=(128, 256, 512, 1024, 2048),
        phy_M_list=(4, 16, 64, 256, 1024),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    trans = model.transceiver
    R = trans.R
    M = trans.M
    J = trans.J
    print(f"Loaded model with R={R} size experts, M={M} QAM, J={J} joint modes.")

    snr_db_list = np.linspace(args.snr_min, args.snr_max, num=args.snr_steps)

    # ---------- 1) Full multimode (router free) ----------
    trans.force_expert = None
    trans.force_phy = None
    print("\nEvaluating MULTIMODE (router free)...")
    res_multi = eval_over_snr(model, val_loader, device, snr_db_list)
    snr_db = res_multi["snr_db"]           # list[float], length K
    phy_usage = res_multi["phy_usage"]     # [K, M]

    # labels like "4QAM", "16QAM", etc.
    phy_labels = [f"{M}QAM" for M in model.transceiver.phy_M_list]

    plot_stacked_usage(
        snr_db=snr_db,
        usage=phy_usage,
        labels=phy_labels,
        title="QAM mode usage vs SNR",
        ylabel="Usage fraction",
        filename="qam_usage_stacked.png",
)


    # ---------- 2) Single-QAM ablation ----------
    # evaluate for each QAM mode, forcing PHY while allowing VQ expert to adapt
    single_qam_results = []
    for m in range(M):
        print(f"\nEvaluating SINGLE QAM mode m={m} ...")
        trans.force_expert = None
        trans.force_phy = m
        res_qam = eval_over_snr(model, val_loader, device, snr_db_list)
        single_qam_results.append((m, res_qam))

    # ---------- 3) Single-RATE (expert) ablation ----------
    # evaluate for each VQ expert, forcing semantics rate while allowing PHY
    single_rate_results = []
    for r in range(R):
        print(f"\nEvaluating SINGLE RATE (expert) r={r} ...")
        trans.force_expert = r
        trans.force_phy = None
        res_rate = eval_over_snr(model, val_loader, device, snr_db_list)
        single_rate_results.append((r, res_rate))

    # Reset forcing
    trans.force_expert = None
    trans.force_phy = None

    # ---------- PLOT: Accuracy vs SNR ----------
    plt.figure(figsize=(8, 5))
    plt.plot(res_multi["snr_db"], res_multi["acc"], marker="o", label="Multimode")

    for m, res_qam in single_qam_results:
        plt.plot(
            res_qam["snr_db"],
            res_qam["acc"],
            linestyle="--",
            marker="x",
            label=f"Single QAM m={m}",
        )

    for r, res_rate in single_rate_results:
        plt.plot(
            res_rate["snr_db"],
            res_rate["acc"],
            linestyle=":",
            marker="s",
            label=f"Single RATE r={r}",
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Accuracy")
    plt.title("Classification accuracy vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("accuracy_vs_snr.png", dpi=200)
    print("Saved accuracy plot -> accuracy_vs_snr.png")

    # ---------- PLOT: Mode usage vs SNR (simple version) ----------
    # Example: plot expert usage and QAM usage (per-SNR average)
    plt.figure(figsize=(8, 5))
    for r in range(R):
        plt.plot(
            res_multi["snr_db"],
            res_multi["expert_usage"][:, r],
            marker="o",
            label=f"expert {r}",
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("P(expert r | SNR)")
    plt.title("Expert usage vs SNR (multimode)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("expert_usage_vs_snr.png", dpi=200)
    print("Saved expert usage plot -> expert_usage_vs_snr.png")

    plt.figure(figsize=(8, 5))
    for m in range(M):
        plt.plot(
            res_multi["snr_db"],
            res_multi["phy_usage"][:, m],
            marker="o",
            label=f"{m}-QAM ",
        )
    plt.xlabel("SNR (dB)")
    plt.ylabel("P(QAM m | SNR)")
    plt.title("QAM usage vs SNR (multimode)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("qam_usage_vs_snr.png", dpi=200)
    print("Saved QAM usage plot -> qam_usage_vs_snr.png")


    trans = model.transceiver
    R = trans.R
    M = trans.M
    J = trans.J

    snr_db = np.array(res_multi["snr_db"])
    expert_usage = res_multi["expert_usage"]   # [K, R]  (K = #SNR points)
    phy_usage = res_multi["phy_usage"]         # [K, M]
    joint_usage = res_multi["joint_usage"]     # [K, J]

    # --------- 1) Overall mean usage across SNR (bar charts) ---------
    exp_mean = expert_usage.mean(axis=0)       # [R]
    phy_mean = phy_usage.mean(axis=0)          # [M]
    joint_mean = joint_usage.mean(axis=0)      # [J]

    # Expert usage
    plt.figure(figsize=(6, 4))
    x = np.arange(R)
    plt.bar(x, exp_mean)
    plt.xticks(x, [f"e_size{r}" for r in range(R)])
    plt.ylabel("Average usage probability")
    plt.title("Expert usage (averaged over SNR)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("expert_usage_bar_overall.png", dpi=200)
    print("Saved expert usage bar chart -> expert_usage_bar_overall.png")

    # QAM / PHY usage
    plt.figure(figsize=(6, 4))
    x = np.arange(M)
    plt.bar(x, phy_mean)
    plt.xticks(x, [f"m{m}" for m in range(M)])
    plt.ylabel("Average usage probability")
    plt.title("QAM mode usage (averaged over SNR)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("qam_usage_bar_overall.png", dpi=200)
    print("Saved QAM usage bar chart -> qam_usage_bar_overall.png")

    # Joint mode usage (optional, if J not too big)
    plt.figure(figsize=(8, 4))
    x = np.arange(J)
    plt.bar(x, joint_mean)
    plt.xticks(x, [f"j{j}" for j in range(J)], rotation=45, ha="right")
    plt.ylabel("Average usage probability")
    plt.title("Joint mode usage (averaged over SNR)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("joint_usage_bar_overall.png", dpi=200)
    print("Saved joint mode usage bar chart -> joint_usage_bar_overall.png")
    # --------- 2) Per-SNR grouped bar charts ---------
    K = len(snr_db)

    # Expert usage per SNR
    plt.figure(figsize=(8, 5))
    x = np.arange(R)
    width = 0.8 / max(K, 1)

    for i in range(K):
        plt.bar(
            x + i * width,
            expert_usage[i],       # [R] at this SNR
            width=width,
            label=f"{snr_db[i]:.1f} dB",
        )

    plt.xticks(x + width * (K - 1) / 2, [f"e{r}" for r in range(R)])
    plt.ylabel("P(expert r | SNR)")
    plt.title("Expert usage per SNR (multimode)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("expert_usage_bar_per_snr.png", dpi=200)
    print("Saved expert usage per-SNR bar chart -> expert_usage_bar_per_snr.png")

    # QAM usage per SNR
    plt.figure(figsize=(8, 5))
    x = np.arange(M)
    width = 0.8 / max(K, 1)

    for i in range(K):
        plt.bar(
            x + i * width,
            phy_usage[i],          # [M] at this SNR
            width=width,
            label=f"{snr_db[i]:.1f} dB",
        )

    plt.xticks(x + width * (K - 1) / 2, [f"m{m}" for m in range(M)])
    plt.ylabel("P(QAM m | SNR)")
    plt.title("QAM usage per SNR (multimode)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("qam_usage_bar_per_snr.png", dpi=200)
    print("Saved QAM usage per-SNR bar chart -> qam_usage_bar_per_snr.png")
    
if __name__ == "__main__":
    main()
