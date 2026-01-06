import os
# import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import argparse
import time
import torch
from datasets import load_dataset
import torch.nn as nn
import numpy as np
from utils_VQVAE import train_one_epoch_cls, eval_one_epoch_cls
from models_2.transceiver_VQ_VAE_JSCC import MODJSCC_MoE_Cls_VQ
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

# Default: anomaly detection OFF (VERY slow). Enable via CLI when debugging.
torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--detect-anomaly', action='store_true', default=False,
                    help='Enable autograd anomaly detection (VERY slow; use only for debugging)')
parser.add_argument('--lambda-vec', default=1.0, type=float,
                    help='Weight for latent distortion loss between encoder and recovered latent')

parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/RoBERTa_MoE/checkpoints/JSCC_MoE_MoE_Cls_1', type=str)
parser.add_argument('--d-model', default=256, type=int)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--epochs', default=30, type=int)
# legacy knobs you can keep tuning
parser.add_argument('--lambda_rate', default=.002, type=float)
parser.add_argument('--lambda_M', default=.02, type=float)   # used as lambda_vq
parser.add_argument('--lambda_mod', default=.002, type=float) 
parser.add_argument('--lambda_ch', default=.002, type=float)
parser.add_argument('--lambda_lat', default=.002, type=float)
parser.add_argument('--vq_warmup_epochs', default=5, type=float) 

parser.add_argument('--data', default='sst2', choices=['sst2', 'text'])
parser.add_argument('--text-train', type=str, default=None)
parser.add_argument('--text-val', type=str, default=None)
parser.add_argument('--max-length', type=int, default=64)

parser.add_argument('--lr', type=float, default=7e-5)
parser.add_argument('--weight-decay', type=float, default=0.5)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--use-amp', action='store_true', help='Use mixed precision')

# SNR control
parser.add_argument('--train-snr-min-db', type=float, default= 00.0,
                    help='Min SNR (dB) for training noise sampling')
parser.add_argument('--train-snr-max-db', type=float, default= 20.0,
                    help='Max SNR (dB) for training noise sampling')
parser.add_argument('--eval-snr-db', type=float, default= 20.0,
                    help='Fixed SNR (dB) used for validation evaluation')
parser.add_argument('--soft-vq-path', action='store_true', default=True,
                    help='Enable soft VQ path during training for differentiable quantization')
parser.add_argument('--alignment-start-epoch', type=int, default=15,
                    help='Epoch to start hard-forward/soft-backward alignment phase')
parser.add_argument('--temp-freeze-floor', type=float, default=0.5,
                    help='Temperature floor during phase 1 (learning), freeze here before alignment')
parser.add_argument('--lambda-entropy', type=float, default=0.2,
                    help='Entropy regularization weight to prevent premature collapse')
parser.add_argument('--lambda-consistency', type=float, default=0.4,
                    help='Consistency loss weight between hard and soft paths during alignment')
parser.add_argument('--rebuild-labeling-every', type=int, default=2,
                    help='Rebuild VQ labeling every N epochs (0 disables; can be slow)')

args = parser.parse_args()

# Enable anomaly detection only if explicitly requested
if args.detect_anomaly:
    torch.autograd.set_detect_anomaly(True)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_sst2(tokenizer, max_length: int):
    """
    Load SST-2 (GLUE) and tokenize.
    Returns HF datasets ready to be wrapped in DataLoader.
    """
    raw_datasets = load_dataset("glue", "sst2")

    def preprocess_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    encoded = raw_datasets.map(preprocess_fn, batched=True)
    # rename label -> labels for our training utils
    encoded = encoded.rename_column("label", "labels")

    encoded["train"].set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    encoded["validation"].set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return encoded["train"], encoded["validation"]


@torch.no_grad()
def plot_signal_distributions(
    model: nn.Module,
    dataloader,
    device: torch.device,
    save_dir: str,
    n_batches: int | None = 20,
):
    """
    Collects encoder outputs y (pre-VQ/JSCC) and feat_pooled (post-JSCC)
    from the classification model and plots their 1D histograms.

    Uses stats["y_target"] and stats["feat_pooled"] that the model forward saves.
    """
    model.eval()

    orig_list = []
    recon_list = []

    batches_seen = 0

    for batch in dataloader:
        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            n_var = batch.get("n_var", 0.01)
        else:
            input_ids, attention_mask, labels = batch[:3]
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            n_var = 0.01

        if torch.is_tensor(n_var):
            n_var = n_var.to(device)

        # forward pass; we only care about stats
        logits, rate_bits, route_hard_tx,Ns_eff, stats = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            n_var=n_var,
            channel="AWGN",
            return_probs=False,
        )

        y = stats["y_target"].detach().cpu()        # [B, d_model]
        y_hat = stats["feat_pooled"].detach().cpu() # [B, d_model]

        orig_list.append(y)
        recon_list.append(y_hat)

        batches_seen += 1
        if (n_batches is not None) and (batches_seen >= n_batches):
            break

    if not orig_list:
        print("No batches to plot distributions from.")
        return

    orig = torch.cat(orig_list, dim=0).view(-1).numpy()
    recon = torch.cat(recon_list, dim=0).view(-1).numpy()

    # Plot histograms
    plt.figure(figsize=(8, 5))
    bins = 100
    plt.hist(orig, bins=bins, density=True, alpha=0.5, label="pre-VQ / encoder y")
    plt.hist(recon, bins=bins, density=True, alpha=0.5, label="post-JSCC / feat_pooled")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Latent distribution: pre-VQ vs post-JSCC")
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "signal_distribution.png")
    plt.savefig(out_path)
    plt.close()

    print(f"Saved signal distribution plot to: {out_path}")


def freeze_roberta_backbone(model: nn.Module):
    """
    Freeze the RoBERTa backbone inside MODJSCC_MoE_Cls_VQ (handles DataParallel).
    """
    core = model.module if isinstance(model,  nn.DataParallel) else model

    if not hasattr(core, "encoder"):
        raise AttributeError("Model has no .encoder attribute")

    if not hasattr(core.encoder, "roberta"):
        raise AttributeError("Encoder has no .roberta attribute")

    for p in core.encoder.roberta.parameters():
        p.requires_grad = False

    print("[INFO] RoBERTa backbone frozen (requires_grad=False).")
def unwrap(m):
    return m.module if hasattr(m, "module") else m


def plot_training_losses(train_metrics_history, val_metrics_history, save_dir: str):
    """
    Plot training and validation losses over epochs.
    
    Args:
        train_metrics_history: list of dicts with 'loss', 'cls_loss', 'vq_loss', 'sym_loss', 'lb_loss'
        val_metrics_history: list of dicts with same keys
        save_dir: directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if not train_metrics_history or not val_metrics_history:
        print("No metrics history to plot")
        return
    
    epochs = range(1, len(train_metrics_history) + 1)
    
    # Extract metrics
    train_total_loss = [m['loss'] for m in train_metrics_history]
    val_total_loss = [m['loss'] for m in val_metrics_history]
    
    train_cls_loss = [m.get('cls_loss', 0) for m in train_metrics_history]
    val_cls_loss = [m.get('cls_loss', 0) for m in val_metrics_history]
    
    train_vq_loss = [m.get('vq_loss', 0) for m in train_metrics_history]
    val_vq_loss = [m.get('vq_loss', 0) for m in val_metrics_history]
    
    train_sym_loss = [m.get('sym_loss', 0) for m in train_metrics_history]
    val_sym_loss = [m.get('sym_loss', 0) for m in val_metrics_history]
    
    train_lb_loss = [m.get('lb_loss', 0) for m in train_metrics_history]
    val_lb_loss = [m.get('lb_loss', 0) for m in val_metrics_history]
    
    train_ch_loss = [m.get('ch_loss', 0) for m in train_metrics_history]
    val_ch_loss = [m.get('ch_loss', 0) for m in val_metrics_history]
    
    train_lat_loss = [m.get('lat_loss', 0) for m in train_metrics_history]
    val_lat_loss = [m.get('lat_loss', 0) for m in val_metrics_history]
    
    # Plot 1: Total Loss
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Metrics Over Epochs', fontsize=16)
    
    # Total Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_total_loss, 'o-', label='Train', linewidth=2)
    ax.plot(epochs, val_total_loss, 's-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Classification Loss
    ax = axes[0, 1]
    ax.plot(epochs, train_cls_loss, 'o-', label='Train', linewidth=2)
    ax.plot(epochs, val_cls_loss, 's-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Classification Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # VQ Loss
    ax = axes[0, 2]
    ax.plot(epochs, train_vq_loss, 'o-', label='Train', linewidth=2)
    ax.plot(epochs, val_vq_loss, 's-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('VQ Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Symbol Loss
    ax = axes[1, 0]
    ax.plot(epochs, train_sym_loss, 'o-', label='Train', linewidth=2)
    ax.plot(epochs, val_sym_loss, 's-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Symbol / Rate Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Load Balance Loss
    ax = axes[1, 1]
    ax.plot(epochs, train_lb_loss, 'o-', label='Train', linewidth=2)
    ax.plot(epochs, val_lb_loss, 's-', label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Load Balance Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Channel + Latent Loss
    ax = axes[1, 2]
    if any(train_ch_loss) or any(val_ch_loss):
        ax.plot(epochs, [c + l for c, l in zip(train_ch_loss, train_lat_loss)], 'o-', label='Train (Ch+Lat)', linewidth=2)
        ax.plot(epochs, [c + l for c, l in zip(val_ch_loss, val_lat_loss)], 's-', label='Val (Ch+Lat)', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Channel + Latent Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_plot_path = os.path.join(save_dir, 'training_losses.png')
    plt.savefig(loss_plot_path, dpi=150)
    plt.close()
    
    print(f"✓ Saved training losses plot to: {loss_plot_path}")


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # ----- Tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # ----- Dataset -----
    if args.data == "sst2":
        train_ds, val_ds = prepare_sst2(tokenizer, args.max_length)
    else:
        raise ValueError("Custom 'text' mode not implemented yet. Use --data sst2 for now.")

    train_loader = DataLoader(
        train_ds,

        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    frozen_roberta=False
    # ----- Model -----
    model = MODJSCC_MoE_Cls_VQ(
        d_model=args.d_model,
        num_labels=2,
        freeze_bert=frozen_roberta,
        # phy_M_list=([4]),#92% with EMA+ze-zedetach
        # vq_codebook_sizes=([4096]), %92% with 20db alone
        vq_codebook_sizes=(256, 1024, 4096 ), 
        
        phy_M_list=(4 ,16 ,64, 256),

    ).to(device)
    
    # Enable soft VQ path if requested
    if args.soft_vq_path:
        unwrap(model).transceiver_dig.soft_vq_path = True
        print("✓ Soft VQ path enabled (fully differentiable pipeline)")

    # Optional: DataParallel if multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model =  nn.DataParallel(model)

    # ----- Optimizer -----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_ckpt_path = os.path.join(args.checkpoint_path, "best_model.pt")
    last_ckpt_path = os.path.join(args.checkpoint_path, "last_model.pt")

    # Track metrics history for plotting
    train_metrics_history = []
    val_metrics_history = []

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        vq_scale = min(1.0, epoch / float(args.vq_warmup_epochs))  # add new arg, e.g. 5
        lambda_vq_cur = args.lambda_M * vq_scale
        
        # ======================================================
        # TWO-PHASE TRAINING STRATEGY
        # ======================================================
        # Phase 1 (epochs 1 to alignment_start): Learn with soft path, anneal to moderate T
        # Phase 2 (alignment_start to end): Freeze T, enable hard-forward/soft-backward
        # ======================================================
        
        alignment_phase = (epoch >= args.alignment_start_epoch)
        
        if not alignment_phase:
            # PHASE 1: LEARNING - anneal down to temp_freeze_floor
            vq_temp = max(args.temp_freeze_floor, 1.0 * (0.97 ** (epoch - 1)))
            symbol_temp = max(args.temp_freeze_floor, 1.0 * (0.97 ** (epoch - 1)))
            use_hard_forward = False
            phase_name = "LEARNING"
        else:
            # PHASE 2: ALIGNMENT - freeze temp, use hard-forward/soft-backward
            vq_temp = args.temp_freeze_floor
            symbol_temp = args.temp_freeze_floor
            use_hard_forward = True
            phase_name = "ALIGNMENT"
        
        unwrap(model).transceiver_dig.vq_temp = vq_temp
        unwrap(model).transceiver_dig.symbol_temp = symbol_temp
        
        # Anneal router temperature too (sharper expert/PHY selection as training progresses)
        unwrap(model).transceiver_dig.routing_temp = symbol_temp
        
        # Keep soft_temp constant (RX-side soft embedding needs to stay soft for gradients)
        unwrap(model).transceiver_dig.soft_temp = 0.7
        
        # Reset debug flag at start of each epoch to log first batch
        unwrap(model).transceiver_dig._debug_logged = False
        
        print(f"[Epoch {epoch}] Phase={phase_name} | vq_T={vq_temp:.3f} sym_T={symbol_temp:.3f} | hard_fwd={use_hard_forward}")
        
        if (epoch > 15) and (not frozen_roberta):
            freeze_roberta_backbone(model)
            frozen_roberta = True


        train_metrics = train_one_epoch_cls(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_vec=args.lambda_vec,
            lambda_sym=args.lambda_rate,
            lambda_lat=args.lambda_lat,
            lambda_ch=args.lambda_ch,
            lambda_vq=lambda_vq_cur,
            lambda_lb=args.lambda_mod,
            lambda_entropy=args.lambda_entropy,
            lambda_consistency=args.lambda_consistency if alignment_phase else 0.0,
            use_hard_forward=use_hard_forward,
            max_grad_norm=1.0,
            use_amp=args.use_amp,
            snr_min_db=args.train_snr_min_db,
            snr_max_db=args.train_snr_max_db,
        )

        val_metrics = eval_one_epoch_cls(
            model=model,
            dataloader=val_loader,
            device=device,
            lambda_vec=args.lambda_vec,
            lambda_sym=args.lambda_rate,
            lambda_vq=lambda_vq_cur,
            lambda_lb=args.lambda_mod,
            eval_snr_db=args.eval_snr_db,
        )

        # Store metrics for later plotting
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)

        elapsed = time.time() - start_time

        if args.rebuild_labeling_every > 0 and epoch >= 1 and (epoch % args.rebuild_labeling_every == 0):
            for expert in unwrap(model).transceiver_analog.vq_moe.experts:
                expert.rebuild_labeling(method="pca_gray", sync_ddp=True)
        # Collapse detection metrics
        entropy_str = ""
        if 'vq_entropy' in train_metrics:
            entropy_str = (f" | VQ_ent: {train_metrics['vq_entropy']:.3f}/{val_metrics['vq_entropy']:.3f} "
                          f"Sym_ent: {train_metrics['sym_entropy']:.3f}/{val_metrics['sym_entropy']:.3f}")
        
        print(
            f"[Epoch {epoch}/{args.epochs}] {phase_name} | time={elapsed:.1f}s\n"
            f"  Train: loss={train_metrics['loss']:.4f} cls={train_metrics['cls_loss']:.4f} "
            f"sym={train_metrics['sym_loss']:.2f} lb={train_metrics['lb_loss']:.3f} "
            f"vq={train_metrics['vq_loss']:.3f} acc={train_metrics['accuracy']:.4f}\n"
            f"  Val:   loss={val_metrics['loss']:.4f} cls={val_metrics['cls_loss']:.4f} "
            f"sym={val_metrics['sym_loss']:.2f} lb={val_metrics['lb_loss']:.3f} "
            f"vq={val_metrics['vq_loss']:.3f} acc={val_metrics['accuracy']:.4f}\n"
            f"  Rate:  bits={val_metrics['avg_bits_per_block']:.1f} syms={val_metrics['avg_syms_per_block']:.1f}"
            f"{entropy_str}"
        )


        # ----- Save best checkpoint by validation accuracy (ALIGNMENT phase only) -----
        val_acc = val_metrics["accuracy"]
        if alignment_phase and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            # unwrap DataParallel if needed for saving
            model_to_save = model.module if isinstance(model,  nn.DataParallel) else model

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_acc,
                    "vq_temp": vq_temp,
                    "symbol_temp": symbol_temp,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"  -> New best model saved (epoch={epoch}, val_acc={best_val_acc:.4f}, vq_T={vq_temp:.3f}, sym_T={symbol_temp:.3f})")
        
        # ----- Always save last checkpoint -----
        model_to_save = model.module if isinstance(model,  nn.DataParallel) else model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "vq_temp": vq_temp,
                "symbol_temp": symbol_temp,
                "args": vars(args),
            },
            last_ckpt_path,
        )
        if epoch == args.epochs:
            print(f"  -> Last model saved (epoch={epoch}, val_acc={val_acc:.4f}, vq_T={vq_temp:.3f}, sym_T={symbol_temp:.3f})")

    print("\n" + "="*70)
    print("Training finished!")
    print(f"  Best val accuracy: {best_val_acc:.4f} (achieved at epoch {best_epoch})")
    print(f"  Best checkpoint: {best_ckpt_path}")
    print(f"  Last checkpoint: {last_ckpt_path}")
    print(f"  Final val accuracy: {val_metrics['accuracy']:.4f} (epoch {args.epochs})")
    print("="*70 + "\n")

    # ----- Plot training losses -----
    print("Plotting training losses...")
    plot_training_losses(train_metrics_history, val_metrics_history, args.checkpoint_path)

    # ----- After training: plot distributions pre-VQ vs post-JSCC -----
    # Use the final model (not necessarily best, but it's fine for analyzing behavior).
    # If you prefer, you can reload the best checkpoint here instead.
    print("Collecting signal distributions on validation set...")
    model_for_plot = model  # already on device
    plot_signal_distributions(
        model=model_for_plot,
        dataloader=val_loader,
        device=device,
        save_dir=args.checkpoint_path,
        n_batches=256,
    )


if __name__ == "__main__":
    main()
