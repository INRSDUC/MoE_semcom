import os
import math
import argparse
from tqdm import tqdm
import time
import torch
from datasets import load_dataset
import torch.nn as nn
import numpy as np
from utils_VQVAE import train_one_epoch_cls, eval_one_epoch_cls
from models_2.transceiver_VQ_VAE_JSCC import MODJSCC_MoE_Cls_VQ
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

# Silence tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/RoBERTa_MoE/checkpoints/JSCC_MoE_MoE_Cls_1', type=str)
parser.add_argument('--d-model', default=256, type=int)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--epochs', default=30, type=int)
# legacy knobs you can keep tuning
parser.add_argument('--lambda_rate', default=.000, type=float)
parser.add_argument('--lambda_M', default=.02, type=float)   # used as lambda_vq
parser.add_argument('--lambda_mod', default=.000, type=float) 
parser.add_argument('--vq_warmup_epochs', default=10, type=float) 

parser.add_argument('--data', default='sst2', choices=['sst2', 'text'])
parser.add_argument('--text-train', type=str, default=None)
parser.add_argument('--text-val', type=str, default=None)
parser.add_argument('--max-length', type=int, default=64)

parser.add_argument('--lr', type=float, default=4e-5)
parser.add_argument('--weight-decay', type=float, default=0.01)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--use-amp', action='store_true', help='Use mixed precision')

# SNR control
parser.add_argument('--train-snr-min-db', type=float, default=0.0,
                    help='Min SNR (dB) for training noise sampling')
parser.add_argument('--train-snr-max-db', type=float, default=10.0,
                    help='Max SNR (dB) for training noise sampling')
parser.add_argument('--eval-snr-db', type=float, default=10.0,
                    help='Fixed SNR (dB) used for validation evaluation')

args = parser.parse_args()
 

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

    # ----- Model -----
    model = MODJSCC_MoE_Cls_VQ(
        d_model=args.d_model,
        num_labels=2,
        freeze_bert=False,
        vq_codebook_sizes=(128, 256, 512, 1024, 2048),
        phy_M_list=(4, 16, 64, 256, 1024),
    ).to(device)

    # Optional: DataParallel if multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # ----- Optimizer -----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.checkpoint_path, "best_model.pt")

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        vq_scale = min(1.0, epoch / float(args.vq_warmup_epochs))  # add new arg, e.g. 5
        lambda_vq_cur = args.lambda_M * vq_scale

        # Train
        # Train
        train_metrics = train_one_epoch_cls(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_sym=args.lambda_rate,   # λ_sym for J_sym
            lambda_lb=args.lambda_mod,       # λ_lb for J_lb
            lambda_vq=lambda_vq_cur,     # weight on VQ loss
            max_grad_norm=1.0,
            use_amp=args.use_amp,
            snr_min_db=args.train_snr_min_db,
            snr_max_db=args.train_snr_max_db,
        )

        # Eval
        val_metrics = eval_one_epoch_cls(
            model=model,
            dataloader=val_loader,
            device=device,
            lambda_sym=args.lambda_rate,
            lambda_lb=args.lambda_mod,
            lambda_vq=lambda_vq_cur,
            eval_snr_db=args.eval_snr_db,
        )


        # Eval
        # val_metrics = eval_one_epoch_cls(
        #     model=model,
        #     dataloader=val_loader,
        #     device=device,
        #     lambda_rate=args.lambda_rate,
        #     lambda_vq=args.lambda_M,
        #     eval_snr_db=args.eval_snr_db,
        # )

        elapsed = time.time() - start_time


        # print(
        #     f"[Epoch {epoch}/{args.epochs}] "
        #     f"time={elapsed:.1f}s | "
        #     f"train_loss={train_metrics['loss']:.4f} "
        #     f"train_acc={train_metrics['accuracy']:.4f} | "
        #     f"val_loss={val_metrics['loss']:.4f} "
        #     f"val_acc={val_metrics['accuracy']:.4f} | "
        #     f"val_bits={val_metrics['avg_bits_per_block']:.1f} "
        #     f"val_syms={val_metrics['avg_syms_per_block']:.1f} "
        #     f"(eval_snr={args.eval_snr_db:.1f}dB)"
        # )
        print(
            f"[Epoch {epoch}/{args.epochs}] time={elapsed:.1f}s | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_cls={train_metrics['cls_loss']:.4f} "
            f"train_sym={train_metrics['sym_loss']:.2f} "
            f"train_lb={train_metrics['lb_loss']:.3f} "
            f"train_vq={train_metrics['vq_loss']:.3f} "
            f"train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_cls={val_metrics['cls_loss']:.4f} "
            f"val_sym={val_metrics['sym_loss']:.2f} "
            f"val_lb={val_metrics['lb_loss']:.3f} "
            f"val_vq={val_metrics['vq_loss']:.3f} "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_bits={val_metrics['avg_bits_per_block']:.1f} "
            f"val_syms={val_metrics['avg_syms_per_block']:.1f}"
        )


        # ----- Save only best checkpoint by validation accuracy -----
        val_acc = val_metrics["accuracy"]
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # unwrap DataParallel if needed for saving
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": best_val_acc,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"  -> New best model saved (val_acc={best_val_acc:.4f}) at {best_ckpt_path}")

    print(f"Training finished. Best val accuracy: {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
