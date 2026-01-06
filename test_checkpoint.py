#!/usr/bin/env python3
"""Test checkpoint evaluation script."""

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from models_2.transceiver_VQ_VAE_JSCC import MODJSCC_MoE_Cls_VQ
from utils_VQVAE import eval_one_epoch_cls
import argparse

def prepare_sst2(tokenizer, max_length: int):
    """Load SST-2 (GLUE) and tokenize."""
    raw_datasets = load_dataset("glue", "sst2")

    def preprocess_fn(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

    encoded = raw_datasets.map(preprocess_fn, batched=True)
    encoded = encoded.rename_column("label", "labels")

    encoded["validation"].set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    return encoded["train"], encoded["validation"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--eval-snr-db', type=float, default=20.0, help='Evaluation SNR')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    
    print(f"\nCheckpoint Info:")
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Val Accuracy: {ckpt['val_accuracy']:.4f}")
    
    # Get saved args
    saved_args = ckpt.get('args', {})
    d_model = saved_args.get('d_model', 256)
    max_length = saved_args.get('max_length', 50)
    
    print(f"  d_model: {d_model}")
    print(f"  max_length: {max_length}")
    
    # Load tokenizer and dataset
    print("\nLoading dataset...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    _, val_ds = prepare_sst2(tokenizer, max_length)
    
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    print("\nCreating model...")
    model = MODJSCC_MoE_Cls_VQ(
        d_model=d_model,
        num_labels=2,
        freeze_bert=False,
        vq_codebook_sizes=(256, 1024, 4096),
        phy_M_list=(4, 16, 64, 256),
    ).to(device)
    
    # Enable soft VQ path if it was used during training
    if hasattr(model, 'transceiver_dig'):
        model.transceiver_dig.soft_vq_path = True
        print("✓ Soft VQ path enabled")
    
    # Load weights (strict=False to handle potential architecture changes)
    missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
    print("✓ Model weights loaded")
    
    # Evaluate
    print(f"\nEvaluating at SNR={args.eval_snr_db} dB...")
    val_metrics = eval_one_epoch_cls(
        model=model,
        dataloader=val_loader,
        device=device,
        lambda_vec=1.0,
        lambda_sym=0.002,
        lambda_vq=0.02,
        lambda_lb=0.002,
        eval_snr_db=args.eval_snr_db,
    )
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Val Loss:      {val_metrics['loss']:.4f}")
    print(f"Val Accuracy:  {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
    print(f"Cls Loss:      {val_metrics['cls_loss']:.4f}")
    print(f"VQ Loss:       {val_metrics['vq_loss']:.4f}")
    print(f"Sym Loss:      {val_metrics['sym_loss']:.2f}")
    print(f"LB Loss:       {val_metrics['lb_loss']:.3f}")
    print(f"Avg Bits:      {val_metrics['avg_bits_per_block']:.1f}")
    print(f"Avg Symbols:   {val_metrics['avg_syms_per_block']:.1f}")
    print("="*60)


if __name__ == "__main__":
    main()
