"""
Legacy (corrupted) version preserved for reference. New implementation follows after this block.
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import argparse
import time
import random
    best_val_nll = float("inf")
    best_epoch = 0
    last_ckpt_path = os.path.join(args.checkpoint_path, "last_model.pt")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        vq_scale = min(1.0, epoch / float(args.vq_warmup_epochs))
        lambda_vq_cur = args.lambda_M * vq_scale

        # ======================================================
        # TWO-PHASE TRAINING STRATEGY (same as classification)
        # ======================================================
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        import argparse
        import math
        import random
        import time
        from pathlib import Path
        from typing import Iterable, List, Tuple

        import numpy as np
        import torch
        import torch.nn as nn
        from datasets import load_dataset, load_from_disk
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer

        from utils_VQVAE import eval_one_epoch_rec, train_one_epoch_rec
        from models_2.transceiver_VQ_VAE_JSCC import MODJSCC_MoE_TextRec_VQ


        # ------------------------------------------------------------
        # Utilities
        # ------------------------------------------------------------
        def set_seed(seed: int) -> None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


        def unwrap(model: nn.Module) -> nn.Module:
            return model.module if isinstance(model, nn.DataParallel) else model


        def get_transceiver(model: nn.Module):
            return unwrap(model).transceiver


        def _mask_pad_as_ignore(input_ids: List[List[int]], pad_token_id: int) -> List[List[int]]:
            labels: List[List[int]] = []
            for row in input_ids:
                labels.append([tok if tok != pad_token_id else -100 for tok in row])
            return labels


        def _tokenize_texts(tokenizer, texts: Iterable[str], max_length: int):
            enc = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            enc["labels"] = _mask_pad_as_ignore(enc["input_ids"], tokenizer.pad_token_id)
            return enc


        def prepare_sst2_reconstruction(tokenizer, max_length: int):
            raw = load_dataset("glue", "sst2")

            def preprocess(batch):
                return _tokenize_texts(tokenizer, batch["sentence"], max_length)

            encoded = raw.map(preprocess, batched=True, remove_columns=["sentence", "label", "idx"])
            encoded["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            encoded["validation"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            return encoded["train"], encoded["validation"]


        def prepare_imdb_reconstruction(tokenizer, max_length: int):
            raw = load_dataset("imdb")

            def preprocess(batch):
                return _tokenize_texts(tokenizer, batch["text"], max_length)

            encoded = raw.map(preprocess, batched=True, remove_columns=["text", "label"])
            encoded["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            encoded["test"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            return encoded["train"], encoded["test"]


        def prepare_wikitext103(tokenizer, max_length: int, processed_root: str, overwrite: bool):
            processed_dir = Path(processed_root)
            processed_dir.mkdir(parents=True, exist_ok=True)

            cache_path = processed_dir / f"wikitext103_maxlen{max_length}"
            if cache_path.exists() and not overwrite:
                ds = load_from_disk(str(cache_path))
                train_ds, val_ds = ds["train"], ds["validation"]
            else:
                raw = load_dataset("wikitext", "wikitext-103-raw-v1")

                def preprocess(batch):
                    return _tokenize_texts(tokenizer, batch["text"], max_length)

                encoded = raw.map(preprocess, batched=True, remove_columns=["text"])
                encoded.save_to_disk(str(cache_path))
                train_ds, val_ds = encoded["train"], encoded["validation"]

            train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
            return train_ds, val_ds


        def parse_int_tuple(text: str) -> Tuple[int, ...]:
            return tuple(int(x.strip()) for x in text.split(",") if x.strip())


        def build_arg_parser() -> argparse.ArgumentParser:
            p = argparse.ArgumentParser()

            # Model
            p.add_argument("--model-name", default="facebook/bart-base", type=str)
            p.add_argument("--N", default=9, type=int, help="Number of latent tokens")
            p.add_argument("--vq-codebook-sizes", default="256,1024,4096", type=str)
            p.add_argument("--phy-M-list", default="4,16,64,256", type=str)
            p.add_argument("--freeze-encoder", action="store_true")
            p.add_argument("--freeze-decoder", action="store_true")
            p.add_argument("--soft-vq-path", action="store_true")

            # Optimization
            p.add_argument("--batch-size", default=32, type=int)
            p.add_argument("--epochs", default=15, type=int)
            p.add_argument("--lr", default=4e-5, type=float)
            p.add_argument("--weight-decay", default=0.01, type=float)
            p.add_argument("--num-workers", default=4, type=int)
            p.add_argument("--use-amp", action="store_true")
            p.add_argument("--max-grad-norm", default=1.0, type=float)
            p.add_argument("--seed", default=42, type=int)
            p.add_argument("--detect-anomaly", action="store_true")

            # Loss weights
            p.add_argument("--lambda-vec", default=1.0, type=float)
            p.add_argument("--lambda-rate", default=0.0, type=float)
            p.add_argument("--lambda-M", default=0.05, type=float)
            p.add_argument("--lambda-mod", default=0.0, type=float)
            p.add_argument("--lambda-lat", default=0.0, type=float)
            p.add_argument("--lambda-ch", default=0.0, type=float)
            p.add_argument("--lambda-entropy", default=0.0, type=float)
            p.add_argument("--lambda-consistency", default=0.0, type=float)

            # Scheduling
            p.add_argument("--vq-warmup-epochs", default=5, type=float)
            p.add_argument("--alignment-start-epoch", default=6, type=int)
            p.add_argument("--temp-init", default=1.0, type=float)
            p.add_argument("--temp-decay", default=0.97, type=float)
            p.add_argument("--temp-freeze-floor", default=0.2, type=float)
            p.add_argument("--gumbel-tau-soft", default=1.0, type=float)
            p.add_argument("--gumbel-tau-hard", default=0.5, type=float)
            p.add_argument("--routing-temp", default=1.0, type=float)
            p.add_argument("--soft-temp", default=0.7, type=float)

            # Data
            p.add_argument("--data", default="sst2", choices=["sst2", "wikitext103", "imdb"], type=str)
            p.add_argument("--max-length", default=64, type=int)
            p.add_argument("--train-max-examples", default=None, type=int)
            p.add_argument("--val-max-examples", default=None, type=int)
            p.add_argument("--processed-dir", default="/home/necphy/ducjunior/processed_datasets", type=str)
            p.add_argument("--overwrite-processed", action="store_true")

            # Noise
            p.add_argument("--train-snr-min-db", default=0.0, type=float)
            p.add_argument("--train-snr-max-db", default=10.0, type=float)
            p.add_argument("--eval-snr-db", default=10.0, type=float)

            # Checkpointing
            p.add_argument("--checkpoint-path", default="/home/necphy/ducjunior/RoBERTa_MoE/checkpoints/textrec_vq", type=str)
            p.add_argument("--rebuild-labeling-every", default=0, type=int, help="epochs between label rebuild; 0 to disable")

            return p


        def build_dataloaders(args, tokenizer):
            if args.data == "sst2":
                train_ds, val_ds = prepare_sst2_reconstruction(tokenizer, args.max_length)
            elif args.data == "wikitext103":
                train_ds, val_ds = prepare_wikitext103(
                    tokenizer, args.max_length, processed_root=args.processed_dir, overwrite=args.overwrite_processed
                )
            elif args.data == "imdb":
                train_ds, val_ds = prepare_imdb_reconstruction(tokenizer, args.max_length)
            else:
                raise ValueError(f"Unsupported dataset: {args.data}")

            if args.train_max_examples is not None and len(train_ds) > args.train_max_examples:
                train_ds = train_ds.select(range(args.train_max_examples))
            if args.val_max_examples is not None and len(val_ds) > args.val_max_examples:
                val_ds = val_ds.select(range(args.val_max_examples))

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
            return train_loader, val_loader


        def configure_model(args, device):
            vq_sizes = parse_int_tuple(args.vq_codebook_sizes)
            phy_list = parse_int_tuple(args.phy_M_list)

            model = MODJSCC_MoE_TextRec_VQ(
                model_name=args.model_name,
                N=args.N,
                vq_codebook_sizes=vq_sizes,
                phy_M_list=phy_list,
                freeze_encoder=args.freeze_encoder,
                freeze_decoder=args.freeze_decoder,
            ).to(device)

            if args.soft_vq_path:
                get_transceiver(model).soft_vq_path = True
                print("✓ Soft VQ path enabled (text reconstruction)")

            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
                model = nn.DataParallel(model)

            return model


        def main():
            parser = build_arg_parser()
            args = parser.parse_args()

            torch.autograd.set_detect_anomaly(bool(args.detect_anomaly))
            set_seed(args.seed)
            torch.backends.cudnn.benchmark = True

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            os.makedirs(args.checkpoint_path, exist_ok=True)

            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            train_loader, val_loader = build_dataloaders(args, tokenizer)

            model = configure_model(args, device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            best_val_nll = float("inf")
            best_epoch = 0
            best_ckpt_path = os.path.join(args.checkpoint_path, "best_model.pt")
            last_ckpt_path = os.path.join(args.checkpoint_path, "last_model.pt")

            print("Starting training (text reconstruction)...")
            for epoch in range(1, args.epochs + 1):
                start_time = time.time()

                vq_scale = min(1.0, epoch / float(args.vq_warmup_epochs))
                lambda_vq_cur = args.lambda_M * vq_scale

                alignment_phase = epoch >= args.alignment_start_epoch
                if not alignment_phase:
                    vq_temp = max(args.temp_freeze_floor, args.temp_init * (args.temp_decay ** (epoch - 1)))
                    symbol_temp = max(args.temp_freeze_floor, args.temp_init * (args.temp_decay ** (epoch - 1)))
                    use_hard_forward = False
                    gumbel_tau = args.gumbel_tau_soft
                    phase_name = "LEARNING"
                else:
                    vq_temp = args.temp_freeze_floor
                    symbol_temp = args.temp_freeze_floor
                    use_hard_forward = True
                    gumbel_tau = args.gumbel_tau_hard
                    phase_name = "ALIGNMENT"

                tx = get_transceiver(model)
                tx.vq_temp = vq_temp
                tx.symbol_temp = symbol_temp
                tx.routing_temp = args.routing_temp
                tx.soft_temp = args.soft_temp
                if hasattr(tx, "_debug_logged"):
                    tx._debug_logged = False

                print(f"[Epoch {epoch}] Phase={phase_name} | vq_T={vq_temp:.3f} sym_T={symbol_temp:.3f} | hard_fwd={use_hard_forward}")

                train_metrics = train_one_epoch_rec(
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
                    gumbel_tau=gumbel_tau,
                    max_grad_norm=args.max_grad_norm,
                    use_amp=args.use_amp,
                    snr_min_db=args.train_snr_min_db,
                    snr_max_db=args.train_snr_max_db,
                )

                val_metrics = eval_one_epoch_rec(
                    model=model,
                    dataloader=val_loader,
                    device=device,
                    lambda_vec=args.lambda_vec,
                    lambda_sym=args.lambda_rate,
                    lambda_vq=lambda_vq_cur,
                    lambda_lb=args.lambda_mod,
                    eval_snr_db=args.eval_snr_db,
                )

                elapsed = time.time() - start_time

                if args.rebuild_labeling_every > 0 and epoch % args.rebuild_labeling_every == 0:
                    tx_unwrapped = unwrap(get_transceiver(model))
                    if hasattr(tx_unwrapped, "vq_moe"):
                        for expert in tx_unwrapped.vq_moe.experts:
                            expert.rebuild_labeling(method="pca_gray", sync_ddp=True)

                entropy_str = ""
                if "vq_entropy" in train_metrics:
                    entropy_str = (
                        f" | VQ_ent: {train_metrics['vq_entropy']:.3f}/{val_metrics.get('vq_entropy', 0):.3f} "
                        f"Sym_ent: {train_metrics['sym_entropy']:.3f}/{val_metrics.get('sym_entropy', 0):.3f}"
                    )

                print(
                    f"[Epoch {epoch}/{args.epochs}] {phase_name} | time={elapsed:.1f}s\n"
                    f"  Train: loss={train_metrics['loss']:.4f} nll={train_metrics['nll_loss']:.4f} "
                    f"ppl={train_metrics['ppl']:.2f} sym={train_metrics['sym_loss']:.2f} "
                    f"lb={train_metrics['lb_loss']:.3f} vq={train_metrics['vq_loss']:.3f} "
                    f"tokacc={train_metrics['token_accuracy']:.3f}\n"
                    f"  Val:   loss={val_metrics['loss']:.4f} nll={val_metrics['nll_loss']:.4f} "
                    f"ppl={val_metrics['ppl']:.2f} sym={val_metrics['sym_loss']:.2f} "
                    f"lb={val_metrics['lb_loss']:.3f} vq={val_metrics['vq_loss']:.3f} "
                    f"tokacc={val_metrics['token_accuracy']:.3f}\n"
                    f"  Rate:  bits={val_metrics['avg_bits_per_block']:.1f} syms={val_metrics['avg_syms_per_block']:.1f}"
                    f"{entropy_str}"
                )

                val_nll = val_metrics["nll_loss"]
                if val_nll < best_val_nll:
                    best_val_nll = val_nll
                    best_epoch = epoch
                    model_to_save = unwrap(model)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model_to_save.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_nll": best_val_nll,
                            "vq_temp": vq_temp,
                            "symbol_temp": symbol_temp,
                            "args": vars(args),
                        },
                        best_ckpt_path,
                    )
                    print(
                        f"  -> New best model saved (epoch={epoch}, val_nll={best_val_nll:.4f}, vq_T={vq_temp:.3f}, sym_T={symbol_temp:.3f})"
                    )

                model_to_save = unwrap(model)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_nll": val_nll,
                        "vq_temp": vq_temp,
                        "symbol_temp": symbol_temp,
                        "args": vars(args),
                    },
                    last_ckpt_path,
                )
                if epoch == args.epochs:
                    print(
                        f"  -> Last model saved (epoch={epoch}, val_nll={val_nll:.4f}, vq_T={vq_temp:.3f}, sym_T={symbol_temp:.3f})"
                    )

            print("\n" + "=" * 70)
            print("Training finished!")
            print(f"  Best val NLL: {best_val_nll:.4f} (achieved at epoch {best_epoch})")
            print(f"  Best checkpoint: {best_ckpt_path}")
            print(f"  Last checkpoint: {last_ckpt_path}")
            print(f"  Final val NLL: {val_metrics['nll_loss']:.4f} (epoch {args.epochs})")
            print("=" * 70)


        if __name__ == "__main__":
            main()
        tx.symbol_temp = symbol_temp
        
        # Keep routing temperature constant (no need to harden expert/PHY selection)
        tx.routing_temp = 1.0
        
        # Keep soft_temp constant (RX-side soft embedding needs to stay soft for gradients)
        tx.soft_temp = 0.7
        
        # Reset debug flag at start of each epoch to log first batch
        tx._debug_logged = False
        
        print(f"[Epoch {epoch}] Phase={phase_name} | vq_T={vq_temp:.3f} sym_T={symbol_temp:.3f} | hard_fwd={use_hard_forward}")

        # ---- Train ----
        train_metrics = train_one_epoch_rec(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_vec=args.lambda_vec,
            lambda_sym=args.lambda_rate,
            lambda_lat=args.lambda_lat,
            lambda_ch=args.lambda_ch,
            lambda_vq=lambda_vq_cur,
        elapsed = time.time() - start_time

        # Optionally rebuild VQ labeling periodically
        if args.rebuild_labeling_every > 0 and epoch >= 1 and (epoch % args.rebuild_labeling_every == 0):
            tx_unwrapped = unwrap(get_transceiver(model))
            if hasattr(tx_unwrapped, 'vq_moe'):
                for expert in tx_unwrapped.vq_moe.experts:
                    expert.rebuild_labeling(method="pca_gray", sync_ddp=True)

        # Collapse detection metrics
        entropy_str = ""
        if 'vq_entropy' in train_metrics:
            entropy_str = (f" | VQ_ent: {train_metrics['vq_entropy']:.3f}/{val_metrics.get('vq_entropy', 0):.3f} "
                          f"Sym_ent: {train_metrics['sym_entropy']:.3f}/{val_metrics.get('sym_entropy', 0):.3f}")

        print(
            f"[Epoch {epoch}/{args.epochs}] {phase_name} | time={elapsed:.1f}s\n"
            f"  Train: loss={train_metrics['loss']:.4f} nll={train_metrics['nll_loss']:.4f} "
            f"ppl={train_metrics['ppl']:.2f} sym={train_metrics['sym_loss']:.2f} "
            f"lb={train_metrics['lb_loss']:.3f} vq={train_metrics['vq_loss']:.3f} "
            f"tokacc={train_metrics['token_accuracy']:.3f}\n"
            f"  Val:   loss={val_metrics['loss']:.4f} nll={val_metrics['nll_loss']:.4f} "
            f"ppl={val_metrics['ppl']:.2f} sym={val_metrics['sym_loss']:.2f} "
            f"lb={val_metrics['lb_loss']:.3f} vq={val_metrics['vq_loss']:.3f} "
            f"tokacc={val_metrics['token_accuracy']:.3f}\n"
            f"  Rate:  bits={val_metrics['avg_bits_per_block']:.1f} syms={val_metrics['avg_syms_per_block']:.1f}"
            f"{entropy_str}
            lambda_sym=args.lambda_rate,
            lambda_vq=lambda_vq_cur,
            lambda_lb=args.lambda_mod,
            eval_snr_db=args.eval_snr_db.lambda_mod,
            eval_snr_db=args.eval_snr_db,
            hard_routing=False,  # deterministic
            gumbel_tau=1.0,
        )
        

        elapsed = time.time() - start_time

        print(
            f"[Epoch {epoch}/{args.epochs}] time={elapsed:.1f}s | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_nll={train_metrics['nll_loss']:.4f} "
            f"train_ppl={train_metrics['ppl']:.2f} "
            f"train_tokacc={train_metrics['token_accuracy']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_nll={val_metrics['nll_loss']:.4f} "
            f"val_ppl={val_metrics['ppl']:.2f} "
            f"val_tokacc={val_metrics['token_accuracy']:.3f} | "
            f"val_bits={val_metrics['avg_bits_per_block']:.1f} "
            f"val_syms={val_metrics['avg_syms_per_block']:.1f} "
            f"(eval_snr={args.eval_snr_db:.1f}dB, lambda_vq={lambda_vq_cur:.4f})"
        )

        # ----- Save best checkpoint by val_nll -----
        val_nll = val_metrics["nll_loss"]
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_epoch = epoch
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_nll": best_val_nll,
                    "vq_temp": vq_temp,
                    "symbol_temp": symbol_temp,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(f"  -> New best model saved (epoch={epoch}, val_nll={best_val_nll:.4f}, vq_T={vq_temp:.3f}, sym_T={symbol_temp:.3f})")
        
        # ----- Always save last checkpoint -----
        model_to_save = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_nll": val_nll,
                "vq_temp": vq_temp,
                "symbol_temp": symbol_temp,
                "args": vars(args),
            },
            last_ckpt_path,
        )
        if epoch == args.epochs:
            print(f"  -> Last model saved (epoch={epoch}, val_nll={val_nll:.4f}, vq_T={vq_temp:.3f}, sym_T={symbol_temp:.3f})")

    print("\n" + "="*70)
    print("Training finished!")
    print(f"  Best val NLL: {best_val_nll:.4f} (achieved at epoch {best_epoch})")
    print(f"  Best checkpoint: {best_ckpt_path}")
    print(f"  Last checkpoint: {last_ckpt_path}")
    print(f"  Final val NLL: {val_metrics['nll_loss']:.4f} (epoch {args.epochs})")
    print("="*70)


if __name__ == "__main__":
    main()

"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import random
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models_2.transceiver_VQ_VAE_JSCC import MODJSCC_MoE_TextRec_VQ
from utils_VQVAE import eval_one_epoch_rec, train_one_epoch_rec


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def get_transceiver(model: nn.Module):
    return unwrap(model).transceiver


def _mask_pad_as_ignore(input_ids: List[List[int]], pad_token_id: int) -> List[List[int]]:
    labels: List[List[int]] = []
    for row in input_ids:
        labels.append([tok if tok != pad_token_id else -100 for tok in row])
    return labels


def _tokenize_texts(tokenizer, texts: Iterable[str], max_length: int):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    enc["labels"] = _mask_pad_as_ignore(enc["input_ids"], tokenizer.pad_token_id)
    return enc


def prepare_sst2_reconstruction(tokenizer, max_length: int):
    raw = load_dataset("glue", "sst2")

    def preprocess(batch):
        return _tokenize_texts(tokenizer, batch["sentence"], max_length)

    encoded = raw.map(preprocess, batched=True, remove_columns=["sentence", "label", "idx"])
    encoded["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    encoded["validation"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return encoded["train"], encoded["validation"]


def prepare_imdb_reconstruction(tokenizer, max_length: int):
    raw = load_dataset("imdb")

    def preprocess(batch):
        return _tokenize_texts(tokenizer, batch["text"], max_length)

    encoded = raw.map(preprocess, batched=True, remove_columns=["text", "label"])
    encoded["train"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    encoded["test"].set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return encoded["train"], encoded["test"]


def prepare_wikitext103(tokenizer, max_length: int, processed_root: str, overwrite: bool):
    processed_dir = Path(processed_root)
    processed_dir.mkdir(parents=True, exist_ok=True)

    cache_path = processed_dir / f"wikitext103_maxlen{max_length}"
    if cache_path.exists() and not overwrite:
        ds = load_from_disk(str(cache_path))
        train_ds, val_ds = ds["train"], ds["validation"]
    else:
        raw = load_dataset("wikitext", "wikitext-103-raw-v1")

        def preprocess(batch):
            return _tokenize_texts(tokenizer, batch["text"], max_length)

        encoded = raw.map(preprocess, batched=True, remove_columns=["text"])
        encoded.save_to_disk(str(cache_path))
        train_ds, val_ds = encoded["train"], encoded["validation"]

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return train_ds, val_ds


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    p.add_argument("--model-name", default="facebook/bart-base", type=str)
    p.add_argument("--N", default=9, type=int, help="Number of latent tokens")
    p.add_argument("--vq-codebook-sizes", default="256,1024,4096", type=str)
    p.add_argument("--phy-M-list", default="4,16,64,256", type=str)
    p.add_argument("--freeze-encoder", action="store_true")
    p.add_argument("--freeze-decoder", action="store_true")
    p.add_argument("--soft-vq-path", action="store_true")

    p.add_argument("--batch-size", default=32, type=int)
    p.add_argument("--epochs", default=15, type=int)
    p.add_argument("--lr", default=4e-5, type=float)
    p.add_argument("--weight-decay", default=0.01, type=float)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--max-grad-norm", default=1.0, type=float)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--detect-anomaly", action="store_true")

    p.add_argument("--lambda-vec", default=1.0, type=float)
    p.add_argument("--lambda-rate", default=0.0, type=float)
    p.add_argument("--lambda-M", default=0.05, type=float)
    p.add_argument("--lambda-mod", default=0.0, type=float)
    p.add_argument("--lambda-lat", default=0.0, type=float)
    p.add_argument("--lambda-ch", default=0.0, type=float)
    p.add_argument("--lambda-entropy", default=0.0, type=float)
    p.add_argument("--lambda-consistency", default=0.0, type=float)

    p.add_argument("--vq-warmup-epochs", default=5, type=float)
    p.add_argument("--alignment-start-epoch", default=6, type=int)
    p.add_argument("--temp-init", default=1.0, type=float)
    p.add_argument("--temp-decay", default=0.97, type=float)
    p.add_argument("--temp-freeze-floor", default=0.2, type=float)
    p.add_argument("--gumbel-tau-soft", default=1.0, type=float)
    p.add_argument("--gumbel-tau-hard", default=0.5, type=float)
    p.add_argument("--routing-temp", default=1.0, type=float)
    p.add_argument("--soft-temp", default=0.7, type=float)

    p.add_argument("--data", default="sst2", choices=["sst2", "wikitext103", "imdb"], type=str)
    p.add_argument("--max-length", default=64, type=int)
    p.add_argument("--train-max-examples", default=None, type=int)
    p.add_argument("--val-max-examples", default=None, type=int)
    p.add_argument("--processed-dir", default="/home/necphy/ducjunior/processed_datasets", type=str)
    p.add_argument("--overwrite-processed", action="store_true")

    p.add_argument("--train-snr-min-db", default=0.0, type=float)
    p.add_argument("--train-snr-max-db", default=10.0, type=float)
    p.add_argument("--eval-snr-db", default=10.0, type=float)

    p.add_argument("--checkpoint-path", default="/home/necphy/ducjunior/RoBERTa_MoE/checkpoints/textrec_vq", type=str)
    p.add_argument("--rebuild-labeling-every", default=0, type=int)

    return p


def build_dataloaders(args, tokenizer):
    if args.data == "sst2":
        train_ds, val_ds = prepare_sst2_reconstruction(tokenizer, args.max_length)
    elif args.data == "wikitext103":
        train_ds, val_ds = prepare_wikitext103(
            tokenizer, args.max_length, processed_root=args.processed_dir, overwrite=args.overwrite_processed
        )
    elif args.data == "imdb":
        train_ds, val_ds = prepare_imdb_reconstruction(tokenizer, args.max_length)
    else:
        raise ValueError(f"Unsupported dataset: {args.data}")

    if args.train_max_examples is not None and len(train_ds) > args.train_max_examples:
        train_ds = train_ds.select(range(args.train_max_examples))
    if args.val_max_examples is not None and len(val_ds) > args.val_max_examples:
        val_ds = val_ds.select(range(args.val_max_examples))

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
    return train_loader, val_loader


def configure_model(args, device):
    vq_sizes = parse_int_tuple(args.vq_codebook_sizes)
    phy_list = parse_int_tuple(args.phy_M_list)

    model = MODJSCC_MoE_TextRec_VQ(
        model_name=args.model_name,
        N=args.N,
        vq_codebook_sizes=vq_sizes,
        phy_M_list=phy_list,
        freeze_encoder=args.freeze_encoder,
        freeze_decoder=args.freeze_decoder,
    ).to(device)

    if args.soft_vq_path:
        get_transceiver(model).soft_vq_path = True
        print("✓ Soft VQ path enabled (text reconstruction)")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    return model


def main_new():
    parser = build_arg_parser()
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(bool(args.detect_anomaly))
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader = build_dataloaders(args, tokenizer)

    model = configure_model(args, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_nll = float("inf")
    best_epoch = 0
    best_ckpt_path = os.path.join(args.checkpoint_path, "best_model.pt")
    last_ckpt_path = os.path.join(args.checkpoint_path, "last_model.pt")

    print("Starting training (text reconstruction)...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        vq_scale = min(1.0, epoch / float(args.vq_warmup_epochs))
        lambda_vq_cur = args.lambda_M * vq_scale

        alignment_phase = epoch >= args.alignment_start_epoch
        if not alignment_phase:
            vq_temp = max(args.temp_freeze_floor, args.temp_init * (args.temp_decay ** (epoch - 1)))
            symbol_temp = max(args.temp_freeze_floor, args.temp_init * (args.temp_decay ** (epoch - 1)))
            use_hard_forward = False
            gumbel_tau = args.gumbel_tau_soft
            phase_name = "LEARNING"
        else:
            vq_temp = args.temp_freeze_floor
            symbol_temp = args.temp_freeze_floor
            use_hard_forward = True
            gumbel_tau = args.gumbel_tau_hard
            phase_name = "ALIGNMENT"

        tx = get_transceiver(model)
        tx.vq_temp = vq_temp
        tx.symbol_temp = symbol_temp
        tx.routing_temp = args.routing_temp
        tx.soft_temp = args.soft_temp
        if hasattr(tx, "_debug_logged"):
            tx._debug_logged = False

        print(
            f"[Epoch {epoch}] Phase={phase_name} | vq_T={vq_temp:.3f} sym_T={symbol_temp:.3f} | hard_fwd={use_hard_forward}"
        )

        train_metrics = train_one_epoch_rec(
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
            gumbel_tau=gumbel_tau,
            max_grad_norm=args.max_grad_norm,
            use_amp=args.use_amp,
            snr_min_db=args.train_snr_min_db,
            snr_max_db=args.train_snr_max_db,
        )

        val_metrics = eval_one_epoch_rec(
            model=model,
            dataloader=val_loader,
            device=device,
            lambda_vec=args.lambda_vec,
            lambda_sym=args.lambda_rate,
            lambda_vq=lambda_vq_cur,
            lambda_lb=args.lambda_mod,
            eval_snr_db=args.eval_snr_db,
        )

        elapsed = time.time() - start_time

        if args.rebuild_labeling_every > 0 and epoch % args.rebuild_labeling_every == 0:
            tx_unwrapped = unwrap(get_transceiver(model))
            if hasattr(tx_unwrapped, "vq_moe"):
                for expert in tx_unwrapped.vq_moe.experts:
                    expert.rebuild_labeling(method="pca_gray", sync_ddp=True)

        entropy_str = ""
        if "vq_entropy" in train_metrics:
            entropy_str = (
                f" | VQ_ent: {train_metrics['vq_entropy']:.3f}/{val_metrics.get('vq_entropy', 0):.3f} "
                f"Sym_ent: {train_metrics['sym_entropy']:.3f}/{val_metrics.get('sym_entropy', 0):.3f}"
            )

        print(
            f"[Epoch {epoch}/{args.epochs}] {phase_name} | time={elapsed:.1f}s\n"
            f"  Train: loss={train_metrics['loss']:.4f} nll={train_metrics['nll_loss']:.4f} "
            f"ppl={train_metrics['ppl']:.2f} sym={train_metrics['sym_loss']:.2f} "
            f"lb={train_metrics['lb_loss']:.3f} vq={train_metrics['vq_loss']:.3f} "
            f"tokacc={train_metrics['token_accuracy']:.3f}\n"
            f"  Val:   loss={val_metrics['loss']:.4f} nll={val_metrics['nll_loss']:.4f} "
            f"ppl={val_metrics['ppl']:.2f} sym={val_metrics['sym_loss']:.2f} "
            f"lb={val_metrics['lb_loss']:.3f} vq={val_metrics['vq_loss']:.3f} "
            f"tokacc={val_metrics['token_accuracy']:.3f}\n"
            f"  Rate:  bits={val_metrics['avg_bits_per_block']:.1f} syms={val_metrics['avg_syms_per_block']:.1f}"
            f"{entropy_str}"
        )

        val_nll = val_metrics["nll_loss"]
        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_epoch = epoch
            model_to_save = unwrap(model)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_nll": best_val_nll,
                    "vq_temp": vq_temp,
                    "symbol_temp": symbol_temp,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
            print(
                f"  -> New best model saved (epoch={epoch}, val_nll={best_val_nll:.4f}, vq_T={vq_temp:.3f}, sym_T={symbol_temp:.3f})"
            )

        model_to_save = unwrap(model)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_nll": val_nll,
                "vq_temp": vq_temp,
                "symbol_temp": symbol_temp,
                "args": vars(args),
            },
            last_ckpt_path,
        )
        if epoch == args.epochs:
            print(
                f"  -> Last model saved (epoch={epoch}, val_nll={val_nll:.4f}, vq_T={vq_temp:.3f}, sym_T={symbol_temp:.3f})"
            )

    print("\n" + "=" * 70)
    print("Training finished!")
    print(f"  Best val NLL: {best_val_nll:.4f} (achieved at epoch {best_epoch})")
    print(f"  Best checkpoint: {best_ckpt_path}")
    print(f"  Last checkpoint: {last_ckpt_path}")
    print(f"  Final val NLL: {val_metrics['nll_loss']:.4f} (epoch {args.epochs})")
    print("=" * 70)


if __name__ == "__main__":
    main_new()
