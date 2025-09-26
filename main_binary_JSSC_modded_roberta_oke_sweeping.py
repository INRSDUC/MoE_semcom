
# import os

# import argparse
# import time
# import torch
# from datasets import load_dataset
# import torch.nn as nn
# import numpy as np
# from utils_oke import SNR_to_noise, val_step_with_smart_simple_JSCC,  train_step_modulated_budget, train_step_modulated_adv, train_step_router_budget_v3, train_step_router_budget_v4, train_step_faithful, train_step_router_budget_v5
# #type 1 transmit a bunch
# # from models_2.transceiver_JSCC_type_1 import JSCC_DeepSC

# #type 2 transmit only the CLS
# from models_2.transceiver_modulation_JSCC_type_2_oke import   MODJSCC_WithHyperprior_real_bit, MODJSCC_WithHyperprior_real_bit_MoE, MODJSCC_MoE_Faithful_2
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from transformers import AutoTokenizer
# import torch
# torch.autograd.set_detect_anomaly(True)

# # Initialize the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("roberta-base")
# parser = argparse.ArgumentParser()

# parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/RoBERTa_MoE/checkpoints/JSCC_MoE_faith_2', type=str)
# # parser.add_argument('--loadcheckpoint-path', default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_v12_sanity', type=str)
# parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
# parser.add_argument('--d-model', default=256, type=int)
# # parser.add_argument('--dff', default=512, type=int)
# parser.add_argument('--batch-size', default=128, type=int)
# parser.add_argument('--epochs', default=5, type=int)
# parser.add_argument('--alpha', default=1, type=float)
# parser.add_argument('--lambda_rate', default=.001, type=float)
# parser.add_argument('--lambda_M', default=.1, type=float)
# parser.add_argument('--lambda_mod', default=.05, type=float)


# args = parser.parse_args()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available())


# def preprocess_sst2(example):
#     return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=64)
# ds =   load_dataset("glue", "sst2")
# ds_encoded = ds.map(preprocess_sst2, batched=True)
# ds_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
# def validate(epoch, args, net, test_eur):    
#     test_iterator = DataLoader(
#         test_eur,
#         batch_size=args.batch_size,
#         num_workers=0,
#         pin_memory=True,
#         shuffle=True
#     )
#     net.eval()
#     pbar = tqdm(test_iterator)
#     total_loss = 0
#     total_samples = 0
#     all_acc = []
#     all_precisions = []
#     all_recalls = []
#     all_f1s = []
#     all_rate_loss = []
#     all_val_bit_y=[]
#     all_val_bit_total=[]

#     with torch.no_grad():
#         for batch in pbar:
#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             labels = batch['label'].to(device)
#             bs = input_ids.size(0)

#             noise_val = np.random.uniform(SNR_to_noise(1), SNR_to_noise(15))
#             n_var = torch.full((bs,),
#                        noise_val,
#                        device=device,
#                        dtype=torch.float)
#             loss, accuracy, precision, recall, f1, rate_loss, val_bit_y, val_bit_total = val_step_with_smart_simple_JSCC(
#                 net, labels, criterion, input_ids, attention_mask, channel=args.channel, n_var=n_var, lambda_rate=args.lambda_rate, lambda_M=args.lambda_M
#             )

#             total_loss = total_loss+ loss 
#             total_samples = total_samples+ labels.size(0)

#             all_acc.append(accuracy)
#             all_precisions.append(precision)
#             all_recalls.append(recall)
#             all_f1s.append(f1)
#             all_rate_loss.append(rate_loss)
#             all_val_bit_y.append(val_bit_y)
#             all_val_bit_total.append(val_bit_total)

#             pbar.set_description(f'Epoch: {epoch + 1}; Type: VAL; Loss: {loss:.5f}')

#     avg_loss = total_loss / len(test_iterator)
#     avg_accuracy = sum(all_acc)/len(all_acc)
#     avg_precision = sum(all_precisions) / len(all_precisions)
#     avg_recall = sum(all_recalls) / len(all_recalls)
#     avg_f1 = sum(all_f1s) / len(all_f1s)
#     avg_rate_loss = sum(all_rate_loss) / len(all_rate_loss)
#     avg_bit_y = sum(all_val_bit_y) / len(all_val_bit_y)
#     avg_bit_total = sum(all_val_bit_total) / len(all_val_bit_total)
#     return avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1,avg_rate_loss, avg_bit_y, avg_bit_total



# def train(epoch, args, train_dataset, net,criterion,  opt, mi_net=None):
#     train_iterator = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0,
#                                  pin_memory=True,   shuffle=False)
#     pbar = tqdm(train_iterator)

#     total_loss = 0
#     net.train()  
#     for batch in pbar:
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch['label'].to(device)

#         bs = input_ids.size(0)
#         labels = labels.to(device)

#         noise_val = np.random.uniform(SNR_to_noise(1), SNR_to_noise(15))
#         n_var = torch.full((bs,),
#                        noise_val,
#                        device=device,
#                        dtype=torch.float)
#         stats = train_step_router_budget_v5(net, input_ids, attention_mask, labels, opt, criterion, n_var=n_var,channel = args.channel, 
#                                              lambda_rate=0.01, lambda_bud=0.1e-2, beta_lb=1)
#         total_loss = total_loss +  stats['total']

#         pbar.set_description( 
#     f"Epoch: {epoch + 1}; Type: Train; Loss: {stats['total']:.5f}, "
#     f"Acc: {stats['acc']:.5f} ori_loss: {stats['cls']:.5f}, buget: {stats['budget']:.5f},"
#     f"rate_loss: {stats['rate']:.5f},"
#     f"mod_loss: {stats['E[bps]']:.3f}")
#     return total_loss/len(train_iterator)

# class WarmUpScheduler:
#     def __init__(self, optimizer, warmup_steps, total_steps): 
#         self.warmup_steps = warmup_steps
#         self.total_steps = total_steps
#         self.step_num = 0

#     def step(self):
#         self.step_num += 1
#         if self.step_num <= self.warmup_steps:
#             lr = self.step_num / self.warmup_steps * self.optimizer.param_groups[0]['initial_lr']
#         else:
#             lr = self.optimizer.param_groups[0]['initial_lr'] * (1 - (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] = lr



# if __name__ == '__main__':
#     args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     train_dataset = ds_encoded["train"]
#     test_eur = ds_encoded["validation"]
  
#     deepsc = MODJSCC_MoE_Faithful_2(args.d_model, freeze_bert=False).to(args.device)
#     # deepsc = MODJSCC_WithHyperprior_real_bit_MoE(args.d_model, freeze_bert=False).to(args.device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(deepsc.parameters(),
#                                  lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
#     total_steps = len(train_dataset) // args.batch_size * args.epochs
#     warmup_steps = 5   
#     scheduler = WarmUpScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
#     best_acc = -1.0
#     best_epoch = -1
#     best_ckpt_path = None
#     for epoch in range(args.epochs):
#         start = time.time()
#         train_loss = train(epoch, args,train_dataset=train_dataset, net=deepsc,criterion=criterion, opt=optimizer)
#         val_loss, val_acc,avg_precision,avg_recall, avg_f1,avg_rate, val_bit_y, val_bit_total = validate(epoch, args, deepsc, test_eur=test_eur)
#         if val_acc > best_acc:

#             best_acc = val_acc
#             best_epoch = epoch + 1
#             best_ckpt_path = os.path.join(args.checkpoint_path, "checkpoint_best.pth")
#             print(f"✅ New best model saved: {best_ckpt_path} (epoch {best_epoch}, val_acc={best_acc:.4f})")
#             torch.save(deepsc.state_dict(), best_ckpt_path)
#         print(f"Epoch {epoch + 1}/{args.epochs}, Time: {time.time() - start:.2f}s, "
#             f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val prec: {avg_precision:.4f}, Val y: {val_bit_y:.4f}, Val total: {val_bit_total:.4f}, Val rate: {avg_rate:.4f}")


#     print("Training complete!")
#     print(f"🏁 Best checkpoint: epoch {best_epoch} with val_acc={best_acc:.4f} at {best_ckpt_path}")

# main.py  (drop-in)
import os
import argparse
import time
import csv
import random
import torch
from datasets import load_dataset
import torch.nn as nn
import numpy as np
from utils_oke import (
    SNR_to_noise, val_step_with_smart_simple_JSCC,
    train_step_router_budget_v5
)
from models_2.transceiver_modulation_JSCC_type_2_oke import MODJSCC_MoE_Faithful_2
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

torch.autograd.set_detect_anomaly(True)

# ----------------------
# Args
# ----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/RoBERTa_MoE/checkpoints/JSCC_MoE_faith_2', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help='Choose AWGN, Rayleigh, or Rician')
parser.add_argument('--d-model', default=256, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--alpha', default=1.0, type=float)
parser.add_argument('--lambda_rate', default=0.001, type=float)
parser.add_argument('--lambda_M', default=0.1, type=float)
parser.add_argument('--lambda_mod', default=0.05, type=float)

# NEW: experiment control
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--freeze-bert', action='store_true')

# NEW: expert selection & per-mod training
parser.add_argument('--expert', type=int, default=None, help='Train only this expert index (post-order)')
parser.add_argument('--train-all', action='store_true', help='Train every expert individually (re-init per expert)')

# NEW: modulation order control (requires model support; safe fallbacks if absent)
parser.add_argument('--mod-order', type=str, choices=['asc','desc','shuffle','custom'], default='asc')
parser.add_argument('--custom-perm', type=str, default=None, help='Comma-separated perm, e.g. "2,3,4,5,1,0"')
parser.add_argument('--shuffle-seed', type=int, default=None)

args = parser.parse_args()

# ----------------------
# Utils
# ----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def preprocess_sst2(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=64)

# ----------------------
# Data
# ----------------------
ds = load_dataset("glue", "sst2")
ds_encoded = ds.map(preprocess_sst2, batched=True)
ds_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
train_dataset = ds_encoded["train"]
val_dataset = ds_encoded["validation"]

# ----------------------
# Schedulers
# ----------------------
class WarmUpScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.optimizer = optimizer  # FIXED: keep a handle
        self.warmup_steps = int(warmup_steps)
        self.total_steps = max(int(total_steps), 1)
        self.step_num = 0
        # capture initial lr(s)
        for pg in self.optimizer.param_groups:
            pg.setdefault('initial_lr', pg['lr'])

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            scale = self.step_num / max(1, self.warmup_steps)
        else:
            remain = max(1, self.total_steps - self.warmup_steps)
            scale = max(0.0, 1 - (self.step_num - self.warmup_steps) / remain)
        for pg in self.optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * scale

# ----------------------
# Helpers around model features (robust to missing attributes)
# ----------------------
def maybe_set_modulation_order(model, order: str, custom_perm, shuffle_seed):
    if hasattr(model, "set_modulation_order"):
        kw = {}
        if custom_perm is not None:
            try:
                perm = [int(x) for x in custom_perm.split(",")]
            except Exception:
                raise ValueError("--custom-perm must be like '2,3,4,5,1,0'")
            kw["custom_perm"] = perm
        if order == "shuffle":
            kw["seed_for_shuffle"] = shuffle_seed
        model.set_modulation_order(order=order, **kw)
        try:
            print("Active M order:", model.current_M_order())
        except Exception:
            pass
    else:
        # Fallback: just print and continue; order won't change inside model.
        if custom_perm:
            print("[WARN] Model lacks set_modulation_order(); ignoring --custom-perm.")
        if order != "asc":
            print(f"[WARN] Model lacks set_modulation_order(); ignoring --mod-order={order}.")

def maybe_force_expert(model, idx: int):
    if hasattr(model, "force_expert"):
        setattr(model, "force_expert", idx)
        print(f"[Info] force_expert set to {idx}")
    else:
        print("[WARN] Model has no 'force_expert'; training will rely on router.")

def num_experts_from(model) -> int:
    if hasattr(model, "K"):
        return int(model.K)
    if hasattr(model, "M_list"):
        try:
            return len(model.M_list)
        except Exception:
            pass
    # Last resort default; adjust if needed
    return 6

def build_loaders(batch_size: int):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0,
                              pin_memory=True, shuffle=True)  # SHUFFLE train
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0,
                            pin_memory=True, shuffle=False)
    return train_loader, val_loader

# ----------------------
# Core train / val
# ----------------------
@torch.no_grad()
def validate(epoch, args, net, val_loader, criterion):
    net.eval()
    pbar = tqdm(val_loader)
    total_loss = 0.0
    total_samples = 0
    all_acc, all_precisions, all_recalls, all_f1s = [], [], [], []
    all_rate_loss, all_val_bit_y, all_val_bit_total = [], [], []

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch['label'].to(device)
        bs = input_ids.size(0)

        noise_val = np.random.uniform(SNR_to_noise(1), SNR_to_noise(15))
        n_var = torch.full((bs,), noise_val, device=device, dtype=torch.float)

        loss, accuracy, precision, recall, f1, rate_loss, val_bit_y, val_bit_total = \
            val_step_with_smart_simple_JSCC(
                net, labels, criterion, input_ids, attention_mask,
                channel=args.channel, n_var=n_var,
                lambda_rate=args.lambda_rate, lambda_M=args.lambda_M
            )

        total_loss += loss
        total_samples += bs
        all_acc.append(accuracy); all_precisions.append(precision)
        all_recalls.append(recall); all_f1s.append(f1)
        all_rate_loss.append(rate_loss)
        all_val_bit_y.append(val_bit_y)
        all_val_bit_total.append(val_bit_total)
        pbar.set_description(f'Epoch: {epoch + 1}; Type: VAL; Loss: {loss:.5f}')

    avg = lambda xs: sum(xs) / max(1, len(xs))
    avg_loss = total_loss / max(1, len(val_loader))
    return (avg_loss, avg(all_acc), avg(all_precisions), avg(all_recalls),
            avg(all_f1s), avg(all_rate_loss), avg(all_val_bit_y), avg(all_val_bit_total))

def train_one_epoch(epoch, args, train_loader, net, criterion, opt, scheduler=None):
    net.train()
    pbar = tqdm(train_loader)
    total_loss = 0.0

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch['label'].to(device)
        bs = input_ids.size(0)

        noise_val = np.random.uniform(SNR_to_noise(1), SNR_to_noise(15))
        n_var = torch.full((bs,), noise_val, device=device, dtype=torch.float)

        stats = train_step_router_budget_v5(
            net, input_ids, attention_mask, labels, opt, criterion,
            n_var=n_var, channel=args.channel,
            lambda_rate=args.lambda_rate, lambda_bud=0.1e-2, beta_lb=1
        )

        total_loss += stats['total']
        pbar.set_description(
            f"Epoch: {epoch + 1}; Type: Train; Loss: {stats['total']:.5f}, "
            f"Acc: {stats['acc']:.5f} ori_loss: {stats['cls']:.5f}, budget: {stats['budget']:.5f}, "
            f"rate_loss: {stats['rate']:.5f}, mod_loss: {stats['E[bps]']:.3f}"
        )

        if scheduler is not None:
            scheduler.step()

    return total_loss / max(1, len(train_loader))  # FIXED: return after loop

def init_model_and_opt(args, expert_idx=None):
    model = MODJSCC_MoE_Faithful_2(args.d_model, freeze_bert=args.freeze_bert).to(device)
    maybe_set_modulation_order(model, args.mod_order, args.custom_perm, args.shuffle_seed)
    if expert_idx is not None:
        maybe_force_expert(model, expert_idx)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

    steps_per_epoch = max(1, len(train_dataset) // args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(5 * steps_per_epoch, total_steps // 10) if total_steps > 0 else 0
    scheduler = WarmUpScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    return model, criterion, optimizer, scheduler

def train_single_run(args, expert_idx=None, out_dir=None):
    set_seed(args.seed)
    train_loader, val_loader = build_loaders(args.batch_size)

    model, criterion, optimizer, scheduler = init_model_and_opt(args, expert_idx)
    K = num_experts_from(model)

    os.makedirs(out_dir, exist_ok=True)
    best_acc, best_epoch = -1.0, -1
    best_ckpt_path = os.path.join(out_dir, "checkpoint_best.pth")

    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train_one_epoch(epoch, args, train_loader, model, criterion, optimizer, scheduler)
        val_loss, val_acc, avg_precision, avg_recall, avg_f1, avg_rate, val_bit_y, val_bit_total = \
            validate(epoch, args, model, val_loader, criterion)

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"✅ New best model saved: {best_ckpt_path} (epoch {best_epoch}, val_acc={best_acc:.4f})")

        print(
            f"Epoch {epoch + 1}/{args.epochs}, Time: {time.time() - start:.2f}s, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Val prec: {avg_precision:.4f}, Val y: {val_bit_y:.4f}, Val total: {val_bit_total:.4f}, Val rate: {avg_rate:.4f}"
        )

    print("Training complete!")
    print(f"🏁 Best checkpoint: epoch {best_epoch} with val_acc={best_acc:.4f} at {best_ckpt_path}")
    return {"K": K, "best_epoch": best_epoch, "best_acc": best_acc, "ckpt": best_ckpt_path}

# ----------------------
# Entry
# ----------------------
if __name__ == '__main__':
    # Prepare root checkpoint dir
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # If only one expert requested
    if args.train_all is False:
        exp_dir = os.path.join(args.checkpoint_path, "single_expert" if args.expert is None else f"expert_{args.expert}")
        result = train_single_run(args, expert_idx=args.expert, out_dir=exp_dir)

    else:
        # Train each expert individually
        # Use a temp model to discover K (robust)
        temp_model = MODJSCC_MoE_Faithful_2(args.d_model, freeze_bert=args.freeze_bert).to(device)
        K = num_experts_from(temp_model)
        del temp_model

        # CSV summary
        csv_path = os.path.join(args.checkpoint_path, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["expert_idx", "epoch_best", "val_acc_best", "ckpt"])

        for i in range(K):
            print(f"\n=== Training expert {i}/{K-1} ===")
            exp_dir = os.path.join(args.checkpoint_path, f"expert_{i}")
            result = train_single_run(args, expert_idx=i, out_dir=exp_dir)
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([i, result["best_epoch"], f"{result['best_acc']:.6f}", result["ckpt"]])

        print(f"\nAll experts trained. Summary saved to: {csv_path}")
