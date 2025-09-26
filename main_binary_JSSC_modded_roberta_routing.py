
import os
import math
import argparse
import time
import torch
from datasets import load_dataset
import torch.nn as nn
import numpy as np
from utils_oke import SNR_to_noise, val_step_with_smart_simple_JSCC,  train_step_modulated_budget, train_step_modulated_adv, train_step_router_budget_v3, train_step_router_budget_v4, train_step_faithful, train_step_router_budget_v5, train_step_router_budget_v6,val_step_JSCC_router
#type 1 transmit a bunch
# from models_2.transceiver_JSCC_type_1 import JSCC_DeepSC

#type 2 transmit only the CLS
from models_2.transceiver_modulation_JSCC_type_2_oke import   MODJSCC_MoE_Faithful,MODJSCC_MoE_Faithful_2
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
torch.autograd.set_detect_anomaly(True)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint-path', default='/home/necphy/ducjunior/RoBERTa_MoE/checkpoints/JSCC_MoE_faith_1_64QAM', type=str)
# parser.add_argument('--loadcheckpoint-path', default='/home/necphy/ducjunior/BERT_Backdoor/checkpoints/deepsc_v12_sanity', type=str)
parser.add_argument('--channel', default='AWGN', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--d-model', default=256, type=int)
# parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--alpha', default=1, type=float)
parser.add_argument('--lambda_rate', default=.001, type=float)
parser.add_argument('--lambda_M', default=.1, type=float)
parser.add_argument('--lambda_mod', default=.05, type=float)


args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
def sample_snr_seq_ar1(batch_size, T, snr_db_low=1.0, snr_db_high=15.0,
                       coherence_frames=3.0, sigma_db=1.5, device=None):
    """
    Returns:
      snr_seq_lin: [B, T] linear SNR (γ) per mini-frame, AR(1) in dB
    AR(1) in dB: x_t = μ + ρ (x_{t-1} - μ) + σ * sqrt(1-ρ^2) * ε_t
    ρ = exp(-1 / coherence_frames).
    """
    device = device or torch.device("cpu")
    B, Tm = int(batch_size), int(T)
    mu = 0.5 * (snr_db_low + snr_db_high)
    rho = math.exp(-1.0 / max(coherence_frames, 1e-6))
    x = torch.empty(B, Tm, device=device)

    # init from uniform range
    x[:, 0] = torch.empty(B, device=device).uniform_(snr_db_low, snr_db_high)

    if Tm > 1:
        for t in range(1, Tm):
            eps = torch.randn(B, device=device)
            x[:, t] = mu + rho * (x[:, t-1] - mu) + sigma_db * math.sqrt(max(1e-8, 1.0 - rho**2)) * eps
            x[:, t] = x[:, t].clamp_(snr_db_low, snr_db_high)

    # convert dB -> linear SNR γ
    snr_seq_lin = (10.0 ** (x / 10.0))
    return snr_seq_lin

def preprocess_sst2(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=64)
ds =   load_dataset("glue", "sst2")
ds_encoded = ds.map(preprocess_sst2, batched=True)
ds_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
def validate(epoch, args, net, test_eur):    
    test_iterator = DataLoader(
        test_eur,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
        shuffle=True
    )
    net.eval()
    pbar = tqdm(test_iterator)
    total_loss = 0
    total_samples = 0
    all_acc = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_rate_loss = []
    all_val_bit_y=[]
    all_val_bit_total=[]

    with torch.no_grad():
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch['label'].to(device)
            bs = input_ids.size(0)

            noise_val = np.random.uniform(SNR_to_noise(1), SNR_to_noise(15))
            n_var = torch.full((bs,),
                       noise_val,
                       device=device,
                       dtype=torch.float)
            loss, accuracy, precision, recall, f1, rate_loss, val_bit_y, val_bit_total = val_step_JSCC_router(
                net, labels, criterion, input_ids, attention_mask, channel=args.channel, n_var=n_var, lambda_rate=args.lambda_rate, lambda_M=args.lambda_M
            )

            total_loss = total_loss+ loss 
            total_samples = total_samples+ labels.size(0)

            all_acc.append(accuracy)
            all_precisions.append(precision)
            all_recalls.append(recall)
            all_f1s.append(f1)
            all_rate_loss.append(rate_loss)
            all_val_bit_y.append(val_bit_y)
            all_val_bit_total.append(val_bit_total)

            pbar.set_description(f'Epoch: {epoch + 1}; Type: VAL; Loss: {loss:.5f}')

    avg_loss = total_loss / len(test_iterator)
    avg_accuracy = sum(all_acc)/len(all_acc)
    avg_precision = sum(all_precisions) / len(all_precisions)
    avg_recall = sum(all_recalls) / len(all_recalls)
    avg_f1 = sum(all_f1s) / len(all_f1s)
    avg_rate_loss = sum(all_rate_loss) / len(all_rate_loss)
    avg_bit_y = sum(all_val_bit_y) / len(all_val_bit_y)
    avg_bit_total = sum(all_val_bit_total) / len(all_val_bit_total)
    return avg_loss, avg_accuracy, avg_precision, avg_recall, avg_f1,avg_rate_loss, avg_bit_y, avg_bit_total



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
#         # loss, ori_loss, rate_loss, mod_loss,smooth_loss, acc = train_step_modulated_adv(net, input_ids, attention_mask, labels, opt, criterion, n_var=n_var,lambda_rate=args.lambda_rate,lambda_mod=args.lambda_mod,   epsilon=1e-5, alpha=args.alpha, channel=args.channel )

#         # stats = train_step_router_budget_v3(net, input_ids, attention_mask, labels, opt, criterion, n_var=n_var,channel = args.channel, 
#         #                                     lambda_rate=0.0006, lambda_bud=0.3e-6, epsilon=1e-5, beta_lb=1, alpha = 0)
#         # stats = train_step_router_budget_v4(net, input_ids, attention_mask, labels, opt, criterion, n_var=n_var,channel = args.channel, 
#         #                                     lambda_rate=0.0006, lambda_bud=0.8e-6, beta_lb=1)
#         stats = train_step_router_budget_v6(net, input_ids, attention_mask, labels, opt, criterion, n_var=n_var,channel = args.channel, 
#                                              lambda_rate=0.0005, lambda_bud=0.1e-3, beta_lb=1)
#         # stats = train_step_modulated_budget(net, input_ids, attention_mask, labels,
#         total_loss = total_loss +  stats['total']
#         # pbar.set_description(f'Epoch: {epoch + 1}; Type: Train; Loss: {stats['total']:.5f}, Acc: {stats['acc']:.5f} ori_loss: {stats['cls']:.5f}, smooth_loss: {stats['smooth']:.5f}, rate_loss: {stats['rate']:.5f}, mod_loss: {stats['E[bps]']:.5f}')
# #         pbar.set_description( 
# #     f"Epoch: {epoch + 1}; Type: Train; Loss: {stats['total']:.5f}, "
# #     f"Acc: {stats['acc']:.5f} ori_loss: {stats['cls']:.5f}, buget: {stats['budget']:.5f},"
# #     f"smooth_loss: {stats['smooth']:.5f}, rate_loss: {stats['rate']:.5f}, load_balancing: {stats['lb']:.5f},"
# #     f"mod_loss: {stats['E[bps]']:.5f}"
# # )
#         pbar.set_description( 
#     f"Epoch: {epoch + 1}; Type: Train; Loss: {stats['total']:.5f}, "
#     f"Acc: {stats['acc']:.5f} ori_loss: {stats['cls']:.5f}, buget: {stats['budget']:.5f},"
#     f"rate_loss: {stats['rate']:.5f},"
#     f"mod_loss: {stats['E[bps]']:.3f}")
#     return total_loss/len(train_iterator)
def train(epoch, args, train_dataset, net, criterion, opt, mi_net=None):
    from torch.utils.data import DataLoader
    import numpy as np
    pbar = tqdm(DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1,
                           pin_memory=True, shuffle=False))
    total_loss = 0.0
    net.train()

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        bs = input_ids.size(0)

        # --- NEW: per-frame SNR sequence for the router ---
        T = getattr(net, "T_max", 6)
        snr_seq = sample_snr_seq_ar1(
            batch_size=bs, T=T, snr_db_low=1.0, snr_db_high=15.0,
            coherence_frames=3.0, sigma_db=1.5, device=device
        )  # [B,T] linear γ

        # Use mean γ to set a single AWGN variance per burst
        n_var = (1.0 / snr_seq.mean(dim=1)).to(torch.float32)  # [B]

        stats = train_step_router_budget_v6(
            net, input_ids, attention_mask, labels, opt, criterion,
            n_var=n_var, channel=args.channel,
            lambda_rate=0.001, lambda_bud=0.1e-2, beta_lb=1,
            # pass the sequence ↓↓↓
            snr_seq=snr_seq
        )

        total_loss += stats["total"]
        pbar.set_description(
            f"Epoch: {epoch + 1}; Type: Train; Loss: {stats['total']:.5f}, "
            f"Acc: {stats['acc']:.5f} ori_loss: {stats['cls']:.5f}, buget: {stats['budget']:.5f}, "
            f"rate_loss: {stats['rate']:.5f}, mod_loss: {stats['B_sum_mean']:.3f}/{stats['R_tot_mean']:.3f}"
        )

    return total_loss / len(pbar)


class WarmUpScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps): 
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.step_num / self.warmup_steps * self.optimizer.param_groups[0]['initial_lr']
        else:
            lr = self.optimizer.param_groups[0]['initial_lr'] * (1 - (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr



if __name__ == '__main__':
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset = ds_encoded["train"]
    test_eur = ds_encoded["validation"]
  
    deepsc = MODJSCC_MoE_Faithful_2(args.d_model, freeze_bert=False).to(args.device)
    # deepsc = MODJSCC_WithHyperprior_real_bit_MoE(args.d_model, freeze_bert=False).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(deepsc.parameters(),
                                 lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    total_steps = len(train_dataset) // args.batch_size * args.epochs
    warmup_steps = 5   
    scheduler = WarmUpScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    best_acc = -1.0
    best_epoch = -1
    best_ckpt_path = None
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = train(epoch, args,train_dataset=train_dataset, net=deepsc,criterion=criterion, opt=optimizer)
        val_loss, val_acc,avg_precision,avg_recall, avg_f1,avg_rate, val_bit_y, val_bit_total = validate(epoch, args, deepsc, test_eur=test_eur)
        if val_acc > best_acc:

            best_acc = val_acc
            best_epoch = epoch + 1
            best_ckpt_path = os.path.join(args.checkpoint_path, "checkpoint_best.pth")
            # if not os.path.exists(args.checkpoint_path):
            #     os.makedirs(args.checkpoint_path)
            print(f"✅ New best model saved: {best_ckpt_path} (epoch {best_epoch}, val_acc={best_acc:.4f})")
            # checkpoint_file = os.path.join(args.checkpoint_path, f'checkpoint_full{epoch + 1:02d}.pth')
            torch.save(deepsc.state_dict(), best_ckpt_path)
        print(f"Epoch {epoch + 1}/{args.epochs}, Time: {time.time() - start:.2f}s, "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val prec: {avg_precision:.4f}, Val y: {val_bit_y:.4f}, Val total: {val_bit_total:.4f}, Val rate: {avg_rate:.4f}")


    print("Training complete!")
    print(f"🏁 Best checkpoint: epoch {best_epoch} with val_acc={best_acc:.4f} at {best_ckpt_path}")

