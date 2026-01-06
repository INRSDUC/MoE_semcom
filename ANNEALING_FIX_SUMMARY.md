# Temperature Annealing Fix Summary

## Problem Diagnosis

Your 30-epoch classification run showed:
- **Best validation accuracy: 90.71%** (saved early in training)
- **Final validation accuracy: 52.06%** (epoch 30)
- **Temperature at epoch 30**: vq_T=0.413, sym_T=0.413

### What Went Wrong

1. **Best checkpoint was early**: The 90.71% was achieved around epoch 5-10 when temperatures were still high (~0.6-0.8), allowing the soft path to work well.

2. **Hard path never learned**: As temperatures dropped to 0.41 by epoch 30:
   - VQ assignments hardened (good)
   - Symbol selection hardened (good)
   - BUT the hard path couldn't carry the classification task
   - Performance collapsed to 52%

3. **Training/validation mismatch**: 
   - Training used soft path (differentiable, smooth gradients)
   - Validation used hard path (discrete, no gradient signal to improve it)
   - Hard path never got strong enough to work at low temperatures

4. **Loss behavior issues**:
   - Classification loss went UP instead of down (hard path features degraded)
   - VQ loss and symbol metrics looked "weird" (conflicting objectives)

## Fixes Applied

### 1. Save Both Best AND Last Checkpoints
```python
best_ckpt_path = os.path.join(args.checkpoint_path, "best_model.pt")
last_ckpt_path = os.path.join(args.checkpoint_path, "last_model.pt")
```
- Now you can compare the early high-temperature model (best) vs final low-temperature model (last)
- Track which epoch the best was achieved
- Save temperature values in checkpoints for debugging

### 2. Raise Temperature Floor from 0.1 → 0.5
```python
# OLD: Too aggressive, hard path collapses
vq_temp = max(0.1, 1.0 * (0.97 ** (epoch - 1)))

# NEW: Higher floor prevents collapse
vq_temp = max(0.5, 1.0 * (0.97 ** (epoch - 1)))
```

**Why this helps**:
- At 0.5, distributions are still soft enough for gradients
- Hard path gets more training signal before full discretization
- Performance should stay more stable through epoch 30

### 3. Better Logging
- Print epoch number when best model is saved
- Show temperature values at checkpoint save time
- Clear final summary showing both best and last results

Example output:
```
  -> New best model saved (epoch=8, val_acc=0.9071, vq_T=0.620, sym_T=0.620)
  ...
  -> Last model saved (epoch=30, val_acc=0.8500, vq_T=0.500, sym_T=0.500)

======================================================================
Training finished!
  Best val accuracy: 0.9071 (achieved at epoch 8)
  Best checkpoint: .../best_model.pt
  Last checkpoint: .../last_model.pt
  Final val accuracy: 0.8500 (epoch 30)
======================================================================
```

## Expected Improvements

With temperature floor at 0.5:
- **Best accuracy**: Should still reach ~90% (early epochs unchanged)
- **Final accuracy**: Should stay >85% instead of dropping to 52%
- **Smaller gap**: Best vs last should be <5% instead of >40%
- **Stable losses**: Classification loss should keep decreasing, not go up

## How to Use

### Load Best Checkpoint (Early High-Temp Model)
```python
ckpt = torch.load("checkpoints/JSCC_MoE_Cls_1/best_model.pt")
print(f"Best model from epoch {ckpt['epoch']}")
print(f"  Val acc: {ckpt['val_accuracy']:.4f}")
print(f"  Temps: vq={ckpt['vq_temp']:.3f}, sym={ckpt['symbol_temp']:.3f}")
model.load_state_dict(ckpt['model_state_dict'])
```

### Load Last Checkpoint (Final Low-Temp Model)
```python
ckpt = torch.load("checkpoints/JSCC_MoE_Cls_1/last_model.pt")
print(f"Last model from epoch {ckpt['epoch']}")
print(f"  Val acc: {ckpt['val_accuracy']:.4f}")
print(f"  Temps: vq={ckpt['vq_temp']:.3f}, sym={ckpt['symbol_temp']:.3f}")
model.load_state_dict(ckpt['model_state_dict'])
```

## Further Improvements (If Needed)

If performance still drops too much:

1. **Even higher floor (0.7)**:
   ```python
   vq_temp = max(0.7, 1.0 * (0.97 ** (epoch - 1)))
   ```

2. **Stop annealing early**:
   ```python
   # Stop annealing after epoch 15
   if epoch <= 15:
       vq_temp = max(0.5, 1.0 * (0.97 ** (epoch - 1)))
   else:
       vq_temp = 0.5  # freeze at floor
   ```

3. **Enable soft path during eval** (keep gradient flow):
   ```python
   # In eval_one_epoch_cls
   model.eval()  # but keep soft_vq_path=True
   ```

4. **Auxiliary loss** to align hard and soft paths:
   ```python
   # Force hard VQ outputs toward soft VQ outputs
   kl_loss = F.kl_div(hard_logits.log_softmax(-1), 
                      soft_logits.softmax(-1), 
                      reduction='batchmean')
   loss = cls_loss + lambda_kl * kl_loss + ...
   ```

## Next Steps

1. **Rerun training** with new settings (floor=0.5)
2. **Monitor logs** to see if final accuracy stays high
3. **Compare checkpoints**:
   - Load best → test on eval set
   - Load last → test on eval set
   - Gap should be <5% now
4. **If text reconstruction** shows same issue, same fixes already applied there too!
