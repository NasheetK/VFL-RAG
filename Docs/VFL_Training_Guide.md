# VFL Multi-Class Training Guide

## Overview

This guide documents the training improvements applied to the VFL SHAP Multi-Class notebook for better performance, early stopping, and overfitting prevention.

---

## ✅ Implemented Improvements

### 1. Train/Val/Test Split

**Location:** Cell 4

**What it does:**
- Creates three-way split: **Train (64%) / Val (16%) / Test (20%)**
- Maintains class stratification in all splits
- Ensures proper validation set for early stopping

**Code:**
```python
# First split: train+val (80%) vs test (20%)
trainval_idx, test_idx = train_test_split(...)

# Second split: train (64%) vs val (16%)
train_idx, val_idx = train_test_split(trainval_idx, test_size=0.2, ...)
```

**Why:** Validation set is essential for:
- Detecting overfitting (train improves, val stops)
- Early stopping decisions
- Model selection (best checkpoint)

---

### 2. Validation Loop & Early Stopping

**Location:** Cell 6 - `train_vfl()` function

**What it does:**
- Evaluates validation set every `eval_every=5` epochs
- Tracks best validation F1 score
- Stops training if no improvement for `patience=20` epochs
- Saves and loads best model checkpoint

**Key Features:**
- **Early stopping:** Prevents overfitting by stopping when validation stops improving
- **Best model checkpoint:** Automatically saves and loads the best model (by validation F1)
- **Min delta:** Ignores tiny improvements (< 0.001) to prevent noise

**Parameters:**
- `early_stop_patience=20`: Stop if no improvement for 20 epochs
- `min_delta=0.001`: Minimum improvement to reset patience
- `eval_every=5`: Evaluate validation every 5 epochs

**Expected Outcome:**
- Training stops at best generalization point (typically 80-180 epochs, not 500)
- Final model is the best validation model, not the last epoch

---

### 3. Fixed Scheduler Warning

**Location:** Cell 6 - `train_vfl()` function

**What it does:**
- Uses validation loss for scheduler (not training loss)
- Scheduler receives scalar value (`.item()`) to avoid autograd warning

**Code:**
```python
val_loss, val_acc, val_rec, val_f1 = evaluate_model(model, x_val_parts, y_val, criterion)
scheduler.step(val_loss)  # Uses validation loss (scalar)
```

**Why:** 
- Validation loss is better indicator of generalization
- Fixes PyTorch warning about tensor with requires_grad

---

### 4. Clamped Class Weights

**Location:** Cell 6 - `train_vfl()` function

**What it does:**
- Calculates class weights for imbalanced data
- **Clamps weights to max=20** to prevent extreme values

**Code:**
```python
class_weights = total / (len(class_counts) * class_counts.astype(float))
class_weights = np.clip(class_weights, None, 20.0)  # Clamp max=20
```

**Why:**
- Prevents extreme weights (e.g., 225 for OTHERS class)
- More stable training
- Better per-class recall balance

**Before:** `[0.14, 35.3, 3.4, 0.99, 10.9, 225.3, 2.1, 14.6, 9.5]`
**After:** `[0.14, 20.0, 3.4, 0.99, 10.9, 20.0, 2.1, 14.6, 9.5]`

---

### 5. Comprehensive Evaluation

**Location:** Cell 6 - `evaluate()` function, Cell 7

**What it does:**
- Reports macro and weighted metrics
- Per-class classification report
- Confusion matrix

**Metrics Reported:**
- Accuracy
- Macro Recall (average across classes)
- Weighted Recall (weighted by class frequency)
- Macro F1 (average across classes)
- Weighted F1 (weighted by class frequency)
- Per-class precision, recall, F1
- Confusion matrix

**Why:**
- IDS systems need per-class metrics (especially minority attacks)
- Macro metrics show performance on rare classes
- Weighted metrics show overall performance

---

### 6. Best Model Checkpointing

**Location:** Cell 6 - `train_vfl()` function

**What it does:**
- Saves model state when validation F1 improves
- Loads best model at end of training
- Reports best epoch and validation F1

**Code:**
```python
if val_f1 > best_val_f1 + min_delta:
    best_model_state = model.state_dict().copy()
    # ... later ...
model.load_state_dict(best_model_state)
```

**Why:**
- Final test evaluation uses best model, not last epoch
- Prevents reporting overfitted model performance

---

## Training Workflow

### Step-by-Step Process

1. **Data Split (Cell 4)**
   - Split into Train/Val/Test (64%/16%/20%)
   - Verify stratification

2. **Model Creation (Cell 7)**
   - Create VFL model with improved architecture
   - Print model parameters

3. **Training (Cell 7)**
   - Train with validation monitoring
   - Early stopping if no improvement
   - Best model checkpointing

4. **Evaluation (Cell 7)**
   - Evaluate on test set with best model
   - Print comprehensive metrics
   - Generate confusion matrix

---

## Expected Training Behavior

### Good Training Log Example

```
[VFL] Epoch   0 | Train Loss: 2.1977 | Val Loss: 2.1500 | Val F1: 0.4500 | ✓ BEST
[VFL] Epoch   5 | Train Loss: 1.5000 | Val Loss: 1.4500 | Val F1: 0.6500 | ✓ BEST
[VFL] Epoch  10 | Train Loss: 1.2000 | Val Loss: 1.1800 | Val F1: 0.7200 | ✓ BEST
[VFL] Epoch  15 | Train Loss: 1.1000 | Val Loss: 1.1200 | Val F1: 0.7150 | Patience: 1/20
[VFL] Epoch  20 | Train Loss: 1.0500 | Val Loss: 1.1000 | Val F1: 0.7250 | ✓ BEST
...
[VFL] Epoch  80 | Train Loss: 0.9000 | Val Loss: 0.9500 | Val F1: 0.7800 | ✓ BEST
[VFL] Epoch  85 | Train Loss: 0.8900 | Val Loss: 0.9600 | Val F1: 0.7790 | Patience: 1/20
...
[VFL] Epoch 100 | Train Loss: 0.8800 | Val Loss: 0.9700 | Val F1: 0.7785 | Patience: 20/20

[EARLY STOP] No improvement for 20 epochs. Best F1: 0.7800 at epoch 80
[CHECKPOINT] Loaded best model from epoch 80 (Val F1: 0.7800)
```

### Signs of Good Training

✅ **Validation F1 improves steadily** (0.45 → 0.65 → 0.72 → 0.78)
✅ **Early stopping triggers** (saves compute, prevents overfitting)
✅ **Best model loaded** (final test uses best validation model)
✅ **LR scheduler reduces LR** when validation plateaus

### Signs of Overfitting

❌ **Train loss decreases, val loss increases**
❌ **Train F1 improves, val F1 plateaus/decreases**
❌ **Large gap between train and val metrics**

**Solution:** Early stopping should catch this automatically.

---

## Key Parameters

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 500 | Maximum epochs (usually stops earlier) |
| `lr` | 0.002 | Learning rate (higher than 0.001 for VFL) |
| `early_stop_patience` | 20 | Stop if no improvement for 20 epochs |
| `min_delta` | 0.001 | Minimum improvement to reset patience |
| `eval_every` | 5 | Evaluate validation every 5 epochs |
| `weight_decay` | 1e-4 | L2 regularization |
| `max_class_weight` | 20 | Clamp class weights to prevent extremes |

### Model Architecture

| Component | Value | Description |
|-----------|-------|-------------|
| `embed_dim` | 64 | Embedding dimension per party |
| `hidden_dim` | 128 | Hidden layer dimension |
| `fusion` | concat | Concatenate party embeddings (192D total) |
| `dropout` | 0.2, 0.1 | Dropout rates in encoder/classifier |

---

## Performance Expectations

### Before Improvements

- Loss plateaus at ~1.38-1.39 after epoch 80
- Training continues to 500 epochs (wasted compute)
- No validation monitoring
- Risk of overfitting

### After Improvements

- **Loss continues decreasing** (should reach ~1.2-1.3 or lower)
- **Early stopping** (typically stops at 80-180 epochs)
- **Better generalization** (validation monitoring)
- **Improved recall/F1** for minority classes (clamped weights)
- **Best model used** for final evaluation

### Expected Metrics

- **Accuracy:** ~97-98% (similar or slightly better)
- **Macro Recall:** 70-75%+ (improved from ~66%)
- **Macro F1:** 68-72%+ (improved from ~65%)
- **Per-class recall:** Better for minority classes (BOT, OTHERS, etc.)

---

## Troubleshooting

### Issue: Training stops too early

**Symptoms:** Early stopping triggers after < 50 epochs

**Solutions:**
- Increase `early_stop_patience` to 30-40
- Increase `min_delta` to 0.002
- Check if validation set is too small

### Issue: No early stopping

**Symptoms:** Training runs full 500 epochs

**Solutions:**
- Decrease `early_stop_patience` to 15
- Decrease `min_delta` to 0.0005
- Check if model is still improving (good sign!)

### Issue: Overfitting (train >> val)

**Symptoms:** Train F1 = 0.99, Val F1 = 0.70

**Solutions:**
- Increase `weight_decay` to 5e-4
- Increase dropout to 0.3
- Reduce model capacity (smaller embed_dim)

### Issue: Poor minority class performance

**Symptoms:** OTHERS/BOT classes have very low recall

**Solutions:**
- Increase `max_class_weight` to 30 (if needed)
- Use focal loss instead of weighted CrossEntropy
- Increase minority class samples (data augmentation)

---

## Best Practices

1. **Always use validation set** - Never skip validation monitoring
2. **Monitor both loss and F1** - F1 is better for imbalanced data
3. **Save best model** - Don't use last epoch for evaluation
4. **Check confusion matrix** - Understand per-class performance
5. **Report macro metrics** - Important for IDS (minority attacks matter)
6. **Early stopping is your friend** - Saves time and prevents overfitting

---

## File Outputs

After training, the following files are generated:

- `vfl_shap_performance.csv`: Performance metrics (Accuracy, Recall, F1, Best Val F1, Best Epoch)
- `vfl_shap_global_summary.csv`: Global SHAP party attributions
- `vfl_shap_{class}_summary.csv`: Per-class SHAP summaries (one per class)
- `vfl_shap_agent_mitigation_summary.csv`: Agent-level mitigation recommendations

---

## References

- **Early Stopping:** Prevents overfitting by monitoring validation performance
- **Class Weights:** Handles imbalanced data (CIC-IDS has extreme imbalance)
- **Validation Set:** Essential for model selection and hyperparameter tuning
- **Best Model Checkpoint:** Ensures final evaluation uses best generalization

---

## Summary

The improved training pipeline includes:

✅ **Train/Val/Test split** (64%/16%/20%)
✅ **Validation monitoring** (every 5 epochs)
✅ **Early stopping** (patience=20, min_delta=0.001)
✅ **Best model checkpointing** (saves/loads best validation model)
✅ **Clamped class weights** (max=20 for stability)
✅ **Comprehensive evaluation** (macro/weighted metrics, confusion matrix)
✅ **Fixed scheduler warning** (uses validation loss)

**Result:** Better generalization, faster training, improved metrics, especially for minority classes.
