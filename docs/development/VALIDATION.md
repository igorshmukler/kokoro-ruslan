# Validation Loop and Overfitting Monitoring

## Overview

The training pipeline includes a comprehensive validation loop to monitor model performance on unseen data and detect overfitting. This helps ensure your model generalizes well beyond the training set.

## Features

- **Automatic Train/Validation Split**: Configurable percentage of data reserved for validation
- **Per-Epoch Validation**: Validates model performance after each training epoch (configurable)
- **Early Stopping**: Automatically stops training when validation performance plateaus
- **Overfitting Detection**: Monitors validation vs training loss to detect overfitting
- **Best Model Saving**: Automatically saves checkpoints when validation improves

## Configuration

### Basic Configuration

In `config.py`, the following parameters control validation:

```python
validation_split: float = 0.1  # 10% of data for validation
validation_interval: int = 1  # Run validation every N epochs
early_stopping_patience: int = 10  # Stop if no improvement for N epochs
early_stopping_min_delta: float = 0.001  # Minimum improvement to count
```

### Command Line Arguments

```shell
# Train with 10% validation split (default)
kokoro-train --corpus ./ruslan_corpus --output ./my_model

# Train with 20% validation split
kokoro-train --corpus ./ruslan_corpus --val-split 0.2

# Disable validation completely (use all data for training)
kokoro-train --corpus ./ruslan_corpus --no-validation

# Custom early stopping patience
kokoro-train --corpus ./ruslan_corpus --early-stopping-patience 15

# Run validation every 2 epochs instead of every epoch
kokoro-train --corpus ./ruslan_corpus --validation-interval 2
```

## How It Works

### Data Splitting

1. **Random Split**: Dataset is randomly split into train/validation sets
   - Uses fixed seed (42) for reproducibility
   - Split happens before any training

2. **Stratified by Length**: Both sets maintain the sorted-by-length property for efficient batching

3. **Separate Dataloaders**: Train and validation use separate dataloaders
   - Training: Shuffled batches, drop last incomplete batch
   - Validation: Sequential batches, use all data

### Validation Loop

During each validation run:

1. **Model Evaluation Mode**: Sets `model.eval()` to disable dropout, etc.
2. **No Gradient Computation**: Uses `torch.no_grad()` for efficiency
3. **Full Dataset Pass**: Processes all validation batches
4. **Comprehensive Metrics**: Computes all losses (mel, duration, stop token, pitch, energy)

### Early Stopping

The early stopping mechanism:

1. **Tracks Best Validation Loss**: Maintains record of best validation performance
2. **Monitors Improvement**: Checks if validation loss improved by at least `min_delta`
3. **Patience Counter**: Increments when no improvement, resets on improvement
4. **Automatic Termination**: Stops training when patience exceeded

### Best Model Saving

- **Automatic Checkpoint**: Saves model when validation improves
- **Overrides Periodic Saves**: Best model takes priority over periodic checkpoints
- **Resume-Friendly**: Can resume from best checkpoint

## Interpreting Results

### Training Logs

```
Epoch 15 completed. Avg Total Loss: 2.3456, Avg Mel Loss: 1.8901, ...
Validation Epoch 15 - Loss: 2.5123, Mel: 2.0345, Dur: 0.3456, Stop: 0.1322
✓ Validation loss improved by 0.0234 - saving best model
```

### Overfitting Indicators

**Good Signs:**
```
✓ Validation loss improved by 0.0234
```
- Validation loss is decreasing
- Model is learning generalizable patterns

**Warning Signs:**
```
⚠ No validation improvement for 3 epoch(s) (best: 2.4521)
```
- Validation loss stopped improving
- Training may continue improving while validation plateaus
- Indicates potential overfitting

**Critical Signs:**
```
Early stopping triggered after 10 epochs without improvement
Best validation loss: 2.4521
```
- Training automatically stopped
- Model is no longer learning useful patterns
- Use checkpoint with best validation loss

### End-of-Training Summary

```
============================================================
VALIDATION SUMMARY
============================================================
Best Validation Loss: 2.4521
Final Validation Loss: 2.6789
Total Validation Runs: 50
⚠ Potential overfitting detected:
  Final validation loss is 9.2% higher than best
  Consider using the checkpoint from epoch with best validation loss
============================================================
```

## Best Practices

### Recommended Split Ratios

- **Large Dataset (>10,000 samples)**: 10% validation is sufficient
- **Medium Dataset (1,000-10,000 samples)**: 15-20% validation
- **Small Dataset (<1,000 samples)**: Consider k-fold cross-validation (not currently implemented)

### Patience Settings

- **Start Conservative**: Begin with patience=10
- **Adjust Based on Training Dynamics**:
  - Fast convergence: Reduce patience to 5-7
  - Slow/noisy convergence: Increase patience to 15-20
  - Very large datasets: Can use smaller patience (5)

### Validation Interval

- **Default (every epoch)**: Best for most cases
- **Every 2-3 epochs**: Use for very large datasets to save time
- **Every epoch**: Mandatory if using early stopping

## Monitoring Overfitting

### Training vs Validation Gap

Monitor the gap between training and validation loss:

```python
# Good: Small gap (< 10%)
Train Loss: 2.30, Val Loss: 2.50  → Gap: 8.7%

# Concerning: Medium gap (10-20%)
Train Loss: 2.00, Val Loss: 2.45  → Gap: 22.5%

# Overfitting: Large gap (> 20%)
Train Loss: 1.50, Val Loss: 2.50  → Gap: 66.7%
```

### Trends to Watch

1. **Diverging Losses**: Training decreases, validation increases → Overfitting
2. **Plateau**: Both losses stopped decreasing → Convergence (good)
3. **Parallel Decrease**: Both decreasing together → Healthy training

## Troubleshooting

### "No validation improvement for many epochs"

**Possible Causes:**
- Model has converged (check if training loss also plateaued)
- Learning rate too low (check current LR in logs)
- Validation set too small or noisy

**Solutions:**
- Check training loss trends
- Adjust learning rate schedule
- Increase validation split
- Review data quality

### "Validation loss much higher than training"

**Diagnosis:** Clear overfitting

**Solutions:**
1. Add regularization (increase dropout)
2. Reduce model size
3. Use more training data
4. Implement data augmentation
5. Train for fewer epochs (use early stopping)

### "Validation loss fluctuating wildly"

**Possible Causes:**
- Validation set too small
- Batch size too small
- Very diverse/noisy data

**Solutions:**
- Increase validation split
- Increase batch size
- Use longer validation interval (every 2-3 epochs)

## Advanced Usage

### Disable Validation for Final Training

Once you've tuned hyperparameters with validation, you can train a final model on all data:

```shell
kokoro-train --corpus ./ruslan_corpus --no-validation --epochs 100
```

### Resume with Validation

When resuming training, validation state is preserved:

```shell
kokoro-train --corpus ./ruslan_corpus --resume auto
```

The trainer will:
- Restore best validation loss
- Continue early stopping counter
- Maintain same train/validation split

## Implementation Details

### Random Seed

The train/validation split uses a fixed random seed (42) for reproducibility. The same split will be used across runs unless the dataset changes.

### Memory Efficiency

Validation uses:
- `torch.no_grad()` to disable gradient computation
- No mixed precision (for consistency)
- Same memory management as training

### Metrics Tracked

All training metrics are also computed for validation:
- Mel spectrogram loss (L1)
- Duration loss (MSE)
- Stop token loss (BCE)
- Pitch loss (MSE, if variance predictor enabled)
- Energy loss (MSE, if variance predictor enabled)

## Expected Improvements

With proper validation monitoring, you should see:
- **Better Generalization**: Model performs well on unseen data
- **Prevent Overfitting**: Catch overfitting early
- **Optimal Training Time**: Stop at the right moment
- **Reproducibility**: Consistent results across runs
- **Confidence**: Know when your model is production-ready

## Example Training Session

```shell
# Start training with validation
kokoro-train --corpus ./ruslan_corpus --output ./my_model --val-split 0.1

# Logs show:
# Dataset split: 19998 training, 2222 validation samples
#
# Epoch 1 completed. Avg Total Loss: 5.2341, ...
# Validation Epoch 1 - Loss: 5.4567, ...
# ✓ Validation loss improved by 0.0000 - saving best model
#
# Epoch 2 completed. Avg Total Loss: 4.8921, ...
# Validation Epoch 2 - Loss: 5.1234, ...
# ✓ Validation loss improved by 0.3333 - saving best model
# ...
# Epoch 25 completed. Avg Total Loss: 2.1234, ...
# Validation Epoch 25 - Loss: 2.4567, ...
# ⚠ No validation improvement for 10 epoch(s) (best: 2.3456)
# Early stopping triggered after 10 epochs without improvement
# Best validation loss: 2.3456
```

The best model (from epoch ~15) is automatically saved and ready to use!
