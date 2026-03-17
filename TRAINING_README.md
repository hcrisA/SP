# StereoPilot Training Guide

This guide explains how to train the StereoPilot model for stereo video conversion using the optimized training script.

## Overview

The training script (`train.py`) enables fine-tuning StereoPilot on your own stereo video dataset using reconstruction loss. It's optimized for memory efficiency and GPU utilization.

## Features

- **Memory Efficient**: Frame-by-frame VAE encoding, gradient checkpointing, mixed precision
- **GPU Optimized**: Supports multiple GPU configurations with automatic memory management
- **Comprehensive Logging**: Tensorboard integration, progress tracking, GPU memory monitoring
- **Robust Training**: NaN detection, gradient clipping, automatic checkpointing
- **Clean Architecture**: Well-documented, modular design following best practices

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- 16GB+ GPU memory (24GB recommended)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install flash attention (optional but recommended)
pip install flash-attn --no-build-isolation
```

## Data Preparation

### Directory Structure

Organize your training data as follows:

```
../SP_Data/mono_train/
├── left/          # Left view images
│   ├── frame_00001.jpg
│   ├── frame_00002.jpg
│   └── ...
└── right/         # Right view images
    ├── frame_00001.jpg
    ├── frame_00002.jpg
    └── ...
```

### Data Requirements

- **Image pairs**: Left and right images must have matching filenames
- **Resolution**: Images will be resized to 832×480
- **Quantity**: At least 81 images per sequence (81 frames = 1 training sample)
- **Format**: JPG, JPEG, PNG, or BMP
- **Alignment**: Left and right images must be properly aligned

### Data Validation

The dataset loader automatically:
- Validates image pairs
- Checks for corrupted files
- Resizes images to target resolution
- Normalizes pixel values to [-1, 1]

## Configuration

### Model Configuration (toml/infer.toml)

```toml
[model]
type = 'stereopilot'
ckpt_path = '../SP_Data/ckpt/Wan2.1-T2V-1.3B'      # Base Wan2.1 model
transformer_path = '../SP_Data/ckpt/StereoPilot.safetensors'  # Pretrained weights
pretrained_path = '../SP_Data/ckpt/StereoPilot.safetensors'   # Domain embeddings
dtype = 'bfloat16'                                   # Model dtype
transformer_dtype = 'bfloat16'                       # Transformer dtype
```

**Important**: Ensure the paths point to your downloaded model files.

### Training Hyperparameters

```python
# In train.py or command line arguments
batch_size = 1                      # Batch size per GPU
learning_rate = 3e-4                # Initial learning rate
num_epochs = 10                     # Training epochs
gradient_accumulation_steps = 4     # Gradient accumulation
mixed_precision = "bf16"            # Mixed precision mode
```

## Basic Training

### Command Line Arguments

```bash
python train.py \
    --config toml/infer.toml \              # Model config file
    --train_dir ../SP_Data/mono_train \    # Training data
    --output_dir ../SP_Data/checkpoints \  # Output directory
    --batch_size 1 \                       # Batch size
    --learning_rate 3e-4 \                 # Learning rate
    --epochs 10 \                          # Number of epochs
    --gradient_accumulation_steps 4 \      # Gradient accumulation
    --mixed_precision bf16                 # Mixed precision
```

### Example Commands for Different GPUs

**RTX 3090/4090 (24GB):**
```bash
python train.py \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --mixed_precision bf16 \
    --epochs 10
```

**A100/A800 (40-80GB):**
```bash
python train.py \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --mixed_precision bf16 \
    --epochs 15
```

**RTX 5090 (32GB):**
```bash
python train.py \
    --batch_size 1 \
    --gradient_accumulation_steps 6 \
    --mixed_precision bf16 \
    --epochs 12
```

## Training Process

### What Happens During Training

1. **Data Loading**: The script loads 81-frame sequences from your left/right directories
2. **VAE Encoding**: Each video is encoded to latent space (frame-by-frame for memory efficiency)
3. **Text Encoding**: Fixed prompt "This is a video viewed from the left perspective" is pre-computed
4. **Forward Pass**: Transformer processes left latents to predict right latents
5. **Loss Calculation**: MSE loss between predicted and actual right latents
6. **Backward Pass**: Gradients accumulated and applied with mixed precision
7. **Optimization**: AdamW optimizer with cosine annealing scheduler
8. **Logging**: Metrics logged to console, file, and Tensorboard
9. **Checkpointing**: Model saved every epoch + best model tracked

### Training Objective

The model learns to reconstruct the right view from the left view using:
- **Fixed timestep**: t = 0.001 (near-zero noise for reconstruction)
- **Reconstruction loss**: MSE in latent space
- **Frozen components**: VAE and text encoder remain frozen
- **Trainable components**: Transformer + domain embeddings

### Domain Embeddings

The model uses two domain-specific embeddings:
- `parall_embedding`: For parallel stereo format
- `converge_embedding`: For converged stereo format (used in training)

During training, `domain_label=1` (converge) is used for stereo conversion.

## Monitoring Training

### Console Output

```
Epoch 1/10
100%|██████████| 125/125 [05:23<00:00,  2.58s/it, loss=0.0234, lr=3.00e-4]
Epoch 1 completed - Avg Loss: 0.0256
```

Shows:
- Progress bar with ETA
- Current loss and learning rate
- Batch processing time
- Epoch summary

### Tensorboard

Launch Tensorboard to monitor training:

```bash
tensorboard --logdir ../SP_Data/checkpoints/logs
```

View metrics at: http://localhost:6006

Tracked metrics:
- Training loss
- Learning rate
- Gradient norm
- GPU memory usage
- Epoch statistics

### Log Files

- `training.log`: Detailed text log with timestamps
- Check console for real-time GPU memory usage (every 100 steps)

## Outputs

### Checkpoints

**Regular Checkpoints:**
- `checkpoint_epoch_XXX.pt`: Full checkpoint (model + optimizer + scheduler)
- `stereopilot_epoch_XXX.safetensors`: Model weights only

**Best Model:**
- `best_checkpoint.pt`: Best performing checkpoint
- `best_model.safetensors`: Best model weights

### Directory Structure

```
../SP_Data/checkpoints/
├── checkpoint_epoch_001.pt
├── stereopilot_epoch_001.safetensors
├── checkpoint_epoch_002.pt
├── stereopilot_epoch_002.safetensors
├── best_checkpoint.pt          # Best model
├── best_model.safetensors
└── logs/
    └── events.out.tfevents...  # Tensorboard logs
```

## Memory Optimization

### Automatic Optimizations

The script automatically applies:
1. **Gradient Checkpointing**: Reduces memory by recomputing activations
2. **Mixed Precision**: bfloat16/float16 training
3. **Frame-by-Frame Encoding**: VAE processes one frame at a time
4. **Gradient Accumulation**: Simulates larger batch sizes
5. **Memory Cleanup**: Regular cache clearing

### Manual Optimization Tips

**If you encounter OOM (Out of Memory):**

1. **Reduce batch size** (most effective):
   ```bash
   --batch_size 1  # Minimum
   ```

2. **Increase gradient accumulation**:
   ```bash
   --gradient_accumulation_steps 8  # Or higher
   ```

3. **Use mixed precision**:
   ```bash
   --mixed_precision bf16  # For RTX 30xx/40xx/50xx
   --mixed_precision fp16  # For older GPUs
   ```

4. **Close other applications** using GPU

5. **Monitor memory usage:**
   ```bash
   nvidia-smi -l 1  # Check GPU memory in real-time
   ```

### GPU Memory Requirements

| GPU Model | Memory | Batch Size | Gradient Accumulation | Mixed Precision |
|-----------|--------|------------|----------------------|-----------------|
| RTX 3090/4090 | 24GB | 1 | 8 | bf16 |
| RTX 5090 | 32GB | 1 | 6 | bf16 |
| A100/A800 | 40GB | 2 | 4 | bf16 |
| A100/A800 | 80GB | 4 | 2 | bf16 |

## Troubleshooting

### Out of Memory (OOM)

**Symptoms:** CUDA out of memory error

**Solutions:**
1. Reduce batch size: `--batch_size 1`
2. Increase gradient accumulation: `--gradient_accumulation_steps 8`
3. Enable mixed precision: `--mixed_precision bf16`
4. Close other GPU applications
5. Restart training with smaller batch size

### NaN Loss

**Symptoms:** Loss becomes NaN during training

**Solutions:**
1. Check input data for corrupted images
2. Verify VAE and text encoder are frozen
3. Reduce learning rate: `--learning_rate 1e-4`
4. Ensure timestep is fixed to 0.001
5. Check for gradient explosion (should be clipped to max_norm=1.0)

### Slow Training

**Symptoms:** Low GPU utilization, slow iteration speed

**Solutions:**
1. Increase num_workers in DataLoader (if CPU bottleneck)
2. Check disk I/O speed (SSD recommended)
3. Reduce validation frequency
4. Ensure GPU is not thermal throttling

### Poor Convergence

**Symptoms:** Loss not decreasing or decreasing very slowly

**Solutions:**
1. Verify data quality and alignment
2. Check that timestep is fixed to 0.001 (not variable)
3. Ensure domain_label=1 for stereo conversion
4. Increase learning rate: `--learning_rate 5e-4`
5. Train for more epochs: `--epochs 15`
6. Check for data leakage or preprocessing issues

### Checkpoint Loading Issues

**Symptoms:** Cannot load checkpoints or weights

**Solutions:**
1. Ensure config matches training config
2. Check file permissions
3. Verify model architecture hasn't changed
4. Use appropriate dtype (bfloat16/float16/float32)

## Advanced Usage

### Custom Loss Functions

Modify the `compute_loss` method in `train.py`:

```python
def compute_loss(self, pred_latents: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
    # MSE loss
    mse_loss = F.mse_loss(pred_latents, target_latents)
    
    # Add perceptual loss (optional)
    # perceptual_loss = self.perceptual_loss(pred_latents, target_latents)
    
    # Combined loss
    total_loss = mse_loss  # + 0.1 * perceptual_loss
    
    return total_loss
```

### Resume Training

To resume from a checkpoint, modify the trainer:

```python
def load_checkpoint(self, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    self.transformer.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['global_step']
```

### Multi-GPU Training

The script automatically supports multi-GPU training via Accelerate. To use multiple GPUs:

```bash
# No changes needed - Accelerate handles multi-GPU automatically
python train.py --batch_size 1  # Will use all available GPUs
```

For manual multi-GPU setup:

```bash
accelerate launch train.py [arguments]
```

### Custom Scheduling

Modify the scheduler in `setup_optimizer`:

```python
# Example: Linear warmup + cosine decay
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup_scheduler = LinearLR(self.optimizer, start_factor=0.1, total_iters=500)
main_scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps - 500)
self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[500])
```

## Inference with Trained Model

After training, use your model for inference:

```bash
python sample.py \
    --config toml/infer.toml \
    --input /path/to/left_video.mp4 \
    --output_folder /path/to/output \
    --device cuda:0
```

**Important**: Update the config to point to your trained weights:

```toml
transformer_path = '../SP_Data/checkpoints/best_model.safetensors'
pretrained_path = '../SP_Data/checkpoints/best_model.safetensors'
```

## Citation

If you use this training script in your research, please cite the original StereoPilot paper:

```bibtex
@misc{shen2025stereopilot,
  title={StereoPilot: Learning Unified and Efficient Stereo Conversion via Generative Priors},
  author={Shen, Guibao and Du, Yihua and Ge, Wenhang and He, Jing and Chang, Chirui and Zhou, Donghao and Yang, Zhen and Wang, Luozhou and Tao, Xin and Chen, Ying-Cong},
  year={2025},
  eprint={2512.16915},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2512.16915}, 
}
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the training log for specific errors
3. Monitor GPU memory usage with `nvidia-smi`
4. Verify data quality and alignment

## License

This training script follows the same license as the StereoPilot project.
