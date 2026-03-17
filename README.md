## ⚙️ Requirements

Our inference environment:
- Python 3.12
- CUDA 12.1
- PyTorch 2.4.1
- GPU: NVIDIA A800 (only ~23GB VRAM required)

## 🛠️ Installation

**Step 1:** Clone the repository

```bash
git clone https://github.com/hcrisA/SP.git

cd SP
```

**Step 2:** Create conda environment

```bash
conda create -n StereoPilot python=3.12

conda activate StereoPilot
```

**Step 3:** Install dependencies

```bash
pip install -r requirements.txt

# Method 1: Install flash-attn from source (takes longer)
pip install flash-attn==2.7.4.post1 --no-build-isolation

# Method 2 (Recommended): Download pre-built wheel for cu121 + torch2.4 + python3.12
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

**Step 4:** Download model checkpoints

Place the following files in the `../SP_Data/ckpt/` directory:

| File | Description |
|------|-------------|
| [`StereoPilot.safetensors`](https://huggingface.co/KlingTeam/StereoPilot) | StereoPilot model weights |
| [`Wan2.1-T2V-1.3B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | Base Wan2.1 model directory |

Download StereoPilot.safetensor & Wan2.1-1.3B base model:

```bash
pip install "huggingface_hub[cli]"

huggingface-cli download KlingTeam/StereoPilot StereoPilot.safetensors --local-dir ./ckpt

huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./ckpt/Wan2.1-T2V-1.3B
```

---

## 🏋️ LoRA Training

### Why LoRA?

**LoRA (Low-Rank Adaptation)** is a parameter-efficient fine-tuning technique that dramatically reduces the computational cost of training large models. We adopt LoRA for StereoPilot training based on [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe) codebase.

**Key Benefits:**
- **99%+ fewer trainable parameters** - Only train small low-rank matrices instead of full weights
- **Lower GPU memory usage** - Fits on consumer GPUs (16-24GB VRAM)
- **Faster training** - Converges in fewer iterations
- **Preserves original model** - Base model remains frozen, easy to swap LoRAs
- **Tiny checkpoints** - LoRA weights are typically 10-100MB vs GBs for full fine-tuning

**LoRA Principle:**

Instead of updating the full weight matrix `W` (dimension `d × k`), LoRA decomposes the weight update `ΔW` into two low-rank matrices:

```
ΔW = B × A
```

Where:
- `B` is `d × r`
- `A` is `r × k`
- `r` (rank) is much smaller than `d` and `k` (typically `r = 4-32`)

This reduces parameters from `d × k` to `r × (d + k)`, achieving 100x+ reduction.

### Running Training

**With Custom Parameters:**

```bash
python train_lora.py \
    --config toml/infer.toml \
    --train_dir ../SP_Data/mono_train \
    --output_dir ../SP_Data/checkpoints_lora \
    --batch_size 1 \
    --num_frames 21 \
    --lora_rank 8 \
    --learning_rate 3e-4 \
    --epochs 10 \
    --gradient_accumulation_steps 4 \
    --mixed_precision bf16
```

**Resume from Checkpoint:**

```bash
python train_lora.py \
    --config toml/infer.toml \
    --train_dir ../SP_Data/mono_train \
    --output_dir ../SP_Data/checkpoints_lora \
    --resume_from_checkpoint ../SP_Data/checkpoints_lora/checkpoint_epoch_005.pt
```

### Training Parameters

#### Basic Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | `toml/infer.toml` | Path to model config file (TOML format) |
| `--train_dir` | `../SP_Data/mono_train` | Path to training data directory |
| `--output_dir` | `../SP_Data/checkpoints_lora` | Path to save checkpoints |
| `--batch_size` | `1` | Batch size per GPU (reduce if OOM) |
| `--epochs` | `10` | Number of training epochs |
| `--learning_rate` | `3e-4` | Learning rate (recommended: 1e-4 to 5e-4) |
| `--num_frames` | `81` | Number of frames per sequence (reduce to save memory) |
| `--image_size` | `832,480` | Image size as `width,height` |
| `--seed` | `42` | Random seed for reproducibility |

#### Memory Optimization Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--gradient_accumulation_steps` | `4` | Gradient accumulation steps (simulates larger batch) |
| `--mixed_precision` | `bf16` | Mixed precision mode: `bf16`, `fp16`, or `no` |

#### LoRA-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora_rank` | `4` | LoRA rank. Higher = more expressive, more params. Typical: 4-32 |
| `--lora_alpha` | `rank` | LoRA alpha scaling factor (defaults to rank) |
| `--lora_dropout` | `0.0` | LoRA dropout rate (0.0-0.1) |
| `--lora_target_modules` | `None` | Comma-separated target modules (e.g., `attn.q,attn.k,attn.v`) |

#### Checkpoint Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--save_every_n_epochs` | `1` | Save checkpoint every N epochs |
| `--log_every_n_steps` | `50` | Log to tensorboard every N steps |
| `--resume_from_checkpoint` | `None` | Path to checkpoint to resume from |

### Training Data Format

Training data should be organized as follows:

```
SP_Data/
├── ckpt/                   # Model Checkpoints
│   ├── StereoPilot.safetensors
│   └── Wan2.1-T2V-1.3B/
└── mono_train/             # Training Data
    ├── left/               # Left eye view images
    │   ├── 00001.jpg
    │   ├── 00002.jpg
    │   └── ...
    └── right/              # Right eye view images
        ├── 00001.jpg
        ├── 00002.jpg
        └── ...
```

### Output Directory Structure

After training, the output directory will contain the following files:

```
output_dir/
├── checkpoint_epoch_001.pt      # Full checkpoint (model + optimizer + scheduler)
├── checkpoint_epoch_002.pt
├── ...
├── checkpoint_epoch_010.pt
├── lora_weights_epoch_001.safetensors  # LoRA weights only (small, ~10-50MB)
├── lora_weights_epoch_002.safetensors
├── ...
├── lora_weights_epoch_010.safetensors
├── training_config.json         # Training configuration snapshot
└── logs/
    └── tensorboard/             # TensorBoard logs (optional)
```

**File Descriptions:**

| File | Description |
|------|-------------|
| `checkpoint_epoch_XXX.pt` | Full checkpoint for resuming training. Contains model state, optimizer state, scheduler state, and training progress. |
| `lora_weights_epoch_XXX.safetensors` | LoRA weights only. Small files that can be loaded for inference or shared. |
| `training_config.json` | JSON snapshot of all training parameters for reproducibility. |
| `logs/tensorboard/` | TensorBoard event files for monitoring training progress. |

**Loading LoRA Weights for Inference:**

```python
from lora_utils import LoRAManager

# Load LoRA weights into model
lora_manager = LoRAManager(model, rank=8)
lora_manager.load_lora_weights("path/to/lora_weights_epoch_010.safetensors")
```

### Training Framework Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    StereoPilot LoRA Training                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Data Loader │───▶│  VAE Encoder │───▶│  Latents     │      │
│  │  (Videos)    │    │  (Frozen)    │    │  (Compressed)│      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                   │             │
│                                                   ▼             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Text Encoder │───▶│  Conditions  │───▶│  Transformer │      │
│  │  (Frozen)    │    │  (T5)        │    │  + LoRA      │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                   │             │
│                                                   ▼             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  VAE Decoder │◀───│  Noise Pred  │◀───│  Diffusion   │      │
│  │  (Frozen)    │    │  (with LoRA) │    │  Timestep    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Training Pipeline

1. **Data Preparation**
   - Load video pairs (left/right eye views)
   - Extract frames and resize to target resolution
   - Tokenize text prompts with T5 encoder

2. **VAE Encoding** (Frozen, no gradients)
   - Encode videos to latent space
   - Move VAE to GPU temporarily, then back to CPU to save memory

3. **LoRA Injection**
   - Inject LoRA layers into target modules (attention Q/K/V/O, FFN)
   - Freeze original weights, only LoRA parameters are trainable

4. **Diffusion Training**
   - Sample random timesteps
   - Add noise to latents
   - Predict noise with transformer + LoRA
   - Compute MSE loss between predicted and actual noise

5. **Backpropagation**
   - Update only LoRA parameters
   - Use gradient accumulation for effective larger batch
   - Apply gradient checkpointing for memory efficiency

6. **Checkpoint Saving**
   - Save LoRA weights (small files ~10-50MB)
   - Save optimizer state for resuming

### Memory Optimization Tips

| GPU VRAM | Recommended Settings |
|----------|---------------------|
| 16GB | `--num_frames 1-5 --batch_size 1 --gradient_accumulation_steps 8` |
| 24GB | `--num_frames 9-13 --batch_size 1 --gradient_accumulation_steps 4` |
| 40GB+ | `--num_frames 81 --batch_size 1-2` |

**Additional optimizations:**
- Use `--mixed_precision bf16` for RTX 30xx/40xx/50xx GPUs
- Reduce `--num_frames` if OOM during VAE encoding
- Increase `--gradient_accumulation_steps` to maintain effective batch size

---


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Wan2.1](https://github.com/Wan-Video/Wan2.1) - Base video generation model
- [Diffusion Pipe](https://github.com/tdrussell/diffusion-pipe) - Training code base
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation paper


