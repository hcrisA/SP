## 📑 目录

- [📑 目录](#-目录)
- [⚙️ Requirements](#️-requirements)
- [🛠️ Installation](#️-installation)
- [🏋️ LoRA Training](#️-lora-training)
  - [Why LoRA?](#why-lora)
  - [Running Training](#running-training)
  - [Training Parameters](#training-parameters)
    - [Basic Training Parameters](#basic-training-parameters)
    - [Memory Optimization Parameters](#memory-optimization-parameters)
    - [LoRA-Specific Parameters](#lora-specific-parameters)
    - [Checkpoint Parameters](#checkpoint-parameters)
  - [Training Data Format](#training-data-format)
  - [Output Directory Structure](#output-directory-structure)
  - [Training Framework Architecture](#training-framework-architecture)
  - [Training Pipeline](#training-pipeline)
  - [Memory Optimization Tips](#memory-optimization-tips)
- [🔍 Base Model Evaluation](#-base-model-evaluation)
  - [Running Evaluation](#running-evaluation)
  - [Evaluation Parameters](#evaluation-parameters)
  - [Evaluation Output Structure](#evaluation-output-structure)
- [🧪 LoRA Inference Testing](#-lora-inference-testing)
  - [Running Inference Tests](#running-inference-tests)
  - [Inference Test Parameters](#inference-test-parameters)
  - [Test Output Structure](#test-output-structure)
  - [Understanding Test Results](#understanding-test-results)
- [❓ FAQ](#-faq)
  - [推理时的文本提示词在哪里修改？](#推理时的文本提示词在哪里修改)
  - [推理时的输入帧数是多少？](#推理时的输入帧数是多少)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

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
|── mono2stereo-test/ 
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

## 🔍 Base Model Evaluation

Before training LoRA, you can evaluate the base StereoPilot model's performance on the test dataset using `evaluate.py`. This helps establish a baseline for comparison.

### Running Evaluation

**Default evaluation (uses mono2stereo-test dataset):**

```bash
python evaluate.py
```

**Custom paths:**

```bash
python evaluate.py \
    --data_root ../SP_Data/mono2stereo-test \
    --output_folder ../SP_Data/evaluate_output \
    --device cuda:0
```

### Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | `toml/infer.toml` | Path to model config file (TOML format) |
| `--data_root` | `../SP_Data/mono2stereo-test` | Path to test dataset directory |
| `--output_folder` | `../SP_Data/evaluate_output` | Path to save evaluation results |
| `--device` | `cuda:0` | Device to use for inference |

### Evaluation Output Structure

After running evaluation, the output directory will contain:

```
../SP_Data/evaluate_output/
├── evaluation_results.txt      # Human-readable results log
├── metrics_summary.json        # Detailed metrics in JSON format
├── animation/                  # Output images for animation subset
│   ├── img001.png
│   ├── img002.png
│   └── ...
├── complex/                    # Output images for complex subset
├── indoor/                     # Output images for indoor subset
├── outdoor/                    # Output images for outdoor subset
└── simple/                     # Output images for simple subset
```

---

## 🧪 LoRA Inference Testing

After training LoRA weights, you can evaluate the model's performance on the test dataset using the `test_lora.py` script.

### Running Inference Tests

**Basic test with trained LoRA weights:**

```bash
python test_lora.py \
    --config toml/infer.toml \
    --lora_weights ../SP_Data/checkpoints_lora/lora_weights_epoch_010.safetensors \
    --lora_rank 8 \
    --test_data ../SP_Data/mono2stereo-test \
    --output_dir ../SP_Data/test_results
```

**Quick test with limited samples:**

```bash
python test_lora.py \
    --config toml/infer.toml \
    --lora_weights ../SP_Data/checkpoints_lora/lora_weights_latest.safetensors \
    --lora_rank 4 \
    --test_data ../SP_Data/mono2stereo-test \
    --output_dir ../SP_Data/test_results_quick \
    --max_samples 50 \
    --sampling_steps 20
```

**Test with optimized inference:**

```bash
python test_lora.py \
    --config toml/infer.toml \
    --lora_weights ../SP_Data/checkpoints_lora/lora_weights_epoch_010.safetensors \
    --lora_rank 16 \
    --test_data ../SP_Data/mono2stereo-test \
    --output_dir ../SP_Data/test_results_optimized \
    --use_torch_compile \
    --sampling_steps 30
```

### Inference Test Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--config` | `toml/infer.toml` | Path to model config file (TOML format) |
| `--lora_weights` | **Required** | Path to trained LoRA weights (.safetensors) |
| `--lora_rank` | `4` | LoRA rank (must match training configuration) |
| `--lora_alpha` | `lora_rank` | LoRA alpha scaling factor |
| `--test_data` | `../SP_Data/mono2stereo-test` | Path to test dataset directory |
| `--output_dir` | `../SP_Data/test_results` | Path to save test results |
| `--sampling_steps` | `30` | Number of diffusion sampling steps |
| `--guide_scale` | `5.0` | Classifier-free guidance scale |
| `--seed` | `42` | Random seed for reproducibility |
| `--device` | `cuda:0` | Device to use for inference |
| `--max_samples` | `None` | Maximum samples per subset (None = all) |
| `--use_torch_compile` | `False` | Enable torch.compile for faster inference |
| `--debug` | `False` | Enable debug mode with full error traces |

### Test Output Structure

After running inference tests, the output directory will contain:

```
../SP_Data/test_results
├── evaluation_results.txt      # Human-readable results log
├── metrics_summary.json        # Detailed metrics in JSON format
├── animation/                  # Output images for animation subset
│   ├── img001.png
│   ├── img002.png
│   └── ...
├── complex/                    # Output images for complex subset
├── indoor/                     # Output images for indoor subset
├── outdoor/                    # Output images for outdoor subset
└── simple/                     # Output images for simple subset
```

### Understanding Test Results

**Metrics:**

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality in dB. Higher is better (typical range: 20-35 dB).
- **SSIM (Structural Similarity Index)**: Measures perceptual similarity. Range: 0-1, higher is better (typical range: 0.75-0.95).
- **SIoU (Stereo IoU)**: Stereo-specific metric measuring edge consistency between predicted and ground truth right views. Range: 0-1, higher is better.
- **FPS (Frames Per Second)**: Inference speed. Higher is better.

**Interpreting Results:**

- **PSNR > 28 dB**: Good reconstruction quality
- **SSIM > 0.85**: High perceptual similarity
- **SIoU > 0.75**: Good stereo consistency
- **FPS > 10**: Real-time capable

Compare results across different LoRA checkpoints to find the best performing model.

---

## ❓ FAQ

### 推理时的文本提示词在哪里修改？

在 StereoPilot 中，文本提示词通过 `text_encoder` 编码为嵌入向量，用于引导生成过程。

**1. 训练代码 (`train_lora.py`)**

固定提示词位于 `precompute_text_embeddings` 方法中（第 687 行）：

```python
def precompute_text_embeddings(self):
    # ...
    prompt = ["This is a video viewed from the left perspective"]  # 修改这里
```

**2. 测试推理代码 (`test_lora.py`)**

当前使用空字符串。如果需要使用固定提示词，修改第 272 行：

```python
# 当前代码（空字符串）：
context_cache = model.text_encoder([""], device)

# 修改为固定提示词：
context_cache = model.text_encoder(["This is a video viewed from the left perspective"], device)
```

**3. 原始评估代码 (`evaluate.py`)**

同样在第 211 行修改：

```python
# 当前代码：
context_cache = model.text_encoder([""], device)

# 修改为：
context_cache = model.text_encoder(["This is a video viewed from the left perspective"], device)
```

**对比：**

| 文件 | 当前文本 | 用途 |
|------|---------|------|
| `train_lora.py` | `"This is a video viewed from the left perspective"` | 训练时引导 |
| `test_lora.py` | `""` (空字符串) | 测试推理 |
| `evaluate.py` | `""` (空字符串) | 模型评估 |

### 推理时的输入帧数是多少？

**帧数取决于输入类型：**

| 场景 | `frame_num` | 说明 |
|------|-------------|------|
| **单帧图像推理** (`test_lora.py`, `evaluate.py`) | `1` | 处理静态立体图像对，输入是单张左视图图像 |
| **视频生成** (`sample.py`) | `81` | 生成视频序列，输入是视频文件 |

**帧数规则：**

- 帧数必须是 `4n + 1` 的形式（例如：1, 5, 9, 13, 17, 21, ..., 81）
- 默认 `frame_num=81` 约为 5 秒视频（16fps × 5s ≈ 81 帧）
- 单帧图像推理使用 `frame_num=1`，因为输入是静态图像而非视频

**代码示例：**

```python
# 单帧图像推理（test_lora.py 第 394 行）
video_out = model.sample(
    video_condition=latents,  # 单帧图像的 latent
    frame_num=1,              # 单帧
    ...
)

# 视频生成（sample.py 第 84 行）
video_out = model.sample(
    video_condition=input_path,  # 视频文件路径
    frame_num=81,                # 81帧视频序列
    ...
)
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Wan2.1](https://github.com/Wan-Video/Wan2.1) - Base video generation model
- [Diffusion Pipe](https://github.com/tdrussell/diffusion-pipe) - Training code base
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation paper


