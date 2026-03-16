## ⚙️ Requirements

Our inference environment:
- Python 3.12
- CUDA 12.1
- PyTorch 2.4.1
- GPU: NVIDIA A800 (only ~23GB VRAM required)

## 🛠️ Installation

**Step 1:** Clone the repository

```bash
git clone https://github.com/KlingTeam/StereoPilot.git

cd StereoPilot
```

**Step 2:** Create conda environment

```bash
conda create -n StereoPilot python=3.12

conda activate StereoPilot
```

**Step 3:** Install dependencies

```bash
pip install -r requirements.txt

pip install flash-attn==2.7.4.post1 --no-build-isolation
```

**Step 4:** Download model checkpoints

Place the following files in the `ckpt/` directory:

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

## 🚀 Inference 

### Input Requirements

For each input video, you need:
1. **Video file** (`.mp4`): Monocular video, 81 frames, 832×480 resolution, 16fps
2. **Prompt file** (`.txt`): Text description of the video content (same name as video)

Example (you can try the cases in the `sample/` folder):
```
sample/
├── my_video.mp4
└── my_video.txt   
```

### Running Inference

**Basic usage:**

```bash
# Edit toml/infer.toml to customize model paths. If you followed the above steps, there is no need to change
python sample.py \
  --config toml/infer.toml \
  --input /path/to/input_video.mp4 \
  --output_folder /path/to/output \
  --device cuda:0
```

**Using the example script:**

```bash
bash sample.sh
```

### Generate Stereo Visualization

After inference, you can generate Side-by-Side (SBS) and Red-Cyan anaglyph stereo videos for visualization:

```bash
python utils/stereo_video.py \
  --left /path/to/left_eye.mp4 \
  --right /path/to/right_eye.mp4 \
```

**Output files:**
| Output | Description | Viewing Device |
|--------|-------------|----------------|
| `{name}_sbs.mp4` | Side-by-Side stereo video | VR Headset <img src="asset/VR_Glass.png" width="24" height="24"> |
| `{name}_anaglyph.mp4` | Red-Cyan anaglyph stereo video | 3D Glasses <img src="asset/Red_Blue_Glass.png" width="24" height="24"> |

## 📊 Dataset

We introduce **UniStereo**, the first large-scale unified stereo video dataset featuring both parallel and converged stereo formats.

<div align="center">
<img src="asset/parallel_vs_converged.png" width="80%">
</div>

UniStereo consists of two parts:
- **3DMovie** - Converged stereo format from 3D movies
- **Stereo4D** - Parallel stereo format *(coming soon)*

For detailed data processing instructions, please refer to [StereoPilot_Dataprocess](./StereoPilot_Dataprocess/).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Wan2.1](https://github.com/Wan-Video/Wan2.1) - Base video generation model
- [Diffusion Pipe](https://github.com/tdrussell/diffusion-pipe) - Training code base

## 🌟 Citation

If you find our work helpful, please consider citing:

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
