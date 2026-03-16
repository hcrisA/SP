import os
import argparse
import random
import json
import toml
import glob
import sys
import types
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_video
from accelerate import Accelerator
from accelerate.utils import set_seed
import re
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file
import torch.cuda.amp as amp
from PIL import Image

# Add current directory to path
sys.path.append(os.getcwd())

# Import Wan Modules
import models.wan

# Patch WanPipeline.__init__ to avoid hardcoded cuda calls which fail on unsupported devices (e.g. RTX 5090 with PyTorch < 2.5)
import inspect
import textwrap
try:
    init_src = inspect.getsource(models.wan.WanPipeline.__init__)
    init_src = textwrap.dedent(init_src)
    # Comment out hardcoded cuda transfers
    init_src = init_src.replace("self.vae.mean = self.vae.mean.to('cuda')", "# self.vae.mean = self.vae.mean.to('cuda')")
    init_src = init_src.replace("self.vae.std = self.vae.std.to('cuda')", "# self.vae.std = self.vae.std.to('cuda')")
    
    exec_globals = models.wan.WanPipeline.__init__.__globals__
    # Execute the modified source in the original globals
    exec(init_src, exec_globals)
    # Update the class method
    models.wan.WanPipeline.__init__ = exec_globals['__init__']
    print("Successfully patched WanPipeline.__init__ to remove hardcoded cuda calls")
except Exception as e:
    print(f"Failed to patch WanPipeline: {e}")

from models.wan import KEEP_IN_HIGH_PRECISION, sinusoidal_embedding_1d

# Patch WanAttentionBlock.forward to fix RMSNorm float32 output issue with bfloat16 weights
# This prevents "RuntimeError: mat1 and mat2 must have the same dtype"
try:
    if hasattr(models.wan, 'WanAttentionBlock'):
        def patched_forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
            # Ensure modulation matches input dtype (e is from e0 which is bfloat16)
            e = (self.modulation.to(x.dtype) + e).chunk(6, dim=1)
            
            # 1. Self Attention with casted Norm
            # self.norm1 likely returns float32. We need bfloat16 for self_attn weights.
            normed_x = self.norm1(x).to(x.dtype)
            y = self.self_attn(normed_x * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs)
            x = x + y * e[2]

            # 2. Cross Attention
            normed_x3 = self.norm3(x).to(x.dtype)
            # context might be different dtype? Ensure it matches if needed, but cross_attn usually handles context.
            # But x must match weights.
            x = x + self.cross_attn(normed_x3, context, context_lens)

            # 3. FFN
            normed_x2 = self.norm2(x).to(x.dtype)
            y = self.ffn(normed_x2 * (1 + e[4]) + e[3])
            x = x + y * e[5]
            return x
            
        models.wan.WanAttentionBlock.forward = patched_forward
        print("Successfully patched WanAttentionBlock.forward for bfloat16 training stability")
except Exception as e:
    print(f"Failed to patch WanAttentionBlock: {e}")

# Patch WanSelfAttention.forward to fix float output from flash_attn/rope mismatching with bfloat16 linear layer
try:
    from wan.modules.model import rope_apply, WanSelfAttention 
    from wan.modules.attention import flash_attention
    
    def patched_sa_forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs), 
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        x = x.flatten(2)
        
        # FIX: Ensure x matches linear layer weight dtype
        if x.dtype != self.o.weight.dtype:
            x = x.to(self.o.weight.dtype)
            
        x = self.o(x)
        return x

    WanSelfAttention.forward = patched_sa_forward
    print("Successfully patched WanSelfAttention.forward for bfloat16 stability")
except Exception as e:
    print(f"Failed to patch WanSelfAttention: {e}")

# Patch WanLayerNorm.forward to avoid casting input to float32 when weights are bfloat16
try:
    if hasattr(models.wan, 'WanLayerNorm'):
        def patched_ln_forward(self, x):
             # Original WanLayerNorm implementation casts input to float32: super().forward(x.float()).type_as(x)
             # This causes error if weights are bfloat16.
             # We just run in original dtype (bfloat16).
             return super(models.wan.WanLayerNorm, self).forward(x)
        
        models.wan.WanLayerNorm.forward = patched_ln_forward
        print("Successfully patched WanLayerNorm.forward for bfloat16 stability")
except Exception as e:
    print(f"Failed to patch WanLayerNorm: {e}")

# Now import StereoPilot (it uses models.wan)
from models.StereoPilot import StereoPilotPipeline
import models.StereoPilot

# Patch StereoPilotPipeline.__init__ to convert string dtypes to torch.dtype objects
# This fixes: RuntimeError: Invalid device string: 'bfloat16'
def _dtype_converter_wrapper(init_func):
    def wrapper(self, config):
        def parse_dtype(name):
            if not isinstance(name, str): return name
            if name == 'float8':
                return getattr(torch, 'float8_e4m3fn', torch.float16)
            if hasattr(torch, name):
                return getattr(torch, name)
            # Try to handle 'bfloat16' specifically just in case
            if name == 'bfloat16': return torch.bfloat16
            return torch.float32 # fallback

        if 'model' in config:
            if 'dtype' in config['model']:
                config['model']['dtype'] = parse_dtype(config['model']['dtype'])
            if 'transformer_dtype' in config['model']:
                config['model']['transformer_dtype'] = parse_dtype(config['model']['transformer_dtype'])
        
        return init_func(self, config)
    return wrapper

# Apply patch
StereoPilotPipeline.__init__ = _dtype_converter_wrapper(StereoPilotPipeline.__init__)
print("Successfully patched StereoPilotPipeline to handle string dtypes")

# Patch StereoPilotPipeline.load_diffusion_model to remove hardcoded cuda device
try:
    load_src = inspect.getsource(StereoPilotPipeline.load_diffusion_model)
    load_src = textwrap.dedent(load_src)
    # Removing hardcoded cuda
    load_src = load_src.replace('device="cuda"', 'device="cpu"') 
    
    # Execute in the module's globals
    sp_globals = StereoPilotPipeline.load_diffusion_model.__globals__
    exec(load_src, sp_globals)
    StereoPilotPipeline.load_diffusion_model = sp_globals['load_diffusion_model']
    print("Successfully patched StereoPilotPipeline.load_diffusion_model")
except Exception as e:
    print(f"Failed to patch StereoPilotPipeline.load_diffusion_model: {e}")

# =============================================================================
# 2. Dataset
# =============================================================================

class StereoVideoDataset(Dataset):
    def __init__(self, root_dir, width=832, height=480, frames=81):
        self.left_dir = os.path.join(root_dir, 'left')
        self.right_dir = os.path.join(root_dir, 'right')
        self.width = width
        self.height = height
        self.frames = frames
        
        # Helper to find images
        def find_images(directory):
            extensions = ['*.jpg', '*.png', '*.jpeg', '*.bmp']
            files = []
            for ext in extensions:
                files.extend(glob.glob(os.path.join(directory, ext)))
                files.extend(glob.glob(os.path.join(directory, ext.upper())))
            
            # Recursive if empty
            if not files:
                 for ext in extensions:
                    files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
                    files.extend(glob.glob(os.path.join(directory, '**', ext.upper()), recursive=True))
            return sorted(files)

        self.left_images = find_images(self.left_dir)
        self.right_images = find_images(self.right_dir)
        
        if len(self.left_images) == 0:
            print(f"Warning: No images found in {self.left_dir}")
        
        # Ensure lengths match or trim
        min_len = min(len(self.left_images), len(self.right_images))
        if min_len < len(self.left_images):
            print(f"Trimming left images from {len(self.left_images)} to {min_len}")
            self.left_images = self.left_images[:min_len]
        if min_len < len(self.right_images):
            print(f"Trimming right images from {len(self.right_images)} to {min_len}")
            self.right_images = self.right_images[:min_len]
            
        print(f"Found {len(self.left_images)} stereo frames total.")
        
    def __len__(self):
        # Return number of full chunks
        return len(self.left_images) // self.frames

    def __getitem__(self, idx):
        try:
            frames_l = []
            frames_r = []

            # Calculate start position for this chunk
            start_pos = idx * self.frames
            
            for i in range(self.frames):
                img_path_l = self.left_images[start_pos + i]
                img_path_r = self.right_images[start_pos + i]
                
                # Load images
                img_l = Image.open(img_path_l).convert('RGB')
                img_r = Image.open(img_path_r).convert('RGB')
                
                # Resize
                img_l = img_l.resize((self.width, self.height), Image.BILINEAR)
                img_r = img_r.resize((self.width, self.height), Image.BILINEAR)
                
                # Convert to tensor [0, 1] [C, H, W]
                tensor_l = transforms.functional.to_tensor(img_l)
                tensor_r = transforms.functional.to_tensor(img_r)
                
                frames_l.append(tensor_l)
                frames_r.append(tensor_r)
            
            # Stack [T, C, H, W]
            video_l = torch.stack(frames_l)
            video_r = torch.stack(frames_r)
            
            # Normalize to [-1, 1]
            video_l = (video_l * 2.0) - 1.0
            video_r = (video_r * 2.0) - 1.0
            
            return {
                "left": video_l,
                "right": video_r
            }
        except Exception as e:
            print(f"Error loading sequence starting at {idx}: {e}")
            dummy = torch.zeros((self.frames, 3, self.height, self.width))
            return {"left": dummy, "right": dummy}

# =============================================================================
# 3. Main Training Logic
# =============================================================================

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to stereopilot config tensor/toml")
    parser.add_argument("--train_dir", type=str, default="../mono_train")
    parser.add_argument("--output_dir", type=str, default="output/checkpoints")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()
    
    # Check for CUDA availability and support
    use_cpu = False
    if torch.cuda.is_available():
        try:
            # Check for RTX 5090 which is incompatible with current PyTorch binary
            if "5090" in torch.cuda.get_device_name(0):
                 print(f"Detected {torch.cuda.get_device_name(0)}. Current PyTorch version has no kernels for this GPU. Falling back to CPU to enable training.")
                 use_cpu = True
            else:
                 t = torch.zeros(1).cuda()
                 torch.cuda.synchronize()
        except RuntimeError as e:
            use_cpu = True
            print(f"CUDA available but failed to run: {e}. Falling back to CPU.")
    
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, cpu=use_cpu)
    set_seed(42)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Training with config: {args.config}")

    # Load Config
    with open(args.config) as f:
        config = json.loads(json.dumps(toml.load(f)))
        
    if use_cpu:
         print("Overriding config dtypes to float32/bfloat16 for CPU execution")
         if 'model' in config:
              # Prefer bfloat16 if likely supported, else float32. 
              # Using bfloat16 matches typical training precision.
              config['model']['dtype'] = 'bfloat16' 
              config['model']['transformer_dtype'] = 'bfloat16'
    else:
        # GPU execution: Avoid float8 for training as standard nn.Linear fails with bfloat16 inputs
         if 'model' in config:
              if config['model'].get('transformer_dtype') == 'float8':
                   print("Switching transformer_dtype from 'float8' to 'bfloat16' for training compatibility.")
                   config['model']['transformer_dtype'] = 'bfloat16'

    
    # Initialize Pipeline
    # This loads VAE and initializes models
    pipeline = StereoPilotPipeline(config)
    pipeline.load_diffusion_model()
    
    # 获取各个模型组件
    model = pipeline.transformer
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    
    # =========================================================
    # 冻结策略 (Freeze Strategy)
    # =========================================================
    
    # 1. 冻结 VAE
    # WanVAE内部通常有一个.model属性 (encoder/decoder)，我们需要冻结它
    if hasattr(vae, 'model'):
        vae.model.requires_grad_(False)
        vae.model.eval()
    else:
        vae.requires_grad_(False).eval()
    print("VAE frozen.")
        
    # 2. 冻结 Text Encoder (UMT5/T5)
    # T5EncoderModel封装了.model (T5Encoder)
    if hasattr(text_encoder, 'model'):
        text_encoder.model.requires_grad_(False)
        text_encoder.model.eval()
    else:
        text_encoder.requires_grad_(False).eval()
    print("Text Encoder frozen.")
        
    # 3. 配置 Transformer (Fine-tune)
    # 微调 Transformer Decoder (包含原层 + 新增的 domain embedding)
    # 启用梯度
    model.requires_grad_(True)
    model.train()
    print("Transformer set to trainable.")
    
    # 检查新增参数 (domain embeddings) 是否存在并可训练
    if hasattr(model, 'parall_embedding'):
        print("Confirmed: parall_embedding exists in model.")
    if hasattr(model, 'converge_embedding'):
        print("Confirmed: converge_embedding exists in model.")

    # =========================================================
    # 自定义 Forward (Inject Embeddings)
    # =========================================================
    # 这里从StereoPilot代码中复制custom_forward逻辑，
    # 主要是为了注入domain_embedding (parall/converge) 和处理 context
    def new_custom_forward(self, x, t, context, seq_len, clip_fea=None, y=None, domain_label=0):
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
             self.freqs = self.freqs.to(device)

        if y is not None:
             x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Ensure correct dtype
        x = [u.to(self.patch_embedding.weight.dtype) for u in x]

        # Patch Embedding
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        input_seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        
        # Concat batch
        x = torch.cat(x, dim=0) # [B, L, C]
        
        # Time embedding
        # Cast input to match the model weights (e.g. bfloat16), avoiding float32 force-cast which causes mismatch on CPU
        t_emb = sinusoidal_embedding_1d(self.freq_dim, t).to(self.patch_embedding.weight.dtype)
        e = self.time_embedding(t_emb)
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        
        # Domain embedding (StereoPilot特有逻辑)
        # domain_label=0 -> parall (平行), domain_label=1 -> converge (汇聚/交叉)
        if domain_label == 0:
             domain_emb = self.parall_embedding.unsqueeze(0)
        else:
             domain_emb = self.converge_embedding.unsqueeze(0)
        e0 = e0 + domain_emb.to(e0.dtype)
        
        # Process Context (Text Embeddings)
        # Context在这里传入的是list of tensors (from T5Encoder)
        if isinstance(context, list):
             # 确保 context 长度一致或正确 padding (WanModel 的 text_embedding 需要 stack 后的 tensor)
             # WanModel 内部 text_embedding 可能期待 [B, L, C] 或 [sum(L), C]
             # 原StereoPilot逻辑:
            context = self.text_embedding(
                torch.stack([
                    torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ])
            )
        else:
            # 如果已经是tensor
            pass 
            
        kwargs = dict(
             e=e0,
             seq_lens=input_seq_lens,
             grid_sizes=grid_sizes,
             freqs=self.freqs,
             context=context,
             context_lens=None
        )

        for block in self.blocks:
             x = block(x, **kwargs)

        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        
        # 返回Latent (list of floats/tensors)
        return [u.float() for u in x]

    # Bind new forward method to the model instance
    model.forward = types.MethodType(new_custom_forward, model)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Dataloader
    dataset = StereoVideoDataset(args.train_dir, width=832, height=480, frames=81)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    
    # Prepare with Accelerator
    # IMPORTANT: Prepare wraps the model.
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Move frozen components to correct device (since they were loaded on CPU via patch)
    print(f"Moving frozen components to {accelerator.device}...")
    
    # 1. Move VAE
    if hasattr(vae, 'model'):
        vae.model.to(accelerator.device)
    elif isinstance(vae, torch.nn.Module):
        vae.to(accelerator.device)
        
    # Also move VAE scale/mean/std if they exist (they might be tensors or lists of tensors)
    for attr_name in ['scale', 'mean', 'std']:
        if hasattr(vae, attr_name):
            val = getattr(vae, attr_name)
            if isinstance(val, torch.Tensor):
                setattr(vae, attr_name, val.to(accelerator.device))
            elif isinstance(val, list):
                # Check if list contains tensors (e.g. scale is [tensor(shift), tensor(scale)])
                if len(val) > 0 and isinstance(val[0], torch.Tensor):
                    new_val = [v.to(accelerator.device) for v in val]
                    setattr(vae, attr_name, new_val)

    # 2. Move Text Encoder    
    if hasattr(text_encoder, 'model'):
        text_encoder.model.to(accelerator.device)
    elif isinstance(text_encoder, torch.nn.Module):
        text_encoder.to(accelerator.device)

    # Constants
    # t0=0.001 代表极小的噪声水平/接近Clean数据。
    # 这意味着我们是在训练一个 "Late-Stage Refinement" 或 "Restoration" 任务，
    # 即 Learning to map Left(Clean) -> Right(Clean) directly via the transformer structure.
    t0 = torch.tensor([0.001], device=accelerator.device) 
    
    empty_prompt = ["This is a left-eye perspective video."] * args.batch_size
    
    # Pre-compute text embeddings to save optimization time
    print("Pre-computing text embeddings...")
    with torch.no_grad():
        empty_context_list = text_encoder(empty_prompt, accelerator.device) # Returns list of tensors
    
    global_step = 0
    print("Starting Training Loop...")
    
    for epoch in range(args.epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                left_vid = batch['left'].permute(0, 2, 1, 3, 4).to(accelerator.device) # [B, C, T, H, W]
                right_vid = batch['right'].permute(0, 2, 1, 3, 4).to(accelerator.device)
                
                # Check NaNs
                if torch.isnan(left_vid).any() or torch.isnan(right_vid).any():
                    print("NaN in input video, skipping")
                    continue
                
                # Encode Latents (VAE)
                with torch.no_grad():
                    # VAE Encode
                    # Use internal model.encode to match StereoPilot.py logic (with scale)
                    # Input to encode: [B, C, T, H, W]
                    z_left_list = []
                    z_right_list = []
                    
                    batch_size_actual = left_vid.shape[0]
                    for i in range(batch_size_actual):
                        l_in = left_vid[i].unsqueeze(0) # [1, C, T, H, W]
                        r_in = right_vid[i].unsqueeze(0)
                        
                        # Use model.encode with scale
                        # Returns Latents usually? Or distribution? StereoPilot.py implies direct use.
                        z_l = vae.model.encode(l_in, vae.scale)
                        z_r = vae.model.encode(r_in, vae.scale)
                        
                        if isinstance(z_l, list): # Handle if it returns list
                            z_l = z_l[0]
                        if isinstance(z_r, list):
                            z_r = z_r[0]
                            
                        z_left_list.append(z_l.squeeze(0))
                        z_right_list.append(z_r.squeeze(0))
                    
                    z_left = torch.stack(z_left_list) # [B, C, T, H_lat, W_lat]
                    z_right = torch.stack(z_right_list)
                
                # Forward Pass
                # Task: Left View -> Right View Reconstruction
                # t is fixed to nearly 0
                t = t0.repeat(args.batch_size)
                
                # x=z_left (Condition/Input), context=Text, domain_label=1 (Convergent/3D Movie)
                pred_right_list = model(
                    x=z_left,
                    t=t,
                    context=empty_context_list, # Reuse pre-computed
                    seq_len=81, # Should match latent seq len logic roughly? Or Max len? 
                    domain_label=1 
                )
                pred_right = torch.stack(pred_right_list)
                
                # Loss
                recon_loss = F.mse_loss(pred_right, z_right)
                
                loss = recon_loss
                
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
            if global_step % 10 == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch} Step {global_step} | Loss: {loss.item():.4f} (R: {recon_loss.item():.4f})")
                
            global_step += 1
            
        # Checkpointing
        if accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
            # 使用 unwrap_model 获取原始模型进行保存，避免DDP wrap wrapper问题
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train()
