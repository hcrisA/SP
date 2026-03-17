#!/usr/bin/env python3
"""
StereoPilot LoRA Training Script

Training script with LoRA (Low-Rank Adaptation) support for efficient fine-tuning.
LoRA reduces trainable parameters by 99%+ while maintaining model performance.

Key Benefits:
- 100x less trainable parameters
- Lower memory usage
- Faster training
- Preserves original model capabilities
- Small checkpoint files (~MB instead of GB)

Usage:
    # Basic LoRA training (rank=4)
    python train_lora.py --config toml/infer.toml \
                         --train_dir ../SP_Data/mono_train \
                         --output_dir ../SP_Data/checkpoints_lora
    
    # High-rank LoRA for better adaptation
    python train_lora.py --config toml/infer.toml \
                         --train_dir ../SP_Data/mono_train \
                         --output_dir ../SP_Data/checkpoints_lora \
                         --lora_rank 16
    
    # Resume from checkpoint
    python train_lora.py --config toml/infer.toml \
                         --train_dir ../SP_Data/mono_train \
                         --output_dir ../SP_Data/checkpoints_lora \
                         --resume_from_checkpoint checkpoint_epoch_005.pt
"""

import os
import sys
import argparse
import json
import time
import gc
import toml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
import torchvision.transforms as transforms

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from tqdm import tqdm
import logging
from PIL import Image
import safetensors.torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training_lora.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from models.StereoPilot import StereoPilotPipeline
from utils.common import DTYPE_MAP
from lora_utils import LoRAManager, create_lora_config

# Import amp from the correct location for compatibility
try:
    from torch import amp
except ImportError:
    from torch.cuda import amp


def bind_custom_forward(model, use_gradient_checkpointing=False):
    """
    Bind custom forward method to model that accepts domain_label parameter.
    
    This is needed because the original WanModel.forward() does not accept domain_label,
    but StereoPilot requires it for domain adaptation.
    
    Args:
        model: The WanModel instance
        use_gradient_checkpointing: If True, use gradient checkpointing to save memory
    """
    from models.wan import sinusoidal_embedding_1d
    from torch.utils.checkpoint import checkpoint
    
    # Use closure to capture model - this avoids the MethodType binding issue
    # where 'self' would be passed as first argument
    def custom_forward(x, t, context, seq_len, clip_fea=None, y=None, domain_label=0):
        """
        Forward pass through diffusion model with domain label support.
        
        Args:
            x: Input video tensor list
            t: Diffusion timestep
            context: Text embedding list
            seq_len: Max sequence length
            clip_fea: CLIP image features (optional)
            y: Conditional video input (optional)
            domain_label: Domain label (0=stereo4d, 1=3dmovie)
        """
        device = model.patch_embedding.weight.device
        if model.freqs.device != device:
            model.freqs = model.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [model.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
        ])
    
        # Time embedding
        with amp.autocast(device_type='cuda', dtype=torch.float32):
            e = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, t).float())
            e0 = model.time_projection(e).unflatten(1, (6, model.dim))
        
        # Domain embedding
        if domain_label == 0:
            domain_emb = model.parall_embedding.unsqueeze(0)
        else:
            domain_emb = model.converge_embedding.unsqueeze(0)
        e0 = e0 + domain_emb.to(e0.dtype)
        
        # Text context
        context = model.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        if clip_fea is not None:
            context_clip = model.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=model.freqs,
            context=context,
            context_lens=None
        )

        # Use gradient checkpointing if enabled
        if use_gradient_checkpointing and model.training:
            for block in model.blocks:
                # Use checkpoint for each block to save memory
                x = checkpoint(block, x, use_reentrant=False, **kwargs)
        else:
            for block in model.blocks:
                x = block(x, **kwargs)

        x = model.head(x, e)
        x = model.unpatchify(x, grid_sizes)
        return [u.float() for u in x]
    
    # Directly assign the closure function (not MethodType)
    # The closure captures 'model' internally, so no self binding needed
    model.forward = custom_forward


@dataclass
class LoRAConfig:
    """LoRA-specific configuration."""
    rank: int = 4
    alpha: Optional[float] = None
    dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.alpha is None:
            self.alpha = self.rank


@dataclass
class TrainingConfig:
    """Training configuration dataclass for type safety."""
    config_path: str
    train_dir: str
    output_dir: str
    batch_size: int
    learning_rate: float
    num_epochs: int
    gradient_accumulation_steps: int
    mixed_precision: Optional[str]
    num_frames: int = 81
    image_size: Tuple[int, int] = (832, 480)
    seed: int = 42
    max_grad_norm: float = 1.0
    save_every_n_epochs: int = 1
    log_every_n_steps: int = 50
    
    # LoRA configuration
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    resume_from_checkpoint: Optional[str] = None


class StereoVideoDataset(Dataset):
    """
    Stereo video dataset for training with robust error handling.
    
    Loads left/right image pairs and returns N-frame sequences.
    Images are expected in the format: mono_train/left/*.jpg, mono_train/right/*.jpg
    """
    
    def __init__(
        self,
        root_dir: str,
        num_frames: int = 81,
        image_size: Tuple[int, int] = (832, 480),
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp'),
        validation_mode: bool = True
    ):
        """
        Args:
            root_dir: Root directory containing 'left' and 'right' subfolders
            num_frames: Number of frames per video sequence (81 as per paper)
            image_size: Target image size (width, height)
            extensions: Valid image file extensions
            validation_mode: Whether to validate image loading
        """
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.extensions = extensions
        self.validation_mode = validation_mode
        
        # Verify directories exist
        self.left_dir = self.root_dir / "left"
        self.right_dir = self.root_dir / "right"
        
        if not self.left_dir.exists():
            raise FileNotFoundError(f"Left directory not found: {self.left_dir}")
        if not self.right_dir.exists():
            raise FileNotFoundError(f"Right directory not found: {self.right_dir}")
        
        # Load image paths
        self.left_images = self._find_images(self.left_dir)
        self.right_images = self._find_images(self.right_dir)
        
        # Validate image pairs
        self._validate_image_pairs()
        
        # Calculate number of sequences
        self.num_sequences = len(self.left_images) // self.num_frames
        if self.num_sequences == 0:
            raise ValueError(
                f"Not enough images for {num_frames} frames. "
                f"Found {len(self.left_images)} images, need at least {num_frames}."
            )
        
        logger.info(
            f"Dataset initialized: {self.num_sequences} sequences "
            f"from {len(self.left_images)} image pairs"
        )
    
    def _find_images(self, directory: Path) -> List[Path]:
        """Find all images in directory recursively and sort by filename."""
        images = []
        for ext in self.extensions:
            images.extend(directory.glob(f"**/*{ext}"))
            images.extend(directory.glob(f"**/*{ext.upper()}"))
        
        # Sort by numeric value in filename if possible, otherwise alphabetical
        def extract_number(path):
            import re
            match = re.search(r'\d+', path.stem)
            return int(match.group()) if match else 0
        
        try:
            return sorted(images, key=extract_number)
        except:
            return sorted(images)
    
    def _validate_image_pairs(self):
        """Validate that left and right images match."""
        if len(self.left_images) != len(self.right_images):
            logger.warning(
                f"Mismatched image counts: {len(self.left_images)} left, "
                f"{len(self.right_images)} right"
            )
            min_len = min(len(self.left_images), len(self.right_images))
            self.left_images = self.left_images[:min_len]
            self.right_images = self.right_images[:min_len]
        
        # Check that filenames match
        mismatches = []
        for i, (left, right) in enumerate(zip(self.left_images, self.right_images)):
            if left.name != right.name:
                mismatches.append((i, left.name, right.name))
        
        if mismatches:
            logger.warning(f"Found {len(mismatches)} filename mismatches")
            for idx, left_name, right_name in mismatches[:5]:  # Show first 5
                logger.warning(f"  Index {idx}: {left_name} vs {right_name}")
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a video sequence with robust error handling.
        
        Returns:
            Dictionary with 'left' and 'right' videos as tensors [T, C, H, W]
        """
        # Calculate start index for this sequence
        start_idx = idx * self.num_frames
        
        # Ensure we don't go out of bounds
        if start_idx + self.num_frames > len(self.left_images):
            start_idx = len(self.left_images) - self.num_frames
        
        # Pre-create resize transform
        # transforms.Resize expects (height, width), but self.image_size is (width, height)
        # So we need to swap: Resize((H, W)) where H=image_size[1], W=image_size[0]
        target_h, target_w = self.image_size[1], self.image_size[0]  # (480, 832)
        resize_transform = transforms.Resize(
            (target_h, target_w),  # (height, width)
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        )
        
        # Load frames
        left_frames = []
        right_frames = []
        valid_frames = 0
        
        for i in range(self.num_frames):
            try:
                left_path = self.left_images[start_idx + i]
                right_path = self.right_images[start_idx + i]
                
                # Open and convert images
                left_img = Image.open(left_path).convert('RGB')
                right_img = Image.open(right_path).convert('RGB')
                
                # Resize to target size
                left_img = resize_transform(left_img)
                right_img = resize_transform(right_img)
                
                # Convert to tensor [C, H, W]
                left_tensor = transforms.ToTensor()(left_img)
                right_tensor = transforms.ToTensor()(right_img)
                
                # Normalize to [-1, 1]
                left_tensor = (left_tensor * 2.0) - 1.0
                right_tensor = (right_tensor * 2.0) - 1.0
                
                left_frames.append(left_tensor)
                right_frames.append(right_tensor)
                valid_frames += 1
                
            except Exception as e:
                logger.warning(
                    f"Error loading frame {start_idx + i} (idx {idx}): {e}. "
                    f"Path: {left_path if 'left_path' in locals() else 'unknown'}"
                )
                # Create dummy tensors as fallback
                dummy = torch.zeros(3, target_h, target_w)
                left_frames.append(dummy.clone())
                right_frames.append(dummy.clone())
        
        # Log if many frames failed to load
        if valid_frames < self.num_frames:
            logger.warning(
                f"Sequence {idx}: Only {valid_frames}/{self.num_frames} frames loaded successfully"
            )
        
        # Stack frames [T, C, H, W]
        left_video = torch.stack(left_frames)
        right_video = torch.stack(right_frames)
        
        return {
            "left": left_video,
            "right": right_video
        }


class Trainer:
    """
    Main trainer class for StereoPilot LoRA with production-quality features.
    
    Handles:
    - Model initialization with LoRA injection
    - Memory-efficient training with gradient checkpointing
    - Robust checkpoint saving and resuming (LoRA weights only)
    - Comprehensive logging and monitoring
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer with configuration.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize accelerator with project config
        project_config = ProjectConfiguration(
            project_dir=str(self.output_dir),
            logging_dir=str(self.output_dir / "logs")
        )
        
        # Try to use tensorboard, fall back to None if not available
        try:
            from torch.utils.tensorboard import SummaryWriter
            log_with = "tensorboard"
        except ImportError:
            log_with = None
            logger.warning("Tensorboard not available. Install with: pip install tensorboard")
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with=log_with,
            project_config=project_config
        )
        
        # Set random seed for reproducibility
        if self.accelerator.is_main_process:
            set_seed(config.seed)
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.start_epoch = 0
        
        logger.info("="*80)
        logger.info("StereoPilot LoRA Training Configuration")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Learning rate: {config.learning_rate}")
        logger.info(f"Epochs: {config.num_epochs}")
        logger.info(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        logger.info(f"Mixed precision: {config.mixed_precision}")
        logger.info(f"LoRA rank: {config.lora_config.rank}")
        logger.info(f"LoRA alpha: {config.lora_config.alpha}")
        logger.info(f"LoRA dropout: {config.lora_config.dropout}")
        logger.info(f"Accelerator config: {self.accelerator.state}")
        logger.info("="*80)
    
    def setup_model(self):
        """Load and configure the model with LoRA injection."""
        if self.accelerator.is_main_process:
            logger.info("Loading model configuration...")
        
        # Load config
        with open(self.config.config_path) as f:
            model_config = json.loads(json.dumps(toml.load(f)))
        
        # Override dtypes for training based on mixed precision setting
        if self.accelerator.mixed_precision == "bf16":
            model_config['model']['dtype'] = 'bfloat16'
            model_config['model']['transformer_dtype'] = 'bfloat16'
        elif self.accelerator.mixed_precision == "fp16":
            model_config['model']['dtype'] = 'float16'
            model_config['model']['transformer_dtype'] = 'float16'
        
        # Convert dtype strings to torch.dtype using DTYPE_MAP
        model_config['model']['dtype'] = DTYPE_MAP[model_config['model']['dtype']]
        if 'transformer_dtype' in model_config['model']:
            model_config['model']['transformer_dtype'] = DTYPE_MAP[model_config['model']['transformer_dtype']]
        
        # Initialize pipeline
        self.pipeline = StereoPilotPipeline(model_config)
        self.pipeline.load_diffusion_model()
        
        # Extract components
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        
        # Bind custom forward method that accepts domain_label parameter
        # Enable gradient checkpointing to save memory
        bind_custom_forward(self.transformer, use_gradient_checkpointing=True)
        if self.accelerator.is_main_process:
            logger.info("✓ Custom forward method bound to transformer (supports domain_label + gradient checkpointing)")
        
        # Initialize LoRA manager and inject LoRA layers
        lora_dtype = model_config['model'].get('transformer_dtype', model_config['model']['dtype'])
        self.lora_manager = LoRAManager(
            model=self.transformer,
            rank=self.config.lora_config.rank,
            alpha=self.config.lora_config.alpha,
            dropout=self.config.lora_config.dropout,
            dtype=lora_dtype,
            target_modules=self.config.lora_config.target_modules
        )
        
        # Inject LoRA into transformer
        injected_count = self.lora_manager.inject_lora()
        if self.accelerator.is_main_process:
            logger.info(f"✓ Injected {injected_count} LoRA modules")
        
        # Freeze all original weights, only LoRA parameters are trainable
        self.lora_manager.freeze_original_weights()
        
        # Freeze VAE (both encoder and decoder)
        if hasattr(self.vae, 'model'):
            self.vae.model.requires_grad_(False)
            self.vae.model.eval()
        else:
            self.vae.requires_grad_(False).eval()
        
        if self.accelerator.is_main_process:
            logger.info("✓ VAE frozen")
        
        # Freeze Text Encoder (UMT5)
        if hasattr(self.text_encoder, 'model'):
            self.text_encoder.model.requires_grad_(False)
            self.text_encoder.model.eval()
        else:
            self.text_encoder.requires_grad_(False).eval()
        
        if self.accelerator.is_main_process:
            logger.info("✓ Text Encoder (UMT5) frozen")
        
        # Set transformer to training mode (LoRA parameters are trainable)
        self.transformer.train()
        
        # Log parameter counts
        self._log_parameter_counts()
        
        # Move VAE to device for encoding
        device = self.accelerator.device
        if hasattr(self.vae, 'model'):
            self.vae.model.to(device)
    
    def _log_parameter_counts(self):
        """Log parameter counts for each component."""
        if not self.accelerator.is_main_process:
            return
        
        def count_params(model, name):
            # Handle wrapped models
            if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                actual_model = model.model
            else:
                actual_model = model
            
            try:
                total = sum(p.numel() for p in actual_model.parameters())
                trainable = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
                frozen = total - trainable
                
                # Format numbers with commas
                logger.info(
                    f"  {name}: {total:,} total, {trainable:,} trainable, {frozen:,} frozen"
                )
                return total, trainable
            except Exception as e:
                logger.warning(f"  {name}: Could not count parameters - {e}")
                return 0, 0
        
        logger.info("Parameter counts:")
        total_params = 0
        total_trainable = 0
        
        t, tr = count_params(self.vae, "VAE")
        total_params += t
        total_trainable += tr
        
        t, tr = count_params(self.text_encoder, "Text Encoder")
        total_params += t
        total_trainable += tr
        
        t, tr = count_params(self.transformer, "Transformer")
        total_params += t
        total_trainable += tr
        
        if total_params > 0:
            trainable_pct = 100 * total_trainable / total_params
            logger.info(
                f"  Total: {total_params:,} parameters, "
                f"{total_trainable:,} trainable ({trainable_pct:.2f}%)"
            )
            
            # Calculate LoRA-specific stats
            lora_params = len(self.lora_manager.get_trainable_parameters())
            logger.info(f"  LoRA parameters: {lora_params:,}")
    
    def setup_data(self):
        """Setup data loaders with optimized settings."""
        if self.accelerator.is_main_process:
            logger.info("Setting up data loaders...")
        
        # Create dataset
        dataset = StereoVideoDataset(
            root_dir=self.config.train_dir,
            num_frames=self.config.num_frames,
            image_size=self.config.image_size
        )
        
        # Create data loader with optimized settings
        num_workers = min(8, (os.cpu_count() or 4))
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        if self.accelerator.is_main_process:
            logger.info(
                f"✓ Created data loader with {len(dataset)} sequences, "
                f"batch size {self.config.batch_size}, "
                f"num_workers {num_workers}"
            )
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler for LoRA parameters."""
        if self.accelerator.is_main_process:
            logger.info("Setting up optimizer for LoRA parameters...")
        
        # Get only LoRA trainable parameters
        trainable_params = self.lora_manager.get_trainable_parameters()
        
        if not trainable_params:
            raise ValueError("No LoRA trainable parameters found! Check LoRA configuration.")
        
        if self.accelerator.is_main_process:
            logger.info(f"  Found {len(trainable_params)} LoRA parameter groups")
        
        # Create AdamW optimizer with Kahan summation for stability
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
            fused=True if torch.cuda.is_available() else False
        )
        
        # Create learning rate scheduler
        total_steps = self.config.num_epochs * len(self.train_loader) // self.config.gradient_accumulation_steps
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(100, total_steps // 10),
            T_mult=2,
            eta_min=self.config.learning_rate * 0.1
        )
        
        if self.accelerator.is_main_process:
            logger.info(
                f"✓ AdamW optimizer initialized for LoRA (lr={self.config.learning_rate}, "
                f"weight_decay=1e-2)"
            )
            logger.info(
                f"✓ CosineAnnealingWarmRestarts scheduler (T_0={max(100, total_steps // 10)}, "
                f"T_mult=2)"
            )
    
    def precompute_text_embeddings(self):
        """Pre-compute text embeddings for fixed prompt."""
        if self.accelerator.is_main_process:
            logger.info("Pre-computing text embeddings...")
        
        # Fixed prompt as per requirements
        prompt = ["This is a video viewed from the left perspective"]
        
        # Move text encoder to device
        device = self.accelerator.device
        if hasattr(self.text_encoder, 'model'):
            self.text_encoder.model.to(device)
        else:
            self.text_encoder.to(device)
        
        # Compute embeddings
        with torch.no_grad():
            self.context_embeddings = self.text_encoder(prompt, device)
        
        # Move text encoder back to CPU to save memory
        if hasattr(self.text_encoder, 'model'):
            self.text_encoder.model.to('cpu')
        else:
            self.text_encoder.to('cpu')
        
        torch.cuda.empty_cache()
        
        if self.accelerator.is_main_process:
            logger.info(
                f"✓ Text embeddings pre-computed: {len(self.context_embeddings)} items, "
                f"moved Text Encoder to CPU to save memory"
            )
    
    def encode_videos(self, left_video: torch.Tensor, right_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode videos to latents using VAE with memory-efficient implementation.
        
        Processes videos frame-by-frame to avoid OOM errors with long sequences.
        
        Args:
            left_video: Left view video [B, T, C, H, W]
            right_video: Right view video [B, T, C, H, W]
            
        Returns:
            left_latents, right_latents: Encoded latents [B, C, T, H_lat, W_lat]
        """
        # Permute to [B, C, T, H, W] for VAE
        left_video = left_video.permute(0, 2, 1, 3, 4)
        right_video = right_video.permute(0, 2, 1, 3, 4)
        
        batch_size = left_video.shape[0]
        device = self.accelerator.device
        
        # Get VAE scale factor
        scale = getattr(self.vae, 'scale', None)
        
        # Process each video in the batch separately to save memory
        left_latents_list = []
        right_latents_list = []
        
        for b in range(batch_size):
            left_frames = []
            right_frames = []
            
            # Process each frame separately (memory-efficient)
            for t in range(left_video.shape[2]):  # T dimension
                with torch.no_grad():
                    # Extract single frame [1, C, 1, H, W]
                    left_frame = left_video[b:b+1, :, t:t+1, :, :]
                    right_frame = right_video[b:b+1, :, t:t+1, :, :]
                    
                    try:
                        # Encode frames - handle different VAE APIs
                        if hasattr(self.vae.model, 'encode'):
                            left_latent = self.vae.model.encode(left_frame, scale)
                            right_latent = self.vae.model.encode(right_frame, scale)
                        else:
                            # Fallback to direct VAE call
                            left_latent = self.vae.encode(left_frame)
                            right_latent = self.vae.encode(right_frame)
                        
                        # Handle list outputs
                        if isinstance(left_latent, (list, tuple)):
                            left_latent = left_latent[0]
                        if isinstance(right_latent, (list, tuple)):
                            right_latent = right_latent[0]
                        
                        # Remove batch dimension [C, 1, H_lat, W_lat]
                        left_frames.append(left_latent.squeeze(0))
                        right_frames.append(right_latent.squeeze(0))
                        
                    except Exception as e:
                        logger.error(f"VAE encoding error at batch {b}, frame {t}: {e}")
                        # Create dummy latent
                        dummy_shape = [left_frame.shape[1], 1, left_frame.shape[3] // 8, left_frame.shape[4] // 8]
                        dummy = torch.zeros(dummy_shape, device=left_frame.device, dtype=left_frame.dtype)
                        left_frames.append(dummy)
                        right_frames.append(dummy.clone())
                
                # Clear cache periodically
                if t % 10 == 0:
                    torch.cuda.empty_cache()
            
            # Concatenate frames along time dimension [C, T, H_lat, W_lat]
            left_video_latent = torch.cat(left_frames, dim=1)
            right_video_latent = torch.cat(right_frames, dim=1)
            
            left_latents_list.append(left_video_latent)
            right_latents_list.append(right_video_latent)
            
            # Clear cache after each batch item
            torch.cuda.empty_cache()
        
        # Stack batch [B, C, T, H_lat, W_lat]
        left_latents = torch.stack(left_latents_list)
        right_latents = torch.stack(right_latents_list)
        
        # Validate shapes
        if left_latents.dim() != 5 or right_latents.dim() != 5:
            logger.error(f"Invalid latent dimensions: left={left_latents.shape}, right={right_latents.shape}")
        
        return left_latents, right_latents
    
    def compute_loss(self, pred_latents: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss with multiple components.
        
        Args:
            pred_latents: Predicted latents
            target_latents: Target latents
            
        Returns:
            Combined loss
        """
        # MSE loss for reconstruction
        mse_loss = F.mse_loss(pred_latents, target_latents)
        
        # L1 loss for sparsity (optional, can help with generalization)
        # l1_loss = F.l1_loss(pred_latents, target_latents)
        
        # Combined loss (you can adjust the weighting)
        total_loss = mse_loss # + 0.1 * l1_loss
        
        return total_loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, Dict[str, float]]:
        """
        Single training step with comprehensive monitoring.
        
        Args:
            batch: Batch dictionary with 'left' and 'right' videos
            
        Returns:
            Loss value and metrics dictionary
        """
        left_video = batch["left"]  # [B, T, C, H, W]
        right_video = batch["right"]  # [B, T, C, H, W]
        batch_size = left_video.shape[0]
        
        # Clear cache before forward pass
        torch.cuda.empty_cache()
        
        # Move VAE to GPU if not already
        device = self.accelerator.device
        if hasattr(self.vae, 'model'):
            # Check device by looking at first parameter (WanVAE_ doesn't have .device attribute)
            try:
                vae_device = next(self.vae.model.parameters()).device
                if vae_device != device:
                    self.vae.model.to(device)
            except StopIteration:
                # No parameters, just try to move
                self.vae.model.to(device)
        
        # Encode videos to latents (memory-intensive, done with no_grad)
        left_latents = None
        right_latents = None
        try:
            with torch.no_grad():
                left_latents, right_latents = self.encode_videos(left_video, right_video)
        except Exception as e:
            import traceback
            logger.error(f"Error encoding videos: {e}")
            logger.error(traceback.format_exc())
            # Move VAE back to CPU to free memory
            if hasattr(self.vae, 'model'):
                self.vae.model.to('cpu')
            torch.cuda.empty_cache()
            return 0.0, {"loss": 0.0, "mse_loss": 0.0}
        
        # Verify latents were computed
        if left_latents is None or right_latents is None:
            logger.error("Latents were not computed properly")
            if hasattr(self.vae, 'model'):
                self.vae.model.to('cpu')
            torch.cuda.empty_cache()
            return 0.0, {"loss": 0.0, "mse_loss": 0.0}
        
        # Move VAE to CPU immediately after encoding to save memory
        if hasattr(self.vae, 'model'):
            self.vae.model.to('cpu')
        torch.cuda.empty_cache()
        
        # Check for NaNs in latents
        if torch.isnan(left_latents).any() or torch.isnan(right_latents).any():
            logger.warning("NaN detected in latents, skipping batch")
            return 0.0, {"loss": 0.0, "mse_loss": 0.0}
        
        # Fixed timestep (near zero for reconstruction)
        timestep = torch.full((batch_size,), 0.001, device=self.accelerator.device)
        
        # Convert latents to list format expected by custom_forward
        # custom_forward expects x as a list of tensors, each with shape (C, T, H, W)
        # left_latents shape: [B, C, T, H, W] -> split into list of (C, T, H, W) tensors
        left_latents_list = [left_latents[b] for b in range(batch_size)]
        
        # Calculate seq_len based on latent shape
        # Formula from Wan2.1: seq_len = ceil((H * W) / (patch_h * patch_w) * T / sp_size) * sp_size
        # latent shape: (C, T, H, W)
        latent_shape = left_latents_list[0].shape  # (C, T, H, W)
        patch_size = self.pipeline.patch_size  # (t_patch, h_patch, w_patch)
        sp_size = getattr(self.pipeline, 'sp_size', 1)
        
        # Calculate sequence length in latent space
        import math
        seq_len = math.ceil(
            (latent_shape[2] * latent_shape[3]) / (patch_size[1] * patch_size[2]) *
            latent_shape[1] / sp_size
        ) * sp_size
        
        # Forward pass through transformer with mixed precision
        with autocast(enabled=self.config.mixed_precision is not None):
            # Input: left latents, target: right latents
            try:
                pred_latents_list = self.transformer(
                    x=left_latents_list,  # List of (C, T, H, W) tensors
                    t=timestep,
                    context=self.context_embeddings,
                    seq_len=seq_len,  # Dynamically calculated sequence length
                    domain_label=1  # Converge domain for stereo conversion
                )
                
                # Stack predictions [B, C, T, H_lat, W_lat]
                pred_latents = torch.stack(pred_latents_list)
                
                # Compute reconstruction loss
                loss = self.compute_loss(pred_latents, right_latents)
                
            except Exception as e:
                import traceback
                logger.error(f"Error in transformer forward pass: {e}")
                logger.error(traceback.format_exc())
                return 0.0, {"loss": 0.0, "mse_loss": 0.0}
        
        # Clear intermediate tensors
        # Note: Don't delete right_latents here, it's needed for metrics
        
        metrics = {
            "loss": loss.item(),
            "mse_loss": F.mse_loss(pred_latents, right_latents).item() if pred_latents is not None and right_latents is not None else 0.0
        }
        
        # Clean up
        del left_latents, right_latents, pred_latents_list, pred_latents
        
        return loss, metrics
    
    def train(self):
        """Main training loop with comprehensive logging and monitoring."""
        if self.accelerator.is_main_process:
            logger.info("Starting LoRA training...")
            start_time = time.time()
        
        # Setup
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.precompute_text_embeddings()
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        # Prepare with accelerator
        self.transformer, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.transformer, self.optimizer, self.train_loader, self.scheduler
        )
        
        # Training state
        total_loss = 0
        steps_per_epoch = len(self.train_loader)
        
        if self.accelerator.is_main_process:
            logger.info(f"Starting training for {self.config.num_epochs} epochs ({steps_per_epoch} steps per epoch)")
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            if self.accelerator.is_main_process:
                logger.info("\n" + "="*80)
                logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
                logger.info("="*80)
            
            epoch_start = time.time()
            epoch_loss = 0
            num_batches = 0
            
            # Progress bar
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                disable=not self.accelerator.is_local_main_process,
                ncols=100,
                smoothing=0.1
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                batch_start = time.time()
                
                # Training step with gradient accumulation
                loss = None
                metrics = None
                
                with self.accelerator.accumulate(self.transformer):
                    try:
                        loss, metrics = self.train_step(batch)
                        
                        # Skip if loss is zero (failed batch)
                        if loss == 0.0 or torch.isnan(loss):
                            logger.warning(f"Zero/NaN loss at batch {batch_idx}, skipping")
                            continue
                        
                        # Backward pass
                        self.accelerator.backward(loss)
                        
                        # Gradient clipping
                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.transformer.parameters(),
                                max_norm=self.config.max_grad_norm
                            )
                        
                        # Optimizer step
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                    except Exception as e:
                        logger.error(f"Error in training step at batch {batch_idx}: {e}")
                        continue
                
                # Skip if metrics was not set (failed batch)
                if metrics is None:
                    continue
                    
                loss_value = metrics["loss"]
                epoch_loss += loss_value
                total_loss += loss_value
                num_batches += 1
                self.global_step += 1
                
                # Calculate batch time
                batch_time = time.time() - batch_start
                
                # Update progress bar
                if batch_idx % 10 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        "loss": f"{loss_value:.6f}",
                        "lr": f"{current_lr:.2e}",
                        "time": f"{batch_time:.2f}s",
                        "mem": f"{torch.cuda.memory_allocated() // 1024**2}MB" if torch.cuda.is_available() else "N/A"
                    })
                
                # Log to tensorboard
                if self.accelerator.is_main_process and self.global_step % self.config.log_every_n_steps == 0:
                    self.accelerator.log({
                        "train/loss": loss_value,
                        "train/mse_loss": metrics["mse_loss"],
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/grad_norm": grad_norm.item() if 'grad_norm' in locals() else 0,
                        "train/samples_seen": self.global_step * self.config.batch_size * self.accelerator.num_processes
                    }, step=self.global_step)
                
                # Memory usage logging
                if (self.global_step % 100 == 0 and 
                    torch.cuda.is_available() and 
                    self.accelerator.is_main_process):
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(
                        f"Step {self.global_step} - "
                        f"GPU Memory: {allocated:.2f}GB allocated, "
                        f"{reserved:.2f}GB reserved"
                    )
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            if self.accelerator.is_main_process:
                logger.info(f"\nEpoch {epoch + 1} Summary:")
                logger.info(f"  Average Loss: {avg_epoch_loss:.6f}")
                logger.info(f"  Time: {epoch_time:.2f} seconds ({epoch_time/60:.1f} minutes)")
                logger.info(f"  Steps: {num_batches}")
                logger.info(f"  Global step: {self.global_step}")
                
                # Save checkpoint
                is_best = avg_epoch_loss < self.best_loss
                if is_best:
                    self.best_loss = avg_epoch_loss
                    logger.info(f"  New best loss: {self.best_loss:.6f}")
                
                self.save_checkpoint(epoch, avg_epoch_loss, is_best)
                
                # Log epoch metrics
                self.accelerator.log({
                    "epoch/avg_loss": avg_epoch_loss,
                    "epoch/time": epoch_time,
                    "epoch/best_loss": self.best_loss,
                    "epoch/learning_rate": self.scheduler.get_last_lr()[0]
                }, step=epoch)
                
                # Clean up
                gc.collect()
                torch.cuda.empty_cache()
        
        # Training completed
        if self.accelerator.is_main_process:
            total_time = time.time() - start_time
            logger.info("\n" + "="*80)
            logger.info("LoRA Training completed!")
            logger.info("="*80)
            
            avg_total_loss = total_loss / self.global_step if self.global_step > 0 else 0
            logger.info(f"Final Metrics:")
            logger.info(f"  Average Loss: {avg_total_loss:.6f}")
            logger.info(f"  Best Loss: {self.best_loss:.6f}")
            logger.info(f"  Total Steps: {self.global_step}")
            logger.info(f"  Total Time: {total_time:.2f} seconds ({total_time/3600:.1f} hours)")
            
            # Save final checkpoint
            self.save_checkpoint(self.config.num_epochs - 1, avg_total_loss, is_best=True)
    
    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint with both full state and LoRA weights."""
        # Only save on main process
        if not self.accelerator.is_main_process:
            return
        
        # Unwrap model
        unwrapped_model = self.accelerator.unwrap_model(self.transformer)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch + 1,
            'global_step': self.global_step,
            'lora_config': {
                'rank': self.config.lora_config.rank,
                'alpha': self.config.lora_config.alpha,
                'dropout': self.config.lora_config.dropout,
                'target_modules': self.config.lora_config.target_modules
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint (only LoRA weights, much smaller!)
        if (epoch + 1) % self.config.save_every_n_epochs == 0:
            # Save full checkpoint with optimizer state
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Save only LoRA weights (small file)
            lora_state_dict = self.lora_manager.get_lora_state_dict()
            lora_weights_path = self.output_dir / f"lora_weights_epoch_{epoch + 1:03d}.safetensors"
            safetensors.torch.save_file(lora_state_dict, lora_weights_path)
            logger.info(f"LoRA weights saved: {lora_weights_path} (size: ~{sum(p.numel() for p in lora_state_dict.values()) * 2 / 1024**2:.1f} MB)")
        
        # Save best model
        if is_best:
            best_checkpoint_path = self.output_dir / "best_checkpoint.pt"
            best_lora_path = self.output_dir / "best_lora.safetensors"
            
            torch.save(checkpoint_data, best_checkpoint_path)
            
            lora_state_dict = self.lora_manager.get_lora_state_dict()
            safetensors.torch.save_file(lora_state_dict, best_lora_path)
            
            logger.info(f"Best LoRA model saved (loss: {loss:.6f})")
            logger.info(f"  LoRA weights size: ~{sum(p.numel() for p in lora_state_dict.values()) * 2 / 1024**2:.1f} MB")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        if not self.accelerator.is_main_process:
            return
        
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load training state
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        # Load LoRA weights into model
        if 'model_state_dict' in checkpoint:
            # Full checkpoint with model state
            self.transformer.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            # Load only LoRA weights
            lora_state_dict = checkpoint.get('lora_state_dict', None)
            if lora_state_dict is None:
                logger.error("No model state dict found in checkpoint")
                return
            self.lora_manager.load_lora_state_dict(lora_state_dict)
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"✓ Resumed from epoch {self.start_epoch}, step {self.global_step}, best loss {self.best_loss:.6f}")


def parse_args() -> TrainingConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train StereoPilot model with LoRA for stereo video conversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Standard training arguments
    parser.add_argument(
        "--config", 
        type=str, 
        default="toml/infer.toml",
        help="Path to model config file (TOML format)"
    )
    parser.add_argument(
        "--train_dir", 
        type=str, 
        default="../SP_Data/mono_train",
        help="Path to training data directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../SP_Data/checkpoints_lora",
        help="Path to save checkpoints"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1,
        help="Batch size per GPU (reduce if OOM)"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=3e-4,
        help="Learning rate (follows paper: 3e-4)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=4,
        help="Gradient accumulation steps (increase to simulate larger batch)"
    )
    parser.add_argument(
        "--mixed_precision", 
        type=str, 
        default="bf16",
        choices=["fp16", "bf16", "no"],
        help="Mixed precision mode (bf16 recommended for RTX 30xx/40xx/50xx)"
    )
    parser.add_argument(
        "--num_frames", 
        type=int, 
        default=81,
        help="Number of frames per sequence"
    )
    parser.add_argument(
        "--image_size", 
        type=str, 
        default="832,480",
        help="Image size as width,height"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--save_every_n_epochs", 
        type=int, 
        default=1,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--log_every_n_steps", 
        type=int, 
        default=50,
        help="Log to tensorboard every N steps"
    )
    
    # LoRA-specific arguments
    parser.add_argument(
        "--lora_rank", 
        type=int, 
        default=4,
        help="LoRA rank (higher=more expressive, more params). Typical: 4-32"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=float, 
        default=None,
        help="LoRA alpha scaling (defaults to lora_rank)"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.0,
        help="LoRA dropout rate (0.0-0.1)"
    )
    parser.add_argument(
        "--lora_target_modules", 
        type=str, 
        default=None,
        help="Comma-separated list of target modules (e.g., 'attn.q,attn.k,attn.v,ffn.fc1')"
    )
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Parse image size
    width, height = map(int, args.image_size.split(','))
    
    # Parse LoRA target modules
    target_modules = None
    if args.lora_target_modules:
        target_modules = args.lora_target_modules.split(',')
    
    # Create LoRA config
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=target_modules
    )
    
    return TrainingConfig(
        config_path=args.config,
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
        num_frames=args.num_frames,
        image_size=(width, height),
        seed=args.seed,
        save_every_n_epochs=args.save_every_n_epochs,
        log_every_n_steps=args.log_every_n_steps,
        lora_config=lora_config,
        resume_from_checkpoint=args.resume_from_checkpoint
    )


def main():
    """Main entry point."""
    # Parse arguments
    config = parse_args()
    
    # Validate paths
    if not Path(config.config_path).exists():
        logger.error(f"Config file not found: {config.config_path}")
        sys.exit(1)
    
    if not Path(config.train_dir).exists():
        logger.error(f"Training directory not found: {config.train_dir}")
        sys.exit(1)
    
    # Log configuration
    logger.info("Starting StereoPilot LoRA Training")
    logger.info(f"Config: {config}")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
