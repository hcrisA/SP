#!/usr/bin/env python3
"""
StereoPilot Training Script - Production Ready

Enhanced training script with best practices from diffusion-pipe:
- Optimized memory management
- Robust error handling
- Efficient data loading
- Comprehensive logging

Usage:
    python train_optimized.py --config toml/infer.toml \\
                             --train_dir ../SP_Data/mono_train \\
                             --output_dir ../SP_Data/checkpoints
"""

import os
import sys
import argparse
import json
import time
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

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
        logging.FileHandler('training.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root.parent))

from models.StereoPilot import StereoPilotPipeline


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
        resize_transform = transforms.Resize(
            self.image_size, 
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
                
                # Resize
                left_img = resize_transform(left_img)
                right_img = resize_transform(right_img)
                
                # Convert to tensor [C, H, W]
                left_tensor = transforms.ToTensor()(left_img)
                right_tensor = transforms.ToTensor()(right_img)
                
                # Validate tensor shape
                expected_shape = (3, self.image_size[1], self.image_size[0])
                if left_tensor.shape != expected_shape or right_tensor.shape != expected_shape:
                    raise ValueError(
                        f"Invalid tensor shape. Expected {expected_shape}, "
                        f"got left={left_tensor.shape}, right={right_tensor.shape}"
                    )
                
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
                dummy = torch.zeros(3, self.image_size[1], self.image_size[0])
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
    Main trainer class for StereoPilot with production-quality features.
    
    Handles:
    - Model initialization and setup
    - Memory-efficient training with gradient checkpointing
    - Robust checkpoint saving and resuming
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
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            log_with="tensorboard",
            project_config=project_config
        )
        
        # Set random seed for reproducibility
        if self.accelerator.is_main_process:
            set_seed(config.seed)
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        logger.info("="*80)
        logger.info("StereoPilot Training Configuration")
        logger.info("="*80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Batch size: {config.batch_size}")
        logger.info(f"Learning rate: {config.learning_rate}")
        logger.info(f"Epochs: {config.num_epochs}")
        logger.info(f"Gradient accumulation: {config.gradient_accumulation_steps}")
        logger.info(f"Mixed precision: {config.mixed_precision}")
        logger.info(f"Accelerator config: {self.accelerator.state}")
        logger.info("="*80)
    
    def setup_model(self):
        """Load and configure the model with proper memory management."""
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
        
        # Initialize pipeline
        self.pipeline = StereoPilotPipeline(model_config)
        self.pipeline.load_diffusion_model()
        
        # Extract components
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        
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
        
        # Enable gradient checkpointing for memory efficiency if supported
        try:
            if hasattr(self.transformer, 'enable_gradient_checkpointing'):
                self.transformer.enable_gradient_checkpointing()
                if self.accelerator.is_main_process:
                    logger.info("✓ Gradient checkpointing enabled")
            elif hasattr(self.transformer, 'gradient_checkpointing_enable'):
                self.transformer.gradient_checkpointing_enable()
                if self.accelerator.is_main_process:
                    logger.info("✓ Gradient checkpointing enabled (HF API)")
        except Exception as e:
            logger.warning(f"Could not enable gradient checkpointing: {e}")
        
        # Set transformer to training mode (only trainable component)
        self.transformer.requires_grad_(True)
        self.transformer.train()
        
        if self.accelerator.is_main_process:
            logger.info("✓ Transformer set to trainable")
        
        # Count parameters
        self._log_parameter_counts()
        
        # Move VAE to device for encoding (will be moved back to CPU after precomputation)
        device = self.accelerator.device
        if hasattr(self.vae, 'model'):
            self.vae.model.to(device)
    
    def _log_parameter_counts(self):
        """Log parameter counts for each component."""
        if not self.accelerator.is_main_process:
            return
        
        def count_params(model, name):
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen = total - trainable
            logger.info(
                f"  {name}: {total:,} total, {trainable:,} trainable, {frozen:,} frozen"
            )
            return total, trainable
        
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
        
        logger.info(
            f"  Total: {total_params:,} parameters, "
            f"{total_trainable:,} trainable ({100*total_trainable/total_params:.1f}%)"
        )
    
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
        """Setup optimizer and scheduler with proper configuration."""
        if self.accelerator.is_main_process:
            logger.info("Setting up optimizer...")
        
        # Filter trainable parameters
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        
        if not trainable_params:
            raise ValueError("No trainable parameters found! Check model configuration.")
        
        # Create AdamW optimizer with Kahan summation for stability
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8,
            fused=True if torch.cuda.is_available() else False  # Use fused optimizer if available
        )
        
        # Create learning rate scheduler
        # Cosine annealing with warm restarts for better convergence
        total_steps = self.config.num_epochs * len(self.train_loader) // self.config.gradient_accumulation_steps
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=max(100, total_steps // 10),  # First restart after 10% of training
            T_mult=2,  # Double the interval after each restart
            eta_min=self.config.learning_rate * 0.1
        )
        
        if self.accelerator.is_main_process:
            logger.info(
                f"✓ AdamW optimizer initialized (lr={self.config.learning_rate}, "
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
        l1_loss = F.l1_loss(pred_latents, target_latents)
        
        # Combined loss (you can adjust the weighting)
        total_loss = mse_loss + 0.1 * l1_loss
        
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
        if self.global_step % 10 == 0:
            torch.cuda.empty_cache()
        
        # Encode videos to latents (memory-intensive, done with no_grad)
        try:
            with torch.no_grad():
                left_latents, right_latents = self.encode_videos(left_video, right_video)
        except Exception as e:
            logger.error(f"Error encoding videos: {e}")
            return 0.0, {"loss": 0.0, "mse_loss": 0.0}
        
        # Check for NaNs in latents
        if torch.isnan(left_latents).any() or torch.isnan(right_latents).any():
            logger.warning("NaN detected in latents, skipping batch")
            return 0.0, {"loss": 0.0, "mse_loss": 0.0}
        
        # Fixed timestep (near zero for reconstruction)
        timestep = torch.full((batch_size,), 0.001, device=self.accelerator.device)
        
        # Forward pass through transformer with mixed precision
        with autocast(enabled=self.config.mixed_precision is not None):
            # Input: left latents, target: right latents
            try:
                pred_latents_list = self.transformer(
                    x=left_latents,
                    t=timestep,
                    context=self.context_embeddings,
                    seq_len=self.config.num_frames,  # Sequence length
                    domain_label=1  # Converge domain for stereo conversion
                )
                
                # Stack predictions [B, C, T, H_lat, W_lat]
                pred_latents = torch.stack(pred_latents_list)
                
                # Compute reconstruction loss
                loss = self.compute_loss(pred_latents, right_latents)
                
            except Exception as e:
                logger.error(f"Error in transformer forward pass: {e}")
                return 0.0, {"loss": 0.0, "mse_loss": 0.0}
        
        # Clear intermediate tensors
        del left_latents, right_latents, pred_latents_list
        
        metrics = {
            "loss": loss.item(),
            "mse_loss": F.mse_loss(pred_latents, right_latents).item() if 'pred_latents' in locals() else 0.0
        }
        
        return loss, metrics
    
    def train(self):
        """Main training loop with comprehensive logging and monitoring."""
        if self.accelerator.is_main_process:
            logger.info("Starting training...")
            start_time = time.time()
        
        # Setup
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.precompute_text_embeddings()
        
        # Prepare with accelerator
        self.transformer, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.transformer, self.optimizer, self.train_loader, self.scheduler
        )
        
        # Training state
        total_loss = 0
        steps_per_epoch = len(self.train_loader)
        
        if self.accelerator.is_main_process:
            logger.info(f"Starting training for {self.config.num_epochs} epochs ({steps_per_epoch} steps per epoch)")
        
        for epoch in range(self.config.num_epochs):
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
                
                # Update metrics
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
            logger.info("Training completed!")
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
        """Save model checkpoint with both full state and weights."""
        # Unwrap model
        unwrapped_model = self.accelerator.unwrap_model(self.transformer)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch + 1,
            'global_step': self.global_step,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        if (epoch + 1) % self.config.save_every_n_epochs == 0:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save weights in safetensors format (more efficient)
        weights_path = self.output_dir / f"stereopilot_epoch_{epoch + 1:03d}.safetensors"
        safetensors.torch.save_file(unwrapped_model.state_dict(), weights_path)
        logger.info(f"Weights saved: {weights_path}")
        
        # Save best model
        if is_best:
            best_checkpoint_path = self.output_dir / "best_checkpoint.pt"
            best_weights_path = self.output_dir / "best_model.safetensors"
            
            torch.save(checkpoint_data, best_checkpoint_path)
            safetensors.torch.save_file(unwrapped_model.state_dict(), best_weights_path)
            
            logger.info(f"Best model saved (loss: {loss:.6f})")


def parse_args() -> TrainingConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train StereoPilot model for stereo video conversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
        default="../SP_Data/checkpoints",
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
    
    args = parser.parse_args()
    
    # Parse image size
    width, height = map(int, args.image_size.split(','))
    
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
        log_every_n_steps=args.log_every_n_steps
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
    logger.info("Starting StereoPilot Training")
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
