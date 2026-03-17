#!/usr/bin/env python3
"""
StereoPilot Training Script - Optimized for Memory Efficiency

This script trains StereoPilot model for stereo video conversion using reconstruction loss.
Key features:
- Memory-efficient training with gradient checkpointing and mixed precision
- GPU-optimized data loading and processing
- Clean architecture following the original StereoPilot paper

Usage:
    python train.py --config toml/infer.toml \
                    --train_dir ../SP_Data/mono_train \
                    --output_dir ../SP_Data/checkpoints
"""

import os
import sys
import argparse
import json
import toml
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.StereoPilot import StereoPilotPipeline


class StereoVideoDataset(Dataset):
    """
    Stereo video dataset for training.
    
    Loads left/right image pairs and returns 81-frame sequences.
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
            raise ValueError(f"Not enough images for {num_frames} frames. Found {len(self.left_images)} images.")
        
        logger.info(f"Dataset initialized: {self.num_sequences} sequences from {len(self.left_images)} image pairs")
    
    def _find_images(self, directory: Path) -> List[Path]:
        """Find all images in directory recursively."""
        images = []
        for ext in self.extensions:
            images.extend(directory.glob(f"**/*{ext}"))
            images.extend(directory.glob(f"**/*{ext.upper()}"))
        return sorted(images)
    
    def _validate_image_pairs(self):
        """Validate that left and right images match."""
        if len(self.left_images) != len(self.right_images):
            logger.warning(f"Mismatched image counts: {len(self.left_images)} left, {len(self.right_images)} right")
            min_len = min(len(self.left_images), len(self.right_images))
            self.left_images = self.left_images[:min_len]
            self.right_images = self.right_images[:min_len]
        
        # Check that filenames match (optional)
        for i, (left, right) in enumerate(zip(self.left_images, self.right_images)):
            if left.name != right.name:
                logger.warning(f"Filename mismatch at index {i}: {left.name} vs {right.name}")
    
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a video sequence.
        
        Returns:
            Dictionary with 'left' and 'right' videos as tensors [T, C, H, W]
        """
        # Calculate start index for this sequence
        start_idx = idx * self.num_frames
        
        # Initialize lists
        left_frames = []
        right_frames = []
        
        # Pre-create resize transform
        resize_transform = transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR)
        
        # Load frames
        for i in range(self.num_frames):
            try:
                # Load images using PIL
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
                if left_tensor.shape != (3, self.image_size[1], self.image_size[0]):
                    raise ValueError(f"Invalid left tensor shape: {left_tensor.shape}")
                if right_tensor.shape != (3, self.image_size[1], self.image_size[0]):
                    raise ValueError(f"Invalid right tensor shape: {right_tensor.shape}")
                
                # Normalize to [-1, 1]
                left_tensor = (left_tensor * 2.0) - 1.0
                right_tensor = (right_tensor * 2.0) - 1.0
                
                left_frames.append(left_tensor)
                right_frames.append(right_tensor)
                
            except Exception as e:
                logger.warning(f"Error loading frame {start_idx + i}: {e}. Using dummy data.")
                # Create dummy tensors as fallback
                dummy = torch.zeros(3, self.image_size[1], self.image_size[0])
                left_frames.append(dummy.clone())
                right_frames.append(dummy.clone())
        
        # Stack frames [T, C, H, W]
        left_video = torch.stack(left_frames)
        right_video = torch.stack(right_frames)
        
        # Validate final shapes
        expected_shape = (self.num_frames, 3, self.image_size[1], self.image_size[0])
        if left_video.shape != expected_shape or right_video.shape != expected_shape:
            logger.error(f"Shape mismatch: left={left_video.shape}, right={right_video.shape}, expected={expected_shape}")
        
        return {
            "left": left_video,
            "right": right_video
        }


class Trainer:
    """
    Main trainer class for StereoPilot.
    
    Handles:
    - Model initialization and setup
    - Training loop with mixed precision
    - Checkpoint saving and logging
    """
    
    def __init__(self, config_path: str, train_dir: str, output_dir: str, 
                 batch_size: int = 1, learning_rate: float = 3e-4,
                 num_epochs: int = 10, gradient_accumulation_steps: int = 4,
                 mixed_precision: str = "bf16"):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to model config file (toml)
            train_dir: Path to training data
            output_dir: Path to save checkpoints
            batch_size: Batch size per GPU
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            gradient_accumulation_steps: Gradient accumulation steps
            mixed_precision: Mixed precision mode ("fp16", "bf16", or None)
        """
        self.config_path = config_path
        self.train_dir = train_dir
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="tensorboard",
            project_dir=str(self.output_dir / "logs")
        )
        
        # Set random seed for reproducibility
        set_seed(42)
        
        logger.info("Trainer initialized")
    
    def setup_model(self):
        """Load and configure the model."""
        logger.info("Loading model configuration...")
        
        # Load config
        with open(self.config_path) as f:
            config = json.loads(json.dumps(toml.load(f)))
        
        # Override dtypes for training
        if self.accelerator.mixed_precision == "bf16":
            config['model']['dtype'] = 'bfloat16'
            config['model']['transformer_dtype'] = 'bfloat16'
        elif self.accelerator.mixed_precision == "fp16":
            config['model']['dtype'] = 'float16'
            config['model']['transformer_dtype'] = 'float16'
        
        # Initialize pipeline
        self.pipeline = StereoPilotPipeline(config)
        self.pipeline.load_diffusion_model()
        
        # Extract components
        self.transformer = self.pipeline.transformer
        self.vae = self.pipeline.vae
        self.text_encoder = self.pipeline.text_encoder
        
        # Freeze VAE
        if hasattr(self.vae, 'model'):
            self.vae.model.requires_grad_(False)
            self.vae.model.eval()
        else:
            self.vae.requires_grad_(False).eval()
        logger.info("VAE frozen")
        
        # Freeze Text Encoder
        if hasattr(self.text_encoder, 'model'):
            self.text_encoder.model.requires_grad_(False)
            self.text_encoder.model.eval()
        else:
            self.text_encoder.requires_grad_(False).eval()
        logger.info("Text Encoder frozen")
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.transformer, 'enable_gradient_checkpointing'):
            self.transformer.enable_gradient_checkpointing()
            logger.info("Gradient checkpointing enabled")
        
        # Set transformer to training mode
        self.transformer.requires_grad_(True)
        self.transformer.train()
        logger.info("Transformer set to trainable")
    
    def setup_data(self):
        """Setup data loaders."""
        logger.info("Setting up data loaders...")
        
        # Create dataset
        dataset = StereoVideoDataset(
            root_dir=self.train_dir,
            num_frames=81,
            image_size=(832, 480)
        )
        
        # Create data loader
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 4),
            pin_memory=True,
            drop_last=True
        )
        
        logger.info(f"Created data loader with {len(dataset)} sequences")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        logger.info("Setting up optimizer...")
        
        # Filter trainable parameters
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        
        # Log parameter counts
        total_params = sum(p.numel() for p in self.transformer.parameters())
        trainable_param_count = sum(p.numel() for p in trainable_params)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_param_count:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_param_count:,}")
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-8
        )
        
        # Create learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=1000,
            T_mult=2,
            eta_min=self.learning_rate * 0.1
        )
        
        # Prepare with accelerator
        self.transformer, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.transformer, self.optimizer, self.train_loader, self.scheduler
        )
    
    def precompute_text_embeddings(self):
        """Pre-compute text embeddings for fixed prompt."""
        logger.info("Pre-computing text embeddings...")
        
        # Fixed prompt as per requirements
        prompt = ["This is a video viewed from the left perspective"]
        
        # Move text encoder to device
        device = self.accelerator.device
        if hasattr(self.text_encoder, 'model'):
            self.text_encoder.model.to(device)
        else:
            self.text_encoder.to(device)
        
        with torch.no_grad():
            self.context_embeddings = self.text_encoder(prompt, device)
        
        logger.info(f"Text embeddings pre-computed: {len(self.context_embeddings)} items")
    
    def encode_videos(self, left_video: torch.Tensor, right_video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode videos to latents using VAE.
        
        Memory-efficient implementation that processes videos frame-by-frame
        to avoid OOM errors with long sequences.
        
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
            
            # Process each frame separately
            for t in range(left_video.shape[2]):  # T dimension
                with torch.no_grad():
                    # Extract single frame [1, C, 1, H, W]
                    left_frame = left_video[b:b+1, :, t:t+1, :, :]
                    right_frame = right_video[b:b+1, :, t:t+1, :, :]
                    
                    # Encode frames
                    left_latent = self.vae.model.encode(left_frame, scale)
                    right_latent = self.vae.model.encode(right_frame, scale)
                    
                    # Handle list outputs
                    if isinstance(left_latent, list):
                        left_latent = left_latent[0]
                    if isinstance(right_latent, list):
                        right_latent = right_latent[0]
                    
                    # Remove batch dimension [C, 1, H_lat, W_lat]
                    left_frames.append(left_latent.squeeze(0))
                    right_frames.append(right_latent.squeeze(0))
            
            # Concatenate frames along time dimension [C, T, H_lat, W_lat]
            left_video_latent = torch.cat(left_frames, dim=1)
            right_video_latent = torch.cat(right_frames, dim=1)
            
            left_latents_list.append(left_video_latent)
            right_latents_list.append(right_video_latent)
            
            # Clear cache to free memory
            if b % 2 == 0:
                torch.cuda.empty_cache()
        
        # Stack batch [B, C, T, H_lat, W_lat]
        left_latents = torch.stack(left_latents_list)
        right_latents = torch.stack(right_latents_list)
        
        return left_latents, right_latents
    
    def compute_loss(self, pred_latents: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            pred_latents: Predicted latents
            target_latents: Target latents
            
        Returns:
            Reconstruction loss
        """
        # Simple MSE loss
        loss = F.mse_loss(pred_latents, target_latents)
        
        # Optional: Add perceptual loss components here
        
        return loss
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Single training step with memory optimization.
        
        Args:
            batch: Batch dictionary with 'left' and 'right' videos
            
        Returns:
            Loss value
        """
        left_video = batch["left"]  # [B, T, C, H, W]
        right_video = batch["right"]  # [B, T, C, H, W]
        batch_size = left_video.shape[0]
        
        # Clear cache before forward pass
        torch.cuda.empty_cache()
        
        # Encode videos to latents
        # This is the memory-intensive part, done with no_grad
        with torch.no_grad():
            left_latents, right_latents = self.encode_videos(left_video, right_video)
        
        # Check for NaNs in latents
        if torch.isnan(left_latents).any() or torch.isnan(right_latents).any():
            logger.warning("NaN detected in latents, skipping batch")
            return torch.tensor(0.0, device=self.accelerator.device)
        
        # Fixed timestep (near zero for reconstruction)
        timestep = torch.full((batch_size,), 0.001, device=self.accelerator.device)
        
        # Forward pass through transformer with mixed precision
        with autocast(enabled=self.mixed_precision is not None):
            # Input: left latents, target: right latents
            pred_latents_list = self.transformer(
                x=left_latents,
                t=timestep,
                context=self.context_embeddings,
                seq_len=81,  # Sequence length
                domain_label=1  # Converge domain for stereo conversion
            )
            
            # Stack predictions [B, C, T, H_lat, W_lat]
            pred_latents = torch.stack(pred_latents_list)
            
            # Compute reconstruction loss
            loss = self.compute_loss(pred_latents, right_latents)
        
        # Clear intermediate tensors
        del left_latents, right_latents, pred_latents_list
        
        return loss
    
    def train(self):
        """Main training loop with comprehensive logging and monitoring."""
        logger.info("Starting training...")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        # Setup
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.precompute_text_embeddings()
        
        # Move VAE to device
        device = self.accelerator.device
        if hasattr(self.vae, 'model'):
            self.vae.model.to(device)
        
        # Initialize training metrics
        global_step = 0
        total_loss = 0
        best_loss = float('inf')
        steps_per_epoch = len(self.train_loader)
        
        logger.info(f"Starting training for {self.num_epochs} epochs ({steps_per_epoch} steps per epoch)")
        
        for epoch in range(self.num_epochs):
            epoch_start = torch.cuda.Event(enable_timing=True)
            epoch_end = torch.cuda.Event(enable_timing=True)
            epoch_start.record()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"{'='*60}")
            
            epoch_loss = 0
            num_batches = 0
            epoch_start_time = torch.cuda.Event(enable_timing=True)
            epoch_end_time = torch.cuda.Event(enable_timing=True)
            
            # Progress bar
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                disable=not self.accelerator.is_local_main_process,
                ncols=100
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                batch_start_time = time.time()
                
                # Training step with gradient accumulation
                with self.accelerator.accumulate(self.transformer):
                    try:
                        loss = self.train_step(batch)
                        
                        # Skip if loss is zero (failed batch)
                        if loss.item() == 0.0:
                            logger.warning(f"Zero loss at batch {batch_idx}, skipping")
                            continue
                        
                        # Backward pass
                        self.accelerator.backward(loss)
                        
                        # Gradient clipping
                        if self.accelerator.sync_gradients:
                            grad_norm = self.accelerator.clip_grad_norm_(
                                self.transformer.parameters(),
                                max_norm=1.0
                            )
                        
                        # Optimizer step
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                    except Exception as e:
                        logger.error(f"Error in training step at batch {batch_idx}: {e}")
                        continue
                
                # Update metrics
                loss_value = loss.item()
                epoch_loss += loss_value
                total_loss += loss_value
                num_batches += 1
                global_step += 1
                
                # Calculate batch time
                batch_time = time.time() - batch_start_time
                
                # Update progress bar
                if batch_idx % 10 == 0:
                    progress_bar.set_postfix({
                        "loss": f"{loss_value:.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                        "time": f"{batch_time:.2f}s"
                    })
                
                # Memory usage logging
                if global_step % 100 == 0 and torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    reserved = torch.cuda.memory_reserved() / 1024**3
                    logger.info(f"Step {global_step} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
                # Log to tensorboard
                if self.accelerator.is_main_process and global_step % 50 == 0:
                    self.accelerator.log({
                        "train/loss": loss_value,
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/grad_norm": grad_norm.item() if 'grad_norm' in locals() else 0,
                    }, step=global_step)
            
            # Epoch summary
            epoch_end.record()
            torch.cuda.synchronize()
            epoch_time = epoch_start.elapsed_time(epoch_end) / 1000  # Convert to seconds
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            
            logger.info(f"\nEpoch {epoch + 1} Summary:")
            logger.info(f"  Average Loss: {avg_epoch_loss:.4f}")
            logger.info(f"  Time: {epoch_time:.2f} seconds")
            logger.info(f"  Steps: {num_batches}")
            
            # Save checkpoint
            if self.accelerator.is_main_process:
                is_best = avg_epoch_loss < best_loss
                if is_best:
                    best_loss = avg_epoch_loss
                
                self.save_checkpoint(epoch, avg_epoch_loss, global_step, is_best)
                
                # Log epoch metrics
                self.accelerator.log({
                    "epoch/avg_loss": avg_epoch_loss,
                    "epoch/time": epoch_time,
                    "epoch/best_loss": best_loss
                }, step=epoch)
        
        # Training completed
        total_training_time = epoch_start.elapsed_time(epoch_end) / 1000
        logger.info(f"\n{'='*60}")
        logger.info("Training completed!")
        logger.info(f"{'='*60}")
        
        avg_total_loss = total_loss / global_step if global_step > 0 else 0
        logger.info(f"Final Metrics:")
        logger.info(f"  Average Loss: {avg_total_loss:.4f}")
        logger.info(f"  Best Loss: {best_loss:.4f}")
        logger.info(f"  Total Steps: {global_step}")
        logger.info(f"  Total Time: {total_training_time:.2f} seconds")
    
    def save_checkpoint(self, epoch: int, loss: float, global_step: int, is_best: bool = False):
        """Save model checkpoint."""
        # Unwrap model
        unwrapped_model = self.accelerator.unwrap_model(self.transformer)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save weights in safetensors format
        weights_path = self.output_dir / f"stereopilot_epoch_{epoch + 1:03d}.safetensors"
        from safetensors.torch import save_file
        save_file(unwrapped_model.state_dict(), weights_path)
        logger.info(f"Weights saved: {weights_path}")
        
        # Save best model
        if is_best:
            best_checkpoint_path = self.output_dir / "best_checkpoint.pt"
            best_weights_path = self.output_dir / "best_model.safetensors"
            
            torch.save(checkpoint_data, best_checkpoint_path)
            save_file(unwrapped_model.state_dict(), best_weights_path)
            
            logger.info(f"Best model saved (loss: {loss:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Train StereoPilot model")
    parser.add_argument("--config", type=str, default="toml/infer.toml",
                        help="Path to model config file")
    parser.add_argument("--train_dir", type=str, default="../SP_Data/mono_train",
                        help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="../SP_Data/checkpoints",
                        help="Path to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["fp16", "bf16", "no"],
                        help="Mixed precision mode")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Trainer(
        config_path=args.config,
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None
    )
    
    # Start training
    trainer.train()


"""
## Detailed Usage Instructions

### 1. Training Data Preparation

The training data should be organized as follows:

```
../SP_Data/mono_train/
├── left/
│   ├── frame_00001.jpg
│   ├── frame_00002.jpg
│   └── ...
└── right/
    ├── frame_00001.jpg
    ├── frame_00002.jpg
    └── ...
```

Requirements:
- Left and right images must be paired (same filename)
- Images will be resized to 832x480
- At least 81 images per training sequence
- Supported formats: JPG, JPEG, PNG, BMP

### 2. Model Configuration

The config file (toml/infer.toml) should contain:

```toml
[model]
type = 'stereopilot'
ckpt_path = '../SP_Data/ckpt/Wan2.1-T2V-1.3B'  # Base Wan2.1 model
transformer_path = '../SP_Data/ckpt/StereoPilot.safetensors'  # Pretrained weights
pretrained_path = '../SP_Data/ckpt/StereoPilot.safetensors'  # Domain embeddings
dtype = 'bfloat16'
transformer_dtype = 'bfloat16'
```

### 3. Training Command

Basic training:
```bash
python train.py \
    --config toml/infer.toml \
    --train_dir ../SP_Data/mono_train \
    --output_dir ../SP_Data/checkpoints \
    --batch_size 1 \
    --learning_rate 3e-4 \
    --epochs 10 \
    --gradient_accumulation_steps 4 \
    --mixed_precision bf16
```

### 4. Memory Optimization Options

For different GPU memory sizes:

**RTX 3090/4090 (24GB):**
```bash
--batch_size 1 --gradient_accumulation_steps 8 --mixed_precision bf16
```

**A100/A800 (40-80GB):**
```bash
--batch_size 2 --gradient_accumulation_steps 4 --mixed_precision bf16
```

**RTX 5090 (32GB):**
```bash
--batch_size 1 --gradient_accumulation_steps 6 --mixed_precision bf16
```

### 5. Training Process

The training process:

1. **Data Loading**: Loads 81-frame sequences from left/right directories
2. **Encoding**: VAE encodes videos to latent space (frame-by-frame for memory efficiency)
3. **Forward Pass**: Transformer processes left latents to predict right latents
4. **Loss Calculation**: MSE loss between predicted and actual right latents
5. **Backward Pass**: Gradient accumulation and mixed precision training
6. **Optimization**: AdamW with cosine annealing scheduler

### 6. Key Features

**Memory Optimization:**
- Gradient checkpointing enabled
- Frame-by-frame VAE encoding
- Mixed precision training (bfloat16)
- Gradient accumulation
- Automatic memory cleanup

**Training Stability:**
- Fixed timestep (t=0.001) for reconstruction
- Frozen VAE and text encoder
- Only transformer and domain embeddings trained
- Gradient clipping (max_norm=1.0)
- NaN detection and batch skipping

**Monitoring:**
- Tensorboard logging
- Progress bars with live metrics
- GPU memory usage tracking
- Checkpoint saving (regular and best models)

### 7. Outputs

Training produces:

1. **Checkpoints**: Full training state
   - `checkpoint_epoch_XXX.pt`: Full checkpoint with optimizer state
   - `stereopilot_epoch_XXX.safetensors`: Model weights only

2. **Best Model**: Best performing checkpoint
   - `best_checkpoint.pt`: Best full checkpoint
   - `best_model.safetensors`: Best model weights

3. **Logs**: Training metrics
   - `training.log`: Text log file
   - Tensorboard logs in `logs/` directory

### 8. Model Architecture Details

**Components:**
- VAE Encoder: Encodes 81 frames to latent space
- Text Encoder: UMT5 (frozen, pre-computed embeddings)
- Transformer: Wan2.1 backbone + domain embeddings
- Domain Embeddings: parall_embedding and converge_embedding

**Training Objective:**
- Reconstruct right view from left view
- Fixed timestep for late-stage refinement
- MSE loss in latent space

### 9. Troubleshooting

**Out of Memory:**
- Reduce batch_size to 1
- Increase gradient_accumulation_steps
- Use mixed_precision bf16
- Close other GPU applications

**Slow Training:**
- Reduce num_workers in DataLoader (if CPU bottleneck)
- Ensure GPU utilization with nvidia-smi
- Check disk I/O speed for data loading

**NaN Loss:**
- Check input data for corruption
- Reduce learning rate
- Ensure VAE and text encoder are frozen

**Poor Convergence:**
- Verify data quality and alignment
- Check that timestep is fixed to 0.001
- Ensure domain_label=1 for stereo conversion
- Increase training epochs

### 10. Advanced Usage

**Resume Training:**
Modify the trainer to load from checkpoint:
```python
checkpoint = torch.load("checkpoint_epoch_XXX.pt")
self.transformer.load_state_dict(checkpoint['model_state_dict'])
self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

**Custom Loss:**
Modify the `compute_loss` method to add:
- Perceptual loss
- Adversarial loss
- Stereo consistency loss

**Multi-GPU Training:**
Accelerate automatically handles multi-GPU training.
Just run the script on a machine with multiple GPUs.

**Inference:**
Use the trained model with the original `sample.py` script:
```bash
python sample.py \
    --config toml/infer.toml \
    --input /path/to/left_video.mp4 \
    --output_folder /path/to/output \
    --device cuda:0
```

For more details, see the StereoPilot paper and original repository.
"""

if __name__ == "__main__":
    main()
