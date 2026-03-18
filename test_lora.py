#!/usr/bin/env python3
"""
StereoPilot LoRA Inference Testing Script

Evaluates trained LoRA weights on the mono2stereo-test dataset.
Computes PSNR, SSIM, SIoU metrics and inference speed (FPS).

Usage:
    # Test with latest LoRA weights
    python test_lora.py \
        --config toml/infer.toml \
        --lora_weights ../SP_Data/checkpoints_lora/lora_weights_latest.safetensors \
        --test_data ../SP_Data/mono2stereo-test \
        --output_dir ../SP_Data/test_results

    # Test with specific epoch
    python test_lora.py \
        --config toml/infer.toml \
        --lora_weights ../SP_Data/checkpoints_lora/lora_weights_epoch_010.safetensors \
        --test_data ../SP_Data/mono2stereo-test \
        --output_dir ../SP_Data/test_results_epoch010

    # Test with different LoRA rank
    python test_lora.py \
        --config toml/infer.toml \
        --lora_weights ../SP_Data/checkpoints_lora/lora_weights_epoch_010.safetensors \
        --lora_rank 8 \
        --test_data ../SP_Data/mono2stereo-test \
        --output_dir ../SP_Data/test_results
"""

import os
import sys
import glob
import time
import json
import toml
import torch
import cv2
import numpy as np
import argparse
import traceback
from tqdm import tqdm
from PIL import Image, ImageOps
from torchvision import transforms
from pathlib import Path
import logging

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Ensure we can import from StereoPilot modules
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root.parent))

from utils.common import DTYPE_MAP
from lora_utils import LoRAManager, LoRAConfig, LoRALinear

try:
    from models import StereoPilot
except ImportError:
    logger.error("Could not import StereoPilot. Make sure you are in the StereoPilot directory.")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Metric functions (from evaluate.py)
# ------------------------------------------------------------------------------

def detect_edges(image, low, high):
    """Detect edges using Canny edge detector."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, low, high)
    return edges


def edge_overlap(edge1, edge2):
    """Calculate edge overlap ratio."""
    intersection = np.logical_and(edge1, edge2).sum()
    union = np.logical_or(edge1, edge2).sum()
    if union == 0:
        return 0
    return intersection / union


def compute_siou(pred, target, left):
    """Compute Stereo IoU metric."""
    # Ensure inputs are uint8 numpy arrays for edge detection
    pred_uint8 = pred.astype(np.uint8)
    target_uint8 = target.astype(np.uint8)
    left_uint8 = left.astype(np.uint8)
    
    left_edges = detect_edges(left_uint8, 100, 200)
    pred_edges = detect_edges(pred_uint8, 100, 200)
    right_edges = detect_edges(target_uint8, 100, 200)
    
    # Calculate overlaps
    edge_overlap_gr = edge_overlap(pred_edges, right_edges)
    
    # Calculate difference overlap
    diff_gl = np.abs(pred.astype(np.float32) - left.astype(np.float32))
    diff_rl = np.abs(target.astype(np.float32) - left.astype(np.float32))
    
    if len(diff_gl.shape) == 3:
        diff_gl = cv2.cvtColor(diff_gl.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif diff_gl.ndim == 2:
        diff_gl = diff_gl.astype(np.uint8)
    
    if len(diff_rl.shape) == 3:
        diff_rl = cv2.cvtColor(diff_rl.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif diff_rl.ndim == 2:
        diff_rl = diff_rl.astype(np.uint8)
    
    diff_gl_ = np.zeros_like(diff_gl)
    diff_rl_ = np.zeros_like(diff_rl)
    diff_gl_[diff_gl > 5] = 1
    diff_rl_[diff_rl > 5] = 1
    
    diff_overlap_grl = edge_overlap(diff_gl_, diff_rl_)
    
    return 0.75 * edge_overlap_gr + 0.25 * diff_overlap_grl


def eval_metrics(pred, target, left):
    """Compute evaluation metrics (PSNR, SSIM, SIoU)."""
    # pred, target, left should be numpy arrays (H, W, C) in range 0-255
    
    # MSE/RMSE
    diff = pred.astype(np.float32) - target.astype(np.float32)
    mse_err = np.mean(diff ** 2)
    rmse = np.sqrt(mse_err)
    
    # PSNR
    max_pixel = 255.0
    if rmse == 0:
        psnr = 100.0
    else:
        psnr = 20 * np.log10(max_pixel / rmse)
    
    # SSIM
    min_dim = min(pred.shape[0], pred.shape[1])
    win_size = 7
    if win_size > min_dim:
        win_size = min_dim if min_dim % 2 == 1 else min_dim - 1
    
    ssim_value = 0.0
    if win_size >= 3:
        try:
            ssim_ret = ssim(pred, target, full=True, channel_axis=2, win_size=win_size)
            if isinstance(ssim_ret, tuple):
                ssim_value = ssim_ret[0]
            else:
                ssim_value = ssim_ret
        except TypeError:
            try:
                ssim_ret = ssim(pred, target, full=True, multichannel=True, win_size=win_size)
                if isinstance(ssim_ret, tuple):
                    ssim_value = ssim_ret[0]
                else:
                    ssim_value = ssim_ret
            except Exception:
                ssim_value = 0.0
        except Exception:
            ssim_value = 0.0
    
    # SIoU
    siou_value = compute_siou(pred, target, left)
    
    return {'psnr': psnr, 'ssim': ssim_value, 'siou': siou_value}


# ------------------------------------------------------------------------------
# Image preprocessing
# ------------------------------------------------------------------------------

def convert_crop_and_resize(pil_img, width_and_height):
    """Convert, crop and resize image (Center Crop)."""
    if pil_img.mode not in ['RGB', 'RGBA'] and 'transparency' in pil_img.info:
        pil_img = pil_img.convert('RGBA')
    
    if pil_img.mode == 'RGBA':
        canvas = Image.new('RGBA', pil_img.size, (255, 255, 255))
        canvas.alpha_composite(pil_img)
        pil_img = canvas.convert('RGB')
    else:
        pil_img = pil_img.convert('RGB')
    
    return ImageOps.fit(pil_img, width_and_height)


def set_config_defaults(config):
    """Set config defaults."""
    model_config = config['model']
    model_config['dtype'] = DTYPE_MAP[model_config['dtype']]
    if 'transformer_dtype' in model_config:
        model_config['transformer_dtype'] = DTYPE_MAP[model_config['transformer_dtype']]


# ------------------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------------------

def run_evaluation(args):
    """Run LoRA evaluation on test dataset."""
    
    # Setup paths
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    log_file_path = output_root / "evaluation_results.txt"
    metrics_file_path = output_root / "metrics_summary.json"
    
    logger.info(f"Loading config from {args.config}")
    if not Path(args.config).exists():
        logger.error(f"Config file not found: {args.config}")
        return
    
    with open(args.config) as f:
        config = json.loads(json.dumps(toml.load(f)))
    set_config_defaults(config)
    
    # Load model
    logger.info("Loading StereoPilot model...")
    device = args.device
    try:
        model = StereoPilot.StereoPilotPipeline(config)
        model.load_diffusion_model()
        model.register_custom_op()
        
        # Move components to device
        model.transformer.eval()
        model.transformer.to(device)
        model.vae.model.to(device)
        model.vae.mean = model.vae.mean.to(device)
        model.vae.std = model.vae.std.to(device)
        
        # Inject LoRA weights
        if args.lora_weights and Path(args.lora_weights).exists():
            logger.info(f"Loading LoRA weights from {args.lora_weights}")
            lora_config = LoRAConfig(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=0.0,
                target_modules=None
            )
            lora_manager = LoRAManager(model.transformer, **lora_config.__dict__)
            lora_manager.inject_lora()
            lora_manager.load_lora_weights(args.lora_weights)
            logger.info(f"LoRA injected successfully. Trainable parameters: {lora_manager.get_trainable_params_count():,}")
        else:
            logger.warning(f"LoRA weights not found at {args.lora_weights}. Running base model.")
        
        # Pre-compute text embeddings and unload Text Encoder
        logger.info("Pre-computing text embeddings and unloading Text Encoder...")
        model.text_encoder.model.to(device)
        with torch.no_grad():
            context_cache = model.text_encoder(["This is a video viewed from the left perspective"], device)
        
        context_lens_cache = [c.shape[0] for c in context_cache]
        
        # Unload Text Encoder
        del model.text_encoder.model
        del model.text_encoder.tokenizer
        del model.text_encoder
        torch.cuda.empty_cache()
        logger.info("Text Encoder unloaded.")
        
        # Optional: Enable optimizations
        if args.use_torch_compile:
            logger.info("Enabling torch.compile for faster inference...")
            model.transformer = torch.compile(model.transformer, mode="reduce-overhead")
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        traceback.print_exc()
        return
    
    target_size = (832, 480)
    
    # Find test subsets
    if not Path(args.test_data).exists():
        logger.error(f"Test data not found: {args.test_data}")
        return
    
    subsets = [d for d in os.listdir(args.test_data) if os.path.isdir(os.path.join(args.test_data, d))]
    data_subsets = []
    for s in subsets:
        left_dir = os.path.join(args.test_data, s, 'left')
        right_dir = os.path.join(args.test_data, s, 'right')
        if os.path.exists(left_dir) and os.path.exists(right_dir):
            data_subsets.append(s)
    
    data_subsets.sort()
    logger.info(f"Found test subsets: {data_subsets}")
    
    # Initialize metrics tracking
    total_metrics = {'psnr': 0, 'ssim': 0, 'siou': 0, 'fps': 0, 'count': 0}
    subset_results = {}
    
    # Write header to log file
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"LoRA Evaluation Results\n")
        log_file.write(f"Test Time: {time.ctime()}\n")
        log_file.write(f"LoRA Weights: {args.lora_weights}\n")
        log_file.write(f"LoRA Rank: {args.lora_rank}\n")
        log_file.write(f"Test Data: {args.test_data}\n")
        log_file.write("=" * 80 + "\n")
    
    # Process each subset
    for subset in data_subsets:
        logger.info(f"\nProcessing subset: {subset}")
        
        subset_metrics = {'psnr': 0, 'ssim': 0, 'siou': 0, 'fps': 0, 'count': 0}
        
        left_dir = os.path.join(args.test_data, subset, 'left')
        right_dir = os.path.join(args.test_data, subset, 'right')
        output_subset_dir = output_root / subset
        output_subset_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = sorted(glob.glob(os.path.join(left_dir, "*")))
        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if args.max_samples and args.max_samples > 0:
            image_files = image_files[:args.max_samples]
        
        logger.info(f"Found {len(image_files)} images in {subset}")
        
        # Process each image
        for img_path in tqdm(image_files, desc=f"Testing {subset}"):
            try:
                img_name = os.path.basename(img_path)
                name_no_ext = os.path.splitext(img_name)[0]
                
                # Find corresponding right image
                right_img_path = os.path.join(right_dir, img_name)
                if not os.path.exists(right_img_path):
                    # Try with different extension
                    candidates = glob.glob(os.path.join(right_dir, name_no_ext + ".*"))
                    if candidates:
                        right_img_path = candidates[0]
                    else:
                        logger.warning(f"Right image not found for {img_name}, skipping.")
                        continue
                
                # Load and preprocess ground truth images
                pil_left = Image.open(img_path)
                pil_right = Image.open(right_img_path)
                
                gt_left_processed = convert_crop_and_resize(pil_left, target_size)
                gt_right_processed = convert_crop_and_resize(pil_right, target_size)
                
                np_left = np.array(gt_left_processed)
                np_right = np.array(gt_right_processed)
                
                # Prepare input for model
                pixel_tf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                pixel_values = pixel_tf(gt_left_processed).to(device)
                video_input = pixel_values.unsqueeze(1).unsqueeze(0)  # [1, 3, 1, H, W]
                
                # Encode to latents
                with torch.no_grad():
                    latents = model.vae.model.encode(video_input, model.vae.scale).float()
                
                # Run inference
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    video_out = model.sample(
                        prompt=None,
                        context=context_cache,
                        context_lens=context_lens_cache,
                        video_condition=latents,
                        size=target_size,
                        frame_num=1,
                        shift=5.0,
                        sample_solver='unipc',
                        sampling_steps=args.sampling_steps,
                        guide_scale=args.guide_scale,
                        n_prompt="",
                        seed=args.seed,
                        domain_label=1
                    )
                
                torch.cuda.synchronize()
                end_time = time.time()
                inference_time = end_time - start_time
                fps = 1.0 / inference_time if inference_time > 0 else 0.0
                
                # Post-process output
                if isinstance(video_out, torch.Tensor):
                    frame_tensor = video_out.float().cpu()
                    
                    # Handle dimensions [C, T, H, W] or [B, C, T, H, W]
                    if frame_tensor.ndim == 4:
                        if frame_tensor.shape[1] == 1:  # [C, 1, H, W]
                            frame_tensor = frame_tensor.squeeze(1)  # [C, H, W]
                    elif frame_tensor.ndim == 5:  # [B, C, 1, H, W]
                        frame_tensor = frame_tensor.squeeze(0).squeeze(1)  # [C, H, W]
                    
                    if frame_tensor.ndim == 3:
                        # Clamp and scale
                        frame_tensor = frame_tensor.clamp(-1, 1)
                        frame_tensor = (frame_tensor + 1) / 2.0
                        frame_numpy = frame_tensor.permute(1, 2, 0).numpy()
                        frame_numpy = (frame_numpy * 255).clip(0, 255).astype(np.uint8)
                    else:
                        logger.error(f"Unexpected tensor shape after processing: {frame_tensor.shape}")
                        continue
                else:
                    logger.error(f"Model output is not a tensor: {type(video_out)}")
                    continue
                
                # Calculate metrics
                metrics = eval_metrics(frame_numpy, np_right, np_left)
                
                # Save output image
                output_path = output_subset_dir / f"{name_no_ext}.png"
                Image.fromarray(frame_numpy).save(output_path)
                
                # Accumulate metrics
                subset_metrics['psnr'] += metrics['psnr']
                subset_metrics['ssim'] += metrics['ssim']
                subset_metrics['siou'] += metrics['siou']
                subset_metrics['fps'] += fps
                subset_metrics['count'] += 1
                
            except Exception as e:
                logger.error(f"Error processing {img_name}: {e}")
                if args.debug:
                    traceback.print_exc()
                continue
        
        # Log subset results
        if subset_metrics['count'] > 0:
            count = subset_metrics['count']
            avg_psnr = subset_metrics['psnr'] / count
            avg_ssim = subset_metrics['ssim'] / count
            avg_siou = subset_metrics['siou'] / count
            avg_fps = subset_metrics['fps'] / count
            
            msg = f"Subset: {subset:12s} | Count: {count:4d} | PSNR: {avg_psnr:6.4f} | SSIM: {avg_ssim:6.4f} | SIoU: {avg_siou:6.4f} | FPS: {avg_fps:6.2f}"
            logger.info(msg)
            
            with open(log_file_path, 'a') as log_file:
                log_file.write(msg + "\n")
            
            # Store subset results
            subset_results[subset] = {
                'count': count,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'siou': avg_siou,
                'fps': avg_fps
            }
            
            # Add to total
            total_metrics['psnr'] += subset_metrics['psnr']
            total_metrics['ssim'] += subset_metrics['ssim']
            total_metrics['siou'] += subset_metrics['siou']
            total_metrics['fps'] += subset_metrics['fps']
            total_metrics['count'] += count
        else:
            logger.warning(f"Subset {subset} had no valid items processed.")
    
    # Log total results
    if total_metrics['count'] > 0:
        count = total_metrics['count']
        avg_psnr = total_metrics['psnr'] / count
        avg_ssim = total_metrics['ssim'] / count
        avg_siou = total_metrics['siou'] / count
        avg_fps = total_metrics['fps'] / count
        
        msg = f"\n{'='*80}\nOverall Average | Count: {count:4d} | PSNR: {avg_psnr:6.4f} | SSIM: {avg_ssim:6.4f} | SIoU: {avg_siou:6.4f} | FPS: {avg_fps:6.2f}"
        logger.info(msg)
        
        with open(log_file_path, 'a') as log_file:
            log_file.write("=" * 80 + "\n")
            log_file.write(msg + "\n")
        
        # Save detailed metrics to JSON
        final_results = {
            'overall': {
                'count': count,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'siou': avg_siou,
                'fps': avg_fps
            },
            'subsets': subset_results,
            'config': {
                'lora_weights': args.lora_weights,
                'lora_rank': args.lora_rank,
                'lora_alpha': args.lora_alpha,
                'sampling_steps': args.sampling_steps,
                'guide_scale': args.guide_scale,
                'seed': args.seed,
                'max_samples': args.max_samples
            }
        }
        
        with open(metrics_file_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"\nEvaluation Complete!")
        logger.info(f"Results saved to: {log_file_path}")
        logger.info(f"Metrics JSON saved to: {metrics_file_path}")
        logger.info(f"Output images saved to: {output_root}")
    else:
        logger.error("No samples were processed successfully.")


# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate StereoPilot LoRA weights on test dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument(
        "--config",
        type=str,
        default="toml/infer.toml",
        help="Path to model config file (TOML format)"
    )
    
    # LoRA configuration
    parser.add_argument(
        "--lora_weights",
        type=str,
        required=True,
        help="Path to LoRA weights file (.safetensors)"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="LoRA rank (must match training configuration)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA alpha scaling (defaults to lora_rank)"
    )
    
    # Test data
    parser.add_argument(
        "--test_data",
        type=str,
        default="../SP_Data/mono2stereo-test",
        help="Path to test dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../SP_Data/test_results",
        help="Path to save test results"
    )
    
    # Inference parameters
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=30,
        help="Number of sampling steps for diffusion"
    )
    parser.add_argument(
        "--guide_scale",
        type=float,
        default=5.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference"
    )
    
    # Testing options
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to test per subset (None = all)"
    )
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Enable torch.compile for faster inference (requires PyTorch 2.0+)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with full error traces"
    )
    
    return parser.parse_args()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    args = parse_args()
    
    # Validate inputs
    if not Path(args.lora_weights).exists():
        logger.error(f"LoRA weights not found: {args.lora_weights}")
        sys.exit(1)
    
    if not Path(args.test_data).exists():
        logger.error(f"Test data not found: {args.test_data}")
        sys.exit(1)
    
    # Set default alpha if not specified
    if args.lora_alpha is None:
        args.lora_alpha = args.lora_rank
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("StereoPilot LoRA Inference Test")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"LoRA Weights: {args.lora_weights}")
    logger.info(f"LoRA Rank: {args.lora_rank}")
    logger.info(f"LoRA Alpha: {args.lora_alpha}")
    logger.info(f"Test Data: {args.test_data}")
    logger.info(f"Output Dir: {args.output_dir}")
    logger.info(f"Sampling Steps: {args.sampling_steps}")
    logger.info(f"Guidance Scale: {args.guide_scale}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)
    
    # Run evaluation
    try:
        run_evaluation(args)
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
