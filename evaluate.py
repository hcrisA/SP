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
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim

# Ensure we can import from StereoPilot modules
sys.path.append(os.getcwd())

# Import StereoPilot modules
from utils.common import DTYPE_MAP
try:
    from models import StereoPilot
except ImportError:
    print("Error: Could not import StereoPilot. Make sure you are in the StereoPilot directory.")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Metric functions
# ------------------------------------------------------------------------------

def detect_edges(image, low, high):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    edges = cv2.Canny(gray, low, high)
    return edges

def edge_overlap(edge1, edge2):
    intersection = np.logical_and(edge1, edge2).sum()
    union = np.logical_or(edge1, edge2).sum()
    if union == 0:
        return 0
    return intersection / union

def compute_siou(pred, target, left):
    # Ensure inputs are uint8 numpy arrays for edge detection
    pred_uint8 = pred.astype(np.uint8)
    target_uint8 = target.astype(np.uint8)
    left_uint8 = left.astype(np.uint8)
    
    left_edges = detect_edges(left_uint8, 100, 200)
    pred_edges = detect_edges(pred_uint8, 100, 200)
    right_edges = detect_edges(target_uint8, 100, 200)

    # Calculate overlaps
    edge_overlap_gr = edge_overlap(pred_edges, right_edges)
    
    # Calculate difference overlap (using float for calculation)
    # diff = |pred - left|
    diff_gl = np.abs(pred.astype(np.float32) - left.astype(np.float32))
    # diff = |target - left|
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
    # pred, target, left should be numpy arrays (H, W, C) in range 0-255
    
    # MSE/RMSE
    # Ensure float32 for diff calculation
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
            # Try valid args for recent skimage
            ssim_ret = ssim(pred, target, full=True, channel_axis=2, win_size=win_size)
            if isinstance(ssim_ret, tuple):
                ssim_value = ssim_ret[0]
            else:
                ssim_value = ssim_ret
        except TypeError:
            try:
                 # Try older skimage args
                ssim_ret = ssim(pred, target, full=True, multichannel=True, win_size=win_size)
                if isinstance(ssim_ret, tuple):
                    ssim_value = ssim_ret[0]
                else:
                    ssim_value = ssim_ret
            except Exception as e:
                # print(f"SSIM computation failed: {e}")
                ssim_value = 0.0
        except Exception as e:
            # print(f"SSIM computation failed: {e}")
            ssim_value = 0.0

    # SIoU
    siou_value = compute_siou(pred, target, left)
    
    return {'psnr': psnr, 'ssim': ssim_value, 'siou': siou_value}

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def convert_crop_and_resize(pil_img, width_and_height):
    """Convert, crop and resize image (Center Crop)"""
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
    """Set config defaults"""
    model_config = config['model']
    model_config['dtype'] = DTYPE_MAP[model_config['dtype']]
    if 'transformer_dtype' in model_config:
        model_config['transformer_dtype'] = DTYPE_MAP[model_config['transformer_dtype']]

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="StereoPilot Evaluation on Mono2Stereo")
    parser.add_argument("--config", type=str, default="toml/infer.toml", help="Config file path")
    parser.add_argument("--data_root", type=str, default="/root/autodl-tmp/mono2stereo", help="Root of Mono2Stereo dataset")
    parser.add_argument("--output_folder", type=str, default="mono_test", help="Output directory name")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--logs_dir", type=str, default="logs", help="Logs directory")
    args = parser.parse_args()

    # Directories
    output_root = os.path.join(os.getcwd(), args.output_folder)
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    
    log_file_path = os.path.join(args.logs_dir, 'evaluation_results.txt')

    print(f"Loading config from {args.config}")
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return

    with open(args.config) as f:
        config = json.loads(json.dumps(toml.load(f)))
    set_config_defaults(config)
    
    print("Loading model...")
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
        
        # Optimization: Pre-compute text embeddings and unload Text Encoder to save VRAM (~9GB)
        print("Pre-computing text embeddings and unloading Text Encoder...")
        model.text_encoder.model.to(device)
        with torch.no_grad():
            # Compute embedding for empty prompt used in evaluation
            context_cache = model.text_encoder([""], device)
        
        # Save context lengths
        context_lens_cache = [c.shape[0] for c in context_cache]
        
        # Unload Text Encoder components
        del model.text_encoder.model
        del model.text_encoder.tokenizer
        del model.text_encoder
        torch.cuda.empty_cache()
        print("Text Encoder unloaded.")

    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return

    target_size = (832, 480)
    
    # Find subsets
    if not os.path.exists(args.data_root):
        print(f"Data root not found: {args.data_root}")
        return

    subsets = [d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))]
    data_subsets = []
    for s in subsets:
        if os.path.exists(os.path.join(args.data_root, s, 'left')):
            data_subsets.append(s)
            
    data_subsets.sort()
    
    print(f"Found subsets: {data_subsets}")
    
    total_metrics = {'psnr': 0, 'ssim': 0, 'siou': 0, 'fps': 0, 'count': 0}
    
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Evaluation started at {time.ctime()}\n")
        log_file.write(f"Data Root: {args.data_root}\n")
        log_file.write("--------------------------------------------------\n")

    for subset in data_subsets:
        subset_metrics = {'psnr': 0, 'ssim': 0, 'siou': 0, 'fps': 0, 'count': 0}
        
        subset_path_left = os.path.join(args.data_root, subset, 'left')
        subset_path_right = os.path.join(args.data_root, subset, 'right')
        output_subset_dir = os.path.join(output_root, subset)
        os.makedirs(output_subset_dir, exist_ok=True)
        
        image_files = sorted(glob.glob(os.path.join(subset_path_left, "*")))
        image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"\nProcessing subset: {subset} ({len(image_files)} images)")
        
        for img_path in tqdm(image_files, desc=f"Evaluating {subset}"):
            try:
                img_name = os.path.basename(img_path)
                
                # Find corresponding right image
                right_img_path = os.path.join(subset_path_right, img_name)
                if not os.path.exists(right_img_path):
                    name_no_ext = os.path.splitext(img_name)[0]
                    candidates = glob.glob(os.path.join(subset_path_right, name_no_ext + ".*"))
                    if candidates:
                        right_img_path = candidates[0]
                    else:
                        pass
                        # print(f"Warning: Right image not found for {img_name}, skipping.")
                        # continue
                
                # Load and Resize GTs for metric calculation
                # If right image not found, we can't compute full metrics, but can run inference.
                # But requirement says "process... output metrics". Assuming pair exists.
                if not os.path.exists(right_img_path):
                     continue

                pil_left = Image.open(img_path)
                pil_right = Image.open(right_img_path)
                
                gt_left_processed = convert_crop_and_resize(pil_left, target_size)
                gt_right_processed = convert_crop_and_resize(pil_right, target_size)
                
                np_left = np.array(gt_left_processed)
                np_right = np.array(gt_right_processed)
                
                # Prepare Latents
                pixel_tf = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
                pixel_values = pixel_tf(gt_left_processed).to(device) # [3, H, W]
                # Add Batch and Time dims: [1, 3, 1, H, W]
                video_input = pixel_values.unsqueeze(1).unsqueeze(0) 
                
                with torch.no_grad():
                     latents = model.vae.model.encode(video_input, model.vae.scale).float()
                     # latents: [1, C, T, H, W]

                # Inference
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
                        sampling_steps=30,
                        guide_scale=5.0,
                        n_prompt="",
                        seed=42,
                        domain_label=1
                    )
                
                torch.cuda.synchronize()
                end_time = time.time()
                inference_time = end_time - start_time
                fps = 1.0 / inference_time if inference_time > 0 else 0.0
                
                # Post-process Output
                if isinstance(video_out, torch.Tensor):
                    frame_tensor = video_out.float().cpu()
                    
                    # Handle dimensions
                    if frame_tensor.ndim == 4: # [B, C, H, W] or [C, T, H, W]
                         # Assuming [C, T, H, W] or similar. 
                         # Model usually returns [C, T, H, W]. T=1.
                         if frame_tensor.shape[1] == 1:
                              frame_tensor = frame_tensor.squeeze(1) # [C, H, W]
                    elif frame_tensor.ndim == 5: # [B, C, T, H, W]
                         frame_tensor = frame_tensor.squeeze(0).squeeze(1)

                    if frame_tensor.ndim == 3:
                        # [C, H, W] -> [H, W, C]
                        # Clamp and Scale
                        frame_tensor = frame_tensor.clamp(-1, 1)
                        frame_tensor = (frame_tensor + 1) / 2.0
                        frame_numpy = frame_tensor.permute(1, 2, 0).numpy()
                        frame_numpy = (frame_numpy * 255).clip(0, 255).astype(np.uint8)
                    else:
                        print(f"Unexpected tensor shape: {frame_tensor.shape}")
                        continue
                        
                else:
                    print(f"Model output is not a tensor: {type(video_out)}")
                    continue
                
                # Calculate Metrics
                metrics = eval_metrics(frame_numpy, np_right, np_left)
                
                # Save Output
                output_path = os.path.join(output_subset_dir, os.path.splitext(img_name)[0] + ".png")
                Image.fromarray(frame_numpy).save(output_path)
                
                subset_metrics['psnr'] += metrics['psnr']
                subset_metrics['ssim'] += metrics['ssim']
                subset_metrics['siou'] += metrics['siou']
                subset_metrics['fps'] += fps
                subset_metrics['count'] += 1
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                traceback.print_exc()
                continue

        # Log Subset Results
        if subset_metrics['count'] > 0:
            count = subset_metrics['count']
            avg_psnr = subset_metrics['psnr'] / count
            avg_ssim = subset_metrics['ssim'] / count
            avg_siou = subset_metrics['siou'] / count
            avg_fps = subset_metrics['fps'] / count
            
            msg = f"Subset: {subset} | Count: {count} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | SIoU: {avg_siou:.4f} | FPS: {avg_fps:.2f}"
            print(msg)
            
            with open(log_file_path, 'a') as log_file:
                log_file.write(msg + "\n")
            
            # Add to total
            total_metrics['psnr'] += subset_metrics['psnr']
            total_metrics['ssim'] += subset_metrics['ssim']
            total_metrics['siou'] += subset_metrics['siou']
            total_metrics['fps'] += subset_metrics['fps']
            total_metrics['count'] += count
        else:
            print(f"Subset {subset} had no valid items processed.")

    # Log Total Results
    if total_metrics['count'] > 0:
        count = total_metrics['count']
        avg_psnr = total_metrics['psnr'] / count
        avg_ssim = total_metrics['ssim'] / count
        avg_siou = total_metrics['siou'] / count
        avg_fps = total_metrics['fps'] / count
        
        msg = f"\nOverall Average | Count: {count} | PSNR: {avg_psnr:.4f} | SSIM: {avg_ssim:.4f} | SIoU: {avg_siou:.4f} | FPS: {avg_fps:.2f}"
        print(msg)
        
        with open(log_file_path, 'a') as log_file:
            log_file.write("--------------------------------------------------\n")
            log_file.write(msg + "\n")
    
    print(f"Evaluation Complete. Results saved to {log_file_path}")

if __name__ == "__main__":
    main()
