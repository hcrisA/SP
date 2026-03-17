#!/bin/bash

# StereoPilot Training Example Script
# This script demonstrates various training configurations for different GPU setups

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base paths (modify these according to your setup)
CONFIG_PATH="toml/infer.toml"
TRAIN_DIR="../SP_Data/mono_train"
OUTPUT_DIR="../SP_Data/checkpoints"

# =============================================================================
# TRAINING CONFIGURATIONS
# =============================================================================

echo "StereoPilot Training Example Script"
echo "===================================="
echo ""
echo "This script provides example training commands for different GPU configurations."
echo "Uncomment the configuration that matches your GPU."
echo ""

# -------------------------------------------------------------------------
# Configuration 1: RTX 3090/4090 (24GB VRAM) - DEFAULT
# ------------------------------------------------------------------------
echo "# Configuration 1: RTX 3090/4090 (24GB VRAM)"
echo "python train.py \\"
echo "    --config $CONFIG_PATH \\"
echo "    --train_dir $TRAIN_DIR \\"
echo "    --output_dir $OUTPUT_DIR \\"
echo "    --batch_size 1 \\"
echo "    --learning_rate 3e-4 \\"
echo "    --epochs 10 \\"
echo "    --gradient_accumulation_steps 8 \\"
echo "    --mixed_precision bf16"
echo ""
echo ""

# -------------------------------------------------------------------------
# Configuration 2: RTX 5090 (32GB VRAM)
# ------------------------------------------------------------------------
echo "# Configuration 2: RTX 5090 (32GB VRAM)"
echo "python train.py \\"
echo "    --config $CONFIG_PATH \\"
echo "    --train_dir $TRAIN_DIR \\"
echo "    --output_dir $OUTPUT_DIR \\"
echo "    --batch_size 1 \\"
echo "    --learning_rate 3e-4 \\"
echo "    --epochs 12 \\"
echo "    --gradient_accumulation_steps 6 \\"
echo "    --mixed_precision bf16"
echo ""
echo ""

# -------------------------------------------------------------------------
# Configuration 3: A100/A800 (40GB VRAM)
# ------------------------------------------------------------------------
echo "# Configuration 3: A100/A800 (40GB VRAM)"
echo "python train.py \\"
echo "    --config $CONFIG_PATH \\"
echo "    --train_dir $TRAIN_DIR \\"
echo "    --output_dir $OUTPUT_DIR \\"
echo "    --batch_size 2 \\"
echo "    --learning_rate 3e-4 \\"
echo "    --epochs 15 \\"
echo "    --gradient_accumulation_steps 4 \\"
echo "    --mixed_precision bf16"
echo ""
echo ""

# -------------------------------------------------------------------------
# Configuration 4: A100/A800 (80GB VRAM)
# ------------------------------------------------------------------------
echo "# Configuration 4: A100/A800 (80GB VRAM)"
echo "python train.py \\"
echo "    --config $CONFIG_PATH \\"
echo "    --train_dir $TRAIN_DIR \\"
echo "    --output_dir $OUTPUT_DIR \\"
echo "    --batch_size 4 \\"
echo "    --learning_rate 3e-4 \\"
echo "    --epochs 20 \\"
echo "    --gradient_accumulation_steps 2 \\"
echo "    --mixed_precision bf16"
echo ""
echo ""

# -------------------------------------------------------------------------
# Configuration 5: Quick Test (2 epochs, faster training)
# ------------------------------------------------------------------------
echo "# Configuration 5: Quick Test (2 epochs)"
echo "python train.py \\"
echo "    --config $CONFIG_PATH \\"
echo "    --train_dir $TRAIN_DIR \\"
echo "    --output_dir $OUTPUT_DIR \\"
echo "    --batch_size 1 \\"
echo "    --learning_rate 3e-4 \\"
echo "    --epochs 2 \\"
echo "    --gradient_accumulation_steps 8 \\"
echo "    --mixed_precision bf16"
echo ""
echo ""

# -------------------------------------------------------------------------
# Configuration 6: High Quality (more epochs, lower LR)
# ------------------------------------------------------------------------
echo "# Configuration 6: High Quality Training"
echo "python train.py \\"
echo "    --config $CONFIG_PATH \\"
echo "    --train_dir $TRAIN_DIR \\"
echo "    --output_dir $OUTPUT_DIR \\"
echo "    --batch_size 1 \\"
echo "    --learning_rate 1e-4 \\"
echo "    --epochs 25 \\"
echo "    --gradient_accumulation_steps 8 \\"
echo "    --mixed_precision bf16"
echo ""
echo ""

# =============================================================================
# MONITORING
# =============================================================================

echo ""
echo "To monitor training with Tensorboard:"
echo "tensorboard --logdir $OUTPUT_DIR/logs"
echo ""
echo "To monitor GPU memory usage:"
echo "nvidia-smi -l 1"
echo ""

# =============================================================================
# NOTES
# =============================================================================

echo ""
echo "Notes:"
echo "- Adjust epochs based on your dataset size and convergence"
echo "- Monitor training loss - it should decrease steadily"
echo "- Check GPU memory usage - reduce batch_size if OOM"
echo "- Best model is automatically saved as best_model.safetensors"
echo ""

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

echo ""
echo "To start training with Configuration 1 (RTX 3090/4090):"
echo "python train.py \\"
echo "    --config $CONFIG_PATH \\"
echo "    --train_dir $TRAIN_DIR \\"
echo "    --output_dir $OUTPUT_DIR \\"
echo "    --batch_size 1 \\"
echo "    --learning_rate 3e-4 \\"
echo "    --epochs 10 \\"
echo "    --gradient_accumulation_steps 8 \\"
echo "    --mixed_precision bf16"
echo ""
echo "For more details, see TRAINING_README.md"
echo ""
