"""
LoRA (Low-Rank Adaptation) utilities for StereoPilot.

Implements LoRA layers that can be injected into Wan2.1 transformer model
to enable efficient fine-tuning with minimal parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import List, Dict, Optional, Union
import re
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation).
    
    Attributes:
        rank: LoRA rank (typically 4-64)
        alpha: LoRA alpha scaling factor (defaults to rank)
        dropout: Dropout rate for LoRA layers
        target_modules: List of module names to inject LoRA into
    """
    rank: int = 4
    alpha: Optional[float] = None
    dropout: float = 0.0
    target_modules: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.alpha is None:
            self.alpha = self.rank


class LoRALinear(nn.Module):
    """
    LoRA linear layer implementation.
    
    Replaces a linear layer with a frozen original weight plus trainable
    low-rank adaptation matrices.
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 4,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Initialize LoRA linear layer.
        
        Args:
            original_linear: Original linear layer to wrap
            rank: LoRA rank (typically 4-64)
            alpha: LoRA alpha scaling factor (defaults to rank)
            dropout: Dropout rate for LoRA layers
            dtype: Data type for LoRA parameters
        """
        super().__init__()
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.scaling = self.alpha / self.rank
        
        # Get device from original linear layer
        device = original_linear.weight.device
        
        # Freeze original weights
        self.weight = original_linear.weight
        self.bias = original_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        
        # LoRA matrices (low rank decomposition: W + BA)
        # Create on the same device as the original layer
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, dtype=dtype, device=device))
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=dropout)
        
        # Initialize LoRA weights
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        """Initialize LoRA weights using Kaiming initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining frozen weights and LoRA adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Original frozen linear
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA adaptation
        if self.rank > 0:
            lora_x = self.lora_dropout(x)
            lora_result = (lora_x @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
            result = result + self.scaling * lora_result
        
        return result


class LoRAManager:
    """
    Manages LoRA injection into transformer model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        rank: int = 4,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.bfloat16,
        target_modules: Optional[List[str]] = None
    ):
        """
        Initialize LoRA manager.
        
        Args:
            model: Transformer model to inject LoRA into
            rank: LoRA rank
            alpha: LoRA alpha scaling
            dropout: Dropout rate
            dtype: Data type
            target_modules: List of module names to inject LoRA into
                         If None, uses default list for Wan2.1
        """
        self.model = model
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank
        self.dropout = dropout
        self.dtype = dtype
        self.target_modules = target_modules or self.get_default_target_modules()
        
        self.lora_modules = {}
        self.original_modules = {}
    
    def get_default_target_modules(self) -> List[str]:
        """
        Get default target modules for Wan2.1 transformer.
        
        Based on typical transformer architecture, these are the modules
        that benefit most from LoRA adaptation:
        - Q, K, V projections in self-attention
        - Feed-forward network layers
        """
        return [
            "attn.q",
            "attn.k", 
            "attn.v",
            "attn.o",
            "ffn.fc1",
            "ffn.fc2",
        ]
    
    def create_module_filter(self) -> callable:
        """Create filter function for matching module names."""
        target_patterns = []
        for module_name in self.target_modules:
            # Convert to regex pattern
            # e.g., "attn.q" -> ".*\.attn\.q$"
            pattern = ".*" + module_name.replace(".", "\\.") + "$"
            target_patterns.append(re.compile(pattern))
        
        def filter_func(name: str) -> bool:
            """Check if module name matches any target pattern."""
            return any(pattern.match(name) for pattern in target_patterns)
        
        return filter_func
    
    def inject_lora(self):
        """
        Inject LoRA layers into the model.
        
        This replaces target linear layers with LoRALinear layers.
        """
        filter_func = self.create_module_filter()
        replaced_count = 0
        
        # Get the device from model's first parameter
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Recursively traverse model and replace modules
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and filter_func(name):
                logger = logging.getLogger(__name__)
                logger.info(f"Injecting LoRA into: {name} (shape: {module.weight.shape})")
                
                # Create LoRA wrapper
                lora_module = LoRALinear(
                    module,
                    rank=self.rank,
                    alpha=self.alpha,
                    dropout=self.dropout,
                    dtype=self.dtype
                )
                
                # Move LoRA module to the same device as the model
                lora_module = lora_module.to(device)
                
                # Replace module
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                module_name = name.rsplit(".", 1)[1] if "." in name else name
                
                if parent_name:
                    parent = dict(self.model.named_modules())[parent_name]
                    setattr(parent, module_name, lora_module)
                else:
                    # Root level module (unlikely)
                    setattr(self.model, module_name, lora_module)
                
                self.lora_modules[name] = lora_module
                replaced_count += 1
        
        logger = logging.getLogger(__name__)
        logger.info(f"✓ Injected LoRA into {replaced_count} modules on {device}")
        return replaced_count
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (LoRA parameters only)."""
        trainable_params = []
        
        # Collect LoRA parameters
        for module in self.lora_modules.values():
            trainable_params.extend([p for p in module.parameters() if p.requires_grad])
        
        # Also include domain embeddings if they exist
        if hasattr(self.model, 'parall_embedding'):
            if self.model.parall_embedding.requires_grad:
                trainable_params.append(self.model.parall_embedding)
        if hasattr(self.model, 'converge_embedding'):
            if self.model.converge_embedding.requires_grad:
                trainable_params.append(self.model.converge_embedding)
        
        return trainable_params
    
    def get_trainable_params_count(self) -> int:
        """Get total count of trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())
    
    def freeze_original_weights(self):
        """
        Freeze all original model weights, keeping only LoRA parameters trainable.
        """
        frozen_count = 0
        trainable_count = 0
        
        for name, param in self.model.named_parameters():
            if "lora_" in name or name in ['parall_embedding', 'converge_embedding']:
                # Keep LoRA and domain embeddings trainable
                param.requires_grad = True
                trainable_count += 1
            else:
                # Freeze everything else
                param.requires_grad = False
                frozen_count += 1
        
        logger = logging.getLogger(__name__)
        logger.info(
            f"✓ Frozen {frozen_count} parameters, kept {trainable_count} trainable"
        )
    
    def get_lora_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get state dict containing only LoRA parameters.
        
        This is useful for saving only the LoRA weights (small file size).
        """
        lora_state_dict = {}
        
        for name, module in self.lora_modules.items():
            for param_name, param in module.named_parameters():
                if "lora_" in param_name:
                    full_name = f"{name}.{param_name}"
                    lora_state_dict[full_name] = param.cpu()
        
        # Include domain embeddings
        if hasattr(self.model, 'parall_embedding'):
            lora_state_dict['parall_embedding'] = self.model.parall_embedding.cpu()
        if hasattr(self.model, 'converge_embedding'):
            lora_state_dict['converge_embedding'] = self.model.converge_embedding.cpu()
        
        return lora_state_dict
    
    def load_lora_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load LoRA parameters from state dict.
        
        Args:
            state_dict: State dict containing LoRA parameters
        """
        # Get the device from model's first parameter
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_state_dict = self.model.state_dict()
        
        for name, param in state_dict.items():
            if name in model_state_dict:
                # Move param to the same device as model parameter
                model_state_dict[name].copy_(param.to(device))
            else:
                logger = logging.getLogger(__name__)
                logger.warning(f"Parameter {name} not found in model state dict")
        
        logger = logging.getLogger(__name__)
        logger.info(f"✓ Loaded LoRA state dict with {len(state_dict)} parameters to {device}")
    
    def load_lora_weights(self, weights_path: str):
        """
        Load LoRA weights from safetensors file.
        
        Args:
            weights_path: Path to .safetensors file containing LoRA weights
        """
        import safetensors.torch
        
        logger = logging.getLogger(__name__)
        logger.info(f"Loading LoRA weights from {weights_path}")
        
        # Load safetensors file
        state_dict = safetensors.torch.load_file(weights_path)
        
        # Load into model
        self.load_lora_state_dict(state_dict)
    
    def merge_lora_weights(self):
        """
        Merge LoRA weights into original weights for inference.
        
        This combines W = W_original + BA * scaling, allowing the model
        to run without LoRA overhead during inference.
        """
        merged_count = 0
        
        for name, module in self.lora_modules.items():
            if hasattr(module, 'weight') and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Merge: W = W + B * A * scaling
                with torch.no_grad():
                    delta_weight = module.lora_B @ module.lora_A * module.scaling
                    module.weight += delta_weight
                
                # Zero out LoRA matrices to prevent double-counting
                module.lora_A.zero_()
                module.lora_B.zero_()
                
                merged_count += 1
        
        logger = logging.getLogger(__name__)
        logger.info(f"✓ Merged {merged_count} LoRA modules into original weights")


def create_lora_config(
    rank: int = 4,
    alpha: Optional[float] = None,
    dropout: float = 0.0,
    dtype: str = "bfloat16",
    target_modules: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Create LoRA configuration dictionary.
    
    Args:
        rank: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout rate
        dtype: Data type string ("bfloat16", "float16", "float32")
        target_modules: Target module names
    
    Returns:
        LoRA configuration dictionary
    """
    DTYPE_MAP = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    
    return {
        "rank": rank,
        "alpha": alpha if alpha is not None else rank,
        "dropout": dropout,
        "dtype": DTYPE_MAP.get(dtype, torch.bfloat16),
        "target_modules": target_modules
    }


import torch.nn.functional as F

# Add import to __all__ if this module is imported elsewhere
__all__ = [
    "LoRAConfig",
    "LoRALinear",
    "LoRAManager", 
    "create_lora_config"
]
