import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import random
import os
import json
import math
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.nn.utils as nn_utils
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
from omegaconf import OmegaConf
from dataclasses import dataclass
import wandb
import matplotlib.pyplot as plt
from safetensors.torch import save_file, load_file
from pathlib import Path
from visualize_flame import render_flame_params
import inspect
import time

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

from FLAME.FLAME import FLAME
from renderer.renderer import Renderer
from modules.mp_2_flame import MP_2_FLAME

class RoPE2D(nn.Module):
    """2D Rotary Position Embeddings for patch grids."""
    def __init__(self, dim, base=100.0):
        super().__init__()
        assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D RoPE"
        self.dim = dim
        self.base = base
        
        # Compute frequencies for rotary embeddings
        half_dim = dim // 2
        freqs = base ** (torch.arange(0, half_dim, 2).float() / half_dim)
        self.register_buffer('freqs', freqs)
        
    def forward(self, x, H=20, W=20):
        """
        Apply 2D RoPE to input tensor.
        x: [B, N, D] where N = H*W (patches)
        """
        B, N, D = x.shape
        assert N == H * W, f"Number of patches {N} doesn't match H*W={H*W}"
        
        # Create 2D position grid normalized to [-1, 1]
        coords_h = torch.linspace(-1, 1, H, device=x.device)
        coords_w = torch.linspace(-1, 1, W, device=x.device)
        grid = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)
        coords = grid.reshape(-1, 2)  # [HW, 2]
        
        # Compute angles for rotary embedding
        # Split dimensions for x and y coordinates
        angles = coords[..., None] / self.freqs[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(-2)  # [HW, D//2]
        angles = 2 * math.pi * angles
        
        # Compute sin and cos
        cos = torch.cos(angles).repeat(1, 2)  # [HW, D]
        sin = torch.sin(angles).repeat(1, 2)  # [HW, D]
        
        # Expand for batch dimension
        cos = cos.unsqueeze(0)  # [1, HW, D]
        sin = sin.unsqueeze(0)  # [1, HW, D]
        
        # Apply rotary embedding using complex number rotation
        # Split features for rotation
        x_reshaped = x.reshape(B, N, -1, 2)  # [B, N, D//2, 2]
        x_real = x_reshaped[..., 0]  # [B, N, D//2]
        x_imag = x_reshaped[..., 1]  # [B, N, D//2]
        
        # Apply rotation: (real + i*imag) * (cos + i*sin)
        cos_reshaped = cos.reshape(1, N, -1, 2)[..., 0]  # [1, N, D//2]
        sin_reshaped = sin.reshape(1, N, -1, 2)[..., 0]  # [1, N, D//2]
        
        x_rotated_real = x_real * cos_reshaped - x_imag * sin_reshaped
        x_rotated_imag = x_real * sin_reshaped + x_imag * cos_reshaped
        
        # Combine back
        x_rotated = torch.stack([x_rotated_real, x_rotated_imag], dim=-1)  # [B, N, D//2, 2]
        x_rotated = x_rotated.reshape(B, N, D)  # [B, N, D]
        
        return x_rotated

class FaceSyntheticsDataset(Dataset):
    """Dataset for preprocessed face images with SMIRK and MediaPipe outputs."""
    def __init__(self, data_dir, split='train', config=None, manifest_path=None):
        self.root_dir = data_dir
        self.is_train = (split == 'train')
        self.split = split
        
        # Use manifest path from config or default
        if manifest_path is None:
            manifest_path = os.path.join(data_dir, 'manifest.json')
        
        # Load or create manifest
        if os.path.exists(manifest_path):
            print(f"Loading existing manifest from {manifest_path}")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        else:
            print(f"Creating new manifest at {manifest_path}")
            manifest = self._create_manifest(data_dir, config)
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
        
        # Get samples for this split
        self.sample_names = manifest[split]
        
        # Validate that files exist
        self._validate_samples()
        
        # Print statistics
        self._print_stats(manifest)
    
    def _create_manifest(self, data_dir, config):
        """Create train/val split manifest."""
        # Get all sample files
        all_samples = sorted([f.replace('.npz', '') for f in os.listdir(data_dir) 
                            if f.endswith('.npz')])
        
        # Shuffle with fixed seed for reproducibility
        random.seed(42)  # Fixed seed for reproducible splits
        random.shuffle(all_samples)
        
        # Calculate split sizes
        train_split = config.data.train_split if config else 0.9
        n_train = int(len(all_samples) * train_split)
        
        # Create manifest
        manifest = {
            'train': all_samples[:n_train],
            'val': all_samples[n_train:],
            'metadata': {
                'total_samples': len(all_samples),
                'train_samples': n_train,
                'val_samples': len(all_samples) - n_train,
                'train_split_ratio': train_split,
                'creation_seed': 42
            }
        }
        
        # Optionally, categorize by source for tracking
        train_sources = self._categorize_by_source(manifest['train'], data_dir)
        val_sources = self._categorize_by_source(manifest['val'], data_dir)
        
        manifest['metadata']['train_sources'] = train_sources
        manifest['metadata']['val_sources'] = val_sources
        
        return manifest
    
    def _categorize_by_source(self, sample_names, data_dir):
        """Count samples by source type."""
        sources = {'facesynthetics': 0, 'real': 0}
        for sample_name in sample_names:
            try:
                data = np.load(os.path.join(data_dir, f'{sample_name}.npz'))
                source = str(data.get('source', 'unknown'))
                if source in sources:
                    sources[source] += 1
            except:
                pass
        return sources
    
    def _validate_samples(self):
        """Check that all samples in manifest exist."""
        missing_samples = []
        for sample_name in self.sample_names:
            img_path = os.path.join(self.root_dir, f'{sample_name}.png')
            npz_path = os.path.join(self.root_dir, f'{sample_name}.npz')
            if not os.path.exists(img_path) or not os.path.exists(npz_path):
                missing_samples.append(sample_name)
        
        if missing_samples:
            print(f"Warning: {len(missing_samples)} samples missing from disk")
            # Remove missing samples
            self.sample_names = [s for s in self.sample_names if s not in missing_samples]
    
    def _print_stats(self, manifest):
        """Print dataset statistics."""
        metadata = manifest.get('metadata', {})
        print(f"\n{self.split.upper()} Dataset Statistics:")
        print(f"  Samples: {len(self.sample_names)}")
        
        if self.split in ['train', 'val']:
            sources = metadata.get(f'{self.split}_sources', {})
            if sources:
                print(f"  FaceSynthetics: {sources.get('facesynthetics', 0)}")
                print(f"  Real images: {sources.get('real', 0)}")

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, idx):
        """
        Load a sample. Training data gets shuffled by DataLoader,
        but the actual samples in train vs val remain consistent.
        """
        try:
            # Load preprocessed image and parameters
            sample_name = self.sample_names[idx]
            img_path = os.path.join(self.root_dir, f'{sample_name}.png')
            params_path = os.path.join(self.root_dir, f'{sample_name}.npz')
            
            # Load image (already 320x320 from preprocessing)
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load precomputed parameters
            data = np.load(params_path)
            
            # Validate shapes
            expression_params = data['expression']
            assert expression_params.shape == (55,), f"Invalid expression shape: {expression_params.shape}"
            
            shape_params = data['shape']
            assert shape_params.shape == (300,), f"Invalid shape shape: {shape_params.shape}"
            
            pose_params = data['pose']
            assert pose_params.shape == (6,), f"Invalid pose shape: {pose_params.shape}"
            
            blendshape_scores = data['blendshape_scores']
            assert blendshape_scores.shape == (52,), f"Invalid blendshape shape: {blendshape_scores.shape}"
            
            # Normalize image for DinoV3
            image_normalized = image.astype(np.float32) / 255.0
            image_normalized = (image_normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
            
            # Convert to tensors
            expression_tensor = torch.tensor(expression_params, dtype=torch.float32)
            blendshape_tensor = torch.tensor(blendshape_scores, dtype=torch.float32)
            shape_tensor = torch.tensor(shape_params, dtype=torch.float32)
            pose_tensor = torch.tensor(pose_params, dtype=torch.float32)
            
            return image_tensor, expression_tensor, blendshape_tensor, shape_tensor, pose_tensor
            
        except Exception as e:
            print(f"Error loading sample {sample_name}: {e}")
            # Try next sample
            return self.__getitem__((idx + 1) % len(self))


def create_dataloaders(config):
    """
    Create train and validation dataloaders with persistent splits.
    The DataLoader handles shuffling of training data each epoch.
    """
    # Create datasets (will use same manifest)
    train_dataset = FaceSyntheticsDataset(
        config.data.data_dir, 
        split='train', 
        config=config
    )
    
    val_dataset = FaceSyntheticsDataset(
        config.data.data_dir, 
        split='val', 
        config=config
    )
    
    # Create dataloaders
    # shuffle=True on train_loader means data gets shuffled each epoch
    # But the actual samples in train vs val remain the same
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.data.batch_size,
        shuffle=True,  # This shuffles training data each epoch
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.data.val_batch_size,
        shuffle=False,  # Never shuffle validation
        num_workers=config.data.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False
    )
    
    return train_loader, val_loader

class TransformerDecoderBlock(nn.Module):
    """Single Transformer Decoder Block with self-attention and cross-attention."""
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Self-attention for queries
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=0
        )
        self.self_attn_norm = nn.LayerNorm(d_model)
        
        # Cross-attention: queries attend to patches
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=0
        )
        self.cross_attn_norm = nn.LayerNorm(d_model)
        
        # MLP
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, queries, keys, values, return_attention=False):
        """
        Args:
            queries: [B, N_queries, d_model] - learnable parameter queries
            keys: [B, N_patches, d_model] - patch features for cross-attention
            values: [B, N_patches, d_model] - patch features for cross-attention
        Returns:
            queries: [B, N_queries, d_model] - refined queries
        """
        # Self-attention: queries attend to each other
        queries_self, self_attn_weights = self.self_attention(
            query=queries,
            key=queries,
            value=queries,
            need_weights=return_attention
        )
        queries = self.self_attn_norm(queries + queries_self)
        
        # Cross-attention: queries attend to patches
        queries_cross, cross_attn_weights = self.cross_attention(
            query=queries,
            key=keys,
            value=values,
            need_weights=return_attention
        )
        queries = self.cross_attn_norm(queries + queries_cross)
        
        # Feed-forward
        queries_ffn = self.ffn(queries)
        queries = self.ffn_norm(queries + queries_ffn)
        
        if return_attention:
            return queries, (self_attn_weights, cross_attn_weights)
        return queries

class FLAMEExpressionEncoder(nn.Module):
    """
    Transformer decoder-based FLAME expression parameter estimator.
    Uses frozen DinoV3 backbone with learnable queries for each expression parameter.
    """
    def __init__(self, 
                 model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
                 num_expression_params=55,  # Can be 55 (base) or 111 (extended)
                 hidden_dim=512,
                 num_heads=8,
                 num_decoder_blocks=2,
                 freeze_backbone=True,
                 rope_base=100.0):
        super().__init__()
        
        # Expression parameter configuration
        self.num_expression_params = num_expression_params
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_decoder_blocks = num_decoder_blocks
        
        # Load DinoV3 backbone (frozen)
        self.backbone = AutoModel.from_pretrained(model_name)
        backbone_dim = self.backbone.config.hidden_size  # 1024 for ViT-L
        
        # Freeze backbone as per strategy
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Using frozen DinoV3 backbone")
        
        # Separate key and value projections: 1024 -> 512
        self.k_proj = nn.Linear(backbone_dim, hidden_dim)
        self.v_proj = nn.Linear(backbone_dim, hidden_dim)
        
        # RoPE for keys (relative positioning)
        self.rope = RoPE2D(hidden_dim, base=rope_base)
        
        # Learnable expression queries - one per parameter
        self.expression_queries = nn.Parameter(
            torch.randn(num_expression_params, hidden_dim) * 0.02
        )
        
        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(hidden_dim, num_heads, hidden_dim * 4)
            for _ in range(num_decoder_blocks)
        ])
        
        # Final projection to scalar values
        self.to_params = nn.Linear(hidden_dim, 1)
        
        # Pre-compute grid size
        self._init_grid_size()
        self._init_weights()
        
    def _init_grid_size(self):
        """Calculate grid size from model config."""
        patch_size = self.backbone.config.patch_size  # 16
        image_size = 320
        self.grid_size = image_size // patch_size  # 20
        self.num_patches = self.grid_size * self.grid_size  # 400
        
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Initialize to predict zeros
        self.to_params.weight.data.zero_()
        self.to_params.bias.data.zero_()
    
    def forward(self, pixel_values, return_attention=False):
        # Get DinoV3 features
        outputs = self.backbone(pixel_values=pixel_values, return_dict=True)
        hidden_states = outputs.last_hidden_state  # [B, 1 + num_register_tokens + num_patches, hidden_dim]
        B = hidden_states.size(0)
        
        # Extract only patch tokens (skip CLS token + register tokens)
        num_register_tokens = self.backbone.config.num_register_tokens  # Usually 4
        patch_tokens = hidden_states[:, 1 + num_register_tokens:]  # [B, 400, 1024]
        
        # Project patches to hidden dimension
        keys_proj = self.k_proj(patch_tokens)  # [B, 400, 512]
        values_proj = self.v_proj(patch_tokens)  # [B, 400, 512]
        
        # Apply RoPE to keys
        keys_rope = self.rope(keys_proj, H=self.grid_size, W=self.grid_size)  # [B, 400, 512]
        
        # Expand expression queries for batch
        queries = self.expression_queries.unsqueeze(0).expand(B, -1, -1)  # [B, 55, 512]
        
        # Pass through decoder blocks
        attention_weights = []
        for decoder_block in self.decoder_blocks:
            if return_attention:
                queries, attn_weights = decoder_block(queries, keys_rope, values_proj, return_attention=True)
                attention_weights.append(attn_weights)
            else:
                queries = decoder_block(queries, keys_rope, values_proj)
        
        # Final projection to expression parameters
        expression_params = self.to_params(queries).squeeze(-1)  # [B, 55]
        
        if return_attention:
            return expression_params, attention_weights
        return expression_params

# Removed obsolete remapping functions - model now outputs 111 params directly

def compute_extended_loss(predictions, smirk_targets, mp2flame_params):
    """
    Compute loss for extended model (111 parameters) according to plan.md:
    - Indices 0-49: SMIRK expression[0:50] - weight 1.0
    - Indices 50-99: MP2FLAME expression[50:100] - weight 0.5
    - Indices 100-101: SMIRK eyelid (expression[50:52]) - weight 1.0
    - Indices 102-104: SMIRK jaw (expression[52:55]) - weight 1.0
    - Indices 105-110: MP2FLAME eye_pose - weight 1.0
    
    Args:
        predictions: Model predictions [B, 111]
        smirk_targets: SMIRK parameters [B, 55]
        mp2flame_params: Dict with MP2FLAME outputs
    
    Returns:
        loss: Weighted MSE loss
    """
    # Define weights for each component
    weights = {
        'smirk_first_50': 1.0,
        'mp2flame_50': 0.5,
        'smirk_eyelid': 1.1,
        'smirk_jaw': 1.1,
        'mp2flame_eye_pose': 1.0
    }
    
    # Compute individual losses with weights
    loss_smirk_first_50 = F.mse_loss(predictions[:, :50], smirk_targets[:, :50]) * weights['smirk_first_50']
    loss_mp2flame_50 = F.mse_loss(predictions[:, 50:100], mp2flame_params['expression'][:, 50:100]) * weights['mp2flame_50']
    loss_smirk_eyelid = F.mse_loss(predictions[:, 100:102], smirk_targets[:, 50:52]) * weights['smirk_eyelid']
    loss_smirk_jaw = F.mse_loss(predictions[:, 102:105], smirk_targets[:, 52:55]) * weights['smirk_jaw']
    loss_mp2flame_eye_pose = F.mse_loss(predictions[:, 105:111], mp2flame_params['eye_pose']) * weights['mp2flame_eye_pose']
    
    # Sum weighted losses and normalize by sum of weights
    total_weight = sum(weights.values())
    combined_loss = (loss_smirk_first_50 + loss_mp2flame_50 + loss_smirk_eyelid + 
                     loss_smirk_jaw + loss_mp2flame_eye_pose) / total_weight
    
    return combined_loss

def generate_inference_example(model, config, epoch, device):
    """Generate FLAME renders comparing SMIRK and trained model predictions."""
    # Initialize FLAME and renderer
    flame = FLAME(n_exp=100).to(device)
    renderer = Renderer().to(device)
    
    # Get a validation sample
    val_dataset = FaceSyntheticsDataset(config.data.data_dir, 'val', config)
    if len(val_dataset) == 0:
        print("Warning: No validation images found for inference example")
        return
    
    # Get dataset sample
    val_image, smirk_expression, blendshape_scores, smirk_shape, smirk_pose = val_dataset[0]
    
    # Initialize MP2FLAME converter if not already done
    
    mp2flame = MP_2_FLAME(mappings_path="modules/mp2flame_mappings/")
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        image_tensor = val_image.unsqueeze(0).to(device)
        
        # Important: synchronize before timing if using GPU
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        predicted_expression = model(image_tensor).squeeze(0)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Calculate and print
        inference_time_ms = (end_time - start_time) * 1000
        print(f"Inference time: {inference_time_ms:.2f} ms")
        
        # Compute MP2FLAME parameters for ground truth visualization
        if blendshape_scores is not None:
            blendshape_tensor = blendshape_scores.unsqueeze(0).to(device) if blendshape_scores.dim() == 1 else blendshape_scores.to(device)
            bs_numpy = blendshape_tensor.cpu().numpy()
            exp_mp, pose_mp, eye_pose_mp = mp2flame.convert(bs_numpy)

            # Convert to torch
            mp2flame_gt = {
                'expression': torch.tensor(exp_mp, device=device, dtype=torch.float32).squeeze(0),
                'eye_pose': torch.tensor(eye_pose_mp, device=device, dtype=torch.float32).squeeze(0)
            }
            # Squeeze batch dimension if needed
            if mp2flame_gt['expression'].dim() > 1:
                mp2flame_gt = {k: v.squeeze(0) for k, v in mp2flame_gt.items()}
    
    # Determine which parameters to use based on config
    use_smirk_shape = config.training.get('use_smirk_shape', True)
    use_smirk_pose = config.training.get('use_smirk_pose', True)
    use_smirk_expression = config.training.get('use_smirk_expression', False)
    
    # Build complete ground truth with both SMIRK and MP2FLAME data
    smirk_extended = torch.zeros(111, device=device)
    smirk_extended[:50] = smirk_expression[:50].to(device)  # First 50 expression from SMIRK
    smirk_extended[50:100] = mp2flame_gt['expression'][50:100]  # Second 50 expression from MP2FLAME
    smirk_extended[100:102] = smirk_expression[50:52].to(device)  # Eyelid from SMIRK
    smirk_extended[102:105] = smirk_expression[52:55].to(device)  # Jaw from SMIRK
    smirk_extended[105:111] = mp2flame_gt['eye_pose']  # Eye pose from MP2FLAME
    
    # SMIRK render (ground truth)
    smirk_render = render_flame_params(
        flame, renderer, 
        expression_params=smirk_extended,
        shape_params=smirk_shape if use_smirk_shape else None,
        pose_params=smirk_pose if use_smirk_pose else None,
        device=device
    )
    
    # Model render (use SMIRK or predicted expression based on config)
    model_render = render_flame_params(
        flame, renderer,
        expression_params=smirk_extended if use_smirk_expression else predicted_expression,
        shape_params=smirk_shape if use_smirk_shape else None,
        pose_params=smirk_pose if use_smirk_pose else None,
        device=device
    )
    
    # Convert to numpy for matplotlib
    if isinstance(smirk_render, torch.Tensor):
        smirk_img = smirk_render.squeeze().permute(1, 2, 0).cpu().numpy()
    if isinstance(model_render, torch.Tensor):
        model_img = model_render.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(smirk_img)
    ax1.set_title('SMIRK Ground Truth')
    ax1.axis('off')
    
    ax2.imshow(model_img)
    ax2.set_title('Model Prediction')
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Save comparison
    os.makedirs(config.training.inference_examples_dir, exist_ok=True)
    output_path = f"{config.training.inference_examples_dir}/flame_comparison_epoch_{epoch+1:03d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Generated FLAME comparison: {output_path}")
    
    # Log to wandb with consistent key name for progression over time
    if config.wandb.enabled:
        wandb.log({
            "flame_comparison": wandb.Image(output_path),  # Same key every time
            "epoch": epoch + 1
        })


# Attention map logging removed


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model and optimizer state from checkpoint files."""
    try:
        print(f"Loading model from: {checkpoint_path}")
        model_state = load_file(checkpoint_path, device=str(device))
        model.load_state_dict(model_state)
        
        # Look for optimizer checkpoint with same base name
        optimizer_path = checkpoint_path.replace('.safetensors', '_optimizer.pth')
        if os.path.exists(optimizer_path):
            print(f"Loading optimizer from: {optimizer_path}")
            optimizer_state = torch.load(optimizer_path, map_location=device)
            optimizer.load_state_dict(optimizer_state)
        else:
            print("Warning: Optimizer checkpoint not found, using fresh optimizer state")
            
        # Extract epoch number from filename (e.g., checkpoint_epoch_050.safetensors -> 50)
        filename = os.path.basename(checkpoint_path)
        if 'epoch_' in filename:
            epoch_str = filename.split('epoch_')[1].split('.')[0]
            start_epoch = int(epoch_str)
        else:
            start_epoch = 0
            
        print(f"Resumed from epoch {start_epoch}")
        return start_epoch
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting from scratch...")
        return 0
    
def manage_checkpoints(checkpoint_dir, max_checkpoints=16):
    """
    Keep only the most recent N checkpoints, deleting older ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
    """
    # Find all checkpoint files (safetensors)
    checkpoint_files = sorted([
        f for f in os.listdir(checkpoint_dir) 
        if f.endswith('.safetensors') and 'checkpoint_epoch_' in f
    ])
    
    # Extract epoch numbers and sort by epoch
    checkpoints_with_epochs = []
    for ckpt in checkpoint_files:
        try:
            # Extract epoch number from filename
            epoch_str = ckpt.split('checkpoint_epoch_')[1].split('.')[0]
            epoch_num = int(epoch_str)
            checkpoints_with_epochs.append((epoch_num, ckpt))
        except:
            continue
    
    # Sort by epoch number
    checkpoints_with_epochs.sort(key=lambda x: x[0])
    
    # If we have more than max_checkpoints, delete the oldest
    if len(checkpoints_with_epochs) > max_checkpoints:
        num_to_delete = len(checkpoints_with_epochs) - max_checkpoints
        
        for i in range(num_to_delete):
            epoch_num, ckpt_file = checkpoints_with_epochs[i]
            
            # Delete model checkpoint
            model_path = os.path.join(checkpoint_dir, ckpt_file)
            if os.path.exists(model_path):
                os.remove(model_path)
                print(f"Deleted old checkpoint: {ckpt_file}")
            
            # Delete associated optimizer state
            optimizer_path = model_path.replace('.safetensors', '_optimizer.pth')
            if os.path.exists(optimizer_path):
                os.remove(optimizer_path)
                print(f"Deleted old optimizer state: {os.path.basename(optimizer_path)}")

def train_model(config_path="config.yaml", restore_checkpoint=None):
    # Load configuration
    config = OmegaConf.load(config_path)
    device = torch.device(config.device)
    
    # Initialize MP2FLAME converter (always needed)
    mp2flame = MP_2_FLAME(mappings_path="modules/mp2flame_mappings/")
    print("Initialized MP2FLAME converter for MediaPipe blendshapes")
    
    # Initialize wandb
    if config.wandb.enabled:
        if config.wandb.api_key:
            wandb.login(key=config.wandb.api_key)
        wandb.init(
                project=config.wandb.project,
                name=config.wandb.name,
                config=OmegaConf.to_container(config, resolve=True),
                tags=config.wandb.tags if config.wandb.tags else None
        )
    
    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    # Dataset with config
    train_dataset = FaceSyntheticsDataset(config.data.data_dir, 'train', config)
    val_dataset = FaceSyntheticsDataset(config.data.data_dir, 'val', config)
    
    train_loader = DataLoader(train_dataset, 
                            batch_size=config.data.batch_size,
                            shuffle=True, 
                            num_workers=config.data.num_workers,
                            pin_memory=True,
                            persistent_workers=True,
                            drop_last=True)  # Drop partial batches for consistent training
    val_loader = DataLoader(val_dataset, 
                          batch_size=config.data.val_batch_size,
                          shuffle=False, 
                          num_workers=config.data.num_workers,
                          pin_memory=True,
                          persistent_workers=True,
                          drop_last=False)  # Keep all validation data for accurate metrics
    
    # Model with attention-based architecture - must be 111 params for extended format
    assert config.model.num_expression_params == 111, "Model must output 111 parameters for extended format"
    model = FLAMEExpressionEncoder(
        model_name=config.model.model_name,
        num_expression_params=111,  # Always 111 for extended format
        hidden_dim=512,
        num_heads=8,                    
        num_decoder_blocks=2,
        freeze_backbone=config.model.freeze_backbone,
        rope_base=100
).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Log model architecture (minimal tracking - no gradients/parameters)
    if config.wandb.enabled:
        wandb.watch(model, log=None)
    
    # Optimizations
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True
    
    # MSE Loss for expression parameters
    criterion = nn.MSELoss()
    
    # Optimizer - only trainable parameters (decoder components)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, 
                                 lr=config.training.lr,
                                 weight_decay=config.training.weight_decay)
    
    print(f"Training {len(trainable_params)} parameter groups")
    
    schedulers = [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.05, end_factor=1.0, total_iters=10),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90, eta_min=2e-5)
    ]

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers, 
        milestones=[10]  # Switch at epoch 10
    )
    
    scaler = GradScaler()
    global_step = 0
    start_epoch = 0
    
    # Load checkpoint if specified
    if restore_checkpoint:
        start_epoch = load_checkpoint(model, optimizer, restore_checkpoint, device)
        print(f"Resuming training from epoch {start_epoch}")
    
    print(f"\nTraining with Transformer Decoder architecture:")
    print(f"  - Model dimension: {model.hidden_dim}")
    print(f"  - Num Decoder Blocks: {model.num_decoder_blocks}")
    print(f"  - Num Heads: {model.num_heads}")
    print(f"  - Num Queries: {model.num_expression_params}")
    print(f"  - Learning rate: {config.training.lr}")
    
    for epoch in range(start_epoch, config.training.num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_idx, (images, targets, blendshape_scores, shape_params, pose_params) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Move blendshapes to device and compute MP2FLAME params
            blendshape_scores = blendshape_scores.to(device, non_blocking=True)
            
            with autocast(dtype=torch.bfloat16):
                predictions = model(images)
                
                # Compute MP2FLAME parameters (always needed)
                with torch.no_grad():
                    # Convert to numpy for MP2FLAME
                    bs_numpy = blendshape_scores.cpu().numpy()
                    exp_mp, pose_mp, eye_pose_mp = mp2flame.convert(bs_numpy)
    
                    # Convert back to torch tensors
                    mp2flame_params = {
                        'expression': torch.tensor(exp_mp, device=device, dtype=torch.float32),
                        'eye_pose': torch.tensor(eye_pose_mp, device=device, dtype=torch.float32)
    }
                
                # Compute loss using extended format
                loss = compute_extended_loss(predictions, targets, mp2flame_params)

            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            grad_norm = nn_utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            
            train_loss += loss.item()
            global_step += 1
            
            # Log to wandb
            if config.wandb.enabled and batch_idx % config.training.log_every == 0:
                log_dict = {
                    'train/loss': loss.item(),
                    'train/grad_norm': grad_norm.item(),
                    'global_step': global_step
                }
                
                if config.model.freeze_backbone:
                    log_dict['train/lr_attention'] = optimizer.param_groups[0]['lr']
                else:
                    log_dict['train/lr_lora'] = optimizer.param_groups[0]['lr']
                    log_dict['train/lr_attention'] = optimizer.param_groups[1]['lr']
                
                wandb.log(log_dict)
            
            if batch_idx % config.training.log_every == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")

        scheduler.step()

        # Validation
        if (epoch + 1) % config.training.val_every == 0:
            model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch_idx, (images, targets, blendshape_scores, shape_params, pose_params) in enumerate(val_loader):
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    # Move blendshapes to device and compute MP2FLAME params
                    blendshape_scores = blendshape_scores.to(device, non_blocking=True)
                    
                    with autocast(dtype=torch.bfloat16):
                        predictions = model(images)
                        
                        # Compute MP2FLAME parameters (always needed)
                        with torch.no_grad():
                            # Convert to numpy for MP2FLAME
                            bs_numpy = blendshape_scores.cpu().numpy()
                            exp_mp, pose_mp, eye_pose_mp = mp2flame.convert(bs_numpy)
                            
                            # Convert back to torch tensors
                            mp2flame_params = {
                                'expression': torch.tensor(exp_mp, device=device, dtype=torch.float32),
                                'eye_pose': torch.tensor(eye_pose_mp, device=device, dtype=torch.float32)
                            }
                        
                        # Compute loss using extended format
                        loss = compute_extended_loss(predictions, targets, mp2flame_params)

                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}")
            
            # Log epoch metrics to wandb
            if config.wandb.enabled:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss/len(train_loader),
                    'val/loss': avg_val_loss,
                    'global_step': global_step
                })
            
            # Save checkpoint every N epochs or on final epoch
            if ((epoch + 1) % config.training.save_every_n == 0) or (epoch == config.training.num_epochs - 1):
                os.makedirs(config.training.checkpoint_dir, exist_ok=True)
                
                # Create checkpoint filename with epoch number
                checkpoint_name = f"expression_weights_{epoch+1:03d}"
                
                # Save model weights in safetensors format
                model_path = f'{config.training.checkpoint_dir}/{checkpoint_name}.safetensors'
                save_file(model.state_dict(), model_path)

                # Delete old checkpoints
                manage_checkpoints(config.training.checkpoint_dir, max_checkpoints=16)
                
                # Save optimizer state in PyTorch format (safetensors doesn't support complex state)
                optimizer_path = f'{config.training.checkpoint_dir}/{checkpoint_name}_optimizer.pth'
                torch.save(optimizer.state_dict(), optimizer_path)
                
                print(f"Saved checkpoint: {checkpoint_name} (Val Loss: {avg_val_loss:.4f})")
                
                # Generate inference example visualization
                try:
                    generate_inference_example(model, config, epoch, device)
                except Exception as e:
                    print(f"Warning: Failed to generate inference example: {e}")
                
                # Switch model back to training mode
                model.train()
    
    if config.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FaceSynthetics landmark detection model")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--restore", help="Path to checkpoint file to restore from")
    
    args = parser.parse_args()
    train_model(config_path=args.config, restore_checkpoint=args.restore)