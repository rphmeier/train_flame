import torch 
import torch.nn as nn
import math

class RoPE2D(nn.Module):
    """2D Rotary Position Embeddings for patch grids."""
    def __init__(self, dim, base=100.0):
        super().__init__()
        assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D RoPE"
        self.dim = dim
        self.base = base
        
    def forward(self, x, H=20, W=20):
        """
        Apply 2D RoPE to input tensor.
        x: [B, N, D] where N = H*W (patches)
        """
        B, N, D = x.shape
        assert N == H * W, f"Number of patches {N} doesn't match H*W={H*W}"

        # Compute frequencies for rotary embeddings
        half_dim = self.dim // 2
        freqs = self.base ** (torch.arange(0, half_dim, 2).float() / half_dim)
        freqs = freqs.to(device=x.device, dtype=x.dtype)
        
        # Create 2D position grid normalized to [-1, 1]
        coords_h = torch.linspace(-1, 1, H, device=x.device)
        coords_w = torch.linspace(-1, 1, W, device=x.device)
        grid = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'), dim=-1)
        coords = grid.reshape(-1, 2)  # [HW, 2]
        
        # Compute angles for rotary embedding
        # Split dimensions for x and y coordinates
        angles = coords[..., None] / freqs[None, None, :]  # [HW, 2, D//4]
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
    
class MultiHeadAttention(nn.Module):
    def __init__(self, dim=512, n_heads=8, rope_base=100.0):
        super().__init__()

        self.q = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)

        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_k = nn.LayerNorm(dim)

        if rope_base is not None:
            self.rope = RoPE2D(dim, rope_base)
        else:
            self.rope = None

        self.dim = dim
        self.n_heads = n_heads

    def forward(self, x, context, grid_size_h, grid_size_w):
        q = self.norm_q(self.q(x)) # [B, LQ, D]
        k = self.norm_k(self.k(context)) # [B, LK, D]
        v = self.v(context) # [B, LK, D]

        if self.rope is not None:
            k = self.rope(k, H=grid_size_h, W=grid_size_w)

        I = torch.eye(self.dim, dtype=q.dtype, device=q.device)
        x, _ = torch.nn.functional.multi_head_attention_forward(
            q.permute(1, 0, 2), # expects (L, B, E)
            k.permute(1, 0, 2),
            v.permute(1, 0, 2),
            self.dim,
            self.n_heads,
            in_proj_weight=I.repeat(3, 1),
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=I,
            out_proj_bias=None,
            training=self.training,
            need_weights=False
        )

        x = x.permute(1, 0, 2) # [L, B, D] -> [B, L, D]
        return self.o(x)


class TransformerDecoderBlock(nn.Module):
    """Single Transformer Decoder Block with self-attention and cross-attention."""
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, rope_base=100.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Self-attention for queries (no RoPE)
        self.self_attention = MultiHeadAttention(d_model, n_heads, rope_base=None) # type: ignore
        
        # Cross-attention: queries attend to patches
        self.cross_attention = MultiHeadAttention(d_model, n_heads, rope_base)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # MLP
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, patch_tokens, grid_size_h, grid_size_w):
        """
        Args:
            x: [B, N_queries, d_model] - landmark sequence
            patch_tokens: [B, N_patches, d_model] - patch features for cross-attention
        Returns:
            x: [B, N_queries, d_model] - refined landmarks
        """

        # Attend to patch tokens first
        x = x + self.cross_attention(self.norm1(x), patch_tokens, grid_size_h, grid_size_w)

        x = x + self.self_attention(self.norm2(x), x, grid_size_h, grid_size_w)
        
        # Feed-forward
        x = x + self.ffn(self.norm3(x))
        
        return x

class FLAMEExpressionEncoder(nn.Module):
    """
    Transformer decoder-based FLAME expression parameter estimator.
    Uses frozen DinoV3 backbone with learnable queries for each expression parameter.
    """
    def __init__(self, 
                 backbone_config,
                 num_expression_params=55,  # Can be 55 (base) or 111 (extended)
                 hidden_dim=512,
                 num_heads=8,
                 num_decoder_blocks=2,
                 rope_base=100.0):
        super().__init__()
        
        # Expression parameter configuration
        self.num_expression_params = num_expression_params
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_decoder_blocks = num_decoder_blocks

        self.backbone_dim = backbone_config.hidden_size
        self.backbone_patch_size = backbone_config.patch_size
        self.backbone_num_register_tokens = backbone_config.num_register_tokens
        
        self.in_proj = nn.Linear(self.backbone_dim, hidden_dim)

        # Learnable expression queries - one per parameter
        self.x_base = nn.Parameter(
            torch.randn(num_expression_params, hidden_dim) * 0.02
        )
        
        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(
                hidden_dim, 
                num_heads, 
                hidden_dim * 4, 
                rope_base,
            )
            for _ in range(num_decoder_blocks)
        ])
        
        # Final projection to scalar values
        self.to_params = nn.Linear(hidden_dim, 1)
        
        # Pre-compute grid size
        self._init_grid_size()
        self._init_weights()
        
    def _init_grid_size(self):
        """Calculate grid size from model config."""
        patch_size = self.backbone_patch_size  # 16
        image_size = 320
        self.grid_size = image_size // patch_size  # 20
        self.num_patches = self.grid_size * self.grid_size  # 400
        
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Initialize to predict zeros
        self.to_params.weight.data.zero_()
        self.to_params.bias.data.zero_()
    
    def forward(self, hidden_states):
        """
        Takes the feature inputs from the backbone and predicts expression parameters.
        Args:
            hidden_states: [B, 1 + num_register_tokens + num_patches, backbone_dim]
        Returns:
            expression_params: [B, num_expression_params]
        """
        # Get DinoV3 features
        B = hidden_states.size(0)
        
        # Extract only patch tokens (skip CLS token + register tokens)
        num_register_tokens = self.backbone_num_register_tokens  # Usually 4
        patch_tokens = hidden_states[:, 1 + num_register_tokens:]  # [B, 400, 1024]
        
        # Project patches to hidden dimension
        patch_tokens = self.in_proj(patch_tokens) # [B, 400, 1024]
        x = self.x_base.unsqueeze(0).expand(B, -1, -1) # [B, L, D]

        # Pass through decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, patch_tokens, self.grid_size, self.grid_size)

        # Final projection to expression parameters
        expression_params = self.to_params(x).squeeze(-1)  # [B, 55]
        
        return expression_params
