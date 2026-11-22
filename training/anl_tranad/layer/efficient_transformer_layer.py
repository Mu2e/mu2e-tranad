"""
Drop-in replacements for torch.nn.TransformerEncoderLayer and
TransformerDecoderLayer that use F.scaled_dot_product_attention
for memory-efficient / flash attention during training and inference.

Usage:
    layer = EfficientEncoderLayer(d_model=512, nhead=8)
    # Same API as nn.TransformerEncoderLayer
    out = layer(src, src_mask=mask, src_key_padding_mask=pad_mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class SDPAMultiheadAttention(nn.Module):
    """MultiheadAttention using F.scaled_dot_product_attention.

    Call signature matches what TransformerEncoderLayer._sa_block and
    TransformerDecoderLayer._mha_block expect:
        (query, key, value, key_padding_mask=, need_weights=, attn_mask=, is_causal=)
        -> (output, None)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        # Packed input projection (Q, K, V in one matmul) for self-attention,
        # but we keep separate projections to handle cross-attention cleanly.
        # TransformerEncoder probes these nn.MultiheadAttention attributes
        # during __init__ to decide on the nested-tensor fast-path.
        # Expose them so we remain a valid drop-in replacement.
        self._qkv_same_embed_dim = True
        self.in_proj_weight = None
        self.in_proj_bias = torch.empty(0) if bias else None

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)


    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> tuple[Tensor, None]:
        # Transpose to batch-first if needed: (seq, batch, dim) -> (batch, seq, dim)
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch, seq_q, _ = query.shape
        seq_k = key.shape[1]

        # Project and reshape to (batch, nhead, seq, head_dim)
        q = self.q_proj(query).view(batch, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch, seq_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch, seq_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Build combined mask from attn_mask, key_padding_mask, and is_causal.
        # SDPA doesn't allow attn_mask and is_causal=True simultaneously,
        # so if we need both, we generate the causal mask explicitly.
        merged_mask = attn_mask

        # Reshape 3D mask (batch*nhead, seq_q, seq_k) -> (batch, nhead, seq_q, seq_k)
        if merged_mask is not None and merged_mask.dim() == 3:
            merged_mask = merged_mask.view(batch, self.num_heads, seq_q, seq_k)

        if is_causal and (key_padding_mask is not None or merged_mask is not None):
            # Generate causal mask explicitly and merge; disable is_causal flag
            causal = torch.triu(
                torch.full((seq_q, seq_k), float("-inf"), device=q.device, dtype=q.dtype),
                diagonal=1,
            )
            merged_mask = causal if merged_mask is None else merged_mask + causal
            is_causal = False

        if key_padding_mask is not None:
            # key_padding_mask: (batch, seq_k) bool, True = ignore
            # Convert to additive: True -> -inf, False -> 0
            pad_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_k)
            pad_mask = torch.zeros_like(pad_mask, dtype=q.dtype).masked_fill(pad_mask, float("-inf"))
            if merged_mask is not None:
                merged_mask = merged_mask + pad_mask
            else:
                merged_mask = pad_mask

        # SDPA: dispatches to flash attention / memory-efficient / math backend
        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=merged_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        # Reshape back: (batch, nhead, seq_q, head_dim) -> (batch, seq_q, embed_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_q, self.embed_dim)

        # Output projection
        output = self.out_proj(attn_out)

        # Transpose back if needed: (batch, seq, dim) -> (seq, batch, dim)
        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, None


class EfficientEncoderLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer with SDPA-based attention.

    Identical norm order, residuals, and feedforward structure.
    Only the attention computation is replaced.  The forward() is
    overridden to skip PyTorch's BetterTransformer fast-path, which
    probes nn.MultiheadAttention-specific attributes.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device=None, dtype=None):
        super().__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps,
            batch_first=batch_first, norm_first=norm_first, bias=bias,
            device=device, dtype=dtype,
        )
        # Replace stock MHA with SDPA version
        self.self_attn = SDPAMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            bias=bias, batch_first=batch_first,
        )
        if device is not None or dtype is not None:
            self.self_attn = self.self_attn.to(device=device, dtype=dtype)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Standard encoder-layer forward, bypassing the BetterTransformer
        fast-path that requires nn.MultiheadAttention internals."""
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x


class EfficientDecoderLayer(nn.TransformerDecoderLayer):
    """TransformerDecoderLayer with SDPA-based attention.

    Identical forward pass, norm order, residuals, and feedforward structure.
    Both self-attention and cross-attention are replaced.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, layer_norm_eps=1e-5, batch_first=False,
                 norm_first=False, bias=True, device=None, dtype=None):
        super().__init__(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps,
            batch_first=batch_first, norm_first=norm_first, bias=bias,
            device=device, dtype=dtype,
        )
        # Replace both attention modules with SDPA versions
        self.self_attn = SDPAMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            bias=bias, batch_first=batch_first,
        )
        self.multihead_attn = SDPAMultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout,
            bias=bias, batch_first=batch_first,
        )
        if device is not None or dtype is not None:
            self.self_attn = self.self_attn.to(device=device, dtype=dtype)
            self.multihead_attn = self.multihead_attn.to(device=device, dtype=dtype)
