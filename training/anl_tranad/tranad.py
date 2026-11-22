"""
implementations of the **TranAD** anomaly‑detection model.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (TransformerDecoder, TransformerDecoderLayer,
                      TransformerEncoder, TransformerEncoderLayer)
from .registry import register_model
from .config import TranADConfig
from .layer import PositionalEncoding, get_output_activation  # Import from layers module
from .layer.efficient_transformer_layer import EfficientEncoderLayer, EfficientDecoderLayer

# ---------------------------------------------------------------------
# TranAD – two-phase self-conditioning Transformer
# ---------------------------------------------------------------------

@register_model(
    name="TranAD",
    arch_field="tranad_arch",
    supports_activation=True
)
class TranAD(nn.Module):
    """
    Two-phase Transformer with self-conditioning for time-series
    anomaly detection.  Hyper-parameters come from a
    TranADConfig instance so they can be managed via
    YAML / JSON with explicit override + bounded auto defaults.
    """

    # --------------------------------------------------------------
    # Model behavior flags (used by data_prepare.py)
    # --------------------------------------------------------------
    is_window_model: bool = True          # this model consumes sliding windows
    keep_window: bool = True              # keep shape (T,W,F_all), do NOT flatten to (T,W*F_all)
    is_forecasting_model: bool = True     # needs last-step target (next-step prediction)
    training_strategy = "tranad"          # mix of reconstruction and self-conditioning

    @staticmethod
    def _auto_d_model(input_dims: int) -> int:
        """Choose a bounded model size instead of linear scaling with input dims."""
        if input_dims <= 128:
            return 256
        if input_dims <= 512:
            return 512
        if input_dims <= 1024:
            return 768
        return 1024

    @staticmethod
    def _auto_nhead(d_model: int) -> int:
        """
        Pick nhead from practical candidates while keeping head_dim in a healthy range.
        Preferred head_dim target is around 32.
        """
        candidates = (4, 8, 12, 16)
        valid = [h for h in candidates if d_model % h == 0 and 16 <= (d_model // h) <= 64]
        if valid:
            return min(valid, key=lambda h: abs((d_model // h) - 32))

        fallback = [h for h in candidates if d_model % h == 0]
        if fallback:
            return max(fallback)
        return 1

    @staticmethod
    def _auto_d_ff(d_model: int) -> int:
        """Use a smaller FFN multiplier for larger models to control memory."""
        if d_model <= 512:
            return 4 * d_model
        return 3 * d_model

    # --------------------------------------------------------------
    # constructor
    # --------------------------------------------------------------
    def __init__(self, input_dims: int, arch_config: TranADConfig):
        super().__init__()

        # ---------- hyper-parameters ----------
        self.name         = "TranAD"
        self.input_dims      = input_dims                       # F
        self.n_window     = arch_config.n_window              # T
        
        # Support explicit values first, then bounded auto defaults.
        # Keep d_model_factor as an opt-in legacy override when non-default.
        if hasattr(arch_config, 'd_model') and arch_config.d_model > 0:
            self.d_model = arch_config.d_model
        elif (
            hasattr(arch_config, "d_model_factor")
            and arch_config.d_model_factor not in (None, 0)
        ):
            self.d_model = max(64, int(arch_config.d_model_factor * input_dims))
        else:
            self.d_model = self._auto_d_model(input_dims)
        
        # If nhead is explicitly set in config and > 0, use it; otherwise auto-select.
        if hasattr(arch_config, 'nhead') and arch_config.nhead > 0:
            self.nhead = arch_config.nhead
        else:
            self.nhead = self._auto_nhead(self.d_model)
        
        # d_ff can be set explicitly or auto-calculated from d_model.
        if hasattr(arch_config, 'd_ff') and arch_config.d_ff > 0:
            self.d_ff = arch_config.d_ff
        else:
            self.d_ff = self._auto_d_ff(self.d_model)
        
        self.dropout      = arch_config.dropout
        self.num_layers   = arch_config.num_layers
        self.output_activation = arch_config.output_activation if hasattr(arch_config, 'output_activation') else 'sigmoid'


        # sanity check: MultiheadAttention requires divisibility
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )

        # ---------- input projection ----------
        # Project concatenated input (2 * n_feats) to model dimension (d_model)
        self.input_projection = nn.Linear(2 * self.input_dims, self.d_model)
        # Project target input (2 * n_feats) to model dimension (d_model) 
        self.target_projection = nn.Linear(2 * self.input_dims, self.d_model)
        
        # ---------- positional encoding ----------
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,  # Now matches transformer encoder dimension
            dropout=self.dropout,
            max_len=self.n_window,
        )

        # ---------- encoder ----------
        self.use_efficient_attention = arch_config.use_efficient_attention

        if self.use_efficient_attention:
            EncLayer, DecLayer = EfficientEncoderLayer, EfficientDecoderLayer
        else:
            EncLayer, DecLayer = TransformerEncoderLayer, TransformerDecoderLayer

        enc_layer = EncLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_ff,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer_encoder = TransformerEncoder(enc_layer, self.num_layers)

        # helper to build symmetric decoders
        def _make_decoder() -> TransformerDecoder:
            dec_layer = DecLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.d_ff,
                dropout=self.dropout,
                batch_first=True,
            )
            return TransformerDecoder(dec_layer, self.num_layers)

        self.transformer_decoder1 = _make_decoder()   # phase-1
        self.transformer_decoder2 = _make_decoder()   # phase-2

        # ---------- final projection ----------
        # Use configurable output activation
        self.output_proj = nn.Sequential(
            nn.Linear(self.d_model, self.input_dims),
            get_output_activation(self.output_activation),
        )

    # --------------------------------------------------------------
    # shared encoder helper
    # --------------------------------------------------------------
    def _encode(self, src: Tensor, ctx: Tensor, tgt: Tensor):
        """
        Concatenate source and conditioning tensor, project to model dimension,
        apply positional encoding, run through the Transformer encoder and return:

            tgt_rep  – projected target for decoder input
            memory   – encoder memory for the decoder
        """
        merged = torch.cat((src, ctx), dim=-1) * math.sqrt(self.input_dims)  # [B, T, 2*F]
        projected = self.input_projection(merged)  # [B, T, d_model]
        encoded = self.pos_encoder(projected)
        memory = self.transformer_encoder(encoded)
        
        # Project target to match decoder expected dimension
        tgt_duplicated = tgt.repeat(1, 1, 2)  # [B, 1, 2*F]
        tgt_rep = self.target_projection(tgt_duplicated)  # [B, 1, d_model]

        return tgt_rep, memory

    # --------------------------------------------------------------
    # forward pass – returns (x1, x2)
    # --------------------------------------------------------------
    def forward(self, src: Tensor, tgt: Tensor):
        """
        src : [B, T, F]  – input window
        tgt : [B, 1, F]  – usually the last time-step as <s> token
        """
        # ----- phase 1: no anomaly scores -----
        ctx = torch.zeros_like(src)
        tgt_rep, mem = self._encode(src, ctx, tgt)
        x1 = self.transformer_decoder1(tgt_rep, mem)
        x1 = self.output_proj(x1)

        # ----- phase 2: self-conditioning -----
        ctx = (x1 - src) ** 2                         # squared error
        tgt_rep, mem = self._encode(src, ctx, tgt)
        x2 = self.transformer_decoder2(tgt_rep, mem)
        x2 = self.output_proj(x2)

        return x1, x2
 
