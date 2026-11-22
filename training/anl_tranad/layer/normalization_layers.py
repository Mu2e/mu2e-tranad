"""Normalization helper layers used across model architectures."""

from __future__ import annotations

import torch.nn as nn


def get_maybe_batchnorm(use_bn: bool, dim: int) -> nn.Module:
    """Return BatchNorm1d when enabled, else Identity."""
    return nn.BatchNorm1d(dim) if use_bn else nn.Identity()
