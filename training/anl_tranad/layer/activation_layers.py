"""Activation helper layers used across model architectures."""

from __future__ import annotations

import torch.nn as nn


def get_output_activation(name: str) -> nn.Module:
    """Get output activation module by name."""
    name = (name or "identity").lower()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    if name in ["linear", "identity", "none"]:
        return nn.Identity()
    raise ValueError(f"Unsupported output activation: {name}")


def get_hidden_activation(name: str) -> nn.Module:
    """Create activation module for hidden layers."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "prelu":
        return nn.PReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation type: {name}")
