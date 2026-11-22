"""
Model Registry Module
Manages model registration and builders
"""

from typing import Dict, Any, Callable
import logging

log = logging.getLogger(__name__)

# ============================================================================
# Model registry
# ============================================================================

MODEL_REGISTRY: Dict[str, Any] = {}
MODEL_BUILDERS: Dict[str, Callable] = {}


def register_model(name=None, arch_field=None, supports_activation=False):
    """
    Model registration:
        • name: framework-level model name (default = class.__name__)
        • arch_field: name of TrainConfig attribute 
        • supports_activation: whether model supports model_activation override
    """
    def decorator(cls):
        key = name or cls.__name__  

        MODEL_REGISTRY[key] = cls
        cls.model_name = key
        cls.arch_field = arch_field
        cls.supports_activation = supports_activation

        MODEL_BUILDERS[key] = lambda dims, cfg: cls(dims, getattr(cfg, arch_field))

        return cls
    return decorator


def resolve_model_key(model_name: str) -> str:
    """Resolve model name in a case-insensitive way to registry key."""
    if model_name in MODEL_REGISTRY:
        return model_name
    key_map = {k.lower(): k for k in MODEL_REGISTRY}
    match = key_map.get(str(model_name).lower())
    if match is None:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}")
    return match


# Export public interfaces
__all__ = [
    'MODEL_REGISTRY',
    'MODEL_BUILDERS',
    'resolve_model_key',
    'register_model',
]
