from dataclasses import dataclass, field
import warnings


# -----------------------------
# TranAD architecture config
# -----------------------------
@dataclass
class TranADConfig:
    """
    Configurations for the TranAD model architecture.

    Dimension resolution rule:
      - If `d_model > 0`, it takes precedence.
      - Else if `d_model_factor > 0`, use `d_model_factor * input_dims`.
      - Else use bounded auto defaults from model ctor (no linear blow-up).
    """
    
    # Determines how the transformer models correlation structure in the input
    correlation_mode: str = field(
        default="flattened",
        metadata={
            "help": (
                "Correlation structure learning mode for Transformer-based models.\n"
                "Options:\n"
                "  - 'channels'   : Learn dependencies *between channels* only "
                "(suitable when each channel has a single feature, D=1).\n"
                "  - 'features'   : Learn dependencies *between features within each channel* "
                "(suitable when C is small but each channel has many features).\n"
                "  - 'flattened'  : Learn full cross-channel × cross-feature interactions "
                "(recommended for multi-feature multi-channel datasets, e.g. GM2_all).\n"
                "If set by the user, it overrides the automatic selection.\n"
                "If None, AutoModelSelector determines the appropriate mode from dataset shape.\n"
                "This parameter is *TranAD-specific* and ignored for DAGMM/USAD/OmniAnomaly.\n"
            )
        },
    )


    n_window: int = field(default=10, metadata={"help": "Time window size."})
    
    
    # If d_model provided, it overrides d_model_factor
    d_model: int = field(
        default=0,
        metadata={
            "help": (
                "Absolute control of transformer dimension.\n"
                "If unset (0), model chooses a bounded default based on input size."
            )
        },
    )

    # Used only when d_model is not provided
    d_model_factor: float = field(
        default=0.0,
        metadata={
            "help": (
                "Optional legacy scaling: when d_model is not provided and d_model_factor > 0,\n"
                "resolve as d_model_factor * input_dims. Leave at 0 to use bounded auto defaults."
            ) 
        },
    )

    nhead: int = field(
        default=0,
        metadata={
            "help": "Number of attention heads. If 0, model auto-selects from practical candidates. d_model must be divisible by nhead.",
        },
    )

    d_ff: int = field(
        default=0,
        metadata={"help": "FFN dimension. If 0, auto-resolved from d_model (4x for <=512, else 3x)."},
    )
    dropout: float = field(default=0.1,  metadata={"help": "Dropout rate in [0,1)."})
    
    output_activation: str = field(
        default=None,
        metadata={
            "help": (
                "Output activation function. Options:\n"
                "- 'sigmoid': For non-negative data in range [0,1]\n"
                "- 'tanh': For data with both positive and negative values centered around 0\n"
                "- 'linear': For other data distributions"
            )
        },
    )
    
    num_layers: int = field(
        default=2, metadata={"help": "Number of encoder and decoder layers."}
    )
    use_efficient_attention: bool = field(
        default=True,
        metadata={"help": "Use SDPA-based attention layers (flash/memory-efficient)."},
    )

    def __post_init__(self):
        # ---- basic value checks ----
        if self.n_window <= 0:
            raise ValueError(f"[TranAD] n_window must be positive, got {self.n_window}")
        if self.num_layers <= 0:
            raise ValueError(f"[TranAD] num_layers must be positive, got {self.num_layers}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"[TranAD] dropout must be in [0,1), got {self.dropout}")

        # ---- mutual guidance between d_model and d_model_factor ----
        # If both are provided (d_model>0 AND factor not 0/None), warn once.
        if self.d_model > 0 and (self.d_model_factor is not None and self.d_model_factor != 0):
            warnings.warn(
                f"[TranAD] Both d_model={self.d_model} and d_model_factor={self.d_model_factor} "
                f"were provided. d_model takes precedence; d_model_factor will be ignored.",
                stacklevel=2
            )

        # ---- divisibility check (only when both numbers are explicit & meaningful) ----
        # If user explicitly set nhead>0 and d_model>0, ensure divisibility.
        if self.d_model > 0 and self.nhead > 0:
            if self.d_model % self.nhead != 0:
                raise ValueError(
                    f"[TranAD] d_model ({self.d_model}) must be divisible by nhead ({self.nhead})."
                )
        
        # ---- validate output activation ----
        valid_activations = {"sigmoid", "tanh", "linear"}
        if self.output_activation is not None and self.output_activation.lower() not in valid_activations:
            raise ValueError(
                f"[TranAD] output_activation must be one of {valid_activations}, got {self.output_activation}"
            )

