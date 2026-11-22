"""
Copy of the TranAD model from the ANL anldq package
(github.com/anl-hep-ai-ml/CrossExperimentalAIDQM, src/anldq).

Copied so mu2e_tranad does not need a CrossExperimentalAIDQM checkout.
Only torch is required. Files were copied unchanged except for the
relative imports:
  deep_learning/models/tranad.py  -> tranad.py
  deep_learning/registry.py       -> registry.py
  deep_learning/layer/*.py        -> layer/
  configs/model_config.py         -> config.py (TranADConfig only)

To update, recopy those files and change ANLDQ_COMMIT.
"""

ANLDQ_COMMIT = "c3b18cc4"  # last CrossExperimentalAIDQM commit touching these files

from .config import TranADConfig
from .tranad import TranAD

__all__ = ["TranAD", "TranADConfig", "ANLDQ_COMMIT"]
