"""tjev: calibrated typed decisions from Qwen3.5 label logits (JAX / Flax NNX / Optax / Grain)."""

import multiprocessing as _mp
import os as _os

__version__ = "0.1.0"

# Grain data workers are spawned processes that import tjev (to unpickle the segment
# function) and, through grain, JAX. They only render and tokenize: keep them off the
# accelerator, which the trainer needs whole (google/grain#820, #1199: ~500 MB per worker).
# The process name is already set when a spawned child re-imports modules.
if _mp.current_process().name != "MainProcess":
    _os.environ["JAX_PLATFORMS"] = "cpu"
