"""Strict, streaming import of HF Qwen3.5 safetensors (text decoder only), and stacking.

Vision tower (``model.visual.*``) and multi-token-prediction head (``mtp.*``) are
skipped by name; any other unexpected tensor, missing tensor or shape mismatch fails.
``lm_head.weight`` is loaded only for untied checkpoints (9B); a tied checkpoint that
still ships one is skipped (it equals the embedding). :func:`stack_layers` then stacks
the layers of each position in the 3 × GDN + 1 × attention period over super-blocks, the
layout the scanned decoder expects.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from safetensors import safe_open

from tjev.model.config import ModelConfig
from tjev.utils import file_hash, fingerprint

TEXT_PREFIXES = ("model.language_model.", "model.")
SKIPPED_PREFIXES = ("model.visual.", "mtp.")


def _inside(folder: Path, name: str) -> Path:
    path = (folder / name).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError(f"Checkpoint index contains an unsafe path: {name}")
    return path


def tensor_files(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    index = folder / "model.safetensors.index.json"
    if index.exists():
        names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [_inside(folder, n) for n in names]
    single = folder / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"No safetensors checkpoint in {folder}")
    return [single]


def expected_shapes(c: ModelConfig) -> dict[str, tuple[int, ...]]:
    shapes = {
        "embed_tokens.weight": (c.vocab_size, c.hidden_size),
        "norm.weight": (c.hidden_size,),
    }
    if not c.tie_word_embeddings:
        shapes["lm_head.weight"] = (c.vocab_size, c.hidden_size)
    for i in range(c.num_layers):
        p = f"layers.{i}."
        shapes[p + "input_layernorm.weight"] = (c.hidden_size,)
        shapes[p + "post_attention_layernorm.weight"] = (c.hidden_size,)
        shapes[p + "mlp.gate_proj.weight"] = (c.intermediate_size, c.hidden_size)
        shapes[p + "mlp.up_proj.weight"] = (c.intermediate_size, c.hidden_size)
        shapes[p + "mlp.down_proj.weight"] = (c.hidden_size, c.intermediate_size)
        if (i + 1) % c.full_attention_interval == 0:
            a = p + "self_attn."
            shapes[a + "q_proj.weight"] = (2 * c.num_heads * c.head_dim, c.hidden_size)
            shapes[a + "k_proj.weight"] = (c.num_kv_heads * c.head_dim, c.hidden_size)
            shapes[a + "v_proj.weight"] = (c.num_kv_heads * c.head_dim, c.hidden_size)
            shapes[a + "o_proj.weight"] = (c.hidden_size, c.num_heads * c.head_dim)
            shapes[a + "q_norm.weight"] = (c.head_dim,)
            shapes[a + "k_norm.weight"] = (c.head_dim,)
        else:
            g = p + "linear_attn."
            shapes[g + "in_proj_qkv.weight"] = (c.conv_dim, c.hidden_size)
            shapes[g + "in_proj_z.weight"] = (c.linear_value_dim, c.hidden_size)
            shapes[g + "in_proj_b.weight"] = (c.linear_num_value_heads, c.hidden_size)
            shapes[g + "in_proj_a.weight"] = (c.linear_num_value_heads, c.hidden_size)
            shapes[g + "out_proj.weight"] = (c.hidden_size, c.linear_value_dim)
            shapes[g + "conv1d.weight"] = (c.conv_dim, 1, c.conv_kernel)
            shapes[g + "A_log"] = (c.linear_num_value_heads,)
            shapes[g + "dt_bias"] = (c.linear_num_value_heads,)
            shapes[g + "norm.weight"] = (c.linear_value_head_dim,)
    return shapes


def read_config(folder: str | Path) -> ModelConfig:
    return ModelConfig.from_hf(json.loads((Path(folder) / "config.json").read_text()))


def load_hf(folder: str | Path, dtype: str = "bfloat16") -> tuple[ModelConfig, dict]:
    """Returns (config, flat tensors keyed without the text prefix) as host numpy."""
    import ml_dtypes

    folder = Path(folder)
    config = read_config(folder)
    expected = expected_shapes(config)
    target = {"bfloat16": ml_dtypes.bfloat16, "float32": np.float32}[dtype]
    tensors: dict[str, np.ndarray] = {}
    for file in tensor_files(folder):
        with safe_open(str(file), framework="numpy") as f:
            for name in f.keys():  # noqa: SIM118 (safe_open is not iterable)
                if name.startswith(SKIPPED_PREFIXES) or (
                    name.startswith("lm_head.") and config.tie_word_embeddings
                ):
                    continue
                prefix = next((p for p in TEXT_PREFIXES if name.startswith(p)), None)
                key = name[len(prefix) :] if prefix else name
                if key not in expected:
                    raise ValueError(
                        f"Unexpected checkpoint tensor {name}; refusing a partial import"
                    )
                if key in tensors:
                    raise ValueError(f"Duplicate tensor {name}")
                shape = tuple(f.get_slice(name).get_shape())
                if shape != expected[key]:
                    raise ValueError(f"Shape mismatch for {name}: {shape} != {expected[key]}")
                value = f.get_tensor(name)
                # Norm/decay parameters stay fp32; matrices go to the storage dtype.
                keep32 = key.endswith(("A_log", "dt_bias", "norm.weight", "layernorm.weight"))
                tensors[key] = value.astype(np.float32 if keep32 else target)
    missing = sorted(set(expected) - set(tensors))
    if missing:
        raise ValueError(f"Missing tensors: {missing[:5]} (+{max(0, len(missing) - 5)})")
    return config, tensors


def snapshot_identity(folder: str | Path) -> dict:
    """Content identity of a snapshot (config + tensor file hashes)."""
    folder = Path(folder)
    files = {p.name: file_hash(p) for p in tensor_files(folder)}
    files["config.json"] = file_hash(folder / "config.json")
    return {"files": files, "identity": fingerprint(files)}


def stack_layers(tensors: dict, config: ModelConfig) -> dict:
    """HF flat names → {'embed_tokens.weight', 'norm.weight', 'blocks': [4 dicts]} where
    blocks[j] holds layer (4s+j) tensors stacked over super-blocks s."""
    interval = config.full_attention_interval
    blocks = []
    for j in range(interval):
        prefix = f"layers.{j}."
        keys = [k[len(prefix) :] for k in tensors if k.startswith(prefix)]
        stacked = {}
        for key in keys:
            parts = [
                tensors.pop(f"layers.{s * interval + j}.{key}")
                for s in range(config.num_super_blocks)
            ]
            xp = np if isinstance(parts[0], np.ndarray) else jnp  # jnp under eval_shape
            stacked[key] = xp.stack(parts)
        blocks.append(stacked)
    leftover = [k for k in tensors if k.startswith("layers.")]
    if leftover:
        raise ValueError(f"Unstacked layer tensors: {leftover[:5]}")
    out = {
        "embed_tokens.weight": tensors.pop("embed_tokens.weight"),
        "norm.weight": tensors.pop("norm.weight"),
        "blocks": blocks,
    }
    if "lm_head.weight" in tensors:
        out["lm_head.weight"] = tensors.pop("lm_head.weight")
    return out
