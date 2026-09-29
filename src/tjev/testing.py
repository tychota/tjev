"""Explicitly marked random fixtures: a tiny Qwen3.5-format snapshot with a byte tokenizer.

Used by CPU tests and smoke runs; never mistaken for a real model (config.json carries
``"tjev_test_fixture": true`` and there is no chat template).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tjev.data.pack import Batch, stack_batches
from tjev.model import ModelConfig, expected_shapes

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]


def byte_tokenizer_json() -> dict:
    """A byte-level BPE with no merges: every byte is a token, specials are atomic."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel

    alphabet = ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(SPECIALS)
    return json.loads(tok.to_str())


def make_tiny_snapshot(folder: str | Path, seed: int = 0, **overrides: Any) -> ModelConfig:
    from safetensors.numpy import save_file

    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    config = ModelConfig.tiny(vocab_size=512, **overrides)
    raw = config.to_hf() | {"tjev_test_fixture": True}
    (folder / "config.json").write_text(json.dumps(raw, indent=2))
    (folder / "tokenizer.json").write_text(json.dumps(byte_tokenizer_json()))
    rng = np.random.default_rng(seed)
    tensors = {}
    for name, shape in expected_shapes(config).items():
        if name.endswith("A_log"):
            value = np.log(rng.uniform(1.0, 8.0, shape))
        elif name.endswith("dt_bias"):
            value = rng.normal(0, 0.5, shape)
        elif "linear_attn.norm" in name:
            value = 1.0 + rng.normal(0, 0.1, shape)
        elif name.endswith(("norm.weight", "layernorm.weight")):
            value = rng.normal(0, 0.1, shape)
        else:
            value = rng.normal(0, 0.05, shape)
        tensors["model." + name] = value.astype(np.float32)
    save_file(tensors, str(folder / "model.safetensors"))
    return config


def tiny_items(n: int, seed: int = 0, lang: str = "en") -> list[dict]:
    """Learnable synthetic decisions: the answer is written in the state."""
    rng = np.random.default_rng(seed)
    colors = ["red", "green", "blue", "amber", "violet"]
    rows = []
    for i in range(n):
        kind = ("choice", "noul", "score")[i % 3]
        if kind == "choice":
            k = int(rng.integers(3, 6))
            answer = colors[int(rng.integers(k))]
            rows.append(
                {
                    "id": f"{seed}-{i}",
                    "state": f"The light is {answer}.",
                    "question": {
                        "type": "choice",
                        "instructions": "Which colour is the light?",
                        "criteria": {c: f"the light is {c}" for c in colors[:k]},
                    },
                    "expected": answer,
                    "source": "colors",
                    "family": "extraction",
                    "lang": lang,
                }
            )
        elif kind == "noul":
            yes = bool(rng.integers(2))
            rows.append(
                {
                    "id": f"{seed}-{i}",
                    "state": "The door is open." if yes else "The door is shut.",
                    "question": {
                        "type": "noul",
                        "instructions": "Is the door open?",
                        "criteria": {"false": "closed", "true": "open"},
                    },
                    "expected": "yes" if yes else "no",
                    "source": "doors",
                    "family": "policy",
                    "lang": lang,
                }
            )
        else:
            level = int(rng.integers(4))
            rows.append(
                {
                    "id": f"{seed}-{i}",
                    "state": f"Rating: {level} stars.",
                    "question": {
                        "type": "score",
                        "instructions": "How many stars?",
                        "criteria": [f"{j} stars" for j in range(4)],
                    },
                    "expected": str(level),
                    "source": "stars",
                    "family": "numeric",
                    "lang": lang,
                }
            )
    return rows


def synthetic_batch(rows: int, length: int, accumulation: int, seg_len: int = 256,
                    slots: int = 32, labels: int = 26, seed: int = 0) -> Batch:  # fmt: skip
    """A packed [A, R, T] step of ``seg_len``-token segments with random tokens (benchmarks)."""
    rng = np.random.default_rng(seed)
    n_seg = length // seg_len

    def one() -> Batch:
        b = Batch(
            tokens=rng.integers(0, 1000, (rows, length)).astype(np.int32),
            segment_ids=np.repeat(np.arange(1, n_seg + 1), seg_len)[None]
            .repeat(rows, 0)
            .astype(np.int32),
            positions=np.tile(np.arange(seg_len), n_seg)[None].repeat(rows, 0).astype(np.int32),
            slots=np.zeros((rows, slots), np.int32),
            label_ids=np.tile(np.arange(32, 32 + labels), (rows, slots, 1)).astype(np.int32),
            label_mask=np.zeros((rows, slots, labels), bool),
            target=np.zeros((rows, slots, labels), np.float32),
            weight=np.zeros((rows, slots), np.float32),
            type_id=np.zeros((rows, slots), np.int32),
            source_id=np.zeros((rows, slots), np.int32),
            lang_id=np.zeros((rows, slots), np.int32),
            family_id=np.zeros((rows, slots), np.int32),
            index=np.full((rows, slots), -1, np.int32),
        )
        for r in range(rows):
            for s in range(min(slots, n_seg)):
                b.slots[r, s] = (s + 1) * seg_len - 1
                b.label_mask[r, s, :4] = True
                b.target[r, s, 0] = 1.0
                b.weight[r, s] = 1.0
        return b

    return stack_batches([one() for _ in range(accumulation)])
