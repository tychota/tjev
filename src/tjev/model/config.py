"""Qwen3.5 text-decoder architecture, parsed fail-closed from a Hugging Face config.json."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    """Qwen3.5 dense text decoder (hybrid Gated DeltaNet + gated attention)."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    conv_kernel: int = 4
    full_attention_interval: int = 4
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    # 0.8B-4B tie the output head to the embedding; 9B has its own lm_head (read at the
    # answer slots only, like the tied rows)
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.num_layers % self.full_attention_interval:
            raise ValueError("num_layers must be a multiple of full_attention_interval")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear value heads must be a multiple of linear key heads")
        if self.rotary_dim % 2 or self.rotary_dim > self.head_dim:
            raise ValueError("invalid partial rotary dimension")

    @property
    def num_super_blocks(self) -> int:
        return self.num_layers // self.full_attention_interval

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_dim(self) -> int:
        return self.linear_num_key_heads * self.linear_key_head_dim

    @property
    def linear_value_dim(self) -> int:
        return self.linear_num_value_heads * self.linear_value_head_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.linear_key_dim + self.linear_value_dim

    @classmethod
    def from_hf(cls, raw: dict[str, Any]) -> ModelConfig:
        """Parse a HF Qwen3.5 config.json; refuse anything this port does not implement."""
        text = raw.get("text_config", raw)
        if raw.get("model_type") not in ("qwen3_5", "qwen3_5_text"):
            raise ValueError(f"Unsupported model_type {raw.get('model_type')!r}; need qwen3_5")
        rope = text.get("rope_parameters") or {}
        checks = {
            "hidden_act": text.get("hidden_act") == "silu",
            "attn_output_gate": text.get("attn_output_gate", True) is True,
            "attention_bias": not text.get("attention_bias", False),
            "mlp_only_layers": not text.get("mlp_only_layers"),
            "dense MLP": text.get("num_experts", 0) in (0, 1, None),
            "rope_type": rope.get("rope_type", "default") == "default",
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"Unsupported Qwen3.5 features: {failed}")
        interval = text["full_attention_interval"]
        expected = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(text["num_hidden_layers"])
        ]
        if text.get("layer_types", expected) != expected:
            raise ValueError("layer_types is not the periodic 3×GDN + 1×attention pattern")
        tied = raw.get("tie_word_embeddings", text.get("tie_word_embeddings"))
        if tied is None:
            raise ValueError("config.json does not say whether the embeddings are tied")
        return cls(
            vocab_size=text["vocab_size"],
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_layers=text["num_hidden_layers"],
            num_heads=text["num_attention_heads"],
            num_kv_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            linear_num_key_heads=text["linear_num_key_heads"],
            linear_num_value_heads=text["linear_num_value_heads"],
            linear_key_head_dim=text["linear_key_head_dim"],
            linear_value_head_dim=text["linear_value_head_dim"],
            conv_kernel=text["linear_conv_kernel_dim"],
            full_attention_interval=interval,
            rope_theta=float(rope.get("rope_theta", text.get("rope_theta", 1e7))),
            partial_rotary_factor=float(rope.get("partial_rotary_factor", 0.25)),
            rms_norm_eps=text["rms_norm_eps"],
            tie_word_embeddings=bool(tied),
        )

    def to_hf(self) -> dict[str, Any]:
        """A minimal HF-compatible text config (tests and tiny fixtures)."""
        return {
            "model_type": "qwen3_5_text",
            "architectures": ["Qwen3_5ForCausalLM"],
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_layers,
            "num_attention_heads": self.num_heads,
            "num_key_value_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "linear_num_key_heads": self.linear_num_key_heads,
            "linear_num_value_heads": self.linear_num_value_heads,
            "linear_key_head_dim": self.linear_key_head_dim,
            "linear_value_head_dim": self.linear_value_head_dim,
            "linear_conv_kernel_dim": self.conv_kernel,
            "full_attention_interval": self.full_attention_interval,
            "layer_types": [
                "full_attention"
                if (i + 1) % self.full_attention_interval == 0
                else "linear_attention"
                for i in range(self.num_layers)
            ],
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": self.rope_theta,
                "partial_rotary_factor": self.partial_rotary_factor,
                "mrope_interleaved": True,
                "mrope_section": [11, 11, 10],
            },
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_act": "silu",
            "attn_output_gate": True,
            "attention_bias": False,
            "tie_word_embeddings": self.tie_word_embeddings,
            "mlp_only_layers": [],
            "max_position_embeddings": 262144,
        }

    @classmethod
    def tiny(cls, **overrides: Any) -> ModelConfig:
        """A small random-weight layout for tests (two super-blocks, GQA and GDN ratio 2)."""
        values: dict[str, Any] = {
            "vocab_size": 512,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_layers": 8,
            "num_heads": 4,
            "num_kv_heads": 2,
            "head_dim": 32,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "full_attention_interval": 4,
        }
        values.update(overrides)
        return cls(**values)
