"""Checks against real Qwen3.5 tokenizers and chat templates (``pytest -m real``).

TJEV_MODELS points at a directory of HF snapshots (default ./models). Deselected by default,
never skipped."""

import os
from pathlib import Path

import pytest

from tjev.data.render import chat
from tjev.data.tokenize import PromptTokenizer

MODELS = Path(os.environ.get("TJEV_MODELS", "models"))
pytestmark = pytest.mark.real


@pytest.mark.parametrize("size", ["0.8B", "2B", "4B"])
def test_chat_framing_equals_hf_template(size):
    from transformers import AutoTokenizer

    folder = MODELS / f"Qwen3.5-{size}"
    hf = AutoTokenizer.from_pretrained(folder)
    assert hf is not None
    messages = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}]
    want = hf.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    assert isinstance(want, str)
    assert chat("SYS", "USER") == want
    tok = PromptTokenizer(folder)
    assert tok.letter_ids == list(range(32, 58))
    assert tok.encode(want) == hf(want, add_special_tokens=False)["input_ids"]
