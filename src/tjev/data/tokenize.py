"""Qwen3.5 tokenizer wrapper (Rust ``tokenizers``; no transformers at runtime)."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer

from tjev.data.render import LETTERS


class PromptTokenizer:
    def __init__(self, folder: str | Path):
        self.tokenizer = Tokenizer.from_file(str(Path(folder) / "tokenizer.json"))
        self.letter_ids = []
        for letter in LETTERS:
            ids = self.tokenizer.encode(letter, add_special_tokens=False).ids
            if len(ids) != 1:
                raise ValueError(f"Letter {letter!r} is not a single token: {ids}")
            self.letter_ids.append(ids[0])
        # The answer letter must be its own token after the assistant header, not merged
        # into the preceding "\n\n".
        probe = self.tokenizer.encode("</think>\n\nA", add_special_tokens=False).ids
        if probe[-1] != self.letter_ids[0]:
            raise ValueError("Answer letter merges with the preceding text; template unusable")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids
