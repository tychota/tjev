"""mix-v7 decontamination: word 8-grams of every new train row against evaluation sets.

A train row (its full text: state, instructions, option labels and descriptions, as
:func:`tjev.data.jevbench.item_text`) is dropped when it shares a word 8-gram (NFKC, lower
case, ``\\w+`` words: :func:`tjev.data.jevbench.word_ngrams`) with a reference set:

``jevbench_public``
    the 231 public JevBench items (every 8-gram, as mix-v6's ``jevbench_8gram``);
``jevbench_hf``
    ``JevBench/jevbench`` on the Hub (questions, cases, rewrites: the metamorphic suite);
eval-only sets
    BFCL, UnSeenTimeQA, Test of Time, sycophancy-eval, LLMBar, AmbiEnt, HateCheck (EN, FR),
    PPE, ANLI dev / test, bbh-fr: never trained on, and nothing derived from them either;
``heldout:<source>``
    our own held-out rows (validation, calibration, test, the held-out generators), per
    source.

The references are read generically (every string of a row, recursively), so a schema
change upstream cannot silently empty a set. Large templated sets share boilerplate
(BFCL's function-doc phrasing, a held-out source's templates): an 8-gram that occurs in
at least ``template_share`` of a set's items (and at least ``TEMPLATE_MIN`` of them) is a
template, not content, and is ignored for that set; the public JevBench items keep all
their 8-grams (mix-v6's rule). Grams are stored as sorted int64 hashes (Python's string
hash: stable within one process, which is all the post-pass needs).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tjev.data.item import Item
from tjev.data.jevbench import item_text, norm_text

N = 8
TEMPLATE_SHARE = 0.01
TEMPLATE_MIN = 5

_WORD = __import__("re").compile(r"\w+")


def gram_hashes(text: str, n: int = N) -> np.ndarray:
    """Sorted unique int64 hashes of the word n-grams of ``text``."""
    words = _WORD.findall(norm_text(text))
    if len(words) < n:
        return np.zeros(0, np.int64)
    return np.unique(
        np.fromiter(
            (hash(" ".join(words[i : i + n])) for i in range(len(words) - n + 1)),
            np.int64,
            len(words) - n + 1,
        )
    )


def strings(value) -> Iterator[str]:
    """Every string inside a (nested) row value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from strings(v)


def row_text(row) -> str:
    """A reference row as one text (every string field; JSON strings are decoded first)."""
    parts = []
    for s in strings(row):
        t = s.strip()
        if t[:1] in "[{" and t[-1:] in "]}":
            try:
                parts.extend(strings(json.loads(t)))
                continue
            except ValueError:
                pass
        parts.append(s)
    return "\n".join(parts)


@dataclass
class RefGrams:
    """The 8-gram hashes of named reference sets, template grams removed per set."""

    grams: dict[str, np.ndarray] = field(default_factory=dict)
    stats: dict[str, dict] = field(default_factory=dict)
    # per set: (min shared grams, min share of the row's grams) for a hit; default (1, 0)
    loose: dict[str, tuple[int, float]] = field(default_factory=dict)
    # per set built by add_groups: each group's own grams (e.g. one origin's held-out rows)
    groups: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    def add(
        self,
        name: str,
        texts: Iterable[str],
        *,
        template_share: float | None = TEMPLATE_SHARE,
        template_min: int = TEMPLATE_MIN,
    ) -> None:
        per_item = [g for t in texts if len(g := gram_hashes(t))]
        if not per_item:
            self.grams[name] = np.zeros(0, np.int64)
            self.stats[name] = {"items": 0, "grams": 0, "template_grams": 0}
            return
        allg, counts = np.unique(np.concatenate(per_item), return_counts=True)
        templates = 0
        if template_share is not None:
            limit = max(template_min, int(np.ceil(template_share * len(per_item))))
            keep = counts < limit
            templates = int((~keep).sum())
            allg = allg[keep]
        self.grams[name] = allg
        self.stats[name] = {
            "items": len(per_item),
            "grams": len(allg),
            "template_grams": templates,
        }

    def add_groups(self, name: str, groups: dict[str, list[str]], *, min_shared: int = 1,
                   min_share: float = 0.0, **kwargs) -> None:  # fmt: skip
        """One set made of several groups (e.g. our held-out rows by source), each with its
        own template grams removed. A row hits it when it shares ``min_shared`` of its
        grams or ``min_share`` of its own grams (whichever comes first)."""
        if min_shared > 1:
            self.loose[name] = (min_shared, min_share)
        parts, stats = [], {"items": 0, "template_grams": 0, "groups": len(groups)}
        for key, texts in groups.items():
            sub = RefGrams()
            sub.add(key, texts, **kwargs)
            parts.append(sub.grams[key])
            self.groups.setdefault(name, {})[key] = sub.grams[key]
            stats["items"] += sub.stats[key]["items"]
            stats["template_grams"] += sub.stats[key]["template_grams"]
        self.grams[name] = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
        self.stats[name] = {**stats, "grams": len(self.grams[name])}

    def hit(self, text: str, *, grams: np.ndarray | None = None,
            ignore: dict[str, np.ndarray] | None = None) -> str | None:  # fmt: skip
        """The first reference set sharing an 8-gram with ``text`` (or its precomputed
        ``grams``), or None. ``ignore[set]``: sorted grams not counted for that set (the
        train side's own templates, :func:`train_templates`)."""
        g = gram_hashes(text) if grams is None else grams
        if not len(g):
            return None
        for name, ref in self.grams.items():
            if len(ref):
                pos = np.searchsorted(ref, g)
                pos[pos == len(ref)] = 0
                shared = g[ref[pos] == g]
                if ignore and name in ignore and len(shared):
                    shared = shared[~_member(shared, ignore[name])]
                need, share = self.loose.get(name, (1, 0.0))
                if len(shared) >= need or (share and len(shared) >= share * len(g)):
                    return name
        return None

    def hit_item(self, item: Item, **kwargs) -> str | None:
        return self.hit(item_text(item), **kwargs)


def _member(values: np.ndarray, sorted_set: np.ndarray) -> np.ndarray:
    if not len(sorted_set):
        return np.zeros(len(values), bool)
    pos = np.searchsorted(sorted_set, values)
    pos[pos == len(sorted_set)] = 0
    return sorted_set[pos] == values


TRAIN_TEMPLATE_SHARE = 0.0005
TRAIN_TEMPLATE_MIN = 20


def train_templates(per_item: list[np.ndarray], share: float = TRAIN_TEMPLATE_SHARE,
                    minimum: int = TRAIN_TEMPLATE_MIN) -> np.ndarray:  # fmt: skip
    """The 8-grams of a source's train rows that occur in at least ``max(minimum, share ·
    rows)`` of them: the source's own templates (a generator's sentence frames, a
    tasksource instruction). Against our held-out rows they are not evidence of a shared
    passage (the same source's held-out rows use the same frames)."""
    parts = [g for g in per_item if len(g)]
    if not parts:
        return np.zeros(0, np.int64)
    allg, counts = np.unique(np.concatenate(parts), return_counts=True)
    return allg[counts >= max(minimum, int(np.ceil(share * len(per_item))))]


# --------------------------------------------------------------------------------------
# The evaluation-only reference sets (downloaded at build time, on the build machine)


def _hf_rows(repo: str, configs=(None,), splits: tuple[str, ...] | None = None) -> list:
    from datasets import get_dataset_split_names, load_dataset

    rows = []
    for config in configs:
        names = splits or get_dataset_split_names(repo, config)
        for split in names:
            ds = load_dataset(repo, config, split=split)
            rows += [row_text(r) for r in ds]
    return rows


def _hf_files(repo: str, suffixes=(".json", ".jsonl"), skip=()) -> list[str]:
    """Every JSON / JSONL file of a dataset repo (repos that are not loadable datasets)."""
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(repo, repo_type="dataset", allow_patterns=[f"*{s}" for s in suffixes])
    )
    texts = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in suffixes or any(k in str(path) for k in skip):
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(raw)
            records = data if isinstance(data, list) else [data]
        except ValueError:
            records = []
            for line in raw.splitlines():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue
        texts += [row_text(r) for r in records]
    return texts


def _github_json(prefix: str, paths: list[str]) -> list[str]:
    import urllib.request

    texts = []
    for path in paths:
        with urllib.request.urlopen(prefix + path, timeout=120) as r:
            texts += [row_text(row) for row in json.loads(r.read())]
    return texts


def _hf_any(repo: str, splits: Callable[[str], bool] = lambda split: True) -> list[str]:
    """Every config and split of a dataset repo (``splits`` filters the split names); a
    repo that is not a loadable dataset (BFCL: JSON files) is read file by file."""
    from datasets import get_dataset_config_names

    try:
        configs = get_dataset_config_names(repo)
    except Exception:
        return _hf_files(repo)
    from datasets import get_dataset_split_names

    texts = []
    try:
        for config in configs:
            names = [s for s in get_dataset_split_names(repo, config) if splits(s)]
            texts += _hf_rows(repo, (config,), tuple(names))
    except Exception:
        if texts:
            raise
        return _hf_files(repo)
    return texts or _hf_files(repo)  # e.g. LLMBar: configs that load no row


# name -> loader of texts; every entry is evaluation-only (never trained on). FiscalQA's
# questions are not on the Hub (and are canaried): nothing to load, and no adapter reads them.
EVAL_REFERENCES: dict[str, Callable[[], list[str]]] = {
    "jevbench_hf": lambda: _hf_any("JevBench/jevbench"),
    "bfcl": lambda: _hf_files("gorilla-llm/Berkeley-Function-Calling-Leaderboard"),
    "unseentimeqa": lambda: _hf_any("nurakib/UnSeenTimeQA"),
    "test_of_time": lambda: _hf_any("baharef/ToT"),
    "sycophancy_eval": lambda: _hf_files("meg-tong/sycophancy-eval"),
    "llmbar": lambda: _github_json(
        "https://raw.githubusercontent.com/princeton-nlp/LLMBar/main/Dataset/",
        [
            f"LLMBar/{p}/dataset.json"
            for p in (
                "Natural",
                "Adversarial/Neighbor",
                "Adversarial/GPTInst",
                "Adversarial/GPTOut",
                "Adversarial/Manual",
            )
        ]
        + [
            f"CaseStudy/{p}/dataset.json"
            for p in ("Constraint", "Negation", "Normal", "Base_9", "Base_10")
        ],
    ),  # a script-only HF repo: the data is on GitHub
    "ambient": lambda: _hf_any("tasksource/ambient"),
    "hatecheck": lambda: _hf_any("Paul/hatecheck"),
    "hatecheck_fr": lambda: _hf_any("Paul/hatecheck-french"),
    "ppe": lambda: _hf_any("lmarena-ai/PPE-Human-Preference-V1"),
    "anli_eval": lambda: _hf_any("facebook/anli", lambda split: not split.startswith("train")),
    "bbh_fr": lambda: _hf_any("le-leadboard/bbh-fr"),
}


def load_references(
    refs: RefGrams,
    names: Iterable[str],
    log: Callable[[str], None] = print,
) -> dict[str, str]:
    """Add each named evaluation set to ``refs``; returns the sets that failed to load (a
    missing reference is reported in the manifest, never silently skipped)."""
    failed = {}
    for name in names:
        try:
            texts = EVAL_REFERENCES[name]()
        except Exception as e:  # recorded: the manifest lists every set actually used
            failed[name] = f"{type(e).__name__}: {str(e)[:200]}"
            log(f"[decontam] {name}: FAILED {failed[name]}")
            continue
        refs.add(name, texts)
        log(f"[decontam] {name}: {refs.stats[name]}")
    return failed
