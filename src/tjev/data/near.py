"""Near-duplicate states: MinHash over word 3-gram shingles, LSH by bands.

128 permutations, 32 bands × 4 rows (candidates from Jaccard ~0.42, recall ≈ 1 at 0.8),
pairs kept at an estimated Jaccard ≥ 0.8. Shared by the builder's train hygiene
(:mod:`tjev.data.hygiene`) and the data-quality report (``scripts/data_report.py``), so
the build drops what the report measures.
"""

from __future__ import annotations

import hashlib
import re
import zlib

import numpy as np

NEAR_THRESHOLD = 0.8
NUM_PERM = 128
BANDS = 32  # × 4 rows: candidates from Jaccard ~0.42, recall ≈ 1 at 0.8
SHINGLE = 3


def normalized(text: str) -> str:
    """Lower case, runs of non-word characters → one space (the key of every exact test)."""
    return re.sub(r"\W+", " ", text.lower()).strip()


def stable_hash(*parts: str) -> int:
    """64-bit hash of a tuple of strings (blake2b), stable across processes."""
    digest = hashlib.blake2b("\x1f".join(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def shingle_hashes(text: str, k: int = SHINGLE) -> np.ndarray:
    """uint64 hashes of the word k-grams of the normalised text (the whole text when it
    has fewer than k words; empty when it has no word)."""
    words = normalized(text).split()
    if not words:
        return np.zeros(0, np.uint64)
    w = np.fromiter((zlib.crc32(x.encode()) for x in words), np.uint64, len(words))
    if len(w) < k:
        w = np.concatenate([w, np.zeros(k - len(w), np.uint64)])
    h = np.zeros(len(w) - k + 1, np.uint64)
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h * np.uint64(0x100000001B3) + w[j : len(w) - k + 1 + j] + np.uint64(j + 1)
    return np.unique(h)


class MinHasher:
    """MinHash signatures with multiply-shift hashing (uint32 per permutation)."""

    def __init__(self, num_perm: int = NUM_PERM, seed: int = 1):
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2**63, num_perm, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 2**63, num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, hashes: np.ndarray) -> np.ndarray:
        if len(hashes) == 0:
            return np.full(self.num_perm, np.iinfo(np.uint32).max, np.uint32)
        with np.errstate(over="ignore"):
            mixed = hashes[:, None] * self.a[None, :] + self.b[None, :]
        return (mixed >> np.uint64(32)).min(axis=0).astype(np.uint32)


def band_keys(sigs: np.ndarray, bands: int = BANDS) -> np.ndarray:
    """[n, bands] uint64 keys: one hash of each band of rows of the signatures."""
    n, perm = sigs.shape
    rows = perm // bands
    parts = sigs[:, : bands * rows].reshape(n, bands, rows).astype(np.uint64)
    keys = np.zeros((n, bands), np.uint64)
    with np.errstate(over="ignore"):
        for r in range(rows):
            keys = keys * np.uint64(0x9E3779B97F4A7C15) + parts[:, :, r] + np.uint64(r + 1)
    return keys


def estimated_jaccard(sigs: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return (sigs[i] == sigs[j]).mean(axis=1)


def _band_pairs(keys: np.ndarray, priority: np.ndarray | None = None):
    """Per band: (first, member) index pairs of every group of equal keys. Groups are
    ordered by ``priority`` (lower first) so that, e.g., train rows lead their groups."""
    n = len(keys)
    pri = np.zeros(n, np.int8) if priority is None else priority
    for band in range(keys.shape[1]):
        k = keys[:, band]
        order = np.lexsort((pri, k))
        ks = k[order]
        new = np.ones(n, bool)
        new[1:] = ks[1:] != ks[:-1]
        group = np.cumsum(new) - 1
        first = order[np.flatnonzero(new)][group]
        mask = first != order
        yield first[mask], order[mask]


def near_matches(
    query: np.ndarray, index: np.ndarray, threshold: float = NEAR_THRESHOLD
) -> np.ndarray:
    """For each query signature, the position of an index signature with estimated
    Jaccard ≥ threshold, or -1."""
    out = np.full(len(query), -1, np.int64)
    if len(query) == 0 or len(index) == 0:
        return out
    sigs = np.concatenate([index, query])
    priority = np.concatenate([np.zeros(len(index), np.int8), np.ones(len(query), np.int8)])
    for first, member in _band_pairs(band_keys(sigs), priority):
        m = (first < len(index)) & (member >= len(index))
        first, member = first[m], member[m]
        ok = estimated_jaccard(sigs, first, member) >= threshold
        q = member[ok] - len(index)
        out[q] = first[ok]
    return out


def near_hits(
    query: np.ndarray,
    index: np.ndarray,
    *,
    query_key: np.ndarray | None = None,
    index_key: np.ndarray | None = None,
    threshold: float = NEAR_THRESHOLD,
    per_group: int = 8,
) -> np.ndarray:
    """For each query signature: is there an index signature with estimated Jaccard ≥
    threshold (and, given keys, the same key)? The keys (uint64, e.g. a hash of the
    instructions) are mixed into every band, so only rows with equal keys are candidates.
    In each band, a query is checked against up to ``per_group`` index rows of its group
    (``near_matches`` checks only the first)."""
    hit = np.zeros(len(query), bool)
    if len(query) == 0 or len(index) == 0:
        return hit
    n_index = len(index)
    sigs = np.concatenate([index, query])
    keys = band_keys(sigs)
    if query_key is not None and index_key is not None:
        extra = np.concatenate([index_key, query_key]).astype(np.uint64)
        with np.errstate(over="ignore"):
            keys = keys * np.uint64(0x100000001B3) + extra[:, None]
    is_query = np.arange(len(sigs)) >= n_index
    for band in range(keys.shape[1]):
        k = keys[:, band]
        order = np.lexsort((is_query, k))  # per key group: index rows first, then queries
        ks = k[order]
        new = np.ones(len(order), bool)
        new[1:] = ks[1:] != ks[:-1]
        group = np.cumsum(new) - 1
        start = np.flatnonzero(new)
        index_in_group = np.bincount(group, weights=~is_query[order]).astype(np.int64)
        qpos = np.flatnonzero(is_query[order] & ~hit[np.maximum(order - n_index, 0)])
        g = group[qpos]
        count = np.minimum(index_in_group[g], per_group)
        for r in range(per_group):
            m = count > r
            if not m.any():
                break
            a, b = order[start[g[m]] + r], order[qpos[m]]
            ok = (sigs[a] == sigs[b]).mean(axis=1) >= threshold
            hit[b[ok] - n_index] = True
    return hit


class BandIndex:
    """A growing LSH index of MinHash signatures (mix-v7's train-side dedup): per band, the
    band keys (optionally mixed with a per-row key, e.g. a question hash) kept sorted, so
    a query is a binary search per band instead of re-sorting the whole index (as
    :func:`near_hits` does) for every source."""

    def __init__(self, num_perm: int = NUM_PERM, bands: int = BANDS, keyed: bool = False):
        self.num_perm, self.bands, self.keyed = num_perm, bands, keyed
        self._sigs: list[np.ndarray] = []
        self._keys: list[np.ndarray] = []
        self._tags: list[np.ndarray] = []
        self._sorted: list[tuple[np.ndarray, np.ndarray]] | None = None
        self.sigs = np.zeros((0, num_perm), np.uint32)
        self.tags = np.zeros(0, np.int32)

    def _band_keys(self, sigs: np.ndarray, key: np.ndarray | None) -> np.ndarray:
        keys = band_keys(sigs, self.bands)
        if self.keyed:
            with np.errstate(over="ignore"):
                keys = keys * np.uint64(0x100000001B3) + key.astype(np.uint64)[:, None]
        return keys

    def add(self, sigs: np.ndarray, key: np.ndarray | None = None, tag: int = 0) -> None:
        if len(sigs) == 0:
            return
        self._sigs.append(sigs)
        self._keys.append(self._band_keys(sigs, key))
        self._tags.append(np.full(len(sigs), tag, np.int32))
        self._sorted = None

    def _build(self) -> None:
        if self._sorted is not None:
            return
        if self._sigs:
            self.sigs = np.concatenate(self._sigs)
            self.tags = np.concatenate(self._tags)
            keys = np.concatenate(self._keys)
            self._sigs, self._tags, self._keys = [self.sigs], [self.tags], [keys]
        else:
            keys = np.zeros((0, self.bands), np.uint64)
        self._sorted = []
        for b in range(self.bands):
            order = np.argsort(keys[:, b], kind="stable")
            self._sorted.append((keys[order, b], order.astype(np.int64)))

    def __len__(self) -> int:
        return sum(len(s) for s in self._sigs)

    def query(
        self,
        sigs: np.ndarray,
        key: np.ndarray | None = None,
        *,
        exclude_tag: int | None = None,
        threshold: float = NEAR_THRESHOLD,
        per_group: int = 8,
    ) -> np.ndarray:
        """For each query signature: is an indexed one (with an equal key, if keyed; with
        another tag than ``exclude_tag``) at estimated Jaccard ≥ threshold? Each band
        checks up to ``per_group`` candidates."""
        hit = np.zeros(len(sigs), bool)
        if len(sigs) == 0 or len(self) == 0:
            return hit
        self._build()
        qkeys = self._band_keys(sigs, key)
        for b, (sorted_keys, order) in enumerate(self._sorted):
            todo = np.flatnonzero(~hit)
            if not len(todo):
                break
            k = qkeys[todo, b]
            lo = np.searchsorted(sorted_keys, k, "left")
            hi = np.searchsorted(sorted_keys, k, "right")
            for r in range(per_group):
                m = lo + r < hi
                if not m.any():
                    break
                q, idx = todo[m], order[lo[m] + r]
                ok = (self.sigs[idx] == sigs[q]).mean(axis=1) >= threshold
                if exclude_tag is not None:
                    ok &= self.tags[idx] != exclude_tag
                hit[q[ok]] = True
        return hit
