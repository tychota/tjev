"""Two mix-level corrections (since mix-v4b): yes/no prior rebalance and a repetition cap.

``rebalance_noul``
    For a source whose train ``noul`` yes-share (gold = argmax of the target, as in
    ``scripts/data_report.py``) is outside ``[1 - hi, hi]``, a seeded subsample of the
    majority class brings the share to ``hi`` (or ``1 - hi``). Train only: the held-out
    parts never change. A source never loses more than ``max_loss`` of its train rows
    (all types): past that, the drop stops at the cap and the share stays above target.
``cap_epochs``
    The effective epochs of every source in a run (``steps`` × ``tokens_per_step``) come
    from :func:`simulate_stream`, the real :class:`tjev.data.pack.BucketPacker` over a
    stream drawn like the training stream. A source over ``max_epochs`` has its weight
    scaled by ``target / epochs``; the freed mass goes to the other sources in proportion
    to their weights. Repeated until no source is over ``max_epochs`` (the packing, and so
    the segments per step, move with the weights).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np

from tjev.data.item import Item, parse_item
from tjev.data.pack import BucketPacker, RowBuilder, Segment, bucket_for
from tjev.data.render import render

# The run the epochs are measured for (scripts/data_report.py: 4B on one v6e chip).
STEPS = 3000
TOKENS_PER_STEP = 65536
MICROBATCH_TOKENS = 4096
BUCKETS = (1024, 2048, 4096)
MAX_SEGMENTS = 32
PACKING_BINS = 16
MAX_LABELS = 26

Lengths = tuple[np.ndarray, np.ndarray, np.ndarray]  # plain tokens, json tokens, options


# --------------------------------------------------------------------------------------
# Yes/no prior rebalance


def is_yes(item: Item) -> bool:
    return item.gold == 1


def rebalance_noul(
    items: list[Item], hi: float = 0.7, *, max_loss: float = 0.6, seed: int = 0
) -> tuple[list[int], dict]:
    """Indices (in order) of ``items`` to keep, and the before/after counts.

    Only noul rows of the majority class are dropped, chosen uniformly with a generator
    seeded by ``seed``; the share lands on ``hi`` (or ``1 - hi``) up to rounding."""
    if not 0.5 <= hi < 1.0:
        raise ValueError(f"rebalance target {hi} not in [0.5, 1)")
    noul = [i for i, it in enumerate(items) if it.question.type == "noul"]
    yes = [i for i in noul if is_yes(items[i])]
    no = [i for i in noul if not is_yes(items[i])]
    stats: dict = {
        "rows_before": len(items),
        "noul_before": len(noul),
        "yes_before": len(yes),
        "yes_share_before": round(len(yes) / len(noul), 4) if noul else None,
    }
    drop: list[int] = []
    if noul:
        share = len(yes) / len(noul)
        if share > hi:
            # keep k yes so that k / (k + no) = hi
            major, minor, target = yes, no, hi
        elif share < 1 - hi:
            major, minor, target = no, yes, hi  # keep k no so that k / (k + yes) = hi
        else:
            major, minor, target = [], [], None
        if target is not None:
            keep = round(len(minor) * target / (1 - target))
            n_drop = min(len(major) - keep, int(max_loss * len(items)))
            if n_drop > 0:
                rng = np.random.default_rng(seed)
                drop = sorted(rng.choice(major, size=n_drop, replace=False).tolist())
            stats["capped"] = len(major) - keep > int(max_loss * len(items))
    dropped = set(drop)
    kept = [i for i in range(len(items)) if i not in dropped]
    yes_after = len(yes) - len(dropped & set(yes))
    noul_after = len(noul) - len(dropped)
    stats |= {
        "rows_after": len(kept),
        "noul_after": noul_after,
        "yes_after": yes_after,
        "yes_share_after": round(yes_after / noul_after, 4) if noul_after else None,
        "dropped": len(dropped),
        "loss": round(len(dropped) / max(len(items), 1), 4),
    }
    return kept, stats


def rebalance_file(
    path: Path, out: Path, hi: float, *, max_loss: float, seed: int
) -> tuple[dict, list[Item]]:
    """Rebalance one train file into ``out`` (kept lines byte-identical, in order)."""
    # split on "\n" only: str.splitlines would also cut at U+2028 inside JSON strings
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    items = [parse_item(json.loads(line), f"{path}:{n}") for n, line in enumerate(lines, 1)]
    kept, stats = rebalance_noul(items, hi, max_loss=max_loss, seed=seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text("".join(lines[i] + "\n" for i in kept), encoding="utf-8")
    tmp.replace(out)
    return stats, [items[i] for i in kept]


# --------------------------------------------------------------------------------------
# Effective epochs and the repetition cap

PACKINGS = ("bucket", "best_fit")


class BestFitPacker(BucketPacker):
    """``train.packing=best_fit`` for the simulation (perf/pack-remat, bfe9f12): a segment
    goes to the open row it fills best, in its own bucket or a longer one; a new row of
    its own bucket is a candidate too (it wins when strictly tighter, or when no open row
    has room). Used only when :class:`tjev.data.pack.BucketPacker` has no ``packing``
    argument yet (this branch predates it); the placement rule is that commit's."""

    def place(self, seg: Segment) -> None:
        self.next_index += 1
        length = bucket_for(len(seg.tokens), self.buckets)
        if length is None or len(seg.label_ids) > self.labels:
            reason = "too_long" if length is None else "too_many_labels"
            self.stats[f"dropped_{reason}"] += 1
            return
        builders = self.open.setdefault(length, [])
        target = self._best_fit(seg, length)
        if target is None:
            if len(builders) >= self.bins:
                fullest = max(range(len(builders)), key=lambda i: builders[i].fill())
                self.ready[length].append(builders.pop(fullest))
            target = self._builder(length)
            target.add(seg)
            builders.append(target)
        if self.bins > 1 and target.fill() >= self.FULL:
            self.open[target.length].remove(target)
            self.ready[target.length].append(target)

    def _best_fit(self, seg: Segment, length: int) -> RowBuilder | None:
        n = len(seg.tokens)
        best: tuple[int, RowBuilder, int] | None = None  # (room left, builder, row)
        for bucket in self.buckets:
            if bucket < length:
                continue
            for builder in self.open.get(bucket, ()):
                limit, used, content = builder.length - n, builder.used, builder.content
                for r in range(builder.rows):
                    room = limit - used[r]
                    if (room >= 0 and (best is None or room < best[0])
                            and len(content[r]) < builder.slots):  # fmt: skip
                        best = (room, builder, r)
        if best is None or length - n < best[0]:
            return None
        _, builder, row = best
        builder.content[row].append(seg)
        builder.used[row] += n
        return builder


def make_packer(packing: str = "bucket", **kwargs) -> BucketPacker:
    """The real packer with ``packing`` (its own argument when it has one, else
    :class:`BestFitPacker`)."""
    import inspect

    if packing not in PACKINGS:
        raise ValueError(f"packing {packing!r} is not one of {PACKINGS}")
    if "packing" in inspect.signature(BucketPacker.__init__).parameters:
        return BucketPacker(**kwargs, packing=packing)
    return BestFitPacker(**kwargs) if packing == "best_fit" else BucketPacker(**kwargs)


def prompt_lengths(items: Iterable[Item], count: Callable[[list[str]], list[int]]) -> Lengths:
    """Rendered prompt tokens of each item, plain and JSON layouts (every option shown,
    the eval render: what ``scripts/data_report.py`` measures), and its option count."""
    from tjev.data.hygiene import EVAL_LAYOUTS

    plain, js, opts, batch = [], [], [], []

    def flush() -> None:
        n = count([t for pair in batch for t in pair])
        plain.extend(n[0::2])
        js.extend(n[1::2])
        batch.clear()

    for it in items:
        batch.append(tuple(render(it, spec)[0] for spec in EVAL_LAYOUTS))
        opts.append(len(it.question.labels))
        if len(batch) >= 512:
            flush()
    if batch:
        flush()
    return np.asarray(plain, np.int64), np.asarray(js, np.int64), np.asarray(opts, np.int64)


def simulate_stream(
    lengths: dict[str, Lengths],
    probs: dict[str, float],
    *,
    steps: int,
    tokens_per_step: int,
    microbatch_tokens: int,
    buckets: tuple[int, ...] = BUCKETS,
    slots: int = MAX_SEGMENTS,
    labels: int = MAX_LABELS,
    bins: int = PACKING_BINS,
    seed: int = 0,
    packing: str = "bucket",
) -> dict:
    """Run the real BucketPacker for ``steps`` steps on length-only segments (``packing``:
    its placement, ``train.packing``: ``bucket`` or ``best_fit``).

    ``lengths[source] = (plain_tokens, json_tokens, n_options)`` per row (a sample is
    fine: rows are drawn uniformly). Returns per-source segments trained on, rows seen at
    least once, segments dropped as too long, and the per-bucket fill and step counts."""
    names = sorted(probs)
    p = np.asarray([probs[n] for n in names], np.float64)
    p /= p.sum()
    accumulation = max(1, tokens_per_step // microbatch_tokens)
    packer = make_packer(
        packing,
        segment_at=lambda i: None,
        buckets=buckets,
        microbatch_tokens=microbatch_tokens,
        accumulation=accumulation,
        slots=slots,
        labels=labels,
        bins=bins,
    )
    rng = np.random.default_rng(seed)
    trained = np.zeros(len(names), np.int64)
    drawn = np.zeros(len(names), np.int64)
    seen = [np.zeros(len(lengths[n][0]), bool) for n in names]
    fill: dict[int, list[float]] = defaultdict(list)
    step_count: Counter = Counter()
    done, batch = 0, 4096
    while done < steps:
        src = rng.choice(len(names), size=batch, p=p)
        u = rng.random(batch)
        layout = rng.random(batch) < 0.5
        for s, r, js in zip(src.tolist(), u.tolist(), layout.tolist(), strict=True):
            plain, jsn, opts = lengths[names[s]]
            j = int(r * len(plain))
            n = int(jsn[j] if js else plain[j])
            drawn[s] += 1
            packer.place(
                Segment(
                    tokens=np.empty(n, np.int8),
                    label_ids=np.empty(int(opts[j]), np.int8),
                    target=None,
                    type_id=0,
                    source_id=s,
                    lang_id=0,
                    family_id=0,
                    index=j,
                )
            )
            for length in packer.buckets:
                if len(packer.ready[length]) >= accumulation:
                    group = packer.ready[length][:accumulation]
                    packer.ready[length] = packer.ready[length][accumulation:]
                    step_count[length] += 1
                    fill[length].append(sum(b.fill() for b in group) / len(group))
                    for b in group:
                        for seg in b.segments():
                            trained[seg.source_id] += 1
                            seen[seg.source_id][seg.index] = True
                    done += 1
                    break
            if done >= steps:
                break
    return {
        "steps": done,
        "packing": packing,
        "accumulation": accumulation,
        "microbatch_tokens": microbatch_tokens,
        "segments": int(trained.sum()),
        "segments_per_step": round(float(trained.sum() / max(done, 1)), 2),
        "dropped_too_long": int(packer.stats.get("dropped_too_long", 0)),
        "steps_by_bucket": {str(k): int(v) for k, v in sorted(step_count.items())},
        "fill_by_bucket": {str(k): round(float(np.mean(v)), 4) for k, v in sorted(fill.items())},
        "fill": round(float(np.mean([x for v in fill.values() for x in v])), 4) if fill else 0.0,
        "by_source": {
            n: {
                "drawn": int(drawn[i]),
                "trained": int(trained[i]),
                "seen_share_of_sample": round(float(seen[i].mean()), 4) if len(seen[i]) else 0.0,
            }
            for i, n in enumerate(names)
        },
    }


def epochs_of(sim: dict, rows: dict[str, int]) -> dict[str, float]:
    return {n: v["trained"] / max(rows[n], 1) for n, v in sim["by_source"].items()}


def rounded(weights: dict[str, float]) -> dict[str, float]:
    """Normalised to 1 and rounded like ``mix.yaml`` (6 decimals)."""
    total = sum(weights.values())
    return {n: round(w / total, 6) for n, w in weights.items()}


def redistribute(
    w: dict[str, float],
    new: dict[str, float],
    block_of: dict[str, str],
    shares: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Capped sources take their ``new`` weights; every other source of a block is scaled
    so that the block keeps its share. A block whose sources are all capped (or whose capped
    sources already exceed its share) cannot absorb: its shortfall goes to the absorbing
    blocks in proportion to their shares. Returns the weights and each non-absorbing
    block's shortfall (share − mass now held)."""
    members: dict[str, list[str]] = defaultdict(list)
    for n in w:
        members[block_of[n]].append(n)
    held = {b: sum(new.get(n, 0.0) for n in ns) for b, ns in members.items()}
    free = {b: sum(w[n] for n in ns if n not in new) for b, ns in members.items()}
    absorbing = {b for b in members if free[b] > 0}
    while True:
        if not absorbing:
            raise ValueError("every source is capped: the cap cannot hold")
        deficit = sum(shares[b] - held[b] for b in members if b not in absorbing)
        total = sum(shares[b] for b in absorbing)
        targets = {b: shares[b] + deficit * shares[b] / total for b in absorbing}
        full = {b for b in absorbing if targets[b] <= held[b]}
        if not full:
            break
        absorbing -= full
    out = {}
    for b, ns in members.items():
        for n in ns:
            if n in new:
                out[n] = new[n]
            else:
                out[n] = w[n] * (targets[b] - held[b]) / free[b]
    overflow = {b: shares[b] - held[b] for b in sorted(members) if b not in absorbing}
    return out, overflow


def cap_epochs(
    lengths: dict[str, Lengths],
    weights: dict[str, float],
    max_epochs: float,
    *,
    block_of: dict[str, str] | None = None,
    rows: dict[str, int] | None = None,
    steps: int = STEPS,
    tokens_per_step: int = TOKENS_PER_STEP,
    microbatch_tokens: int = MICROBATCH_TOKENS,
    margin: float = 0.98,
    iterations: int = 12,
    seed: int = 0,
    caps: dict[str, float] | None = None,
    log: Callable[[str], None] = print,
    packing: str = "bucket",
    also_steps: tuple[int, ...] = (),
) -> dict:
    """Per-source weight multipliers that keep every source at ≤ ``max_epochs``.

    A source over the cap joins the capped set; each round, every capped source is moved
    to ``margin · max_epochs`` (its weight × target / epochs, both ways) and the freed
    mass goes to the uncapped sources of the same block (``block_of``; without it, one
    block) in proportion to their weights, so block shares stay as they were. Only a block
    whose sources are all capped passes its shortfall to the other blocks, in proportion
    to their shares (:func:`redistribute`; recorded under ``overflow``). ``rows`` is the
    full row count per source (``lengths`` may be a sample). Returns the multipliers
    (final weight / input weight, before renormalising), the final weights, epochs, block
    shares and simulation (on the rounded weights, as ``mix.yaml`` stores them). ``caps``:
    per-source caps below ``max_epochs`` (e.g. small noisy sources at 1 epoch)."""
    rows = rows or {n: len(v[0]) for n, v in lengths.items()}
    block_of = block_of or dict.fromkeys(weights, "all")
    base = rounded(weights)
    shares: dict[str, float] = defaultdict(float)
    for n, v in base.items():
        shares[block_of[n]] += v
    w = dict(base)
    limit = {n: min(max_epochs, (caps or {}).get(n, max_epochs)) for n in w}
    capped: set[str] = set()
    history = []
    overflow: dict[str, float] = {}
    sim: dict = {}
    for it in range(iterations + 1):
        w = rounded(w)
        sim = simulate_stream(
            lengths,
            w,
            steps=steps,
            tokens_per_step=tokens_per_step,
            microbatch_tokens=microbatch_tokens,
            seed=seed,
            packing=packing,
        )
        ep = epochs_of(sim, rows)
        top = max(ep, key=lambda n: ep[n] / limit[n])
        history.append(
            {
                "round": it,
                "max_source": top,
                "max_epochs": round(ep[top], 3),
                "capped": len(capped),
                "overflow": {b: round(v, 5) for b, v in overflow.items()},
            }
        )
        log(
            f"[epoch-cap] round {it}: max {top} {ep[top]:.2f}, {len(capped)} capped, "
            f"blocks that cannot absorb: {sorted(overflow) or 'none'}"
        )
        if ep[top] <= limit[top] or it == iterations:
            break
        capped |= {n for n, e in ep.items() if e > margin * limit[n]}
        new = {n: w[n] * margin * limit[n] / max(ep[n], 1e-9) for n in capped}
        w, overflow = redistribute(w, new, block_of, shares)
    ep = epochs_of(sim, rows)
    if any(e > limit[n] for n, e in ep.items()):
        raise ValueError(f"epoch cap {max_epochs} not reached in {iterations} rounds")
    final: dict[str, float] = defaultdict(float)
    for n, v in w.items():
        final[block_of[n]] += v
    # the same final weights measured at other run lengths (e.g. 3,000 steps)
    other = {}
    for k in also_steps:
        s_k = simulate_stream(lengths, w, steps=k, tokens_per_step=tokens_per_step,
                              microbatch_tokens=microbatch_tokens, seed=seed,
                              packing=packing)  # fmt: skip
        other[str(k)] = {
            "epochs": {n: round(e, 3) for n, e in sorted(epochs_of(s_k, rows).items())},
            "simulation": {kk: v for kk, v in s_k.items() if kk != "by_source"},
        }
    return {
        "max_epochs": max_epochs,
        "caps": {n: c for n, c in sorted((caps or {}).items()) if n in w},
        "preserve_blocks": len(set(block_of.values())) > 1,
        "packing": packing,
        **({"at_steps": other} if other else {}),
        "steps": steps,
        "tokens_per_step": tokens_per_step,
        "microbatch_tokens": microbatch_tokens,
        "seed": seed,
        "scale": {n: w[n] / base[n] if base[n] else 1.0 for n in sorted(w)},
        "capped": sorted(capped),
        "weights": w,
        "epochs": {n: round(e, 3) for n, e in sorted(ep.items())},
        "block_shares": {
            b: {"before": round(shares[b], 5), "after": round(final[b], 5)} for b in sorted(shares)
        },
        "overflow": {b: round(v, 5) for b, v in overflow.items()},
        "history": history,
        "simulation": {k: v for k, v in sim.items() if k != "by_source"},
    }


def remeasure(cap: dict, lengths: dict[str, Lengths], rows: dict[str, int], *,
              steps: int = STEPS, tokens_per_step: int = TOKENS_PER_STEP,
              microbatch_tokens: int = MICROBATCH_TOKENS, seed: int = 0,
              packing: str = "bucket", also_steps: tuple[int, ...] = ()) -> dict:  # fmt: skip
    """``cap`` (:func:`cap_epochs`) with its epochs and simulations measured again on new
    ``lengths`` / ``rows`` at the same weights (mix-v7.5: after the min-epoch subsample,
    which must not feed back into the cap)."""
    w = cap["weights"]
    out = dict(cap)

    def sim_at(k: int) -> dict:
        return simulate_stream(lengths, w, steps=k, tokens_per_step=tokens_per_step,
                               microbatch_tokens=microbatch_tokens, seed=seed,
                               packing=packing)  # fmt: skip

    sim = sim_at(steps)
    out["epochs"] = {n: round(e, 3) for n, e in sorted(epochs_of(sim, rows).items())}
    out["simulation"] = {k: v for k, v in sim.items() if k != "by_source"}
    if also_steps:
        out["at_steps"] = {}
        for k in also_steps:
            s_k = sim_at(k)
            out["at_steps"][str(k)] = {
                "epochs": {n: round(e, 3) for n, e in sorted(epochs_of(s_k, rows).items())},
                "simulation": {kk: v for kk, v in s_k.items() if kk != "by_source"},
            }
    out["remeasured_after_subsample"] = True
    return out
