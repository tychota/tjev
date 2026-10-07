"""mix-v7 generators: contrastive one-fact groups, unknowable twins, intermediate
quantities, exact posteriors, distractor opinions, real-text seeds (docs/data.md,
"Generators v7").

``V7_GENERATORS`` maps a source name to a :class:`V7Source`: the base generators it draws
groups from, the roles of a group it emits, its family and licence. ``generate_v7`` has
``generators.generate``'s signature, and ``generators.generate`` dispatches to it, so the
mix builder's ``Builder.generator`` builds a v7 source like any generator. Rows of one
group (a pair, its unknowable twin, its auxiliary questions) are made by one call, so they
always land in the same split; the held-out split uses its own template families.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from collections.abc import Callable, Iterable

import numpy as np

from . import contrastive, distractor, posterior
from .core import ROLES, V7Source, check_group, rows

BASES: dict[str, Callable] = {**contrastive.BASES, **posterior.BASES, **distractor.BASES}
SOLVERS: dict[str, Callable] = {**contrastive.SOLVERS}
FAMILY = {
    "rw": "temporal",
    "sc": "temporal",
    "wd": "temporal",
    "inv": "math",
    "rp": "policy",
    "lrp": "long_policy",
    **posterior.FAMILY,
    **distractor.FAMILY,
}

PAIRS = ("a", "b")
CONTRAST = {"rw": 1.0, "inv": 1.0, "sc": 1.0, "wd": 1.0, "rp": 1.0, "lrp": 0.25}

V7_GENERATORS: dict[str, V7Source] = {
    # 1. contrastive one-fact pairs (both members, same split)
    "gen7_return_window_pairs": V7Source({"rw": 1.0}, PAIRS, "temporal"),
    "gen7_invoice_budget_pairs": V7Source({"inv": 1.0}, PAIRS, "math"),
    "gen7_schedule_pairs": V7Source({"sc": 1.0}, PAIRS, "temporal"),
    "gen7_weekday_pairs": V7Source({"wd": 1.0}, PAIRS, "temporal"),
    "gen7_refund_pairs": V7Source({"rp": 1.0}, PAIRS, "policy"),
    "gen7_long_refund_pairs": V7Source({"lrp": 1.0}, PAIRS, "long_policy"),
    # 2. unknowable twins: one evidence line removed (noul: exact base rate, soft; choice:
    # the abstain option)
    "gen7_unknowable": V7Source(CONTRAST, ("unk",), "unknowable", row_family={"unk": "unknowable"}),
    # 3. intermediate quantities: a choice question about the same state (days elapsed,
    # last day, total, overlap, refund period, date); a separate low-share source
    "gen7_aux_quantities": V7Source(
        CONTRAST, ("aux",), "intermediate", row_family={"aux": "intermediate"}
    ),
    # 5. exact posteriors (probability family, the noul band)
    "gen7_posterior": V7Source(dict.fromkeys(posterior.BASES, 1.0), ("a",), "probability"),
    # 6. distractor opinions / injected instructions, label unchanged
    "gen7_distractor_opinion": V7Source({"opinion": 1.0}, ("a",), "adversarial"),
}

# 3. The fade-out of the intermediate-quantity curriculum. The training stream draws its
# sources with fixed probabilities (tjev.data.mixture.Mixture: segment i draws from
# default_rng([seed, i]), whatever the step), so a share that decreases over training is not
# supported: gen7_aux_quantities is a separate source at the schedule's mean share
# (aux_static_share). aux_share() is the intended schedule, for when the stream gets a
# step-dependent mixture (Mixture.draw would need the segment index).
AUX_FADE = {"start": 0.015, "end_frac": 0.6}  # share of all segments at step 0; zero from 60%


def aux_share(step: int, total_steps: int, start: float = AUX_FADE["start"],
              end_frac: float = AUX_FADE["end_frac"]) -> float:  # fmt: skip
    """The auxiliary source's share of the segments at ``step``: linear from ``start`` to 0
    at ``end_frac`` of training, then 0."""
    return max(0.0, start * (1 - step / (end_frac * total_steps)))


def aux_static_share(start: float = AUX_FADE["start"], end_frac: float = AUX_FADE["end_frac"]):
    """The schedule's mean over training: the share of a fixed-weight source."""
    return start * end_frac / 2


try:  # 4. real-text seeds (their own module: licences per source)
    from . import realtext

    BASES.update(realtext.BASES)
    SOLVERS.update(realtext.SOLVERS)
    FAMILY.update(realtext.FAMILY)
    V7_GENERATORS.update(realtext.SOURCES)
except ImportError:  # pragma: no cover - the module is optional while it is written
    realtext = None


def _normalized(text: str) -> str:  # as tjev.data.mix.normalized
    return re.sub(r"\W+", " ", text.lower()).strip()


def key_of(row: dict) -> str:
    return _normalized(row["state"]) + "\x1f" + _normalized(row["question"]["instructions"])


def groups(name: str, seed: int, *, fr_share: float = 0.3, split: str = "train"):
    """The source's groups, in order, with all their rows (every role): an endless
    iterator of ``(group, rows)``; :func:`generate_v7` selects the roles and dedups."""
    spec = V7_GENERATORS[name]
    rng = np.random.default_rng([seed, zlib.crc32(name.encode()), zlib.crc32(split.encode())])
    srng = np.random.default_rng([seed, zlib.crc32(name.encode()), zlib.crc32(split.encode()), 5])
    bases = list(spec.bases)
    p = np.asarray([spec.bases[b] for b in bases], float)
    p /= p.sum()
    while True:
        base = bases[int(rng.choice(len(bases), p=p))] if len(bases) > 1 else bases[0]
        lang = "fr" if rng.random() < fr_share else "en"
        g = BASES[base](rng, lang, split, srng)
        if g is None:  # a seed source ran out of usable rows for this draw
            continue
        if g.evidence and base in SOLVERS:
            check_group(g, SOLVERS[base])
        yield g, rows(g, ROLES, twin_tag=g.extra.get("twin"))


def generate_v7(
    name: str, n: int, seed: int, *, fr_share: float = 0.3, split: str = "train"
) -> list[dict]:
    """At most ``n`` rows of whole groups (a pair is never cut), distinct keys across
    every row of every group (a group with a repeated key is skipped whole)."""
    spec = V7_GENERATORS[name]
    out: list[dict] = []
    seen: set[str] = set()
    tries = 0
    for gi, (g, all_rows) in enumerate(groups(name, seed, fr_share=fr_share, split=split)):
        tries += 1
        if len(out) >= n or tries > 30 * n + 200:
            break
        keys = [key_of(r) for r in all_rows]
        if len(set(keys)) < len(keys) or seen.intersection(keys):
            continue
        keep = [r for r in all_rows if r["meta"]["role"] in spec.roles]
        if not keep:
            continue
        if len(out) + len(keep) > n:
            break
        seen.update(keys)
        count = Counter()
        for r in keep:
            role = r["meta"]["role"]
            suffix = f"{role}{count[role]}" if role == "aux" else role
            count[role] += 1
            fam = spec.row_family.get(role) or FAMILY.get(g.base, spec.family)
            r["meta"]["group"] = gi
            r.update(id=f"{name}-{split}-{seed}-g{gi}-{suffix}", source=name, family=fam,
                     lang=g.lang)  # fmt: skip
            out.append(r)
    return out


def source_info(name: str) -> dict:
    """What the mix builder records for a v7 source (manifest ``license`` / ``family``)."""
    spec = V7_GENERATORS[name]
    licence, redistributable, note = spec.licence
    return {"license": licence, "commercial_ok": redistributable in ("yes", "share-alike"),
            "family": spec.family, "redistributable": redistributable, "note": note}  # fmt: skip


def group_of(row_id: str) -> str:
    """The group part of a v7 row id (``<source>-<split>-<seed>-g<k>``)."""
    return row_id.rsplit("-", 1)[0]


def drop_orphans(rows_: Iterable[dict], roles: tuple[str, ...] = PAIRS) -> list[dict]:
    """Rows whose group still has every member of ``roles`` among them: after a post-pass
    drops one member of a pair (dedup, contamination), drop the other one too."""
    rows_ = list(rows_)
    have: dict[str, set[str]] = {}
    for r in rows_:
        role = r["id"].rsplit("-", 1)[1]
        have.setdefault(group_of(r["id"]), set()).add(role)
    out = []
    for r in rows_:
        gid, role = group_of(r["id"]), r["id"].rsplit("-", 1)[1]
        if role in roles and not set(roles) <= have[gid]:
            continue
        out.append(r)
    return out
