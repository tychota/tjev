"""The official JevBench evaluation: tjev's ``/v1/systemone`` server under the official harness.

``tjev eval-official TARGET`` serves TARGET (a run: its selected checkpoint and calibration;
or a bare HF snapshot: the base model zero-shot) on a local port, in this process, and runs
the unmodified harness (github.com/fstandhartinger/jevbench, pinned commit) against it as a
subprocess: ``python -m jevbench.cli run --adapter typesafe --endpoint http://127.0.0.1:PORT``.
The harness writes its per-item records; this module scores them with the harness's own
code where it has it (``jevbench.summarize``, ``composite_v13``, ``composite_v14``) and with
the frozen v1.5 method (``docs/METHOD-v1.5.md`` and the equal-types headline addendum, which
the harness does not implement yet) for the rest. See docs/official-eval.md.

What can be computed on the public 231 (easy 48, standard 72, hard 111; no judge tier):

- per tier / type / family accuracy, the harness's v1 summary (Brier, ECE, latency);
- v1.3 Intelligence (chance-corrected, the published per-tier chances) and Calibration
  (hard tier: ECE and TVD to the gold distributions), Speed, Cost;
- v1.5 open-set Intelligence on the public items (per-item chance; noul P(yes) in
  (0.2, 0.8) is an abstention, counted wrong; score on the MAE of the expected level) and
  the typed calibration;
- composites as **public-only proxies**: the official v1.4 and v1.5 Intelligence need the
  sealed set (the evaluator's), so the official composites are ``null``.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from tjev.eval import v15

HARNESS_URL = "https://github.com/fstandhartinger/jevbench"
HARNESS_COMMIT = "bb05a335bc809e61b20c0f745d25499a82b326fc"  # 2026-09-29, v1.5.4 board
METHOD_V15_SHA256 = "c25d3d8b8512e4d93370a9e0c99705d19b2a9389956ca33b8a4bd2b0ec501c07"
# public split file -> tier (the harness's "original" split is the standard tier)
PUBLIC_TIERS = {"easy": "easy", "original": "standard", "hard": "hard"}
# v1.5 M2 price floor: the base model's public market reference price, USD per M tokens
# (input, output); a logit readout generates no output tokens (interpretation I-2)
REFERENCE_PRICES = {
    "Qwen3.5-4B": (0.03, 0.15, "DeepInfra Qwen/Qwen3.5-4B, frozen 25 Sep 2026 snapshot (v1.5 M2)"),
}
V15_TIER_WEIGHTS = v15.TIER_WEIGHTS
V15_TYPE_WEIGHTS = v15.TYPE_WEIGHTS
V15_PRESETS = {  # axis weights I, C, S, $ and the Intelligence gate
    "A": ((0.25, 0.25, 0.25, 0.25), 50.0),  # official headline (with equal types)
    "B": ((0.40, 0.20, 0.20, 0.20), 50.0),
    "C": ((0.25, 0.25, 0.25, 0.25), 60.0),
}
NOUL_BAND = v15.NOUL_BAND


# --- the harness ------------------------------------------------------------------------


def harness_info(root: str | Path) -> dict:
    """Commit, cleanliness and the v1.5 method hash of a harness checkout."""
    root = Path(root)

    def git(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    commit = git("rev-parse", "HEAD")
    method = root / "docs" / "METHOD-v1.5.md"
    method_sha = hashlib.sha256(method.read_bytes()).hexdigest() if method.exists() else None
    return {
        "url": HARNESS_URL,
        "path": str(root),
        "commit": commit or None,
        "pinned_commit": HARNESS_COMMIT,
        "pinned": commit == HARNESS_COMMIT,
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "method_v15_sha256": method_sha,
        "method_v15_frozen": method_sha == METHOD_V15_SHA256,
        "adapter": "typesafe",
    }


def harness_modules(root: str | Path) -> dict[str, Any]:
    """The harness's own scoring modules, imported from the checkout (not modified)."""
    root = str(Path(root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    names = ("tasks", "summarize", "metrics", "composite_v13", "composite_v14")
    mods = {n: importlib.import_module(f"jevbench.{n}") for n in names}
    origin = Path(str(mods["tasks"].__file__)).resolve()
    if not origin.is_relative_to(Path(root)):
        raise ImportError(f"jevbench imported from {origin}, not from {root}")
    return mods


def public_tasks(root: str | Path, mods: dict[str, Any]) -> list[tuple[Any, str, Path]]:
    """(task, tier, file) for the public items, in the harness's file order."""
    out = []
    for split, tier in PUBLIC_TIERS.items():
        path = Path(root) / "datasets" / "public" / f"{split}.jsonl"
        out += [(t, tier, path) for t in mods["tasks"].load_jsonl(str(path))]
    return out


def run_harness(
    root: str | Path,
    tasks_file: Path,
    endpoint: str,
    out: Path,
    *,
    model: str,
    label: str,
    price: tuple[float, float] | None,
    limit: int | None = None,
) -> tuple[Path, Path, int]:
    """``python -m jevbench.cli run --adapter typesafe`` against ``endpoint`` (the harness's
    own CLI, no key, a budget cap that cannot bind locally); returns (results, manifest,
    exit code). The harness refuses result / raw paths inside its checkout."""
    results, manifest = out / "harness-results.jsonl", out / "harness-manifest.json"
    cmd = [
        sys.executable,
        "-m",
        "jevbench.cli",
        "run",
        "--tasks",
        str(tasks_file),
        "--adapter",
        "typesafe",
        "--endpoint",
        endpoint,
        "--model",
        model,
        "--key-env",
        "",
        "--results",
        str(results),
        "--ledger",
        str(out / "harness-ledger.jsonl"),
        "--raw-dir",
        str(out / "raw"),
        "--cap-usd",
        "1000000",
        "--manifest",
        str(manifest),
        "--run-label",
        label,
        "--cost-basis",
        "base_model_reference_price_estimate" if price else "self_hosted_unpriced",
    ]
    if price:
        cmd += ["--price-in-per-m", repr(price[0]), "--price-out-per-m", repr(price[1])]
    if limit:
        cmd += ["--limit", str(limit)]
    env = {**os.environ, "PYTHONPATH": str(Path(root).resolve())}
    with (out / "harness.log").open("w") as log:
        rc = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return results, manifest, rc


# --- the server ---------------------------------------------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextlib.contextmanager
def local_server(
    backend: Any, *, port: int = 0, max_wait_ms: float = 0.0, policy: Any = None
) -> Iterator[str]:
    """tjev's app under uvicorn in a thread of this process; yields its base URL.

    ``max_wait_ms`` 0: the harness is serial, so a micro-batch never waits for company.
    ``policy``: a :class:`tjev.serve.policy.ServingPolicy` (None: off)."""
    import uvicorn

    from tjev.serve.app import create_app

    port = port or free_port()
    app = create_app(backend, max_items=64, max_wait_ms=max_wait_ms, policy=policy)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 120
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("the tjev server did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=30)


# --- scoring --------------------------------------------------------------------------------


_ece = v15.ece  # the harness's ECE (10 equal-width bins, the last one closed)
tvd = v15.tvd
ece_points = v15.ece_points
v15_cc = v15.cc
v15_intelligence = v15.intelligence


def gold_probs(task: Any) -> dict | None:
    return (task.provenance or {}).get("gold_probs")


def v15_item(task: Any, rec: dict) -> dict:
    """One item under v1.5 §3.1 from the harness's own per-item fields (``correct``: its
    argmax; ``ordinal_ev``: its expected level): choice correct / chance 1/options; noul
    thresholded at 0.2 / 0.8 (between: abstention, wrong); score nMAE of the expected
    level and the uniform guess's nMAE. Failed or invalid answers are wrong (score: the
    worst level). Kept independent of :func:`tjev.eval.v15.score_item` (which recomputes
    them from the distribution): tests/eval/test_v15.py checks that the two agree."""
    kind = task.question["type"]
    probs = rec.get("probs") if rec.get("valid") else None
    out: dict[str, Any] = {"type": kind}
    if kind == "choice":
        out |= {"correct": bool(rec.get("correct")), "chance": 1 / len(task.labels)}
    elif kind == "noul":
        p = None if probs is None else float(probs["yes"])
        yes = p is not None and p >= NOUL_BAND[1]
        no = p is not None and p <= NOUL_BAND[0]
        out |= {
            "correct": (yes and task.expected == "yes") or (no and task.expected == "no"),
            "abstained": p is not None and not (yes or no),
            "p_yes": p,
        }
    else:
        k, gold = len(task.labels), int(task.expected)
        ev = rec.get("ordinal_ev") if probs is not None else None
        worst = max(gold, k - 1 - gold) / (k - 1)
        out |= {
            "nmae": worst if ev is None else abs(float(ev) - gold) / (k - 1),
            "nmae_chance": sum(abs(level - gold) for level in range(k)) / (k * (k - 1)),
        }
    return out


def harness_row(task: Any, rec: dict, tier: str = v15.UNTIERED) -> v15.Row:
    """A harness record as a :class:`tjev.eval.v15.Row` (its validated distribution)."""
    labels = tuple(str(label) for label in task.labels)
    probs = rec.get("probs") if rec.get("valid") else None
    gold = gold_probs(task)
    return v15.Row(
        id=str(task.id) if hasattr(task, "id") else "",
        type=task.question["type"],
        labels=labels,
        gold=str(task.expected),
        probs=None if not probs else np.asarray([float(probs[k]) for k in labels]),
        tier=tier,
        family=getattr(task, "family", None) or "unknown",
        gold_probs=None if not gold else np.asarray([float(gold.get(k, 0.0)) for k in labels]),
    )


def v15_calibration(pairs: list[tuple[Any, dict]] | list[tuple[Any, dict, str]]) -> dict:
    """Typed calibration (v1.5 §5, :func:`tjev.eval.v15.calibration`) of harness records:
    (task, record) or (task, record, tier) triples."""
    return v15.calibration([harness_row(*p) for p in pairs])


def harmonic(axes: dict[str, float | None], weights: tuple, gate: float) -> float | None:
    """v1.4 / v1.5 composite: the weighted harmonic mean of I, C, S, $ and the three gates
    (an axis below ``gate`` (Intelligence) or 50 (Speed, Cost) multiplies by (axis/gate)²)."""
    names = ("intelligence", "calibration", "speed", "cost")
    maybe = [(axes.get(n), w) for n, w in zip(names, weights, strict=True) if w > 0]
    pairs = [(v, w) for v, w in maybe if v is not None]
    if len(pairs) < len(maybe):
        return None
    if any(v <= 0 for v, _ in pairs):
        score = 0.0
    else:
        score = sum(w for _, w in pairs) / sum(w / v for v, w in pairs)
    for name, threshold in (("intelligence", gate), ("speed", 50.0), ("cost", 50.0)):
        v = axes[name]
        if v is not None and v < threshold:
            score *= (max(0.0, v) / threshold) ** 2
    return score


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    k = (len(vals) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return vals[lo] if lo == hi else vals[lo] * (hi - k) + vals[hi] * (k - lo)


def tjev_metrics(pairs: list[tuple[Any, dict]]) -> dict:
    """tjev's own metrics (``tjev.eval.metrics.summarize``: accuracy, NLL, Brier, 15-bin
    ECE) on the served distributions, one-hot gold: comparable to ``tjev eval`` reports."""
    from tjev.eval.metrics import summarize

    rows = [(t, r["probs"]) for t, r in pairs if r.get("valid") and r.get("probs")]
    if not rows:
        return {"n": 0}
    width = max(len(t.labels) for t, _ in rows)
    probs, target, mask = (np.zeros((len(rows), width)) for _ in range(3))
    for i, (t, p) in enumerate(rows):
        k = len(t.labels)
        probs[i, :k] = [p[str(label)] for label in t.labels]
        target[i, t.labels.index(str(t.expected))] = 1.0
        mask[i, :k] = 1
    return summarize(probs, target, mask.astype(bool)) | {"failed": len(pairs) - len(rows)}


def score_public(
    pairs: list[tuple[Any, str, dict]],
    mods: dict[str, Any],
    *,
    endpoint_kind: str = "gpu",
    price: tuple[float, float] | None = None,
) -> dict:
    """The official metrics of harness records on the public items. ``pairs``: (task, tier,
    record). ``endpoint_kind`` (the harness's): ``gpu`` / ``cpu`` for an own server (latency
    ×2 + 0.15 s), ``api`` for a production API (unadjusted)."""
    v13, v14 = mods["composite_v13"], mods["composite_v14"]
    tasks = [t for t, _, _ in pairs]
    records = [r for _, _, r in pairs]
    summary = mods["summarize"].summarize(tasks, records)
    tiers = {}
    for tier in ("easy", "standard", "hard"):
        sel = [(t, r) for t, tr, r in pairs if tr == tier]
        if sel:
            m = mods["summarize"].metric([t for t, _ in sel], [r for _, r in sel])
            tiers[tier] = {
                k: m[k]
                for k in (
                    "n_scorable",
                    "n_correct",
                    "accuracy",
                    "schema_validity",
                    "brier_mean",
                    "ordinal_mae",
                )
            }
            tiers[tier]["ece"] = (m["ece"] or {}).get("ece")
            tiers[tier]["tjev"] = tjev_metrics(sel)
    by_type = {}
    for kind in ("choice", "noul", "score"):
        sel = [(t, r) for t, _, r in pairs if t.question["type"] == kind]
        if sel:
            by_type[kind] = {
                "n": len(sel),
                "accuracy": float(np.mean([bool(r.get("correct")) for _, r in sel])),
            }

    # v1.3: chance-corrected tier accuracy (published per-tier chances; judge absent, the
    # weights renormalised); calibration on the hard tier: ECE and TVD to gold distributions
    acc = {t: v["accuracy"] for t, v in tiers.items()}
    hard = [(t, r) for t, tr, r in pairs if tr == "hard" and r.get("valid") and r.get("probs")]
    hard_ece = _ece(
        [max(r["probs"].values()) for _, r in hard],
        [float(bool(r.get("correct"))) for _, r in hard],
    )
    gold = [(t, r, g) for t, r in hard if (g := gold_probs(t))]
    hard_tvd = float(np.mean([tvd(r["probs"], g, t.labels) for t, r, g in gold])) if gold else None
    public_chance = {
        tier: float(np.mean([1 / len(t.labels) for t, tr, _ in pairs if tr == tier]))
        for tier in tiers
    }

    # latency: the harness's wall time per request; Speed on the standard tier (v1.5: the
    # serial standard + judge run; judge is not public)
    def latency(sel: list[dict]) -> dict:
        xs = [r["latency_s"] for r in sel if r.get("ok") and r.get("latency_s") is not None]
        p50, p95 = _percentile(xs, 0.5), _percentile(xs, 0.95)
        return {
            "n": len(xs),
            "p50_s": p50,
            "p95_s": p95,
            "p50_adjusted_s": v13.adjusted_latency(p50, endpoint_kind),
            "p95_adjusted_s": v13.adjusted_latency(p95, endpoint_kind),
            "speed": v13.speed(p50, p95, endpoint_kind) if xs else None,
        }

    standard = [r for _, tr, r in pairs if tr == "standard"]
    lat = {
        "endpoint_kind": endpoint_kind,
        "all": latency(records),
        "standard": latency(standard) if standard else None,
    }
    speed = (lat["standard"] or lat["all"])["speed"]
    # Cost (v1.5 §5, M2): tokens pooled over the answered decisions times the price; the
    # harness's own per-1,000 price is null as soon as one request failed
    ok = [r for r in records if r.get("ok") and isinstance(r.get("usage"), dict)]
    tokens_in = [float(r["usage"].get("input_tokens") or 0) for r in ok]
    tokens_out = [float(r["usage"].get("output_tokens") or 0) for r in ok]
    per_1000 = None
    if ok and price:
        per_1000 = 1000 * (np.mean(tokens_in) * price[0] + np.mean(tokens_out) * price[1]) / 1e6
    harness_cost = [r.get("cost_usd") for r in records]
    cost = {
        "usd_per_1000": None if per_1000 is None else float(per_1000),
        "harness_usd_per_1000": summary.get("price_per_1000_decisions_usd"),
        "input_tokens_mean": float(np.mean(tokens_in)) if ok else None,
        "output_tokens_mean": float(np.mean(tokens_out)) if ok else None,
        "n_priced": sum(c is not None for c in harness_cost),
        "basis": sorted({r.get("cost_basis", "unknown") for r in records}),
        "axis": v13.cost(per_1000) if per_1000 else None,
    }
    c13 = v13.calibration(hard_ece, hard_tvd)
    i13 = v13.intelligence(acc)
    axes13 = {"intelligence": i13, "calibration": c13, "speed": speed, "cost": cost["axis"]}

    scored = [(tr, v15_item(t, r)) for t, tr, r in pairs]
    i15 = v15_intelligence(scored)
    c15 = v15_calibration([(t, r, tr) for t, tr, r in pairs])

    def clip(x: float | None) -> float | None:  # Intelligence = clip(base · penalty, 0, 100)
        return None if x is None else min(100.0, max(0.0, x))

    composites = {}
    for name, (weights, gate) in V15_PRESETS.items():  # the addendum: equal types in A, B, C
        axes = {
            "intelligence": clip(i15["intelligence_equal"]),
            "calibration": c15["calibration_equal"],
            "speed": speed,
            "cost": cost["axis"],
        }
        composites[name] = {"axes": axes, "score": harmonic(axes, weights, gate)}
    frozen_axes = {
        "intelligence": clip(i15["intelligence_frozen"]),
        "calibration": c15["calibration_frozen"],
        "speed": speed,
        "cost": cost["axis"],
    }
    composites["B_frozen_types"] = {
        "axes": frozen_axes,
        "score": harmonic(frozen_axes, V15_PRESETS["B"][0], 50.0),
    }
    sealed_note = "needs the sealed set (evaluator-only): official value not computable"
    return {
        "n": len(records),
        "failed": sum(1 for r in records if not r.get("ok")),
        "accuracy": summary["accuracy"],
        "tiers": tiers,
        "types": by_type,
        "families": {
            k: {"n": v["n_scorable"], "accuracy": v["accuracy"]}
            for k, v in summary["per_family"].items()
        },
        "harness_summary": summary,
        "tjev": tjev_metrics([(t, r) for t, _, r in pairs]),
        "latency": lat,
        "cost": cost,
        "v13": {
            "intelligence_public": i13,
            "intelligence_public_item_chance": v13.intelligence(acc, public_chance),
            "tier_chances": {t: v13.TIER_CHANCES[t] for t in tiers},
            "calibration": {
                "hard_ece": hard_ece,
                "hard_tvd": hard_tvd,
                "tvd_n": len(gold),
                "axis": c13,
            },
            "axes": axes13,
            "score_public": v13.jevbench_score(axes13) if None not in axes13.values() else None,
        },
        "v14": {
            "official": None,
            "note": sealed_note,
            "composite_public_proxy": v14.harmonic(axes13) if None not in axes13.values() else None,
        },
        "v15": {
            "official": None,
            "note": sealed_note + "; proxies take I_sealed = I_open (no gap penalty) and only "
            "the 231 v1.2 public items (not the 601 v1.5 published ones)",
            "noul_band": list(NOUL_BAND),
            "intelligence_open_public": i15,
            "calibration_typed": c15,
            "composites_public_proxy": composites,
        },
    }


# --- the whole evaluation -----------------------------------------------------------------


def model_size(name: str) -> str | None:
    import re

    m = re.search(r"Qwen3\.5-([0-9.]+B)", name)
    return f"Qwen3.5-{m.group(1)}" if m else None


def build_backend(
    target: Path,
    *,
    calibration: Path | None,
    orders: int,
    step: int | None,
    prompt: str,
    buckets: tuple[int, ...],
    warmup: bool,
    scratch: Path,
    noul_order: str = "shuffled",
) -> tuple[Any, dict]:
    """A run directory (its selected step; with tjev's prompt, the ``tjev post`` calibration
    of the same readout, ``post/calibration-jax.json`` or e.g. ``…-k2.json``, when present) or a bare snapshot (the base model zero-shot; a calibration may be a
    ``tjev eval-base --calibrate-on`` report, whose artifact is used)."""
    from tjev.eval.runs import readout_tag
    from tjev.serve.backends import JaxBackend

    is_run = (target / "checkpoints").is_dir() and (target / "config.json").exists()
    default = target / "post" / f"calibration-jax{readout_tag(orders, noul_order)}.json"
    if calibration is None and is_run and prompt == "tjev" and default.exists():
        calibration = default
    cal_file = calibration
    if calibration is not None:
        art = json.loads(Path(calibration).read_text(encoding="utf-8"))
        if "calibration_artifact" in art:
            cal_file = scratch / "calibration-artifact.json"
            cal_file.write_text(json.dumps(art["calibration_artifact"]), encoding="utf-8")
    common: dict[str, Any] = {
        "calibration": cal_file,
        "orders": orders,
        "prompt": prompt,
        "noul_order": noul_order,
        "buckets": buckets,
        "warmup": warmup,
    }
    if is_run:
        backend = JaxBackend(run_dir=target, step=step, **common)
        base = json.loads((target / "config.json").read_text())["config"]["model"]["path"]
    else:
        backend = JaxBackend(model_path=target, **common)
        base = str(target)
    info: dict[str, Any] = {
        "target": str(target),
        "kind": "run" if is_run else "base",
        "model": backend.name,
        "base_model": model_size(base),
        "calibration": str(calibration) if calibration else None,
        "temperatures": backend.temperatures,
        "orders": orders,
        "prompt": prompt,
        "noul_order": noul_order,
    }
    return backend, info


def _price(served: dict, price: tuple[float, float] | None) -> tuple[tuple | None, str | None]:
    if price is not None:
        return price, "given"
    if served.get("base_model") in REFERENCE_PRICES:
        ref = REFERENCE_PRICES[served["base_model"]]
        return ref[:2], ref[2]
    return None, None


def evaluate_official(
    target: str | Path,
    *,
    harness: str | Path,
    out: str | Path,
    suite: str = "jevbench-public",
    calibration: str | Path | None = None,
    orders: int = 1,
    step: int | None = None,
    prompt: str = "tjev",
    limit: int | None = None,
    per_tier: int = 0,
    endpoint_kind: str = "gpu",
    price: tuple[float, float] | None = None,
    allow_unpinned: bool = False,
    buckets: tuple[int, ...] = (1024, 2048, 4096, 8192),
    warmup: bool = True,
    di_kit: str | Path | None = None,
    di_suite: str | Path | None = None,
    backend: Any = None,
    noul_order: str = "shuffled",
    policy: Any = None,
) -> dict:
    """Serve ``target``, run the official harness (``jevbench-public``: the JevBench harness
    on the public items; ``decision-index``: the Decision Index kit's ``http`` engine over a
    built suite), score; writes ``out/report.json`` (and the harness's own files)."""
    if suite not in ("jevbench-public", "decision-index"):
        raise ValueError(f"unknown suite {suite!r}")
    out = Path(out)
    if suite == "decision-index":
        if di_kit is None or di_suite is None:
            raise ValueError("decision-index needs the kit checkout and a built suite directory")
        info = kit_info(di_kit)
    else:
        info = harness_info(harness)
    if not info["pinned"] and not allow_unpinned:
        raise RuntimeError(
            f"{info['url']} at {info['commit']}, pinned {info['pinned_commit']} "
            "(check it out, or allow_unpinned)"
        )
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    if backend is None:
        backend, served = build_backend(
            Path(target),
            calibration=Path(calibration) if calibration else None,
            orders=orders,
            step=step,
            prompt=prompt,
            buckets=buckets,
            warmup=warmup,
            scratch=out,
            noul_order=noul_order,
        )
    else:  # tests: an already built backend
        served: dict[str, Any] = {
            "target": str(target),
            "kind": "given",
            "model": backend.name,
            "base_model": model_size(str(target)),
            "orders": orders,
            "prompt": prompt,
        }
    # the serving policy (None: off; tjev.serve.policy): serving temperatures and the noul
    # commit change what the harness scores
    from tjev.serve.policy import as_policy

    policy = as_policy(policy)
    served["policy"] = None if policy is None else policy.describe()
    if suite == "decision-index":
        t0 = time.perf_counter()
        with local_server(backend, policy=policy) as url:
            result = run_decision_index(  # both checked above
                Path(str(di_kit)),
                Path(str(di_suite)),
                url,
                out,
                model=str(served["model"]),
                limit=limit,
            )
        report = {
            "suite": suite,
            "served": served,
            "kit": info,
            "wall_s": time.perf_counter() - t0,
        } | result
        (out / "report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        return report

    harness = Path(harness)
    mods = harness_modules(harness)
    tasks = public_tasks(harness, mods)
    if per_tier:  # a smoke run: the first N items of each tier
        seen: dict[str, int] = defaultdict(int)
        kept = []
        for t, tier, path in tasks:
            seen[tier] += 1
            if seen[tier] <= per_tier:
                kept.append((t, tier, path))
        tasks = kept
    price, price_source = _price(served, price)
    # one tasks file in tier order (the harness's --tasks takes a comma list, but one file
    # keeps --limit simple)
    tasks_file = out / "tasks.jsonl"
    with tasks_file.open("w", encoding="utf-8") as f:
        for t, _, _ in tasks:
            f.write(t.to_json() + "\n")
    t0 = time.perf_counter()
    with local_server(backend, policy=policy) as url:
        results, manifest, rc = run_harness(
            harness,
            tasks_file,
            url,
            out,
            model=str(served["model"]),
            label=str(served["model"]),
            price=price,
            limit=limit,
        )
    wall = time.perf_counter() - t0
    if not results.exists():
        raise RuntimeError(f"the harness wrote no results (exit {rc}; see {out / 'harness.log'})")
    records = {}
    for line in results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            records[r["task_id"]] = r
    pairs = [(t, tier, records[t.id]) for t, tier, _ in tasks if t.id in records]
    report = {
        "suite": suite,
        "served": served,
        "harness": info
        | {
            "exit_code": rc,
            "manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
        },
        "price_per_m": {"input": price[0], "output": price[1], "source": price_source}
        if price
        else None,
        "wall_s": wall,
        "complete": len(pairs) == len(tasks),
    } | score_public(pairs, mods, endpoint_kind=endpoint_kind, price=price)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


# --- the Decision Index kit ----------------------------------------------------------------

DI_URL = "https://github.com/sinanuozdemir/decision-index"
DI_COMMIT = "87d4650b42b377c0291a89c1f1a879f9b31082bf"  # 2026-09-27, Decision Index 0.2.1


def kit_info(root: str | Path) -> dict:
    info = harness_info(root)
    return info | {
        "url": DI_URL,
        "pinned_commit": DI_COMMIT,
        "pinned": info["commit"] == DI_COMMIT,
        "adapter": "http",
    }


def run_decision_index(
    kit: str | Path,
    suite_dir: str | Path,
    endpoint: str,
    out: Path,
    *,
    model: str,
    limit: int | None = None,
) -> dict:
    """The kit's own ``run --engine http`` (serial, resumable, ``--compact``: no item text
    in the results) and ``score`` over a suite built with ``decision_index suite rebuild``
    + ``suite import``; returns the score brief (``decision_index``, per-benchmark scores,
    areas, completeness)."""
    env = {**os.environ, "PYTHONPATH": str(Path(kit).resolve())}
    runs = out / "decision-index"
    base = [sys.executable, "-m", "decision_index"]
    run = [
        *base,
        "run",
        "--engine",
        "http",
        "--option",
        f"base_url={endpoint}",
        "--option",
        f"model={model}",
        "--suite-dir",
        str(suite_dir),
        "--out",
        str(runs),
        "--compact",
    ]
    if limit:
        run += ["--limit", str(limit)]
    with (out / "decision-index-run.log").open("w") as log:
        rc = subprocess.run(run, cwd=kit, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    score = subprocess.run(
        [
            *base,
            "score",
            "--results",
            str(runs / "results.jsonl"),
            "--suite-dir",
            str(suite_dir),
            "--engine",
            model,
        ],
        cwd=kit,
        env=env,
        capture_output=True,
        text=True,
    )
    (out / "decision-index-score.log").write_text(score.stdout + score.stderr)
    brief = json.loads(score.stdout) if score.returncode == 0 else None
    return {"run_exit_code": rc, "score_exit_code": score.returncode, "decision_index": brief}
