"""mix-v7.5-np: mix-v7.5 without the yes/no rebalance (docs/data.md, mix-v7.5-np).

The rebalance is baked into the base mixes: mix-v6 (`build --rebalance-noul 0.7`: mage,
mlma_en, mlma_fr, openjev, raid) and mix-v7 (`extend`, 0.7 on its new sources: forecasting,
oasst2, openjev_x, scruples). mix-v7.5 rebalanced nothing more. Its dropped rows come back
from builds of the same code without the rebalance (``scripts/build_mix_v75_np.sh``):

  recover   UNREB REB OUT --sources …   the unrebalanced train files of the rebalanced
            sources, checked: the seeded rebalance of each one gives REB's rows, in order
            (the licence fields aside: the build now records them); when today's build
            has a few rows more or fewer, the dropped rows are located by the seeded draw
            (``realign``)
  assemble  MIX_V7 V6_DIR V7_DIR OUT    mix-v7 with those sources' rebalanced train files
            replaced by the unrebalanced ones (the mix-v6 rows licence-annotated as
            `extend` does, checked against mix-v7's), the manifest counts updated
  finalize  V75 V7 V7NP PRE NP           drifted sources frozen to mix-v7.5's train files,
            held-out files copied from mix-v7.5, the checks
  heldout   A B                         sha256 of every held-out file of two mixes
  report    V75 NP                      yes-share per noul source, epochs at 5000 / 2500
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from tjev.data import balance
from tjev.data.item import parse_item, read_jsonl, write_jsonl
from tjev.data.mix import _seed_of, annotate_licence, license_rows

V6_REBALANCED = ("mage", "mlma_en", "mlma_fr", "openjev", "raid")
V7_REBALANCED = ("forecasting", "oasst2", "openjev_x", "scruples")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lines_of(path: Path) -> list[str]:
    # split on "\n" only (balance.rebalance_file): U+2028 may sit inside JSON strings
    return [x for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def unlicensed(line: str) -> dict:
    """A row without the licence fields: the build records them since mix-v7 (mix-v6's
    rows get them from `extend`, ``annotate_licence``)."""
    row = json.loads(line)
    row.pop("license", None)
    row.pop("license_use", None)
    return row


def rebalanced(lines: list[str], name: str, target: float, max_loss: float, seed: int):
    items = [parse_item(json.loads(x)) for x in lines]
    kept, stats = balance.rebalance_noul(items, target, max_loss=max_loss,
                                         seed=_seed_of(name) + seed)  # fmt: skip
    return kept, stats, items


def key(line: str) -> str:
    return json.dumps(unlicensed(line), sort_keys=True, ensure_ascii=False)


def realign(
    lines: list[str], reb_lines: list[str], ref: dict, name: str, target: float, seed: int
) -> tuple[list[str], dict]:
    """When the unrebalanced file of today's code is not exactly the one the rebalance saw
    (a few rows more or fewer from the post-pass), the rows it dropped are found by the
    seeded draw itself: it picks *positions* in the majority-class list (its size is in the
    built mix's table), so the built mix's kept majority rows pin that list's alignment.
    Majority rows not kept are restored, except in the gaps where an extra row of today's
    build sits (ambiguous: none of them is restored). Returns the merged file (the built
    mix's rows as they are, in order, plus the restored rows) and its stats."""
    import numpy as np

    items = [parse_item(json.loads(x)) for x in lines]
    keys = [key(x) for x in lines]
    reb_keys = [key(x) for x in reb_lines]
    kept_set, have = set(reb_keys), set(keys)
    yes_major = ref["yes_share_before"] > target
    n_major = ref["yes_before"] if yes_major else ref["noul_before"] - ref["yes_before"]

    def major(it) -> bool:
        return it.question.type == "noul" and balance.is_yes(it) == yes_major

    rng = np.random.default_rng(_seed_of(name) + seed)
    drawn = set(rng.choice(np.arange(n_major), size=ref["dropped"], replace=False).tolist())
    kept_pos = [p for p in range(n_major) if p not in drawn]
    maj = [i for i, it in enumerate(items) if major(it)]
    q = [j for j, i in enumerate(maj) if keys[i] in kept_set]
    if len(q) != len(kept_pos):
        raise SystemExit(f"{name}: {len(q)} kept majority rows, the draw keeps {len(kept_pos)}")
    shift = [a - b for a, b in zip(q, kept_pos, strict=True)]
    extra = len(maj) - n_major
    if any(b < a for a, b in itertools.pairwise(shift)) or shift[0] < 0 or shift[-1] > extra:
        raise SystemExit(f"{name}: the draw does not align with the built mix")
    ambiguous: set[int] = set()
    bounds = [(-1, 0), *zip(q, shift, strict=True), (len(maj), extra)]
    for (q0, s0), (q1, s1) in itertools.pairwise(bounds):
        if s1 != s0:  # an extra row of today's build sits in this gap
            ambiguous |= set(range(q0 + 1, q1))
    restore = {maj[j] for j in range(len(maj)) if keys[maj[j]] not in kept_set
               and j not in ambiguous}  # fmt: skip
    merged, j = [], 0
    for i, line in enumerate(lines):
        while j < len(reb_lines) and reb_keys[j] not in have:
            merged.append(reb_lines[j])  # a built row today's build no longer has
            j += 1
        if j < len(reb_lines) and keys[i] == reb_keys[j]:
            merged.append(reb_lines[j])
            j += 1
        elif i in restore:
            merged.append(line)
    merged += reb_lines[j:]
    stats = {"extra_majority_rows_today": extra, "ambiguous_not_restored": len(ambiguous)
             - extra, "restored": len(restore), "dropped_by_rebalance": ref["dropped"]}  # fmt: skip
    return merged, stats


def recover(unreb: Path, reb: Path, out: Path, sources: list[str]) -> int:
    man = json.loads((reb / "manifest.json").read_text(encoding="utf-8"))
    cfg = man["rebalance_noul"]
    target, max_loss, seed = cfg["target"], cfg["max_loss"], cfg.get("seed", man.get("seed", 0))
    out.mkdir(parents=True, exist_ok=True)
    table = {}
    for name in sources:
        lines = lines_of(unreb / "train" / f"{name}.jsonl")
        reb_lines = lines_of(reb / "train" / f"{name}.jsonl")
        kept, stats, _ = rebalanced(lines, name, target, max_loss, seed)
        same = [unlicensed(lines[i]) for i in kept] == [unlicensed(x) for x in reb_lines]
        ref = cfg["sources"][name]
        print(f"[recover] {name}: {len(lines)} rows, yes {stats['yes_share_before']} → "
              f"{stats['yes_share_after']}, dropped {stats['dropped']} (built mix: "
              f"{ref['dropped']}), rebalance reproduces it: {same}", flush=True)  # fmt: skip
        if same:
            merged = list(lines)
            for j, i in enumerate(kept):  # the built mix's lines, byte for byte
                merged[i] = reb_lines[j]
            info = {"exact": True, "restored": stats["dropped"]}
        else:
            merged, info = realign(lines, reb_lines, ref, name, target, seed)
            info["exact"] = False
            print(f"  realigned on the seeded draw: {info}", flush=True)
        (out / f"{name}.jsonl").write_text("".join(x + "\n" for x in merged), encoding="utf-8")
        table[name] = info
    (out / "recover.json").write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
    common = sorted({p.name for p in (unreb / "train").glob("*.jsonl")}
                    & {p.name for p in (reb / "train").glob("*.jsonl")}
                    - {f"{n}.jsonl" for n in sources})  # fmt: skip
    diff = [n for n in common if sha(unreb / "train" / n) != sha(reb / "train" / n)
            and [unlicensed(x) for x in lines_of(unreb / "train" / n)]
            != [unlicensed(x) for x in lines_of(reb / "train" / n)]]  # fmt: skip
    print(f"[recover] other sources: {len(common) - len(diff)}/{len(common)} identical"
          + (f"; differ: {diff}" if diff else ""), flush=True)  # fmt: skip
    return 0


def assemble(v7: Path, v6_dir: Path, v7_dir: Path, out: Path) -> int:
    man = json.loads((v7 / "manifest.json").read_text(encoding="utf-8"))
    info = {
        **json.loads((v6_dir / "recover.json").read_text()),
        **json.loads((v7_dir / "recover.json").read_text()),
    }
    if out.exists():
        shutil.rmtree(out)
    for path in sorted(v7.rglob("*")):  # hard links: the base mix is never modified
        dest = out / path.relative_to(v7)
        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, dest)
    restored, ok = {}, True
    for name in (*V6_REBALANCED, *V7_REBALANCED):
        dest = out / "train" / f"{name}.jsonl"
        dest.unlink()
        built = lines_of(v7 / "train" / f"{name}.jsonl")
        if name in V6_REBALANCED:  # extend: base rows licence-annotated, written row for row
            items = [parse_item(unlicensed(x)) for x in lines_of(v6_dir / f"{name}.jsonl")]
            write_jsonl(dest, annotate_licence(items, name))
        else:
            shutil.copyfile(v7_dir / f"{name}.jsonl", dest)
        lines = lines_of(dest)
        items = [parse_item(json.loads(x)) for x in lines]
        built_set = set(built)
        same = [x for x in lines if x in built_set] == built  # mix-v7's rows, byte for byte
        ok &= same
        entry = man["sources"][name]
        before = entry["train"]
        entry["train"] = len(items)
        drops = dict(entry.get("drops", {}))
        undone = drops.pop("noul_rebalance", 0)
        entry["drops"] = drops
        entry["noul_rebalance_undone"] = {"dropped": undone, "restored": len(items) - before,
                                          **info[name]}  # fmt: skip
        if isinstance(entry.get("lang"), dict):
            entry["lang"] = dict(Counter(it.lang for it in items))
        if "license_use_rows" in entry:
            entry["license_use_rows"] = license_rows(items)
        noul = [it for it in items if it.question.type == "noul"]
        yes = sum(balance.is_yes(it) for it in noul)
        restored[name] = {"rows_rebalanced": before, "rows": len(items),
                          "dropped_by_rebalance": undone, "restored": len(items) - before,
                          "yes_share": round(yes / len(noul), 4) if noul else None,
                          "mix_v7_rows_kept_as_is": same, "exact": info[name]["exact"]}  # fmt: skip
        print(f"[assemble] {name}: {before} → {len(items)} rows (+{len(items) - before} of "
              f"{undone} dropped), yes-share {restored[name]['yes_share']}, mix-v7's rows "
              f"as is: {same}", flush=True)  # fmt: skip
    man["rebalance_noul_undone"] = {"from": str(v7), "sources": restored,
                                    "was": man.pop("rebalance_noul", None)}  # fmt: skip
    (out / "manifest.json").unlink()
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n",
                                       encoding="utf-8")  # fmt: skip
    return 0 if ok else 1


def finalize(v75: Path, v7: Path, v7np: Path, pre: Path, np_: Path) -> int:
    """mix-v7.5-np after its derive: the sources the restored rows do not touch but whose
    pre file differs from mix-v7.5-pre's (upstream drift since the build, e.g. compar:IA,
    and jev_distractor, derived from the restored base rows) get mix-v7.5's train file as
    is (their mixture weight unchanged: ``weight_rows`` kept); every held-out file is
    mix-v7.5's, byte for byte. The rebalanced sources' kept rows are checked against
    mix-v7.5-pre's hash."""
    from tjev.data.mix import write_mixture

    ma = json.loads((v75 / "manifest.json").read_text(encoding="utf-8"))
    want = {Path(k).stem: h for k, h in ma["derived_from"]["files"].items()}
    man = json.loads((np_ / "manifest.json").read_text(encoding="utf-8"))
    targets = (*V6_REBALANCED, *V7_REBALANCED)
    report: dict = {"kept_rows_equal_v75_pre": {}, "frozen_from_v75": {}}
    for name in targets:  # the rows mix-v7 kept, as mix-v7.5-pre has them
        built = {json.loads(x)["id"] for x in lines_of(v7 / "train" / f"{name}.jsonl")}
        lines = [
            x for x in lines_of(pre / "train" / f"{name}.jsonl") if json.loads(x)["id"] in built
        ]
        h = hashlib.sha256("".join(x + "\n" for x in lines).encode()).hexdigest()
        report["kept_rows_equal_v75_pre"][name] = h == want.get(name)
        if h != want.get(name):  # today's cleaning differs (scruples: &#x200B; unescaped):
            # the rows mix-v7.5 trains on take its lines, byte for byte
            v75_lines = {json.loads(x)["id"]: x for x in lines_of(v75 / "train" / f"{name}.jsonl")}
            path = np_ / "train" / f"{name}.jsonl"
            cur = lines_of(path)
            new = [v75_lines.get(json.loads(x)["id"], x) for x in cur]
            report.setdefault("aligned_to_v75_lines", {})[name] = sum(
                a != b for a, b in zip(cur, new, strict=True)
            )
            path.write_text("".join(x + "\n" for x in new), encoding="utf-8")
    for name, h in sorted(want.items()):
        p = pre / "train" / f"{name}.jsonl"
        if name in targets or (p.exists() and sha(p) == h):
            continue
        entry = man["sources"][name]
        entry.setdefault("weight_rows", entry["train"])
        before = entry["train"]
        shutil.copyfile(v75 / "train" / f"{name}.jsonl", np_ / "train" / f"{name}.jsonl")
        its = list(read_jsonl(np_ / "train" / f"{name}.jsonl"))
        entry["train"] = len(its)
        if isinstance(entry.get("lang"), dict):
            entry["lang"] = dict(Counter(it.lang for it in its))
        if "license_use_rows" in entry:
            entry["license_use_rows"] = license_rows(its)
        entry["np_frozen_from_v75"] = True
        report["frozen_from_v75"][name] = {"np_rows": before, "v75_rows": len(its)}
    for rel in held_files(v75):  # held-out: mix-v7.5's, byte for byte
        (np_ / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(v75 / rel, np_ / rel)
    for rel in set(held_files(np_)) - set(held_files(v75)):
        (np_ / rel).unlink()
    man["np"] = report
    present = {n: s["block"] for n, s in man["sources"].items() if s.get("train", 0) > 0}
    old = (np_ / "mix.yaml").read_text()
    write_mixture(np_, man, present)
    print(json.dumps(report, indent=1))
    print("mix.yaml unchanged by the swap:", old == (np_ / "mix.yaml").read_text())
    # no train state of the arm equals a held-out state (whitespace / case normalised)
    norm = lambda t: " ".join(str(t).split()).lower()  # noqa: E731
    held = {norm(it.state) for rel in held_files(v75) for it in read_jsonl(v75 / rel)}
    hits = Counter(it.source for p in sorted((np_ / "train").glob("*.jsonl"))
                   for it in read_jsonl(p) if norm(it.state) in held)  # fmt: skip
    hits75 = Counter(it.source for p in sorted((v75 / "train").glob("*.jsonl"))
                     for it in read_jsonl(p) if norm(it.state) in held)  # fmt: skip
    print("train states equal to a held-out state: np", dict(hits), "| v7.5", dict(hits75))
    return 0


def held_files(mix: Path) -> dict[str, str]:
    paths = [*mix.glob("*.jsonl"), *(mix / "test").glob("*.jsonl")]
    return {str(p.relative_to(mix)): sha(p) for p in sorted(paths)}


def heldout(a: Path, b: Path) -> int:
    ha, hb = held_files(a), held_files(b)
    same = [k for k in ha if hb.get(k) == ha[k]]
    diff = sorted(k for k in set(ha) | set(hb) if ha.get(k) != hb.get(k))
    print(f"[heldout] {len(same)}/{len(ha)} files byte-identical ({a.name} vs {b.name})")
    for k in ("validation.jsonl", "calibration.jsonl", "heldout-select.jsonl",
              "validation-v6.jsonl", "calibration-v6.jsonl", "validation-v7.jsonl",
              "calibration-v7.jsonl"):  # fmt: skip
        print(
            f"  {k}: {ha.get(k, '-')[:16]} {hb.get(k, '-')[:16]} "
            f"{'same' if ha.get(k) == hb.get(k) else 'DIFFERENT'}"
        )
    if diff:
        print(f"  DIFFERENT: {diff}")
    return 0 if not diff else 1


def yes_shares(mix: Path) -> dict[str, tuple[int, int]]:
    res = {}
    for path in sorted((mix / "train").glob("*.jsonl")):
        n = y = 0
        for it in read_jsonl(path):
            if it.question.type == "noul":
                n += 1
                y += balance.is_yes(it)
        if n:
            res[path.stem] = (y, n)
    return res


def report(v75: Path, np_: Path) -> int:
    ma = json.loads((v75 / "manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((np_ / "manifest.json").read_text(encoding="utf-8"))
    ya, yb = yes_shares(v75), yes_shares(np_)
    print("## yes-share of the noul train rows (final train files)\n")
    print("| source | tier | v7.5 noul rows | v7.5 yes | np noul rows | np yes | rebalanced |")
    print("|---|---|---|---|---|---|---|")
    reb = set(V6_REBALANCED) | set(V7_REBALANCED)
    tiers = mb.get("tiers", {})
    tot = {"a": [0, 0], "b": [0, 0]}
    for name in sorted(set(ya) | set(yb), key=lambda n: (n not in reb, n)):
        a, b = ya.get(name, (0, 0)), yb.get(name, (0, 0))
        tot["a"][0] += a[0]
        tot["a"][1] += a[1]
        tot["b"][0] += b[0]
        tot["b"][1] += b[1]
        fa = f"{a[0] / a[1]:.1%}" if a[1] else "-"
        fb = f"{b[0] / b[1]:.1%}" if b[1] else "-"
        print(
            f"| {name} | {tiers.get(name, '')} | {a[1]} | {fa} | {b[1]} | {fb} | "
            f"{'yes' if name in reb else ''} |"
        )
    print(
        f"| **all** | | {tot['a'][1]} | {tot['a'][0] / tot['a'][1]:.1%} | {tot['b'][1]} | "
        f"{tot['b'][0] / tot['b'][1]:.1%} | |\n"
    )
    ea, eb = ma["epoch_cap"], mb["epoch_cap"]
    print("## epochs at 5,000 / 2,500 steps (best_fit simulation) and sampled share\n")
    print(
        "| source | tier | v7.5 rows | np rows | v7.5 share % | np share % | v7.5 ep@5000 | "
        "np ep@5000 | v7.5 ep@2500 | np ep@2500 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    wa = ma.get("mixture", {})
    wb = mb.get("mixture", {})

    def at(e, n, k):
        v = e.get("at_steps", {}).get(str(k), {})
        v = v.get("epochs", v) if isinstance(v, dict) else {}
        return v.get(n)

    for name in sorted(eb["epochs"], key=lambda n: (n not in reb, tiers.get(n, ""), n)):
        ra = ma["sources"].get(name, {}).get("train")
        rb = mb["sources"].get(name, {}).get("train")
        sa, sb = wa.get(name), wb.get(name)
        cells = [ea["epochs"].get(name), eb["epochs"].get(name), at(ea, name, 2500),
                 at(eb, name, 2500)]  # fmt: skip
        fmt = [f"{c:.2f}" if isinstance(c, (int, float)) else "-" for c in cells]
        pa = f"{100 * sa:.2f}" if isinstance(sa, (int, float)) else "-"
        pb = f"{100 * sb:.2f}" if isinstance(sb, (int, float)) else "-"
        print(
            f"| {name} | {tiers.get(name, '')} | {ra} | {rb} | {pa} | {pb} | "
            + " | ".join(fmt)
            + " |"
        )
    print(
        f"\nunique train rows: v7.5 {sum(s.get('train', 0) for s in ma['sources'].values())}, "
        f"np {sum(s.get('train', 0) for s in mb['sources'].values())}"
    )
    print(
        f"block shares: v7.5 {ea.get('block_shares')}\n              np   {eb.get('block_shares')}"
    )
    print(f"French sampled: v7.5 {ma.get('fr_sampled_share')}, np {mb.get('fr_sampled_share')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recover")
    r.add_argument("unreb", type=Path)
    r.add_argument("reb", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--sources", required=True)
    a = sub.add_parser("assemble")
    for k in ("v7", "v6_dir", "v7_dir", "out"):
        a.add_argument(k, type=Path)
    h = sub.add_parser("heldout")
    h.add_argument("a", type=Path)
    h.add_argument("b", type=Path)
    f = sub.add_parser("finalize")
    for k in ("v75", "v7", "v7np", "pre", "np_"):
        f.add_argument(k, type=Path)
    p = sub.add_parser("report")
    p.add_argument("v75", type=Path)
    p.add_argument("np_", type=Path)
    args = ap.parse_args()
    if args.cmd == "recover":
        return recover(args.unreb, args.reb, args.out, args.sources.split(","))
    if args.cmd == "assemble":
        return assemble(args.v7, args.v6_dir, args.v7_dir, args.out)
    if args.cmd == "finalize":
        return finalize(args.v75, args.v7, args.v7np, args.pre, args.np_)
    if args.cmd == "heldout":
        return heldout(args.a, args.b)
    return report(args.v75, args.np_)


if __name__ == "__main__":
    sys.exit(main())
