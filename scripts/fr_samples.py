"""Write docs/fr-samples.md: French generator rows (training and held-out templates) and
translated contender rows (jev_fr_pilot), for a native speaker to review.

    uv run python scripts/fr_samples.py            # -> docs/fr-samples.md
"""

from __future__ import annotations

import json
from pathlib import Path

from tjev.data.sources.contenders import PILOT_FILE
from tjev.data.sources.generators import GENERATORS, generate

YES_NO = {"yes": "oui", "no": "non"}
HELD_OUT = (
    "gen_return_window", "gen_weekday", "gen_invoice_total", "gen_schedule_conflict",
    "gen_refund_policy", "gen_base_rate", "gen_table_argmax", "gen_ticket_triage",
    "gen_register",
)  # fmt: skip


def block(row: dict, i: int, tag: str) -> str:
    q = row["question"]
    crit = q["criteria"]
    if isinstance(crit, dict):
        opts = "; ".join(
            v if (v == k or k in ("false", "true")) else f"`{k}` {v}" for k, v in crit.items()
        )
    else:
        opts = " < ".join(f"{j}: {c}" for j, c in enumerate(crit))
    gold = row.get("expected")
    if gold is None:
        top = max(row["target"], key=row["target"].get)
        gold = (
            f"probabilité de « oui » {row['target']['yes']:.2f}"
            if "yes" in row["target"]
            else f"{top} ({row['target'][top]:.2f})"
        )
    else:
        gold = YES_NO.get(gold, gold)
    state = row["state"]
    if len(state) > 900:
        state = state[:420] + " […] " + state[-380:]
    quoted = "> " + state.replace("\n", "  \n> ")
    return (
        f"### {i}. {row['source']} ({tag})\n\n{quoted}\n\n- **Question :** {q['instructions']}\n"
        f"- **Options :** {opts}\n- **Réponse attendue :** {gold}\n"
    )


def main() -> None:
    parts = [
        "# French samples for review (mix-v5)\n",
        (
            "French rows for a native speaker to review: please flag anything unnatural, wrong or "
            "ambiguous (the expected answer must hold under any reasonable reading).\n"
        ),
        (
            "- **1–21: generator training templates**, two per generator (one for the long "
            "policy).\n- **22–30: held-out templates**, which training never sees.\n"
            "- **31–40: translated contender rows** (`jev_fr_pilot`).\n"
        ),
        (
            "Generator labels are computed by code; a template only changes the wording "
            "(`tests/data/test_generators.py::test_a_template_never_changes_the_label`).\n"
        ),
    ]
    i = 1
    for name in GENERATORS:
        seen, picks = set(), []
        for r in generate(name, 300, 11, fr_share=1.0):
            if r["question"]["instructions"] in seen:
                continue
            seen.add(r["question"]["instructions"])
            picks.append(r)
            if len(picks) == (1 if name == "gen_long_refund_policy" else 2):
                break
        for r in picks:
            parts.append(block(r, i, "entraînement"))
            i += 1
    for name in HELD_OUT:
        rows = generate(name, 40, 21, split="heldout", fr_share=1.0)
        other = [
            r for r in rows if r["question"]["instructions"] != rows[0]["question"]["instructions"]
        ]
        parts.append(block((other or rows)[0], i, "held-out"))
        i += 1
    if PILOT_FILE.exists():
        pilot = [json.loads(x) for x in PILOT_FILE.read_text(encoding="utf-8").splitlines()]
        by_origin: dict[str, list[dict]] = {}
        for r in pilot:
            by_origin.setdefault(r["origin"].split("-")[0], []).append(r)
        picked = [r for rows in by_origin.values() for r in rows[:: max(1, len(rows) // 3)][:3]]
        for r in picked[:10]:
            parts.append(block(r, i, f"traduction de {r['origin']}"))
            i += 1
    Path("docs/fr-samples.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"{i - 1} rows -> docs/fr-samples.md")


if __name__ == "__main__":
    main()
