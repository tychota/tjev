"""mix-v7 generators: deterministic, valid, labels recomputed from the text, the
evidence-ablation property, pair integrity across splits, held-out template families,
exact unknowable / posterior targets, distractor rows, diversity against v6."""

import calendar
import datetime as dt
import hashlib
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from math import comb

import pytest

from tjev.data.item import parse_item
from tjev.data.sources.gen_v7 import (
    SOLVERS,
    V7_GENERATORS,
    drop_orphans,
    generate_v7,
    group_of,
    key_of,
)
from tjev.data.sources.gen_v7 import contrastive as ctr
from tjev.data.sources.gen_v7 import distractor as dst
from tjev.data.sources.gen_v7.core import ABSTAIN, Line, Member, edit_words, leaked
from tjev.data.sources.generators import GENERATORS, generate

CODE = [n for n, s in V7_GENERATORS.items() if s.licence[0].startswith("Apache-2.0 (generated")]
PAIRED = [n for n in CODE if n.endswith("_pairs")]
N = 120


def _rows(name, n=N, seed=1, split="train", fr=0.5):
    return generate_v7(name, n, seed, split=split, fr_share=fr)


def _member(row) -> Member:
    lines = [Line(t, tag, tuple(f), tuple(s), pre) for tag, t, f, s, pre in row["meta"]["lines"]]
    return Member(lines, row["meta"]["facts"])


@pytest.mark.parametrize("name", CODE)
def test_valid_deterministic_bilingual(name):
    for split in ("train", "heldout"):
        a = _rows(name, split=split)
        assert a == _rows(name, split=split), name
        assert len(a) >= N - 6
        for r in a:
            parse_item(r)
            assert r["meta"]["license_use"]
        assert {r["lang"] for r in a} == {"en", "fr"}
        assert len({key_of(r) for r in a}) == len(a)  # no repeated (state, instructions)
        assert len({r["id"] for r in a}) == len(a)
    assert generate(name, 30, 4) == generate_v7(name, 30, 4)  # generators.generate dispatches


# ---------------------------------------------------------------------------------------
# Labels recomputed from the text (independent parsers)

MONTHS_EN = {m: i for i, m in enumerate(calendar.month_name) if m}
MONTHS_FR = {m: i + 1 for i, m in enumerate(["janvier", "février", "mars", "avril", "mai", "juin",
             "juillet", "août", "septembre", "octobre", "novembre", "décembre"])}  # fmt: skip
DATE_RES = [
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})"), lambda m: dt.date(*map(int, m.groups()))),
    (re.compile(r"\b(\d{2})/(\d{2})/(\d{4})"),
     lambda m: dt.date(int(m[3]), int(m[2]), int(m[1]))),
    (re.compile(r"(" + "|".join(MONTHS_EN) + r") (\d{1,2}), (\d{4})"),
     lambda m: dt.date(int(m[3]), MONTHS_EN[m[1]], int(m[2]))),
    (re.compile(r"\b(\d{1,2})(?:er)? (" + "|".join([*MONTHS_EN, *MONTHS_FR]) + r") (\d{4})"),
     lambda m: dt.date(int(m[3]), {**MONTHS_EN, **MONTHS_FR}[m[2]], int(m[1]))),
]  # fmt: skip


def dates_in(text):
    found = []
    for rx, conv in DATE_RES:
        for m in rx.finditer(text):
            if not any(s <= m.start() < e for s, e, _ in found):
                found.append((m.start(), m.end(), conv(m)))
    return [d for _, _, d in sorted(found)]


def money_in(text):
    """Amounts as cents: "$1,234.50" / "1 234,50 €"."""
    out = [
        int(m[1].replace(",", "")) * 100 + int(m[2])
        for m in re.finditer(r"\$([\d,]+)\.(\d{2})", text)
    ]
    out += [int(m[1].replace(" ", "").replace(" ", "")) * 100 + int(m[2])
            for m in re.finditer(r"(\d{1,3}(?:[  ]\d{3})*),(\d{2}) €", text)]  # fmt: skip
    return out


def ints_in(text):
    return [int(x) for x in re.findall(r"(?<![\d.,:/-])\d+(?![\d.,:/]\d)", text)]


def line(row, tag):
    return next(t for g, t, *_ in row["meta"]["lines"] if g == tag)


def recompute(row):
    base = row["meta"]["base"]
    if base == "rw":
        w = ints_in(line(row, "rule"))[0]
        dl = line(row, "dates")
        ds = dates_in(dl)
        e = (ds[1] - ds[0]).days if len(ds) == 2 else ints_in(dl)[0]
        return "yes" if e <= w else "no"
    if base == "wd":
        d0 = dates_in(line(row, "start"))[0]
        o = ints_in(line(row, "offset"))[0]
        names = calendar.day_name if row["lang"] == "en" else ctr.WEEKDAYS["fr"]
        return list(names)[(d0 + dt.timedelta(days=o)).weekday()]
    if base == "inv":
        order = line(row, "order")
        budget = money_in(line(row, "rule"))[0]
        qty = [int(m) for m in re.findall(r"(?:^|: |contient |holds |, | and | et )(\d+) ", order)]
        prices = money_in(order)
        assert len(qty) == len(prices), order
        disc = re.search(r"(\d+) ?% ", order)
        d = int(disc[1]) if disc else 0
        gross = sum(q * p for q, p in zip(qty, prices, strict=True))
        net = int((Decimal(gross) * (100 - d) / 100).quantize(Decimal(1), ROUND_HALF_UP))
        ok = net <= budget if row["meta"]["facts"]["dir"] == "le" else net >= budget
        return "yes" if ok else "no"
    if base == "rp":
        tid = row["meta"]["template"].split(".")
        fam = ctr.RP[tid[1], tid[2]][int(tid[3])]
        w = ints_in(line(row, "rule"))[0]
        n = ints_in(line(row, "days"))[0]
        bonus = ints_in(line(row, "member_rule"))[0]
        limit = money_in(line(row, "limit_rule"))[0]
        amount = money_in(line(row, "amount"))[0]
        member = line(row, "membership") == fam["is_member"][0]
        if n > w + (bonus if member else 0):
            return "deny"
        return "escalate" if amount > limit else "approve"
    if base == "sc":
        return _sc(row)
    raise KeyError(base)


def _minutes(text):
    out = []
    for m in re.finditer(r"\b(\d{1,2}):(\d{2})(?: (am|pm))?|\b(\d{1,2}) ?h ?(\d{2})|\b(\d{2})(\d{2})\b",
                         text):  # fmt: skip
        if m[1]:
            h, mm = int(m[1]), int(m[2])
            if m[3] == "pm" and h != 12:
                h += 12
            if m[3] == "am" and h == 12:
                h = 0
        elif m[4]:
            h, mm = int(m[4]), int(m[5])
        else:
            h, mm = int(m[6]), int(m[7])
        out.append(h * 60 + mm)
    return out


def _sc(row):
    ev = _minutes(line(row, "event"))
    pl = line(row, "proposal")
    pt = _minutes(re.sub(r"\d+[- ]minute", "", pl))
    if len(pt) == 1:
        pt.append(pt[0] + int(re.search(r"(\d+)[- ]minute", pl)[1]))
    return "yes" if ev[0] < pt[1] and pt[0] < ev[1] else "no"


@pytest.mark.parametrize("name", ["gen7_return_window_pairs", "gen7_weekday_pairs",
                                  "gen7_invoice_budget_pairs", "gen7_refund_pairs",
                                  "gen7_schedule_pairs"])  # fmt: skip
def test_labels_recomputed_from_the_text(name):
    for split in ("train", "heldout"):
        for r in _rows(name, 300, 2, split):
            m = _member(r)
            assert m.state(sep="\n") == r["state"]  # the tagged lines are the state
            assert recompute(r) == r["expected"], (r["state"], r["expected"])


def test_long_refund_labels_and_one_window_sentence():
    for split in ("train", "heldout"):
        for r in _rows("gen7_long_refund_pairs", 40, 2, split):
            f = r["meta"]["facts"]
            allowed = f["w"] + (f["bonus"] if f["member"] else 0)
            want = (
                "deny"
                if f["n"] > allowed
                else "escalate"
                if f["amount"] > f["limit"]
                else "approve"
            )
            assert r["expected"] == want
            rule = line(r, "rule")
            assert r["state"].count(rule) == 1
            assert len(r["state"]) > 8000


# ---------------------------------------------------------------------------------------
# Evidence ablation, minimal edits, pairs


@pytest.mark.parametrize("name", PAIRED)
def test_evidence_ablation_makes_the_answer_unknowable(name):
    for split in ("train", "heldout"):
        for r in _rows(name, 200, 3, split):
            m, base = _member(r), r["meta"]["base"]
            assert SOLVERS[base](m.known()) == r["expected"]
            assert len(r["meta"]["evidence"]) == 2
            for tag in r["meta"]["evidence"]:
                assert sum(t == tag for t, *_ in r["meta"]["lines"]) == 1
                assert SOLVERS[base](m.known(drop=tag)) is None, (name, tag)
                assert not leaked(m, tag, r["question"]), (name, tag)


@pytest.mark.parametrize("name", PAIRED)
def test_pairs_flip_with_a_minimal_edit_and_share_their_split(name):
    for split in ("train", "heldout"):
        rows = _rows(name, 200, 4, split)
        by_group = {}
        for r in rows:
            assert f"-{split}-" in r["id"]
            by_group.setdefault(group_of(r["id"]), {})[r["meta"]["role"]] = r
        for members in by_group.values():
            assert set(members) == {"a", "b"}  # both members, same call, same split
            a, b = members["a"], members["b"]
            assert a["question"] == b["question"] and a["expected"] != b["expected"]
            old, new = b["meta"]["edit"]
            assert edit_words(old, new) <= 8, (old, new)
            assert a["state"].replace(old, new, 1) == b["state"]
            assert a["meta"]["template"] == b["meta"]["template"]
    train = {group_of(r["id"]) for r in _rows(name, 200, 0)}
    held = {group_of(r["id"]) for r in _rows(name, 200, 101, "heldout")}
    assert not train & held


def test_drop_orphans_keeps_whole_pairs():
    rows = _rows("gen7_return_window_pairs", 40, 5)
    kept = drop_orphans(rows[1:])  # the first group lost its "a"
    assert len(kept) == len(rows) - 2
    assert all(group_of(r["id"]) != group_of(rows[0]["id"]) for r in kept)


# ---------------------------------------------------------------------------------------
# Held-out template families


def _template_lines(row):
    q = row["question"]
    parts = [*re.split(r"(?<=[.!?])\s+|\n", row["state"]), q["instructions"]]
    if isinstance(q["criteria"], dict) and q["type"] != "noul":
        parts += [v for k, v in q["criteria"].items() if v != k]
    out = set()
    for p in parts:
        text = re.sub(r"\d+", "#", re.sub(r"\W+", " ", p.lower()).strip())
        if len(re.findall(r"[^\W\d_]{3,}", text)) >= 2:
            out.add(text)
    return out


@pytest.mark.parametrize("name", CODE)
def test_no_template_leakage_between_train_and_heldout_families(name):
    train = _rows(name, 400, 1)
    held = _rows(name, 150, 2, "heldout")
    t_ids = {r["meta"]["template"] for r in train}
    h_ids = {r["meta"]["template"] for r in held}
    assert not t_ids & h_ids
    if name != "gen7_distractor_opinion":
        assert all(".heldout." in t for t in h_ids) and all(".train." in t for t in t_ids)
    assert not {key_of(r) for r in train} & {key_of(r) for r in held}
    seen = set().union(*(_template_lines(r) for r in train))
    v6 = set().union(*(_template_lines(r) for g in GENERATORS for r in generate(g, 60, 1)))
    shared = {x for r in held for x in _template_lines(r) if x in seen or x in v6}
    assert not shared, sorted(shared)[:5]


def test_families_have_enough_templates():
    for table in (ctr.RW, ctr.INV, ctr.SC, ctr.WD, ctr.RP):
        for lang in ("en", "fr"):
            assert len(table["train", lang]) >= 5 and len(table["heldout", lang]) >= 2


# ---------------------------------------------------------------------------------------
# Unknowable twins


def test_unknowable_twins_drop_one_evidence_line_with_the_right_target():
    for split in ("train", "heldout"):
        for r in _rows("gen7_unknowable", 400, 6, split):
            m, drop = _member(r), r["meta"]["dropped"]
            assert drop in r["meta"]["evidence"]
            assert m.state(drop=drop, sep="\n") == r["state"]
            assert SOLVERS[r["meta"]["base"]](m.known(drop=drop)) is None
            if r["question"]["type"] == "choice":
                assert r["expected"] in ABSTAIN and r["expected"] in r["question"]["criteria"]
            else:
                p = r["target"]["yes"]
                assert 0 < p < 1
                if r["meta"]["base"] == "rw":
                    assert p == pytest.approx(_rw_base_rate_by_enumeration(split))
                else:
                    assert p == 0.5  # a fair coin decides the label of a


def _rw_base_rate_by_enumeration(split):
    """P(yes) by enumerating the sampler: window prior × fair coin × offset, rejecting
    elapsed < 1."""
    if split == "train":
        prior = {}
        for w in ctr.TRAIN_WINDOWS:
            prior[w] = prior.get(w, 0) + 0.6 / len(ctr.TRAIN_WINDOWS)
        for w in ctr.COMMON_WINDOWS:
            prior[w] = prior.get(w, 0) + 0.4 / len(ctr.COMMON_WINDOWS)
    else:
        prior = dict.fromkeys(ctr.HELDOUT_WINDOWS, 0.25)
    joint = {"yes": 0.0, "no": 0.0}
    for w, pw in prior.items():
        for label, offsets in (("yes", range(-10, 1)), ("no", range(1, 12))):
            for d in offsets:
                if w + d >= 1:
                    joint[label] += pw * 0.5 / len(offsets)
    return joint["yes"] / (joint["yes"] + joint["no"])


def test_rw_base_rate_matches_the_sampler_empirically():
    rows = [r for r in _rows("gen7_return_window_pairs", 8000, 7) if r["meta"]["role"] == "a"]
    share = sum(r["expected"] == "yes" for r in rows) / len(rows)
    assert abs(share - ctr.rw_base_rate("train")) < 0.03


# ---------------------------------------------------------------------------------------
# Intermediate quantities


def test_aux_questions_ask_an_exact_quantity_of_the_same_state():
    for r in _rows("gen7_aux_quantities", 400, 8):
        f, q = r["meta"]["facts"], r["meta"]["quantity"]
        assert r["expected"] in r["question"]["criteria"]
        if q == "elapsed":
            want = str((ctr._d(f["t"]) - ctr._d(f["b"])).days)
        elif q == "last_day":
            want = (ctr._d(f["b"]) + dt.timedelta(days=f["w"])).isoformat()
        elif q == "date":
            want = (ctr._d(f["d"]) + dt.timedelta(days=f["o"])).isoformat()
        elif q == "overlap":
            (a, b), (c, d) = f["E"], f["P"]
            want = str(max(0, min(b, d) - max(a, c)))
        elif q == "refund_period":
            want = str(f["w"] + (f["bonus"] if f["member"] else 0))
        else:
            assert q == "total"
            continue
        assert r["expected"] == want, (q, r["state"])
        assert _member(r).state(sep="\n") == r["state"]


# ---------------------------------------------------------------------------------------
# Exact posteriors


def test_posterior_targets_are_exact():
    from scipy.stats import hypergeom

    rows = _rows("gen7_posterior", 400, 9)
    band = 0
    for r in rows:
        f, t, base = r["meta"]["facts"], r["target"], r["meta"]["base"]
        if base == "pw_signals":
            lp, ln = f["prior"], 1 - f["prior"]
            for s, fp, res in zip(f["sens"], f["fpr"], f["res"], strict=True):
                lp, ln = lp * (s if res else 1 - s), ln * (fp if res else 1 - fp)
            want = lp / (lp + ln)
        elif base == "pw_sampling":
            want = hypergeom(f["N"], f["K"], f["m"]).sf(f["r"] - 1)
            assert str(f["N"]) in r["state"] and str(f["K"]) in r["state"]
            assert want == pytest.approx(
                sum(comb(f["K"], k) * comb(f["N"] - f["K"], f["m"] - k)
                    for k in range(f["r"], f["m"] + 1)) / comb(f["N"], f["m"]))  # fmt: skip
        elif base == "pw_table":
            c = f["counts"][f["pick"]]
            assert t == pytest.approx({o: x / sum(c) for o, x in zip(f["outs"], c, strict=True)})
            continue
        elif base == "pw_delay":
            want = sum(s for v, s in zip(f["values"], f["shares"], strict=True)
                       if v <= f["slack"]) / 100  # fmt: skip
            assert sum(f["shares"]) == 100
        else:
            want = f["p_cond"] if f["n"] < f["w"] else 0.0
        assert t["yes"] == pytest.approx(want)
        band += 0.2 < t["yes"] < 0.8
    assert band / len(rows) > 0.3  # the noul band is well covered


# ---------------------------------------------------------------------------------------
# Distractor opinions


def test_distractor_rows_append_one_opinion_and_keep_the_label():
    for split in ("train", "heldout"):
        for r in _rows("gen7_distractor_opinion", 200, 10, split):
            f = r["meta"]["facts"]
            assert f["line"] in r["state"]
            base = f["base_state"]
            assert r["state"].replace(f["line"] + "\n", "", 1).replace("\n" + f["line"], "", 1) \
                == base  # fmt: skip
            assert f["kind"] != "irrelevant" or not re.search(r"\d", f["line"])


def test_distractor_from_real_rows_keeps_targets_and_licence():
    rows = [
        {"id": "certo-1", "source": "certo", "lang": "en", "state": "Rule: A. Fact: B.",
         "question": {"type": "noul", "instructions": "Ok?", "criteria": {"false": "No", "true": "Yes"}},
         "expected": "yes"},
        {"id": "eikos-2", "source": "eikos", "lang": "en", "state": {"k": "v"},
         "question": {"type": "choice", "instructions": "Which?", "criteria": {"x": "X", "y": "Y"}},
         "target": {"x": 0.3, "y": 0.7}},
        {"id": "go-3", "source": "go_emotions", "lang": "en", "state": "text",
         "question": {"type": "noul", "instructions": "?", "criteria": {"false": "No", "true": "Yes"}},
         "expected": "no"},
    ]  # fmt: skip
    out = dst.from_rows(rows, 10, 0)
    assert len(out) == 2  # text-analysis sources are never used
    for r in out:
        parse_item(r)
        base = next(x for x in rows if x["id"] == r["meta"]["base_id"])
        assert r.get("expected") == base.get("expected") and r.get("target") == base.get("target")
        assert r["meta"]["line"] in r["state"] and r["meta"]["license_use"]


# ---------------------------------------------------------------------------------------
# Diversity and pinned output


def test_v7_is_more_diverse_than_v6_on_average():
    from tjev.data.sources.gen_v7.diversity import compare

    rows = compare(300)
    v7 = sum(r["v7_distinct3"] for r in rows) / len(rows)
    v6 = sum(r["v6_distinct3"] for r in rows) / len(rows)
    assert v7 > v6, rows


# Train-split output is training data: a change must be deliberate (update the hash with a
# mix rebuild).
TRAIN_HASHES: dict[str, str] = {
    "gen7_return_window_pairs": "21b20cf002c7a186",
    "gen7_invoice_budget_pairs": "40285d80fd4c5f23",
    "gen7_schedule_pairs": "133b3b3f4a254585",
    "gen7_weekday_pairs": "95552dd963c5cd52",
    "gen7_refund_pairs": "118064abfa22a128",
    "gen7_long_refund_pairs": "d26b55d2848ad15a",
    "gen7_unknowable": "9d14bba45489960b",
    "gen7_aux_quantities": "cbf24ec96390c0c8",
    "gen7_posterior": "fe9b41ee3f69cc09",
    "gen7_distractor_opinion": "f76cc26fb9e5a34f",
}


def test_every_code_source_is_pinned():
    assert set(TRAIN_HASHES) == set(CODE)


@pytest.mark.parametrize("name", sorted(TRAIN_HASHES))
def test_train_output_is_pinned(name):
    rows = generate_v7(name, 20, 0, fr_share=0.5)
    blob = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
    assert hashlib.sha256(blob).hexdigest()[:16] == TRAIN_HASHES[name]
