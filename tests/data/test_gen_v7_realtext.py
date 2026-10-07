"""Real-text seeded v7 generators, on small fixtures (tests/data/fixtures/gen_v7: tau-bench
policies, a few ShARC snippets, KALI and COLD seed excerpts, each with its attribution):
deterministic, labels recomputed independently from the real text, evidence ablation,
pairs kept whole, held-out documents / rules disjoint from training, licence per row."""

import calendar
import datetime as dt
import re
from pathlib import Path

import pytest

from tjev.data.item import parse_item

FIXTURES = Path(__file__).parent / "fixtures" / "gen_v7"
SOURCES = ["gen7_rt_tau_pairs", "gen7_rt_sharc_pairs", "gen7_rt_kali_fr_pairs",
           "gen7_rt_cold_fr_pairs"]  # fmt: skip
LICENCE = {"gen7_rt_tau_pairs": "MIT", "gen7_rt_sharc_pairs": "CC BY-SA 3.0",
           "gen7_rt_kali_fr_pairs": "ODC-By", "gen7_rt_cold_fr_pairs": "CC BY 4.0"}  # fmt: skip


@pytest.fixture(autouse=True)
def fixture_seeds(monkeypatch):
    monkeypatch.setenv("TJEV_GEN7_SEEDS", str(FIXTURES))


def _gen(name, n=60, seed=1, split="train"):
    from tjev.data.sources.gen_v7 import generate_v7

    return generate_v7(name, n, seed, split=split)


@pytest.mark.parametrize("name", SOURCES)
def test_valid_deterministic_and_licensed(name):
    from tjev.data.sources.gen_v7 import V7_GENERATORS, source_info

    for split in ("train", "heldout"):
        rows = _gen(name, split=split)
        assert rows and rows == _gen(name, split=split), (name, split)
        for r in rows:
            parse_item(r)
            assert r["meta"]["license_use"].startswith(LICENCE[name]), r["meta"]["license_use"]
    assert V7_GENERATORS[name].licence[0].lower().startswith(LICENCE[name].lower()[:6])
    assert source_info(name)["commercial_ok"]


@pytest.mark.parametrize("name", SOURCES)
def test_groups_pass_the_contrastive_checks(name):
    """Every group: the rule and the case are both needed (the solver gives None and no
    surface of the dropped line remains), and b's one-line edit (≤ 8 words) flips it."""
    from tjev.data.sources.gen_v7 import SOLVERS, groups
    from tjev.data.sources.gen_v7.core import check_group

    for split in ("train", "heldout"):
        it = groups(name, 3, split=split)
        for _ in range(40):
            g, rows = next(it)
            check_group(g, SOLVERS[g.base])
            unk = next(r for r in rows if r["meta"]["role"] == "unk")
            dropped = next(t for _, t, *_ in [x for x in unk["meta"]["lines"]
                                              if x[0] == unk["meta"]["dropped"]])  # fmt: skip
            assert dropped not in unk["state"]
            if unk["question"]["type"] == "noul":
                assert unk["target"] == {"yes": 0.5, "no": 0.5}
            else:
                key = "insufficient_information" if split == "train" else "cannot_determine"
                assert unk["expected"] == key and key in unk["question"]["criteria"]


@pytest.mark.parametrize("name", SOURCES)
def test_pairs_stay_whole(name):
    from tjev.data.sources.gen_v7 import drop_orphans, group_of

    for split in ("train", "heldout"):
        rows = _gen(name, 61, split=split)
        roles: dict[str, set] = {}
        for r in rows:
            roles.setdefault(group_of(r["id"]), set()).add(r["meta"]["role"])
        assert all(v == {"a", "b", "unk"} for v in roles.values()), roles
        assert drop_orphans(rows) == rows
        assert len(drop_orphans(rows[1:])) == len(rows) - 2  # a's partner b goes with it


def _shingles(text):
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i : i + 3]) for i in range(len(w) - 2)}


@pytest.mark.parametrize("name", SOURCES)
def test_heldout_documents_rules_and_templates_are_disjoint(name):
    train = _gen(name, 150)
    held = _gen(name, 60, seed=2, split="heldout")
    assert not {r["meta"]["template"] for r in train} & {r["meta"]["template"] for r in held}

    def rules(rows):
        return {t for r in rows for tag, t, *_ in r["meta"]["lines"] if tag == "rule"}

    assert not rules(train) & rules(held), name
    assert (
        not {r["question"]["instructions"] for r in train}
        & {r["question"]["instructions"] for r in held}
        or name == "gen7_rt_tau_pairs"
    )  # tau: held out by rule (other questions anyway)
    tsh = [_shingles(r["state"]) for r in train]
    for r in held:  # the mix hygiene drops train rows near (J ≥ 0.8) a held-out state
        h = _shingles(r["state"])
        assert max(len(h & t) / len(h | t) for t in tsh) < 0.8, r["state"][:200]


# ---------------------------------------------------------------------------------------
# Labels recomputed independently, from the real text where the rule is read from it


def _months(d, n):
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return dt.date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _number(tok):
    words = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6,
             "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "quinze": 15, "trente": 30}  # fmt: skip
    return int(tok) if tok.isdigit() else words[tok]


def _rule_text(row):
    return next(t for tag, t, *_ in row["meta"]["lines"] if tag == "rule")


@pytest.mark.parametrize("name", ["gen7_rt_kali_fr_pairs", "gen7_rt_cold_fr_pairs"])
def test_french_legal_labels_recomputed_from_the_excerpt(name):
    for split in ("train", "heldout"):
        for r in _gen(name, 90, split=split):
            if r["meta"]["role"] == "unk":
                continue
            f, text = r["meta"]["facts"], _rule_text(r)
            d1, d2 = dt.date.fromisoformat(f["d1"]), dt.date.fromisoformat(f["d2"])
            if "pairs" in f:  # trial / notice period by category, read from the excerpt
                n, unit = None, None
                for cat, k, u in f["pairs"]:
                    pat = (rf"pour les {re.escape(cat)} ?: ?(\w+) {u}|(\w+) {u} pour les "
                           rf"{re.escape(cat)}")  # fmt: skip
                    m = re.search(pat, text)
                    assert m and _number(m.group(1) or m.group(2)) == k, (cat, text[:300])
                    if cat == f["cat"]:
                        n, unit = k, u
                end = (_months(d1, n) - dt.timedelta(days=1) if unit == "mois" else
                       d1 + dt.timedelta(days=(7 * n if unit == "semaines" else n) - 1))  # fmt: skip
                want = "yes" if d1 <= d2 <= end else "no"
            else:  # délai: N days / months from the start day (excluded), no move
                m = re.search(r"délai (?:maximum |maximal |de rigueur )?(?:de |d')(\w+) "
                              r"(jours|mois)", text)  # fmt: skip
                assert m and _number(m.group(1)) == f["n"] and m.group(2) == f["unit"], text[:300]
                last = (_months(d1, f["n"]) if f["unit"] == "mois"
                        else d1 + dt.timedelta(days=f["n"]))  # fmt: skip
                want = "yes" if d2 <= last else "no"
            assert r["expected"] == want, (r["state"][-300:], f)


def test_tau_labels_recomputed_from_the_policy_text():
    airline = (FIXTURES / "tau" / "airline.md").read_text()
    # the allowance table, parsed from the policy sentence itself
    table = {}
    for tier, body in re.findall(r"If the booking user is a (\w+) member, ([^.]+)\.", airline):
        counts = re.findall(r"(\d+) free checked bags? for each (basic economy|economy|business)",
                            body)  # fmt: skip
        table[tier] = {cabin: int(k) for k, cabin in counts}
    assert set(table) == {"regular", "silver", "gold"}
    fee = int(re.search(r"Each extra baggage is (\d+) dollars", airline).group(1))
    now = dt.datetime.fromisoformat(re.search(r"current time is ([\d-]+ [\d:]+)", airline).group(1))
    for split in ("train", "heldout"):
        for r in _gen("gen7_rt_tau_pairs", 240, split=split):
            f, base = r["meta"]["facts"], r["meta"]["base"]
            if r["meta"]["role"] == "unk":
                continue
            if base == "rt_tau_bags":
                free = table[f["tier"]][f["cabin"]] * f["pax"]
                assert r["expected"] == f"${max(0, f['bags'] - free) * fee}"
            elif base == "rt_tau_cancel":
                booked = dt.datetime.fromisoformat(f["booked"])
                ok = (now - booked <= dt.timedelta(hours=24) or f["airline_cancelled"]
                      or f["cabin"] == "business"
                      or (f["insurance"] and f["reason"] in ("health reasons", "bad weather")))  # fmt: skip
                assert r["expected"] == ("yes" if ok else "no")
            elif base == "rt_tau_status":
                m = re.search(r"can only be \w+ if its status is '(\w+)'", _rule_text(r))
                assert r["expected"] == ("yes" if f["status"] == m.group(1) else "no")
            elif base == "rt_tau_gift":
                assert r["expected"] == ("yes" if f["balance"] >= f["total"] else "no")
            elif base == "rt_tau_comp":
                ok = f["tier"] != "regular" or f["insurance"] or f["cabin"] == "business"
                amount = (0 if not (ok and f["asks"]) else 100 * f["pax"]
                          if f["complaint"] == "cancelled" else 50 * f["pax"] * f["changes"])  # fmt: skip
                assert r["expected"] == f"${amount}"
    # held out by rule: the compensation and gift-card rules never train
    assert {r["meta"]["base"] for r in _gen("gen7_rt_tau_pairs", 150)} == {
        "rt_tau_bags",
        "rt_tau_cancel",
        "rt_tau_status",
    }
    assert {r["meta"]["base"] for r in _gen("gen7_rt_tau_pairs", 90, split="heldout")} == {
        "rt_tau_comp",
        "rt_tau_gift",
    }


def test_sharc_labels_recomputed_from_the_snippet():
    for split in ("train", "heldout"):
        for r in _gen("gen7_rt_sharc_pairs", 60, split=split):
            if r["meta"]["role"] == "unk":
                continue
            f, text = r["meta"]["facts"], _rule_text(r)
            shown = (
                f"{f['value']:,.2f}".rstrip("0").rstrip(".")
                if isinstance(f["value"], float)
                else str(f["value"])
            )
            assert shown.replace(",", "") in text.replace(",", ""), (shown, text[:200])
            x, v = f["x"], f["value"]
            ok = {">=": x >= v, ">": x > v, "<": x < v, "<=": x <= v}[f["op"]]
            assert r["expected"] == ("yes" if ok else "no")


def test_month_arithmetic_is_explicit():
    from tjev.data.sources.gen_v7.realtext import add_months, delay_deadline, period_end

    assert add_months(dt.date(2025, 1, 31), 1) == dt.date(2025, 2, 28)
    assert add_months(dt.date(2024, 1, 31), 1) == dt.date(2024, 2, 29)
    assert add_months(dt.date(2025, 11, 15), 3) == dt.date(2026, 2, 15)
    assert delay_deadline(dt.date(2025, 3, 3), 15, "jours") == dt.date(2025, 3, 18)
    assert delay_deadline(dt.date(2025, 3, 3), 2, "mois") == dt.date(2025, 5, 3)
    assert period_end(dt.date(2025, 3, 3), 1, "mois") == dt.date(2025, 4, 2)
    assert period_end(dt.date(2025, 3, 3), 2, "semaines") == dt.date(2025, 3, 16)


def test_extractors_keep_only_unambiguous_rules():
    from tjev.data.sources.gen_v7.realtext import extract_delay, extract_periods

    ok = (
        "Le salarié doit répondre dans un délai de 15 jours à compter de la réception de la lettre."
    )
    assert extract_delay(ok)["n"] == 15
    assert extract_delay(ok + " " + ok.replace("15", "8")) is None  # two délais
    assert (
        extract_delay("Il répond dans un délai de 8 jours ouvrés à compter de la réception.")
        is None
    )
    assert extract_delay("dans un délai de 8 jours à compter de sa signature.") is None  # pronoun
    lst = ("La période d'essai est fixée comme suit : pour les employés : 1 mois ; pour les "
           "cadres : 3 mois.")  # fmt: skip
    assert extract_periods(lst)["pairs"] == [("employés", 1, "mois"), ("cadres", 3, "mois")]
    assert extract_periods(lst.replace("suit :", "suit, renouvelable une fois :")) is None
    notice = ("Au-delà de la période d'essai, le préavis est fixé ainsi : 1 mois pour les "
              "employés ; 2 mois pour les cadres.")  # fmt: skip
    assert extract_periods(notice)["kind"] == "preavis"  # the keyword nearest the list
