"""Code-labelled generators: valid, deterministic, bilingual; held-out splits share no
key or template line with training; pinned training output."""

import hashlib
import json
import re

import numpy as np
import pytest

from tjev.data.item import parse_item
from tjev.data.sources.generators import GENERATORS, LONG_GENERATORS, generate, item_key

EVERY = sorted({**GENERATORS, **LONG_GENERATORS})


@pytest.mark.parametrize("name", EVERY)
def test_generators_valid_deterministic_and_bilingual(name):
    a = generate(name, 60, seed=1)
    assert a == generate(name, 60, seed=1)
    for row in a:
        parse_item(row)
    assert {r["lang"] for r in a} == {"en", "fr"}
    held = generate(name, 60, seed=2, split="heldout")
    for row in held:
        parse_item(row)
    train_keys = {(r["state"], r["question"]["instructions"]) for r in generate(name, 400, seed=1)}
    overlap = sum((r["state"], r["question"]["instructions"]) in train_keys for r in held)
    assert overlap / len(held) < 0.5, f"{name}: held-out rows mostly memorisable ({overlap})"


def test_generator_labels_are_computed_correctly():
    for row in generate("gen_base_rate", 50, seed=0):
        p = row["target"]["yes"]
        assert 0 < p < 1
    for row in generate("gen_refund_policy", 200, seed=0):
        state = row["state"]
        if "final sale" in state and "marked final sale." in state.split("\n")[-1]:
            assert row["expected"] == "deny"


# Train-split generator output is training data (the mix is built from it): any change
# must be deliberate. Update a hash only together with a mix rebuild (mix-v5: training
# templates per generator × language; mix-v5b: return_window windows and hints, long-policy
# hints).
TRAIN_HASHES = {
    "gen_return_window": "fd96ceeae336a51a",
    "gen_weekday": "737c4daf3737f818",
    "gen_invoice_total": "315e240c4f7ca81d",
    "gen_base_rate": "2a5c2aa4411afc07",
    "gen_schedule_conflict": "b56bcff6f4f3638e",
    "gen_table_argmax": "0c5c78e408f44f2e",
    "gen_refund_policy": "16432f29931f82c0",
    "gen_long_refund_policy": "7c1e8884673382db",
    "gen_ticket_triage": "9a342b6bf76e3146",
    "gen_spelling_error": "3d66f8ec12a0bd42",
    "gen_register": "2f3748cf4130b00b",
}
LONG_TRAIN_HASHES = {"gen_longdoc_refund_policy": "4865536572fe80bd"}


def test_train_generators_are_stable():
    assert set(TRAIN_HASHES) == set(GENERATORS)
    assert set(LONG_TRAIN_HASHES) == set(LONG_GENERATORS)
    for name, want in (TRAIN_HASHES | LONG_TRAIN_HASHES).items():
        rows = generate(name, 20, 0, split="train", fr_share=0.5)
        blob = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
        assert hashlib.sha256(blob).hexdigest()[:16] == want, name


def test_heldout_generators_use_their_own_surface_text():
    for name in ("gen_return_window", "gen_weekday", "gen_schedule_conflict", "gen_table_argmax"):
        train = {r["question"]["instructions"] for r in generate(name, 100, 1, split="train")}
        heldout = {r["question"]["instructions"] for r in generate(name, 100, 1, split="heldout")}
        assert not train & heldout, name


def _template_lines(row):
    """Sentences/lines of a row (numbers masked) that carry template text, not only data."""
    q = row["question"]
    parts = [*re.split(r"(?<=[.!?])\s+|\n", row["state"]), q["instructions"]]
    if q["type"] != "noul":  # criteria wording (skip name-only options such as weekdays)
        crit = q["criteria"]
        pairs = crit.items() if isinstance(crit, dict) else enumerate(crit)
        parts += [v for k, v in pairs if v != k]
    out = set()
    for p in parts:
        line = re.sub(r"\d+", "#", _norm(p))
        if len(re.findall(r"[^\W\d_]{3,}", line)) >= 2:
            out.add(line)
    return out


def _norm(text):
    return re.sub(r"\W+", " ", text.lower()).strip()


@pytest.mark.parametrize("name", EVERY)
def test_heldout_shares_no_key_or_template_line_with_train(name):
    train = generate(name, 400, 1)
    held = generate(name, 150, 2, split="heldout")
    assert not {item_key(r) for r in train} & {item_key(r) for r in held}, name
    seen = set().union(*(_template_lines(r) for r in train))
    shared = {line for r in held for line in _template_lines(r) if line in seen}
    assert not shared, f"{name}: {sorted(shared)[:3]}"


@pytest.mark.parametrize("name", EVERY)
def test_generators_do_not_repeat_keys(name):
    for split, n in (("train", 400), ("heldout", 150)):
        rows = generate(name, n, 3, split=split)
        assert len(rows) == n, (name, split)
        assert len({item_key(r) for r in rows}) == n, (name, split)
        assert len({r["id"] for r in rows}) == n


def test_heldout_template_tables_are_disjoint():
    from tjev.data.sources.generators import REGISTER, TRIAGE, WRITING_TEMPLATES

    for lang in ("en", "fr"):
        wt = WRITING_TEMPLATES[lang]
        assert len(wt["heldout"]) >= 6
        assert not {t for t, _ in wt["train"]} & {t for t, _ in wt["heldout"]}
        for split in ("train", "heldout"):
            for template, errors in wt[split]:
                for word, wrong in errors.items():  # each perturbation hits its template
                    assert re.search(rf"\b{re.escape(word)}\b", template), word
                    assert word != wrong
        for level in REGISTER["train", lang]:
            for part in range(3):
                assert not set(REGISTER["train", lang][level][part]) & set(
                    REGISTER["heldout", lang][level][part]
                )
        li = 0 if lang == "en" else 1
        train = {s for q in TRIAGE["train"].values() for s in q[li]}
        assert not train & {s for q in TRIAGE["heldout"].values() for s in q[li]}
    assert set(TRIAGE["train"]) < set(TRIAGE["heldout"])


def test_long_policy_documents_are_long():
    from tjev.data.sources.generators import POLICY_TEXT

    for split in ("train", "heldout"):
        rows = generate("gen_long_refund_policy", 200, 4, split=split, fr_share=0.5)
        chars = np.array([len(r["state"]) for r in rows])
        assert np.median(chars) >= 12000, (split, np.median(chars))
        assert chars.max() <= 19000, split  # ~3.8k prompt tokens: fits the 4096 bucket
        for r in rows:  # the case's category has exactly one window rule
            lines = r["state"].split("\n")
            case = next(x for x in reversed(lines) if not x.startswith(HINT_PREFIXES))
            variants = POLICY_TEXT[split, r["lang"]]
            found = None
            for v in variants:  # the variant whose case and question the row uses
                if v["q"] != r["question"]["instructions"]:
                    continue
                pre, post = v["category"].split("{d}")
                m = re.search(re.escape(pre) + r"(.+?)" + re.escape(post), case)
                if m:
                    found = v, m.group(1).strip()
                    break
            assert found, case
            v, domain = found
            rule = re.escape(v["window"]).replace(re.escape("{w}"), r"\d+")
            rule = rule.replace(re.escape("{d}"), re.escape(domain))
            assert len(re.findall(rule, r["state"])) == 1, (rule, case)


def test_french_items_have_no_english_products_queues_or_decimals():
    from tjev.data.sources.generators import PRODUCTS, TRIAGE

    for split in ("train", "heldout"):
        fr = {p for p, _ in PRODUCTS[split, "fr"]}
        english = [p for p, _ in PRODUCTS[split, "en"] if p not in fr]  # brands are shared
        english += list(TRIAGE[split])
        for r in generate("gen_ticket_triage", 300, 5, split=split, fr_share=1.0):
            text = r["state"] + " " + " ".join(r["question"]["criteria"].values())
            assert not [w for w in english if re.search(rf"\b{w}\b", text)], text
        for r in generate("gen_base_rate", 50, 5, split=split, fr_share=1.0):
            assert not re.search(r"\d\.\d|\d%", r["state"]), r["state"]
            assert re.search(r"\d,\d+ %", r["state"])


def _pattern(template: str) -> str:
    """A template's regex: each {field} a named lazy group."""
    out = re.escape(template)
    for field in set(re.findall(r"\{(\w+)\}", template)):
        out = out.replace(re.escape("{" + field + "}"), f"(?P<{field}>.+?)", 1)
    return out


def test_longdoc_handbooks_are_long_and_only_the_named_chapter_decides():
    """5–14.5k-token handbooks of 3–5 stores (22k–80k characters at ~4.5–5.5 per token:
    under the 16384 bucket). The case's category has one window rule per chapter; the named
    store's chapter (exactly one) gives the label by refund_policy's rules, and another
    store's rules decide differently whenever some rules could (English rows parsed)."""
    from tjev.data.sources.generators import (
        HANDBOOK_TEXT,
        POLICY_TEXT,
        STORES,
        _refund_decision,
    )

    for split in ("train", "heldout"):
        rows = generate("gen_longdoc_refund_policy", 120, 4, split=split, fr_share=0.5)
        chars = np.array([len(r["state"]) for r in rows])
        assert chars.min() >= 20000 and chars.max() <= 80000, (split, chars.min(), chars.max())
        assert {r["expected"] for r in rows} == {"approve", "deny", "escalate"}
        conflicting = english = 0
        for r in rows:  # the POLICY_TEXT variant the row uses (each has its own question)
            (t,) = [
                v for v in POLICY_TEXT[split, r["lang"]] if v["q"] == r["question"]["instructions"]
            ]
            _, chapter, named = HANDBOOK_TEXT[split, r["lang"]]
            *body, store_line, case = r["state"].split("\n")
            heads = [x for x in body if re.fullmatch(_pattern(chapter), x)]
            assert 3 <= len(heads) <= 5, len(heads)
            store = re.fullmatch(_pattern(named), store_line).group("s")
            assert store in STORES[split]
            own = [
                i for i, h in enumerate(heads) if re.fullmatch(_pattern(chapter), h)["s"] == store
            ]
            assert len(own) == 1
            final = case.endswith(t["words"][2] + ".")
            plain = case.removesuffix(t["words"][2] + ".") + "." if final else case
            case_rx = _pattern(t["case"].replace("{f}", "")).replace(
                "(?P<days>.+?)", r"(?P<days>\d+)"
            )
            c = re.fullmatch(case_rx, plain)
            c = c if r["lang"] == "en" else None
            text = "\n".join(body)
            chapters = re.split("|".join(re.escape(h) for h in heads), text)[1:]
            if c is None:
                continue
            english += 1
            domain = re.fullmatch(_pattern(t["category"]), c["c"])["d"]
            member_w, opened_w, _ = t["words"]
            facts = {"days": int(c["days"]), "member": c["m"] == member_w[0],
                     "opened": c["o"] == opened_w[0], "final_sale": final,
                     "amount": float(c["a"].strip("$").replace(",", ""))}  # fmt: skip
            decisions = []
            for body_text in chapters:
                windows = re.findall(_pattern(t["window"]).replace("(?P<d>.+?)", re.escape(domain)),
                                     body_text)  # fmt: skip
                assert len(windows) == 1, (r["id"], windows)
                rule = {
                    "window": int(windows[0]),
                    "bonus": int(re.search(_pattern(t["member"].rstrip(".")), body_text)["b"]),
                    "limit": float(re.search(_pattern(t["limit"]), body_text)["l"]
                                   .strip("$").replace(",", "")),
                }  # fmt: skip
                decisions.append(_refund_decision(split, domain, **rule, **facts))
            assert decisions[own[0]] == r["expected"], (r["id"], decisions)
            possible = {
                _refund_decision(split, domain, window=w, bonus=b, limit=lim, **facts)
                for w in (14, 30, 45)
                for b in (15, 30)
                for lim in (200, 500, 1000)
            }
            assert len(set(decisions)) > 1 or possible == {r["expected"]}, (r["id"], decisions)
            conflicting += len(set(decisions)) > 1
        assert english > 30 and conflicting > 0.6 * english, (english, conflicting)


SURFACE_TABLES = (
    "RETURN_TEXT", "WEEKDAY_TEXT", "INVOICE_TEXT", "BASE_RATE_TEXT", "SCHEDULE_TEXT",
    "TABLE_TEXT", "POLICY_TEXT", "QUEUE_TEXT", "SPELLING_TEXT", "REGISTER_TEXT",
)  # fmt: skip


@pytest.mark.parametrize("table", SURFACE_TABLES)
def test_surface_tables_have_six_to_eight_train_and_two_to_three_heldout_templates(table):
    from tjev.data.sources import generators

    t = getattr(generators, table)
    for lang in ("en", "fr"):
        train, held = t["train", lang], t["heldout", lang]
        assert 6 <= len(train) <= 8, (table, lang, len(train))
        assert 2 <= len(held) <= 3, (table, lang, len(held))
        as_text = [json.dumps(v, ensure_ascii=False, sort_keys=True) for v in (*train, *held)]
        assert len(set(as_text)) == len(as_text), (table, lang)  # no repeated template


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_a_template_never_changes_the_label(name):
    fn, _ = GENERATORS[name]
    for split in ("train", "heldout"):
        for lang in ("en", "fr"):
            states = set()
            for k in range(25):
                base = fn(np.random.default_rng(k), lang, split, None)
                for t in range(3):
                    other = fn(np.random.default_rng(k), lang, split, np.random.default_rng(t))
                    assert other.get("expected") == base.get("expected"), (name, split, lang)
                    assert other.get("target") == base.get("target"), (name, split, lang)
                    crit = other["question"]["criteria"]
                    keys = list(crit) if isinstance(crit, dict) else len(crit)
                    want = base["question"]["criteria"]
                    assert keys == (list(want) if isinstance(want, dict) else len(want))
                    states.add(other["state"] + other["question"]["instructions"])
            assert len(states) > 25, (name, split, lang)  # the template did vary


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_training_rows_use_many_templates(name):
    rows = generate(name, 400, 0, fr_share=0.5)
    for lang in ("en", "fr"):
        heads = {
            (r["question"]["instructions"] if name != "gen_table_argmax" else r["state"][:6])
            for r in rows
            if r["lang"] == lang
        }
        assert len(heads) >= 5, (name, lang, len(heads))


def test_french_dates_and_times_follow_french_conventions():
    import datetime as dt

    from tjev.data.sources.generators import _clock, _date

    assert _date(dt.date(2025, 3, 1), "fr", 1) == "samedi 1er mars 2025"
    assert _date(dt.date(2025, 3, 3), "fr", 2) == "03/03/2025"
    assert _clock(14 * 60 + 30, "fr", 0) == "14h30"
    assert _clock(14 * 60 + 30, "fr", 1) == "14 h 30"
    for name in ("gen_schedule_conflict", "gen_return_window", "gen_weekday"):
        for split in ("train", "heldout"):
            for r in generate(name, 200, 0, split=split, fr_share=1.0):
                assert not re.search(r"\b(am|pm)\b|[A-Z][a-z]+day\b", r["state"]), r["state"]


def test_a_template_never_changes_the_longdoc_label():
    from tjev.data.sources.generators import LONG_GENERATORS

    fn, _ = LONG_GENERATORS["gen_longdoc_refund_policy"]
    for split in ("train", "heldout"):
        for lang in ("en", "fr"):
            questions = set()
            for k in range(4):
                base = fn(np.random.default_rng(k), lang, split, None)
                for t in range(3):
                    other = fn(np.random.default_rng(k), lang, split, np.random.default_rng(t))
                    assert other["expected"] == base["expected"], (split, lang, k)
                    assert list(other["question"]["criteria"]) == list(base["question"]["criteria"])
                    questions.add(other["question"]["instructions"])
            assert len(questions) > 1, (split, lang)


# mix-v5b: training-only curriculum hints and wider training windows. Held-out rows are
# mix-v5's, byte for byte (validation / calibration / test parts 101 / 202 / 303 and the
# selection / report sets 777 / 4242 of tjev.data.mix and tjev.data.heldout).
HINT_PREFIXES = ("Days elapsed", "Last day of the window", "Jours écoulés", "Dernier jour du",
                 "Case note", "Note de dossier")  # fmt: skip
HELDOUT_HASHES = {
    "gen_base_rate": "8fb07d57a1142f23",
    "gen_invoice_total": "df0848c16b6bf0be",
    "gen_long_refund_policy": "813efe1250ddd825",
    "gen_refund_policy": "bf3b674c8783f3a4",
    "gen_register": "ae3b1cb2b155b18d",
    "gen_return_window": "4471b7a69147cefe",
    "gen_schedule_conflict": "93cb76dd487b9a81",
    "gen_spelling_error": "1dfa3a371b6c059b",
    "gen_table_argmax": "8dfa9f27edcb6aaf",
    "gen_ticket_triage": "045ee79f8189bb40",
    "gen_weekday": "6cd633959b24ac3b",
}


def test_heldout_generator_rows_are_unchanged_since_mix_v5():
    assert set(HELDOUT_HASHES) == set(GENERATORS)
    for name, want in HELDOUT_HASHES.items():
        parts = [
            generate(name, n, seed, split="heldout", fr_share=fr)
            for seed, n, fr in ((101, 150, 0.3), (202, 150, 0.3), (303, 150, 0.3),
                                (777, 20, 0.5), (4242, 20, 0.5))
        ]  # fmt: skip
        blob = json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()
        assert hashlib.sha256(blob).hexdigest()[:16] == want, name


@pytest.mark.parametrize("name", ["gen_return_window", "gen_long_refund_policy"])
def test_hints_are_training_only_and_never_change_the_label(name):
    fn, _ = GENERATORS[name]
    hinted = 0
    for lang in ("en", "fr"):
        for k in range(150):
            for split in ("train", "heldout"):
                a = fn(np.random.default_rng(k), lang, split, np.random.default_rng(k), hints=False)
                b = fn(np.random.default_rng(k), lang, split, np.random.default_rng(k))
                assert a["expected"] == b["expected"] and a["question"] == b["question"]
                if split == "heldout":
                    assert a == b
                    assert not any(x.startswith(HINT_PREFIXES) for x in b["state"].split("\n"))
                elif a["state"] != b["state"]:  # one exact line appended
                    head, line = b["state"].rsplit("\n", 1)
                    assert head == a["state"] and line.startswith(HINT_PREFIXES), line
                    hinted += 1
    assert 0.1 < hinted / 300 < 0.3, hinted  # 300 training rows
    rows = generate(name, 1000, 0, fr_share=0.5)
    share = np.mean([r["state"].rsplit("\n", 1)[-1].startswith(HINT_PREFIXES) for r in rows])
    assert 0.12 < share < 0.25, share


_EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]  # fmt: skip


def _en_dates(text):
    import datetime as dt

    m = "|".join(_EN_MONTHS)
    out = []
    pattern = rf"(\d{{4}})-(\d\d)-(\d\d)|({m}) (\d+), (\d{{4}})|(\d+) ({m}) (\d{{4}})"
    for g in re.findall(pattern, text):
        if g[0]:
            out.append(dt.date(int(g[0]), int(g[1]), int(g[2])))
        elif g[3]:
            out.append(dt.date(int(g[5]), _EN_MONTHS.index(g[3]) + 1, int(g[4])))
        else:
            out.append(dt.date(int(g[8]), _EN_MONTHS.index(g[7]) + 1, int(g[6])))
    return out


def test_return_window_hints_are_exact_and_training_windows_avoid_heldout_ones():
    from tjev.data.sources.generators import HELDOUT_WINDOWS

    rows = generate("gen_return_window", 3000, 0, fr_share=0.0)
    windows, checked = set(), 0
    for r in rows:
        window = int(re.search(r"\d+", r["state"]).group())  # {w} is the first number
        windows.add(window)
        dates = _en_dates(r["state"])
        if len(dates) < 2:  # the "{e} days ago" template
            continue
        bought, today = dates[:2]
        assert r["expected"] == ("yes" if (today - bought).days <= window else "no")
        last = r["state"].rsplit("\n", 1)[-1]
        if last.startswith("Days elapsed"):
            assert int(re.search(r"\d+", last).group()) == (today - bought).days
            checked += 1
        elif last.startswith("Last day of the window"):
            assert dates[2] == bought + (today - bought).__class__(days=window)
            assert r["expected"] == ("yes" if today <= dates[2] else "no")
            checked += 1
    assert checked > 300, checked
    assert not windows & set(HELDOUT_WINDOWS)
    assert len(windows) > 150  # not only the common store windows (the month shortcut)
    held = {int(re.search(r"\d+", r["state"]).group())
            for r in generate("gen_return_window", 300, 1, split="heldout")}  # fmt: skip
    assert held == set(HELDOUT_WINDOWS)


def test_long_policy_hint_points_to_the_window_and_states_the_period():
    from tjev.data.sources.generators import POLICY_HINT

    checked = 0
    for r in generate("gen_long_refund_policy", 400, 0, fr_share=0.5):
        lines = r["state"].split("\n")
        if not lines[-1].startswith(("Case note", "Note de dossier")):
            continue
        checked += 1
        nums = [int(x) for x in re.findall(r"\d+", lines[-1])]
        section, window, period = nums[0], nums[1], nums[-1]
        member = len(nums) == 4
        if member:
            assert nums[1] + nums[2] == period
        plain = POLICY_HINT[r["lang"]]["plain"].split(",")[0]
        assert member != (plain in lines[-1])
        word = "Section" if r["lang"] == "en" else "Article"
        target = next(x for x in lines if x.startswith(f"{word} {section}. "))
        domain = re.search(r"(?:for|catégorie) (.+?) (?:is in|figure)", lines[-1]).group(1)
        assert domain in target and re.search(rf"\b{window}\b", target), (target[:200], lines[-1])
        # the stated period decides the window rule exactly as the label does
        days = int(re.search(r"(\d+) (?:days|jours)", lines[-2]).group(1))
        if days > period:
            assert r["expected"] == "deny"
    assert checked > 40, checked
