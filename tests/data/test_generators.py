"""Code-labelled generators: valid, deterministic, bilingual; held-out splits share no
key or template line with training; pinned training output."""

import hashlib
import json
import re

import numpy as np
import pytest

from tjev.data.item import parse_item
from tjev.data.sources.generators import GENERATORS, generate, item_key


@pytest.mark.parametrize("name", sorted(GENERATORS))
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
# must be deliberate. Update a hash only together with a mix rebuild.
TRAIN_HASHES = {
    "gen_return_window": "180a9c77b4a35d7a",
    "gen_weekday": "2d3a6ced62cbc777",
    "gen_invoice_total": "2e72459cac489d32",
    "gen_base_rate": "0ef483d00e90adab",
    "gen_schedule_conflict": "e0d63e1eb31ce532",
    "gen_table_argmax": "96533860ab4e8741",
    "gen_refund_policy": "39c8e64e0700db3d",
    "gen_long_refund_policy": "10df09cf5bbf63cf",
    "gen_ticket_triage": "001c220b60a7c206",
    "gen_spelling_error": "a927da5e0f832454",
    "gen_register": "3f53710111cc0d8c",
}


def test_train_generators_are_stable():
    assert set(TRAIN_HASHES) == set(GENERATORS)
    for name, want in TRAIN_HASHES.items():
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


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_heldout_shares_no_key_or_template_line_with_train(name):
    train = generate(name, 400, 1)
    held = generate(name, 150, 2, split="heldout")
    assert not {item_key(r) for r in train} & {item_key(r) for r in held}, name
    seen = set().union(*(_template_lines(r) for r in train))
    shared = {line for r in held for line in _template_lines(r) if line in seen}
    assert not shared, f"{name}: {sorted(shared)[:3]}"


@pytest.mark.parametrize("name", sorted(GENERATORS))
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
    for split in ("train", "heldout"):
        rows = generate("gen_long_refund_policy", 200, 4, split=split, fr_share=0.5)
        chars = np.array([len(r["state"]) for r in rows])
        assert np.median(chars) >= 12000, (split, np.median(chars))
        assert chars.max() <= 19000, split  # ~3.8k prompt tokens: fits the 4096 bucket
        for r in rows:  # the case's category has exactly one window rule
            case = r["state"].rsplit("\n", 1)[-1]
            domain = re.search(r"(?:category|catégorie|listed under|rayon) ([^,;]+)[,;]", case)
            assert domain, case
            pattern = {
                ("train", "en"): "Refunds for {d} are accepted",
                ("train", "fr"): "pour la catégorie {d} sont acceptés",
                ("heldout", "en"): "Goods listed under {d} may be returned",
                ("heldout", "fr"): "du rayon {d} peuvent être retournés",
            }[split, r["lang"]].format(d=domain.group(1).strip())
            assert r["state"].count(pattern) == 1, (pattern, case)


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
