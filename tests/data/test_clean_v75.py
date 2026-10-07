"""mix-v7.5 cleaning of mix-v7 rows (text fixes, drops, near-cluster cap). No download."""

from tjev.data.clean_v75 import clean_rows, cluster_cap, fix_item, repr_to_json, repr_to_text
from tjev.data.item import parse_item


def _item(
    state,
    criteria=None,
    lang="en",
    source="scruples",
    expected="a",
    target=None,
    i=0,
    instructions="Which one?",
):
    row = {
        "id": f"x{i}",
        "source": source,
        "lang": lang,
        "state": state,
        "question": {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria or {"a": "A", "b": "B"},
        },
    }
    row |= {"target": target} if target else {"expected": expected}
    return parse_item(row)


def test_html_entities_unescaped_in_prose_sources_only():
    it = _item("Tom &amp; Jerry said &quot;hi&quot; &#39;twice&#39;")
    fixed, fixes = fix_item(it, unescape=True, retag=False)
    assert fixed.state == "Tom & Jerry said \"hi\" 'twice'" and fixes == ["state"]
    twice, _ = fix_item(_item("It&amp;#39;s &amp;amp; done"), unescape=True, retag=False)
    assert twice.state == "It's & done"
    same, fixes = fix_item(it, unescape=False, retag=False)
    assert same.state == it.state and fixes == []


def test_python_reprs_become_text_or_json():
    assert (
        repr_to_text("{'level': 'High', 'definition': 'Clearly harmful'}")
        == "High: Clearly harmful"
    )
    assert repr_to_text("{'value': None}") == ""
    assert repr_to_json("Tool: {'name': 'get_weather', 'args': [1, None]} end") == (
        'Tool: {"name": "get_weather", "args": [1, null]} end'
    )
    assert repr_to_json("a {not python} b") == "a {not python} b"
    assert repr_to_json("x {'a': [1, ...]} y") == "x {'a': [1, ...]} y"  # not JSON: kept
    it = _item("s", {"a": "{'level': 'Low', 'definition': 'fine'}", "b": "B"})
    fixed, _ = fix_item(it, unescape=False, retag=False)
    assert fixed.question.descriptions == ("Low: fine", "B")


def test_language_retag_and_drops():
    fr = (
        "Je pense que la réponse de l'assistant est très bonne et que nous avons "
        "été aidés pour ce projet avec les données dans le tableau."
    )
    rows = [
        _item(fr, lang="en", source="helpsteer3_x"),
        _item("bad � byte", i=1, source="helpsteer3_x"),
        _item("ctrl\x07char", i=2, source="helpsteer3_x"),
    ]
    out, stats = clean_rows(rows, "helpsteer3_x", "judge", train=True)
    assert [it.lang for it in out] == ["fr", "en"] and out[1].state == "ctrlchar"
    assert stats["drop:replacement_char"] == 1 and stats["fix:lang:en->fr"] == 1
    held, stats = clean_rows(rows, "helpsteer3_x", "judge", train=False)
    assert len(held) == 3  # held-out rows are fixed, never dropped
    from tjev.data.clean_v75 import damaged

    assert damaged(_item("Ã‚GE LÃ©GAL")) == "mojibake" and damaged(_item("L'ÂGE LÉGAL")) is None
    assert damaged(_item("Il fait 20Â°C")) == "mojibake"
    soft = _item("s", target={"a": 0.6, "b": 0.4}, source="dices")
    out, stats = clean_rows([soft, _item("t", source="dices", i=1)], "dices", "judge", train=True)
    assert out == [soft] and stats["drop:one_hot"] == 1


def test_cluster_cap_keeps_a_few_rows_per_near_duplicate_state():
    base = "On 12 March 2004 Alice Martin was the chair of the board of the Acme company in Paris"
    rows = [_item(base + f" {k}", i=k, instructions=f"Q{k % 3}") for k in range(6)]
    rows.append(_item("A wholly different text about rivers, mountains and the weather", i=9))
    keep = cluster_cap(rows, 2)
    assert sum(keep[:6]) == 2 and keep[6]
    assert {rows[i].question.instructions for i in range(6) if keep[i]} == {"Q0", "Q1"}
