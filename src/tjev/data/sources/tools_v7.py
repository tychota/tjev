"""mix-v7 sources of the "targeted" block: tool use, rule conversations and legal reading.

==================  ===================================================  ======================
source              what it adds                                         labels
==================  ===================================================  ======================
``sharc``           a rule text + scenario + follow-up history:          exact (annotations)
                    yes / no / irrelevant / ask a follow-up
``when2call``       call a tool, ask a follow-up, or say it cannot help  exact (train splits;
                    (+ which tool), given the tools on offer             the test is BFCL-
                                                                         derived: never read)
``xlam``            which provided tool a request needs (or none)        exact (verified calls)
``xlam_irrelevance``  the same requests with the needed tool removed:    exact (no tool)
                    no listed tool applies
``metatool``        tool selection among dissimilar tools, multi-tool    exact (ToolE labels)
                    membership, "does this request need a tool?"
``casehold``        which holding a cited case stands for (5-way)        exact
``ledgar``          the provision type of a contract clause (8 of 100)   exact
``cxm_fr``          French customer-service decisions (synthetic): agent exact (synthetic
                    quality checks, contact intent, next tool to call    benchmark labels)
==================  ===================================================  ======================

Same contract as the other adapter modules: each adapter maps one row to item dicts.
Every source sets its block ("targeted") and the licence use of its rows; card licences
are in ``LICENCES``. Menus never exceed 26 options and always contain the gold label.
"""

from __future__ import annotations

import base64
import csv
import functools
import hashlib
import io
import json
import re
from pathlib import Path

from tjev.data.sources.base import CACHE, Source

BLOCK = "targeted"
MAX_STATE_CHARS = 10000
MAX_MENU = 26


def _hash01(key: str) -> float:
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / 16**12


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", str(text).lower()).strip()


def _dataset(rows: list[dict]):
    from datasets import Dataset

    return Dataset.from_list(rows)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:60]


def _clip(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


# --- tool descriptions (xLAM / When2Call / CXM formats) ----------------------------------


def _params(spec, required=()) -> list[tuple[str, str, str, bool, str]]:
    """(name, type, description, required, default) of a tool's parameters: xLAM's flat
    ``{name: {type, description, default}}`` or a JSON schema ``{type, properties,
    required}``; ``required`` may also sit next to ``parameters`` (When2Call)."""
    if not isinstance(spec, dict):
        return []
    required = set(required or [])
    if isinstance(spec.get("properties"), dict):
        required |= set(spec.get("required") or [])
        props = spec["properties"]
    else:
        props = spec
    out = []
    for name, p in props.items():
        if isinstance(p, dict):
            kind = str(p.get("type", ""))
            default = p.get("default")
            default = (
                ""
                if default in (None, "")
                else _clip(json.dumps(default) if not isinstance(default, str) else default, 40)
            )
            out.append((str(name), kind.replace(", optional", ""), str(p.get("description", "")),
                        name in required and "optional" not in kind, default))  # fmt: skip
    return out


def tool_text(tool: dict, desc_chars: int = 300, param_chars: int = 120) -> str:
    """One tool as text: ``- name: description`` and its parameters (type, required,
    default)."""
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    line = f"- {fn.get('name')}: {_clip(fn.get('description', ''), desc_chars)}"
    params = _params(fn.get("parameters"), fn.get("required") or tool.get("required"))
    if params:
        parts = []
        for n, t, d, r, default in params:
            tags = ", ".join(
                x
                for x in (t, "required" if r else "", f"default {default}" if default else "")
                if x
            )
            parts.append(f"{n} ({tags})" + (f": {_clip(d, param_chars)}" if d else ""))
        line += "\n  Parameters: " + "; ".join(parts)
    return line


def _tool_name(tool: dict) -> str:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str(fn.get("name") or "")


def _tool_desc(tool: dict) -> str:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return _clip(fn.get("description", ""), 200)


NONE_LABEL = "none_of_these"
NONE_DESC = "None of the listed tools can serve this request"
TOOL_NOUL = {
    "false": "No: this request does not need that tool",
    "true": "Yes: that tool is needed to handle the request",
}
ANY_TOOL_NOUL = {
    "false": "No: none of the available tools can serve it",
    "true": "Yes: at least one available tool serves it",
}


def _tool_state(query: str, tools: list[dict], history: str = "") -> str:
    listed = "\n".join(tool_text(t) for t in tools) if tools else "(none)"
    head = f"{history}\n\n" if history else ""
    return f"{head}User request:\n{query.strip()}\n\nAvailable tools:\n{listed}"


def tool_items(query: str, tools: list[dict], called: set[str], rng, history: str = ""):
    """Items asking which tool a request needs (``called``: the gold tool names; empty
    when no listed tool applies): a choice among the tools (+ "none of these"), a yes/no
    on one tool, or a yes/no on "does any tool apply". Rows whose names are not unique
    identifiers, whose gold is not on the list, or that are too long give no item."""
    names = [_tool_name(t) for t in tools]
    if not all(names) or len(set(names)) != len(names) or NONE_LABEL in names:
        return []
    if not called <= set(names) or len(names) + 1 > MAX_MENU:
        return []
    if any(len(n) > 64 or not re.fullmatch(r"[\w.\-]+", n) for n in names):
        return []
    state = _tool_state(query, tools, history)
    if len(state) > MAX_STATE_CHARS:
        return []
    others = [n for n in names if n not in called]
    u = rng.random()
    if (not called and u < 0.2) or (called and u < 0.05):
        question = {
            "type": "noul",
            "instructions": "Can one of the available tools serve this request?",
            "criteria": ANY_TOOL_NOUL,
        }
        return [{"state": state, "question": question, "expected": "yes" if called else "no"}]
    if names and len(called) <= 1 and (u < 0.6 or not others):
        criteria = {n: _tool_desc(t) for n, t in zip(names, tools, strict=True)}
        criteria[NONE_LABEL] = NONE_DESC
        question = {
            "type": "choice",
            "instructions": "Which of the available tools fits this request best?",
            "criteria": criteria,
        }
        return [{"state": state, "question": question,
                 "expected": next(iter(called)) if called else NONE_LABEL}]  # fmt: skip
    if not others and rng.random() < 0.7:  # every listed tool is needed: mostly left out
        return []
    pick_called = bool(called) and (not others or rng.random() < 0.5)
    pool = sorted(called) if pick_called else others
    name = pool[int(rng.integers(len(pool)))]
    question = {
        "type": "noul",
        "instructions": f"Should the assistant invoke `{name}` for this request?",
        "criteria": TOOL_NOUL,
    }
    return [{"state": state, "question": question, "expected": "yes" if pick_called else "no"}]


# --- ShARC (tasksource/sharc = nikhilweee/sharc_modified) ---------------------------------
# {snippet, scenario, question, history ("Q: …\nA: …" lines), label (Yes / No /
# Irrelevant / Clarification needed), follow_up_question, tree_id}. The follow-up question
# is the answer of the "clarification" rows: never shown.

SHARC_LABELS = ("yes", "no", "irrelevant", "ask_follow_up")
SHARC_CRITERIA = {
    "yes": "Yes: by the rule and what the user said, the answer is yes",
    "no": "No: by the rule and what the user said, the answer is no",
    "irrelevant": "The rule text does not address the user's question",
    "ask_follow_up": "More information is needed: ask the user a follow-up question first",
}


def sharc(row, rng, meta):
    names = meta.get("label_names") or ["Yes", "No", "Irrelevant", "Clarification needed"]
    name = names[int(row["label"])] if isinstance(row["label"], int) else str(row["label"])
    mapping = {"yes": "yes", "no": "no", "irrelevant": "irrelevant",
               "clarification needed": "ask_follow_up", "more": "ask_follow_up"}  # fmt: skip
    label = mapping.get(name.strip().lower())
    if label is None or not str(row["snippet"]).strip():
        return []
    parts = [f"Rule text:\n{row['snippet'].strip()}"]
    if str(row.get("scenario") or "").strip():
        parts.append(f"User scenario: {row['scenario'].strip()}")
    parts.append(f"User question: {row['question'].strip()}")
    if str(row.get("history") or "").strip():
        parts.append(f"Follow-up questions already answered:\n{row['history'].strip()}")
    return [
        {
            "state": "\n\n".join(parts),
            "question": {
                "type": "choice",
                "instructions": "Using only the rule text, the scenario and the answers so far, "
                "how should the user's question be answered?",
                "criteria": SHARC_CRITERIA,
            },
            "expected": label,
        }
    ]


# --- When2Call (nvidia/When2Call: train_sft + train_pref only) ----------------------------
# {tools: [JSON str], messages: [{role, content}], (pref) chosen_response,
# rejected_response}. The gold behaviour is read off the assistant reply: a <TOOLCALL>,
# a follow-up question, or a refusal ("unable…"). The test config is derived from BFCL
# (and others) and is never loaded.

W2C_CRITERIA = {
    "call_tool": "Call one of the available tools: the request is clear and a tool serves it",
    "ask_follow_up": "Ask the user for the missing information a tool call needs",
    "cannot_help": "Say it cannot help: no available tool serves the request",
    "answer_directly": "Answer directly from its own knowledge, without any tool",
}
_TOOLCALL = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)
_UNABLE = re.compile(
    r"\b(unable|can(?:no|')t|not able|don't have the (?:capability|ability)|beyond my)\b",
    re.IGNORECASE,
)
_ASK = re.compile(r"\b(could you|can you (?:please )?(?:provide|specify|tell)|please (?:provide"
                  r"|specify)|i(?:'ll| will)? need)\b", re.IGNORECASE)  # fmt: skip


def when2call_category(reply: str) -> str | None:
    """call_tool / ask_follow_up / cannot_help of an assistant reply, or None (unclear)."""
    if "<TOOLCALL>" in reply:
        return "call_tool"
    if "?" in reply and not _UNABLE.search(reply):
        return "ask_follow_up"
    if _UNABLE.search(reply):
        return "cannot_help"
    if _ASK.search(reply):
        return "ask_follow_up"
    return None


def when2call_called(reply: str) -> set[str]:
    m = _TOOLCALL.search(reply)
    if m is None:
        return set()
    try:
        calls = json.loads(m.group(1))
    except ValueError:
        return set()
    return {str(c.get("name")) for c in calls if isinstance(c, dict) and c.get("name")}


def when2call_load(split: str):
    from datasets import load_dataset

    rows, seen = [], set()
    for config in ("train_pref", "train_sft"):
        for r in load_dataset("nvidia/When2Call", config, split="train"):
            msgs = list(r["messages"])
            reply = r.get("chosen_response") if config == "train_pref" else None
            if reply is None:
                if not msgs or msgs[-1]["role"] != "assistant":
                    continue
                reply, msgs = msgs[-1], msgs[:-1]
            if not msgs or msgs[-1]["role"] != "user":
                continue
            key = _norm(msgs[-1]["content"]) + "\x1f" + "".join(sorted(r["tools"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "tools": json.dumps([json.loads(t) for t in r["tools"]], ensure_ascii=False),
                    "messages": json.dumps(msgs, ensure_ascii=False),
                    "reply": reply["content"],
                    "query": msgs[-1]["content"],
                }
            )
    return _dataset(rows)


def when2call(row, rng, meta):
    tools, msgs = json.loads(row["tools"]), json.loads(row["messages"])
    category = when2call_category(row["reply"])
    if category is None:
        return []
    history = "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in msgs[:-1])
    history = f"Conversation so far:\n{history}" if history else ""
    state = _tool_state(msgs[-1]["content"], tools, history)
    if len(state) > MAX_STATE_CHARS:
        return []
    called = when2call_called(row["reply"])
    names = [_tool_name(t) for t in tools]
    if category == "call_tool" and len(names) >= 2 and len(called) == 1 and rng.random() < 0.4:
        return tool_items(msgs[-1]["content"], tools, called, rng, history)[:1] or []
    labels = ["call_tool", "ask_follow_up", "cannot_help"]
    if rng.random() < 0.5:
        labels.append("answer_directly")
    return [
        {
            "state": state,
            "question": {
                "type": "choice",
                "instructions": "Given the available tools, what should the assistant do next?",
                "criteria": {k: W2C_CRITERIA[k] for k in labels},
            },
            "expected": category,
        }
    ]


# --- xLAM function calling 60k (Salesforce, gated) and xlam-irrelevance-7.5k --------------
# {id?, query, answers: JSON [{name, arguments}], tools: JSON [{name, description,
# parameters}]}. The irrelevance rows are xLAM queries whose needed tool was removed
# (answers []). Both hold out the same queries (group = the normalised query).


def _xlam_group(row) -> str:
    return _norm(row["query"])


def xlam(row, rng, meta):
    try:
        tools, answers = json.loads(row["tools"]), json.loads(row["answers"])
    except (TypeError, ValueError):
        return []
    called = {str(a.get("name")) for a in answers if isinstance(a, dict) and a.get("name")}
    if not called and answers:
        return []
    return tool_items(row["query"], tools, called, rng)


def xlam_irrelevance(row, rng, meta):
    try:
        tools, answers = json.loads(row["tools"]), json.loads(row["answers"])
    except (TypeError, ValueError):
        return []
    if answers or not tools:  # no tool at all: the trivial case is left out
        return []
    return tool_items(row["query"], tools, set(), rng)


# --- MetaTool / ToolE (github.com/HowieHwong/MetaTool, MIT) -------------------------------
# dataset/data/all_clean_data.csv (Query, Tool: 20.6k single-tool queries over 199 tools),
# dataset/data/multi_tool_query_golden.json ([{query, tool: [..]}], 497), the awareness
# set dataset/tmp_dataset/Task1.json ([{query, label positive|negative, tool}], 1040), tool
# descriptions in dataset/big_tool_des.json (merged tools) and dataset/plugin_des.json.
# Many tools overlap (ResearchHelper / ResearchFinder): distractors are drawn only among
# tools that share no name token and few description words with the gold.

METATOOL_SHA = "35e81bb7576826e980c80fed8f8c0a2b4a1e6fbb"
METATOOL_FILES = (
    "dataset/data/all_clean_data.csv",
    "dataset/data/multi_tool_query_golden.json",
    "dataset/tmp_dataset/Task1.json",
    "dataset/big_tool_des.json",
    "dataset/plugin_des.json",
)


def _metatool_file(name: str) -> Path:
    import urllib.request

    path = CACHE / f"metatool-{METATOOL_SHA[:7]}" / name
    if not path.exists():
        url = f"https://raw.githubusercontent.com/HowieHwong/MetaTool/{METATOOL_SHA}/{name}"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
            f.write(r.read())
        tmp.replace(path)
    return path


_METATOOL_DESCS: dict[str, str] = {}


def metatool_descs() -> dict[str, str]:
    """Every ToolE tool's description (merged tools override plugin descriptions)."""
    if not _METATOOL_DESCS:
        for name in ("dataset/plugin_des.json", "dataset/big_tool_des.json"):
            _METATOOL_DESCS.update(json.loads(_metatool_file(name).read_text(encoding="utf-8")))
    return _METATOOL_DESCS


def metatool_load(split: str):
    descs = metatool_descs()
    text = _metatool_file("dataset/data/all_clean_data.csv").read_text(encoding="utf-8")
    rows = [
        {"kind": "single", "query": r["Query"], "tools": [r["Tool"]], "label": ""}
        for r in csv.DictReader(io.StringIO(text))
        if r.get("Tool") in descs and str(r.get("Query", "")).strip()
    ]
    path = _metatool_file("dataset/data/multi_tool_query_golden.json")
    rows += [
        {"kind": "multi", "query": r["query"], "tools": list(r["tool"]), "label": ""}
        for r in json.loads(path.read_text(encoding="utf-8"))
        if r.get("tool") and all(t in descs for t in r["tool"])
    ]
    path = _metatool_file("dataset/tmp_dataset/Task1.json")
    rows += [
        {"kind": "aware", "query": r["query"], "tools": [], "label": r["label"]}
        for r in json.loads(path.read_text(encoding="utf-8"))
        if r.get("label") in ("positive", "negative")
    ]
    return _dataset(rows)


_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def _name_tokens(name: str) -> set[str]:
    return {t.lower() for t in _CAMEL.findall(name) if len(t) > 2 and t.lower() != "tool"}


@functools.lru_cache(maxsize=4096)
def _words(text: str) -> frozenset[str]:
    return frozenset(w for w in _norm(text).split() if len(w) > 3)


def dissimilar_tools(gold: list[str], descs: dict[str, str], limit: float = 0.2) -> list[str]:
    """Tools that share no name token with any gold tool and whose description words have
    a Jaccard ≤ ``limit`` with each gold description (sorted)."""
    out = []
    for name in sorted(descs):
        if name in gold:
            continue
        ok = True
        for g in gold:
            a, b = _words(descs[name]), _words(descs[g])
            if _name_tokens(name) & _name_tokens(g) or (a | b and len(a & b) / len(a | b) > limit):
                ok = False
                break
        if ok:
            out.append(name)
    return out


AWARE_CRITERIA = {
    "false": "No: a language model can answer it alone",
    "true": "Yes: it needs an external tool (real-time or external data, special inputs or "
    "outputs, a domain service, or the user's own data)",
}


def metatool(row, rng, meta, descs: dict[str, str] | None = None):
    descs = descs if descs is not None else metatool_descs()
    query = str(row["query"]).strip()
    if row["kind"] == "aware":
        return [
            {
                "state": f"User request:\n{query}",
                "question": {
                    "type": "noul",
                    "instructions": "Does the assistant need an external tool to answer this "
                    "request well?",
                    "criteria": AWARE_CRITERIA,
                },
                "expected": "yes" if row["label"] == "positive" else "no",
                "family": "tool_awareness",
            }
        ]
    gold = list(row["tools"])
    pool = dissimilar_tools(gold, descs)
    if len(pool) < 4:
        return []
    if row["kind"] == "single" and rng.random() < 0.7:
        k = min(int(rng.integers(4, 10)), len(pool))
        menu = gold + [pool[int(i)] for i in rng.choice(len(pool), size=k, replace=False)]
        menu = [menu[int(i)] for i in rng.permutation(len(menu))]
        criteria = {n: _clip(descs[n], 220) for n in menu}
        expected = gold[0]
        if rng.random() < 0.2:  # the gold tool left off the menu: none of these
            del criteria[gold[0]]
            expected = NONE_LABEL
        if rng.random() < 0.5 or expected == NONE_LABEL:
            criteria[NONE_LABEL] = NONE_DESC
        return [
            {
                "state": f"User request:\n{query}",
                "question": {
                    "type": "choice",
                    "instructions": "Which tool should the assistant use for this request?",
                    "criteria": criteria,
                },
                "expected": expected,
                "family": "tool_selection",
            }
        ]
    right = rng.random() < 0.5
    name = gold[int(rng.integers(len(gold)))] if right else pool[int(rng.integers(len(pool)))]
    return [
        {
            "state": f"User request:\n{query}\n\nTool `{name}`: {_clip(descs[name], 300)}",
            "question": {
                "type": "noul",
                "instructions": f"Is `{name}` one of the tools this request needs?",
                "criteria": {
                    "false": "No: this tool does not serve the request",
                    "true": "Yes: the request needs this tool",
                },
            },
            "expected": "yes" if right else "no",
            "family": "tool_selection_multi" if row["kind"] == "multi" else "tool_selection",
        }
    ]


# --- LexGLUE (coastalcph/lex_glue): case_hold and ledgar ---------------------------------


def casehold(row, rng, meta):
    endings = [str(e).strip() for e in row["endings"]]
    label = int(row["label"])
    if len(endings) != 5 or not 0 <= label < 5 or "<HOLDING>" not in row["context"]:
        return []
    if len(set(endings)) != 5:
        return []
    labels = [f"holding_{i + 1}" for i in range(5)]
    return [
        {
            "state": f"Excerpt of a US court decision:\n{row['context'].strip()}",
            "question": {
                "type": "choice",
                "instructions": "Which holding does the case cited at <HOLDING> stand for?",
                "criteria": dict(zip(labels, endings, strict=True)),
            },
            "expected": labels[label],
        }
    ]


LEDGAR_OPTIONS = 8


def _stems(name: str) -> set[str]:
    return {w[:5] for w in _norm(name).split() if len(w) > 2}


def ledgar(row, rng, meta):
    names = meta.get("label_names") or []
    label = int(row["label"])
    if not names or not 0 <= label < len(names):
        return []
    gold = names[label]
    # distractors share no word stem with the gold ("Assigns" / "Assignments")
    pool = [n for n in names if n != gold and not _stems(n) & _stems(gold)]
    picked = [pool[int(i)] for i in rng.choice(len(pool), size=LEDGAR_OPTIONS - 1, replace=False)]
    menu = [gold, *picked]
    menu = [menu[int(i)] for i in rng.permutation(len(menu))]
    criteria = {_slug(n): n for n in menu}
    if len(criteria) != len(menu):
        return []
    return [
        {
            "state": f"Contract provision:\n{row['text'].strip()}",
            "question": {
                "type": "choice",
                "instructions": "What type of provision is this contract clause?",
                "criteria": criteria,
            },
            "expected": _slug(gold),
        }
    ]


# --- CXM Arena French (sprinklr-huggingface/CXM_Arena_French, CC BY-NC 4.0, synthetic) -----
# Agent_Quality_Adherence {conversation, question_answers: python-literal list of
# {Question, Answer Yes|No|NA, ...}}; Intent_Prediction {Complete Conversations,
# Taxonomy_1 "L1:L2"} with Taxonomy/taxonomy_1 {L1, L2, Description}; Tool_Calling
# {conversation_context, true_tools, tool_candidates16} with Tools_Description {Name,
# Definition (base64 JSON)}. The KB-refinement pairs (positives only) and article recall
# (retrieval over 2.4k articles) are not used. One conversation feeds several configs:
# the held-out group is its first customer line.

CXM_REPO = "sprinklr-huggingface/CXM_Arena_French"
CXM_QA = {"yes": "oui", "no": "non", "na": "sans_objet"}
CXM_QA_CRITERIA = {
    "oui": "Oui : l'agent respecte ce critère dans la conversation",
    "non": "Non : l'agent ne le respecte pas",
    "sans_objet": "Sans objet : la situation décrite ne se présente pas dans la conversation",
}
CXM_INTENT_OPTIONS = 8


def _cxm_group(conversation: str) -> str:
    first = next((line for line in str(conversation).splitlines() if line.strip()), "")
    return _norm(re.sub(r"^\s*(customer|client)(\[\d+\])?\s*:", "", first, flags=re.IGNORECASE))


def _literal(text: str):
    import ast

    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        try:
            return json.loads(text)
        except ValueError:
            return None


def cxm_load(split: str):
    from datasets import load_dataset

    def cfg(name, part="train"):
        return load_dataset(CXM_REPO, name, split=part)

    rows = []
    for r in cfg("Agent_Quality_Adherence"):
        qas = _literal(r["question_answers"]) or []
        for k, qa in enumerate(qas):
            if isinstance(qa, dict) and str(qa.get("Answer", "")).strip().lower() in CXM_QA:
                rows.append(
                    {"kind": "quality", "conversation": r["conversation"],
                     "question": str(qa["Question"]), "gold": str(qa["Answer"]).strip().lower(),
                     "extra": "", "group": _cxm_group(r["conversation"]), "key": f"q{r['conversation_id']}-{k}"}
                )  # fmt: skip
    taxonomy = {f"{t['L1']}:{t['L2']}": t["Description"] for t in cfg("Taxonomy", "taxonomy_1")}
    tax = json.dumps(taxonomy, ensure_ascii=False)
    for i, r in enumerate(cfg("Intent_Prediction")):
        if r["Taxonomy_1"] in taxonomy:
            rows.append(
                {"kind": "intent", "conversation": r["Complete Conversations"], "question": "",
                 "gold": r["Taxonomy_1"], "extra": tax,
                 "group": _cxm_group(r["Complete Conversations"]), "key": f"i{i}"}
            )  # fmt: skip
    tools = {}
    for t in cfg("Tools_Description"):
        definition = t["Definition"]
        if isinstance(definition, str):
            try:
                definition = json.loads(base64.b64decode(definition))
            except ValueError:
                try:
                    definition = json.loads(definition)
                except ValueError:
                    continue
        if isinstance(definition, dict):
            tools[t["Name"]] = definition
    for i, r in enumerate(cfg("Tool_Calling")):
        true = _literal(r["true_tools"]) if isinstance(r["true_tools"], str) else r["true_tools"]
        cands = r["tool_candidates16"]
        cands = _literal(cands) if isinstance(cands, str) else cands
        if not true or not cands:
            continue
        defs = [tools[c] for c in cands if c in tools]
        if len(defs) != len(cands):
            continue
        rows.append(
            {"kind": "tool", "conversation": r["conversation_context"], "question": "",
             "gold": json.dumps(sorted(true)), "extra": json.dumps(defs, ensure_ascii=False),
             "group": _cxm_group(r["conversation_context"]), "key": f"t{i}"}
        )  # fmt: skip
    return _dataset(rows)


def cxm(row, rng, meta):
    conversation = str(row["conversation"]).strip()
    if len(conversation) > MAX_STATE_CHARS - 1500:
        return []
    state = f"Conversation avec le service client :\n{conversation}"
    if row["kind"] == "quality":
        return [
            {
                "state": state,
                "question": {
                    "type": "choice",
                    "instructions": "Contrôle qualité de l'agent. Critère (en anglais) : "
                    f"« {row['question'].strip()} » L'agent le respecte-t-il ?",
                    "criteria": CXM_QA_CRITERIA,
                },
                "expected": CXM_QA[row["gold"]],
                "family": "cxm_fr_synthetic_quality",
            }
        ]
    if row["kind"] == "intent":
        taxonomy = json.loads(row["extra"])
        gold = row["gold"]
        others = sorted(k for k in taxonomy if k != gold)
        same = [k for k in others if k.split(":")[0] == gold.split(":")[0]]
        k = CXM_INTENT_OPTIONS - 1
        n_same = min(len(same), int(rng.integers(0, k // 2 + 1)))
        picked = (
            [same[int(i)] for i in rng.choice(len(same), size=n_same, replace=False)]
            if n_same
            else []
        )
        rest = [o for o in others if o not in picked]
        picked += [rest[int(i)] for i in rng.choice(len(rest), size=k - n_same, replace=False)]
        menu = [gold, *picked]
        menu = [menu[int(i)] for i in rng.permutation(len(menu))]
        criteria = {
            f"motif_{_slug(m)}": f"{m.split(':', 1)[1]} — {_clip(taxonomy[m], 200)}" for m in menu
        }
        if len(criteria) != len(menu):
            return []
        return [
            {
                "state": state,
                "question": {
                    "type": "choice",
                    "instructions": "Quel est le motif de contact du client dans cette conversation ?",
                    "criteria": criteria,
                },
                "expected": f"motif_{_slug(gold)}",
                "family": "cxm_fr_synthetic_intent",
            }
        ]
    defs, true = json.loads(row["extra"]), set(json.loads(row["gold"]))
    names = [_tool_name(d) for d in defs]
    if not true <= set(names) or len(set(names)) != len(names) or len(names) > MAX_MENU:
        return []
    if len(true) == 1:
        criteria = {n: _tool_desc(d) for n, d in zip(names, defs, strict=True)}
        question = {
            "type": "choice",
            "instructions": "Quel outil l'agent doit-il appeler maintenant ?",
            "criteria": criteria,
        }
        return [{"state": state, "question": question, "expected": next(iter(true)),
                 "family": "cxm_fr_synthetic_tool"}]  # fmt: skip
    right = rng.random() < 0.5
    pool = sorted(true) if right else [n for n in names if n not in true]
    if not pool:
        return []
    name = pool[int(rng.integers(len(pool)))]
    tool = defs[names.index(name)]
    return [
        {
            "state": f"{state}\n\nOutil `{name}` : {_tool_desc(tool)}",
            "question": {
                "type": "noul",
                "instructions": f"L'agent doit-il appeler l'outil `{name}` à ce stade ?",
                "criteria": {
                    "false": "Non : cet outil n'est pas utile ici",
                    "true": "Oui : cet outil fait partie des appels nécessaires",
                },
            },
            "expected": "yes" if right else "no",
            "family": "cxm_fr_synthetic_tool",
        }
    ]


# --- licences and registry ---------------------------------------------------------------

LICENCES: dict[str, tuple[str, str, str]] = {
    "sharc": (
        "CC BY-SA 3.0",
        "share-alike",
        (
            "ShARC (UCLNLP/sharc card); tasksource/sharc repackages "
            "nikhilweee/sharc_modified, whose cards state no licence"
        ),
    ),
    "when2call": (
        "CC BY 4.0",
        "yes",
        (
            "nvidia/When2Call card ('ready for commercial use'); train_sft + train_pref "
            "only, the BFCL-derived test config is never read"
        ),
    ),
    "xlam": (
        "CC BY 4.0",
        "yes",
        (
            "Salesforce/xlam-function-calling-60k card tag (gated: terms accepted); APIGen "
            "(ToolBench / RapidAPI tools), not derived from BFCL"
        ),
    ),
    "xlam_irrelevance": (
        "CC BY 4.0",
        "yes",
        "MadeAgents/xlam-irrelevance-7.5k card tag; xLAM queries with the needed tool removed",
    ),
    "metatool": (
        "MIT",
        "yes",
        "github.com/HowieHwong/MetaTool LICENSE; queries written by ChatGPT / GPT-4",
    ),
    "casehold": ("CC BY 4.0", "yes", "coastalcph/lex_glue card tag (case_hold)"),
    "ledgar": ("CC BY 4.0", "yes", "coastalcph/lex_glue card tag (ledgar: SEC filings)"),
    "cxm_fr": (
        "CC BY-NC 4.0",
        "NC",
        (
            "sprinklr-huggingface/CXM_Arena_French card; synthetic (LLM-generated "
            "conversations and labels); non-commercial"
        ),
    ),
}  # fmt: skip


def _source(name, repo, config, train, held, adapter, family, lang, cap, *, use="commercial",
            **kw) -> Source:  # fmt: skip
    return Source(
        name, repo, config, train, held, adapter, family, lang, LICENCES[name][0],
        use == "commercial", cap=cap, block=BLOCK, license_use=use, **kw,
    )  # fmt: skip


TOOLS_V7_SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        _source("sharc", "tasksource/sharc", None, "train", "validation", sharc,
                "rule_conversation", "en", 20000),
        _source("when2call", "nvidia/When2Call", "train_sft+train_pref", "train", None,
                when2call, "tool_decision", "en", 15000, load=when2call_load,
                group=lambda r: _norm(r["query"])),
        _source("xlam", "Salesforce/xlam-function-calling-60k", "dataset", "train", None, xlam,
                "tool_selection", "en", 30000, group=_xlam_group),
        _source("xlam_irrelevance", "MadeAgents/xlam-irrelevance-7.5k", None, "train", None,
                xlam_irrelevance, "tool_irrelevance", "en", 7500, group=_xlam_group),
        _source("metatool", "github:HowieHwong/MetaTool/dataset", None, "train", None,
                metatool, "tool_selection", "en", 20000, load=metatool_load,
                group=lambda r: _norm(r["query"])),
        _source("casehold", "coastalcph/lex_glue", "case_hold", "train", "validation",
                casehold, "legal_holding", "en", 10000),
        _source("ledgar", "coastalcph/lex_glue", "ledgar", "train", "validation", ledgar,
                "contract_provision", "en", 10000),
        _source("cxm_fr", CXM_REPO, None, "train", None, cxm, "cxm_fr_synthetic", "fr", 14000,
                use="non-commercial", load=cxm_load, group=lambda r: r["group"]),
    ]
}  # fmt: skip
