"""Real-text seeds (mix-v7): rules read from real documents, questions made by rules,
labels computed by executing the rule.

Each group is contrastive like the template generators (:mod:`.contrastive`): the
verbatim excerpt of a real document is the rule line, a generated case is the fact line,
the twin ``b`` changes one field of the case, and the unknowable twin drops the case
(noul: the label of ``a`` is a fair coin drawn first, so the base rate is exactly 0.5;
choice: the abstain option). Every row keeps the seed text's licence (``license_use``).

=========================  =====================================  ========================
source                     seed text                              licence (as stated)
=========================  =====================================  ========================
``gen7_rt_tau_pairs``      tau-bench airline / retail agent       MIT (repo LICENSE)
                           policies (``wiki.md``, commit 59a200c)
``gen7_rt_sharc_pairs``    gov.uk / US government guidance        CC BY-SA 3.0
                           snippets from ShARC (train / dev)      (UCLNLP/sharc card)
``gen7_rt_kali_fr_pairs``  KALI collective agreements (French):   odc-by (card tag)
                           trial / notice periods by category,
                           délais
``gen7_rt_cold_fr_pairs``  COLD / LEGI French law articles:       CC BY 4.0 (card tag)
                           délais ("dans un délai de N jours à
                           compter de …")
=========================  =====================================  ========================

Held-out splits: tau-bench by rule (the compensation and gift-card rules are held out,
with their own case wording), ShARC by its own split (train snippets for training, dev
snippets held out), KALI by excerpt hash (articles carry no agreement id; identical
excerpts of several agreements share a split), COLD by ``texte_titre`` with the
``cold_law_fr`` adapter's own rule (sha256[:8] % 1000 < 100), so a title held out there
is held out here.

Seeds live under ``CACHE/gen7`` (``tjev.data.sources.base.CACHE``). tau-bench and ShARC
are downloaded on first use (a few MB); KALI and COLD seeds are extracted from the full
datasets by ``python -m tjev.data.sources.gen_v7.realtext build-seeds`` (300 MB / 2.3 GB
downloads: run it on Colab and copy ``kali.jsonl`` / ``cold.jsonl`` into the cache).
A source whose seeds are absent yields nothing (its base returns None).

Date arithmetic is explicit and stated in each state ("Mode de calcul retenu ici …"):
a délai of N days from D expires at the end of D + N (the start day does not count); a
délai of N months expires on the same day of the month N months later, or on that month's
last day when it is shorter (:func:`add_months`); no move to the next working day. A
period of N months starting on day J ends the day before the same day N months later
(weeks / days: J + 7N − 1 / J + N − 1).
"""

from __future__ import annotations

import calendar
import datetime as dt
import functools
import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

from tjev.data.sources.base import CACHE
from tjev.data.sources.gen_v7.core import (
    ABSTAIN_KEY,
    YES_NO,
    Group,
    Line,
    Member,
    V7Source,
    edit_words,
    with_abstain,
)
from tjev.data.sources.generators import _date

SEEDS = CACHE / "gen7"


class SeedsMissingError(RuntimeError):
    """A real-text source's seed files are absent (build or download them first)."""


def seed_dir() -> Path:
    """The seed directory: ``$TJEV_GEN7_SEEDS`` (the tests point it at their fixtures),
    else ``CACHE/gen7``. tau-bench and ShARC are downloaded into the default one only."""
    return Path(os.environ.get("TJEV_GEN7_SEEDS") or SEEDS)


def _may_download() -> bool:
    return not os.environ.get("TJEV_GEN7_SEEDS")


TAU_COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"
TAU_URL = (
    "https://raw.githubusercontent.com/sierra-research/tau-bench/{c}/tau_bench/envs/{e}/wiki.md"
)
SHARC_URL = (
    "https://huggingface.co/datasets/tasksource/sharc/resolve/main/data/{s}-00000-of-00001.parquet"
)

LIC_TAU = "MIT (tau-bench policy text, sierra-research/tau-bench @59a200c)"
LIC_SHARC = "CC BY-SA 3.0 (ShARC snippets of gov.uk / US government guidance)"
LIC_KALI = "ODC-By (KALI collective agreements, Nicolas-BZRD/KALI_opendata; DILA open data)"
LIC_COLD = "CC BY 4.0 (COLD French Law, harvard-lil/cold-french-law; LEGI, Licence Ouverte)"


def _s(split: str) -> str:
    return "train" if split == "train" else "heldout"


def _hash_frac(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 1000


def add_months(d: dt.date, n: int) -> dt.date:
    """The same day of the month ``n`` months later, or that month's last day."""
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    return dt.date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def delay_deadline(start: dt.date, n: int, unit: str) -> dt.date:
    """The last day of a délai of ``n`` ``unit`` counted from ``start`` (start excluded)."""
    if unit.startswith("jour"):
        return start + dt.timedelta(days=n)
    if unit.startswith("semaine"):
        return start + dt.timedelta(days=7 * n)
    return add_months(start, n)


def period_end(start: dt.date, n: int, unit: str) -> dt.date:
    """The last day of a period of ``n`` ``unit`` that starts on ``start`` (included)."""
    if unit.startswith("jour"):
        return start + dt.timedelta(days=n - 1)
    if unit.startswith("semaine"):
        return start + dt.timedelta(days=7 * n - 1)
    return add_months(start, n) - dt.timedelta(days=1)


def _download(url: str, path: Path) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return True
    except OSError:
        return False


def _rand_date(rng, start=dt.date(2016, 1, 1), span=3600) -> dt.date:
    return start + dt.timedelta(days=int(rng.integers(span)))


def _coin_case(rng, draw, solve, want: str, tries: int = 200):
    """A case (facts dict) drawn by ``draw(rng)`` whose label is ``want`` (rejection)."""
    for _ in range(tries):
        f = draw(rng)
        if solve(f) == want:
            return f
    return None


def _flip(rng, facts: dict, mutate, solve, tries: int = 100):
    """``facts`` with one field changed by ``mutate`` so the label changes, or None."""
    want = solve(facts)
    for _ in range(tries):
        f2 = mutate(rng, dict(facts))
        if f2 is not None and solve(f2) not in (None, want):
            return f2
    return None


# ---------------------------------------------------------------------------------------
# tau-bench policies


def tau_policy(env: str) -> dict[str, list[str]]:
    """``{section title: [bullets]}`` of tau-bench's ``env`` policy (cached download)."""
    return _tau_policy(seed_dir(), env)


@functools.cache
def _tau_policy(root: Path, env: str) -> dict[str, list[str]]:
    path = root / "tau" / f"{env}.md"
    if not path.exists() and not (
        _may_download() and _download(TAU_URL.format(c=TAU_COMMIT, e=env), path)
    ):
        raise SeedsMissingError(f"tau-bench {env} policy: {path}")
    sections: dict[str, list[str]] = {"_intro": []}  # _intro: the text lines before "## "
    title = "_intro"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("## ", "### ")):
            title = line.lstrip("#").strip()
            sections.setdefault(title, [])
        elif line.startswith("- "):
            sections.setdefault(title, []).append(line[2:].strip())
        elif line.strip() and not line.startswith("#") and title == "_intro":
            sections["_intro"].append(line.strip())
    return sections


def _bullet(env: str, section: str, starts: str) -> str | None:
    pol = tau_policy(env)
    hits = [b for b in pol.get(section, []) if b.startswith(starts)]
    return hits[0] if len(hits) == 1 else None


TIERS = ("regular", "silver", "gold")
CABINS = ("basic economy", "economy", "business")
# "Checked bag allowance" (Book flight): free bags per passenger, tier × cabin
BAGS = {"regular": (0, 1, 2), "silver": (1, 2, 3), "gold": (2, 3, 3)}
BAG_FEE = 50


def tau_bags_solve(f: dict) -> str | None:
    if "rule" not in f or "tier" not in f:
        return None
    free = BAGS[f["tier"]][CABINS.index(f["cabin"])] * f["pax"]
    return f"${max(0, f['bags'] - free) * BAG_FEE}"


TAU_BAGS_CASE = [
    (
        "Reservation booked by a {tier} member: {pax} passenger{s} in {cabin}, {bags} checked "
        "bag{bs} in total."
    ),
    (
        "Booking user's membership: {tier}. Cabin: {cabin}. Passengers: {pax}. Checked bags "
        "requested: {bags}."
    ),
]


def _tau_head(env: str, section: str) -> Line:
    title = "Airline Agent Policy" if env == "airline" else "Retail agent policy"
    return Line(f"Excerpt from the {title} — {section}:", "head")


def tau_bags(rng, lang, split, srng) -> Group | None:
    if _s(split) != "train":
        return None
    rule = _bullet("airline", "Book flight", "Checked bag allowance")
    if rule is None:
        return None
    tpl = TAU_BAGS_CASE[int(srng.integers(len(TAU_BAGS_CASE)))]

    def draw(r):
        return {"rule": "bags", "tier": TIERS[int(r.integers(3))],
                "cabin": CABINS[int(r.integers(3))], "pax": int(r.integers(1, 6)),
                "bags": int(r.integers(0, 13))}  # fmt: skip

    facts = draw(rng)

    def case_line(f):
        text = tpl.format(
            tier=f["tier"],
            pax=f["pax"],
            s="" if f["pax"] == 1 else "s",
            cabin=f["cabin"],
            bags=f["bags"],
            bs="" if f["bags"] == 1 else "s",
        )
        return Line(text, "case", ("tier", "cabin", "pax", "bags"), (text,))

    def mutate(r, f):
        k = ("tier", "cabin", "bags")[int(r.integers(3))]
        if k == "tier":
            f["tier"] = TIERS[int(r.integers(3))]
        elif k == "cabin":
            f["cabin"] = CABINS[int(r.integers(3))]
        else:
            f["bags"] = max(0, f["bags"] + int(r.choice([-3, -2, -1, 1, 2, 3])))
        return f

    f2 = _flip(rng, facts, mutate, tau_bags_solve)
    if f2 is None:
        return None
    rule_line = Line(rule, "rule", ("rule",), (rule,))
    other = _bullet("airline", "Book flight", "Passengers:")
    lines = [_tau_head("airline", "Book flight")]
    lines += [Line(other, "context")] if other and rng.random() < 0.5 else []
    lines += [rule_line, case_line(facts)]
    lines_b = [*lines[:-1], case_line(f2)]
    vals = {tau_bags_solve(facts), tau_bags_solve(f2)}
    for t in TIERS:  # the amounts other tiers / cabins would give: plausible wrong options
        for c in CABINS:
            if len(vals) < 5:
                vals.add(tau_bags_solve({**facts, "tier": t, "cabin": c}))
    order = sorted(vals, key=lambda v: int(v[1:]))
    key = ABSTAIN_KEY["train"]
    crit = {v: f"The user pays {v} for checked bags" for v in order}
    if rng.random() < 0.5:
        crit = with_abstain(crit, "en", key)
    q = {"type": "choice", "instructions": "Under this policy, how much does the user pay "
         "for the checked bags of this reservation?", "criteria": crit}  # fmt: skip
    a = Member(lines, facts, tau_bags_solve(facts))
    b = Member(lines_b, f2, tau_bags_solve(f2))
    return Group("rt_tau_bags", "rt_tau_bags.train.en.0", "en", q, a, b, ("rule", "case"),
                 {"rule": {"expected": key}, "case": {"expected": key}},
                 license_use=LIC_TAU, extra={"twin": "rule"})  # fmt: skip


NOW = dt.datetime(2024, 5, 15, 15, 0)  # "The current time is 2024-05-15 15:00:00 EST."
REASONS = ("change of plan", "health reasons", "bad weather", "a conflicting meeting")


def tau_cancel_solve(f: dict) -> str | None:
    if "rule" not in f or "cabin" not in f:
        return None
    booked = dt.datetime.fromisoformat(f["booked"])
    ok = (
        NOW - booked <= dt.timedelta(hours=24)
        or f["airline_cancelled"]
        or f["cabin"] == "business"
        or (f["insurance"] and f["reason"] in ("health reasons", "bad weather"))
    )
    return "yes" if ok else "no"


TAU_CANCEL_CASE = [
    (
        "Reservation in {cabin}, booked on {booked} EST; travel insurance: {ins}. The flight is "
        "{status}; the user wants to cancel because of {reason}."
    ),
    (
        "Cabin {cabin}. Booking time: {booked} EST. Insurance bought: {ins}. Flight status: "
        "{status}. Reason given: {reason}."
    ),
]


def tau_cancel(rng, lang, split, srng) -> Group | None:
    if _s(split) != "train":
        return None
    rule = _bullet("airline", "Cancel flight", "All reservations can be cancelled")
    ins_rule = _bullet("airline", "Book flight", "Travel insurance:")
    intro = [b for b in tau_policy("airline").get("_intro", [])
             if b.startswith("The current time")]  # fmt: skip
    if rule is None or ins_rule is None or len(intro) != 1:
        return None
    tpl = TAU_CANCEL_CASE[int(srng.integers(len(TAU_CANCEL_CASE)))]
    want = "yes" if rng.random() < 0.5 else "no"

    def draw(r):
        hours = int(r.choice([2, 6, 20, 23, 25, 30, 48, 72, 200]))
        booked = NOW - dt.timedelta(hours=hours, minutes=int(r.choice([0, 15, 30, 45])))
        return {"rule": "cancel", "cabin": CABINS[int(r.integers(3))],
                "booked": booked.isoformat(sep=" ", timespec="minutes"),
                "insurance": bool(r.integers(2)), "airline_cancelled": r.random() < 0.15,
                "reason": REASONS[int(r.integers(len(REASONS)))]}  # fmt: skip

    facts = _coin_case(rng, draw, tau_cancel_solve, want)
    if facts is None:
        return None

    def case_line(f):
        text = tpl.format(cabin=f["cabin"], booked=f["booked"],
                          ins="yes" if f["insurance"] else "no",
                          status="cancelled by the airline" if f["airline_cancelled"]
                          else "scheduled", reason=f["reason"])  # fmt: skip
        return Line(text, "case", ("cabin", "booked", "insurance", "airline_cancelled", "reason"),
                    (text,))  # fmt: skip

    def mutate(r, f):
        k = int(r.integers(4))
        if k == 0:
            f["cabin"] = CABINS[int(r.integers(3))]
        elif k == 1:
            f["insurance"] = not f["insurance"]
        elif k == 2:
            f["reason"] = REASONS[int(r.integers(len(REASONS)))]
        else:
            f["booked"] = draw(r)["booked"]
        return f

    f2 = _flip(rng, facts, mutate, tau_cancel_solve)
    if f2 is None:
        return None
    lines = [_tau_head("airline", "Cancel flight"), Line(intro[0], "context"),
             Line(ins_rule, "context"), Line(rule, "rule", ("rule",), (rule,)), case_line(facts)]  # fmt: skip
    q = {"type": "noul", "instructions": "Can the agent cancel this reservation under the "
         "policy?", "criteria": YES_NO["en"]}  # fmt: skip
    a = Member(lines, facts, want)
    b = Member([*lines[:-1], case_line(f2)], f2, tau_cancel_solve(f2))
    return Group("rt_tau_cancel", "rt_tau_cancel.train.en.0", "en", q, a, b, ("rule", "case"),
                 {"rule": {"target": {"yes": 0.5, "no": 0.5}}}, license_use=LIC_TAU,
                 extra={"twin": "rule"})  # fmt: skip


STATUSES = ("pending", "processed", "delivered", "cancelled")
ACTIONS = {  # action -> (section, bullet start, status allowed)
    "cancel": ("Cancel pending order", "An order can only be cancelled", "pending"),
    "modify": ("Modify pending order", "An order can only be modified", "pending"),
    "return": ("Return delivered order", "An order can only be returned", "delivered"),
    "exchange": ("Exchange delivered order", "An order can only be exchanged", "delivered"),
}


def tau_status_solve(f: dict) -> str | None:
    if "rule" not in f or "status" not in f:
        return None
    return "yes" if f["status"] == ACTIONS[f["rule"]][2] else "no"


TAU_STATUS_CASE = [
    "Order #W{n} is in status '{status}'. The user asks to {action} it.",
    "Looked-up order #W{n}: status '{status}'. Request: {action} the order.",
]


def tau_status(rng, lang, split, srng) -> Group | None:
    if _s(split) != "train":
        return None
    action = list(ACTIONS)[int(rng.integers(len(ACTIONS)))]
    section, start, allowed = ACTIONS[action]
    rule = _bullet("retail", section, start)
    if rule is None:
        return None
    tpl = TAU_STATUS_CASE[int(srng.integers(len(TAU_STATUS_CASE)))]
    yes = rng.random() < 0.5
    others = [s for s in STATUSES if s != allowed]
    status = allowed if yes else others[int(rng.integers(len(others)))]
    num = int(rng.integers(10**6, 10**7))
    facts = {"rule": action, "status": status}

    def case_line(st):
        text = tpl.format(n=num, status=st, action=action)
        return Line(text, "case", ("status",), (text,))

    s2 = others[int(rng.integers(len(others)))] if yes else allowed
    f2 = {**facts, "status": s2}
    lines = [
        _tau_head("retail", section),
        Line(rule, "rule", ("rule",), (rule,)),
        case_line(status),
    ]  # no other bullet of the section: some restate the status rule
    q = {"type": "noul", "instructions": f"Can the agent {action} this order under the policy?",
         "criteria": YES_NO["en"]}  # fmt: skip
    a = Member(lines, facts, "yes" if yes else "no")
    b = Member([*lines[:-1], case_line(s2)], f2, tau_status_solve(f2))
    return Group("rt_tau_status", "rt_tau_status.train.en.0", "en", q, a, b, ("rule", "case"),
                 {"rule": {"target": {"yes": 0.5, "no": 0.5}}}, license_use=LIC_TAU,
                 extra={"twin": "rule"})  # fmt: skip


def tau_comp_solve(f: dict) -> str | None:
    if "rule" not in f or "tier" not in f:
        return None
    eligible = f["tier"] in ("silver", "gold") or f["insurance"] or f["cabin"] == "business"
    if not eligible or not f["asks"]:
        return "$0"
    if f["complaint"] == "cancelled":
        return f"${100 * f['pax']}"
    return f"${50 * f['pax']}" if f["changes"] else "$0"


TAU_COMP_CASE = [
    (
        "Caller profile — membership {tier}; {pax} passenger{s} in {cabin}; insurance {ins}. "
        "Complaint: flight {complaint}; the caller wants to {plan} the reservation. Compensation "
        "explicitly requested: {asks}."
    ),
]


def tau_compensation(rng, lang, split, srng) -> Group | None:
    if _s(split) != "heldout":
        return None
    bullets = tau_policy("airline").get("Refund", [])
    if len(bullets) != 3:
        return None
    rule = " ".join(bullets)

    def draw(r):
        return {"rule": "comp", "tier": TIERS[int(r.integers(3))],
                "cabin": CABINS[int(r.integers(3))], "insurance": bool(r.integers(2)),
                "pax": int(r.integers(1, 5)),
                "complaint": "cancelled" if r.random() < 0.5 else "delayed",
                "changes": bool(r.integers(2)), "asks": r.random() < 0.85}  # fmt: skip

    facts = draw(rng)

    def case_line(f):
        text = TAU_COMP_CASE[0].format(tier=f["tier"], pax=f["pax"],
                                       s="" if f["pax"] == 1 else "s", cabin=f["cabin"],
                                       ins="bought" if f["insurance"] else "not bought",
                                       complaint=f["complaint"],
                                       plan="change" if f["changes"] else "keep",
                                       asks="yes" if f["asks"] else "no")  # fmt: skip
        return Line(text, "case", ("tier", "cabin", "insurance", "pax", "complaint", "changes",
                                   "asks"), (text,))  # fmt: skip

    def mutate(r, f):
        k = int(r.integers(4))
        if k == 0:
            f["tier"] = TIERS[int(r.integers(3))]
        elif k == 1:
            f["insurance"] = not f["insurance"]
        elif k == 2:
            f["complaint"] = "delayed" if f["complaint"] == "cancelled" else "cancelled"
        else:
            f["pax"] = int(r.integers(1, 5))
        return f

    f2 = _flip(rng, facts, mutate, tau_comp_solve)
    if f2 is None:
        return None
    vals = {"$0", tau_comp_solve(facts), tau_comp_solve(f2)}
    for n in (50, 100):
        vals.add(f"${n * facts['pax']}")
    key = ABSTAIN_KEY["heldout"]
    crit = {v: f"A certificate worth {v}" if v != "$0" else "No certificate" for v in
            sorted(vals, key=lambda v: int(v[1:]))}  # fmt: skip
    if rng.random() < 0.5:
        crit = with_abstain(crit, "en", key)
    lines = [_tau_head("airline", "Refund"), Line(rule, "rule", ("rule",), (rule,)),
             case_line(facts)]  # fmt: skip
    q = {
        "type": "choice",
        "instructions": "Which gesture may the agent offer this caller?",
        "criteria": crit,
    }
    a = Member(lines, facts, tau_comp_solve(facts))
    b = Member([*lines[:-1], case_line(f2)], f2, tau_comp_solve(f2))
    return Group("rt_tau_comp", "rt_tau_comp.heldout.en.0", "en", q, a, b, ("rule", "case"),
                 {"rule": {"expected": key}, "case": {"expected": key}}, license_use=LIC_TAU,
                 extra={"twin": "rule"})  # fmt: skip


def tau_gift_solve(f: dict) -> str | None:
    if "rule" not in f or "balance" not in f:
        return None
    return "yes" if f["balance"] >= f["total"] else "no"


def tau_giftcard(rng, lang, split, srng) -> Group | None:
    if _s(split) != "heldout":
        return None
    rule = _bullet("retail", "Modify payment", "If the user wants the modify the payment method")
    if rule is None:
        return None
    yes = rng.random() < 0.5
    total = int(rng.integers(2000, 300000))  # cents
    gap = int(rng.integers(1, 5000))
    bal = total + gap if yes else total - gap
    if rng.random() < 0.1 and yes:
        bal = total
    facts = {"rule": "gift", "total": total, "balance": max(bal, 1)}

    def case_line(f):
        text = (
            f"Pending order total: ${f['total'] / 100:,.2f}. The user wants to pay with "
            f"gift card {card} instead; its balance is ${f['balance'] / 100:,.2f}."
        )
        return Line(text, "case", ("total", "balance"), (text,))

    card = f"gift_card_{int(rng.integers(10**6, 10**7))}"
    if yes:
        f2 = {**facts, "balance": max(1, total - int(rng.integers(1, 5000)))}
    else:
        f2 = {**facts, "balance": total + int(rng.integers(0, 5000))}
    lines = [_tau_head("retail", "Modify payment"), Line(rule, "rule", ("rule",), (rule,)),
             case_line(facts)]  # fmt: skip
    q = {"type": "noul", "instructions": "Is the payment change to this gift card allowed?",
         "criteria": YES_NO["en"]}  # fmt: skip
    a = Member(lines, facts, tau_gift_solve(facts))
    b = Member([*lines[:-1], case_line(f2)], f2, tau_gift_solve(f2))
    return Group("rt_tau_gift", "rt_tau_gift.heldout.en.0", "en", q, a, b, ("rule", "case"),
                 {"rule": {"target": {"yes": 0.5, "no": 0.5}}}, license_use=LIC_TAU,
                 extra={"twin": "rule"})  # fmt: skip


# ---------------------------------------------------------------------------------------
# ShARC snippets: one numeric threshold (age, weekly hours, an amount in pounds)

SHARC_PATTERNS = {
    # kind -> regex; only requirement wording counts ("must", "to qualify", "if you",
    # "eligible", "only"), within one sentence, and the snippet must hold exactly one match
    "age": re.compile(r"\b(?:must|qualify|eligible|only if|if you(?:'re| are)?)[^.\n]{0,60}?"
                      r"\b(?:aged |be )?(\d{2}) or (?:over|older)\b"),
    "hours": re.compile(r"\b(?:must|qualify|eligible|if you|only|work|working)[^.\n]{0,80}?"
                        r"\b(at least|more than|less than|fewer than) (\d{1,2}) hours "
                        r"(?:a|per|each) week\b"),
    "money": re.compile(r"\b(?:if|must|only|eligible|qualify)[^.\n]{0,80}?\b(less than|more than|"
                        r"at least|up to|over|under) £([\d,]+(?:\.\d\d)?)"
                        r"(?: (?:a|per) (week|month|year))?"),
}  # fmt: skip
OPS = {"or over": ">=", "at least": ">=", "more than": ">", "over": ">", "less than": "<",
       "fewer than": "<", "under": "<", "up to": "<="}  # fmt: skip


def sharc_seeds(split: str) -> list[dict]:
    """Snippets with exactly one threshold of one kind (ShARC train for train, dev for
    held-out)."""
    return _sharc_seeds(seed_dir(), "train" if split == "train" else "validation")


@functools.cache
def _sharc_snippets(root: Path, src: str) -> frozenset[str]:
    jsonl, path = root / "sharc" / f"{src}.jsonl", root / "sharc" / f"{src}.parquet"
    if jsonl.exists():  # test fixtures: {"snippet": ...} rows
        return frozenset(json.loads(x)["snippet"].strip()
                         for x in jsonl.read_text().splitlines() if x)  # fmt: skip
    if not path.exists() and not (_may_download() and _download(SHARC_URL.format(s=src), path)):
        raise SeedsMissingError(f"ShARC {src}: {path}")
    import pandas as pd

    return frozenset(s.strip() for s in pd.read_parquet(path, columns=["snippet"])["snippet"])


@functools.cache
def _sharc_seeds(root: Path, src: str) -> list[dict]:
    snippets = _sharc_snippets(root, src)
    if src == "train":  # ShARC's train and dev share a few snippets: those stay held out
        snippets = snippets - _sharc_snippets(root, "validation")
    snippets = sorted(snippets)
    out = []
    for snip in snippets:
        hits = {k: list(p.finditer(snip)) for k, p in SHARC_PATTERNS.items()}
        found = [(k, m) for k, ms in hits.items() for m in ms]
        if len(found) != 1 or len(snip) > 1500:
            continue
        kind, m = found[0]
        if kind == "age":
            op, value, unit = ">=", int(m.group(1)), ""
            if not 15 <= value <= 80:
                continue
        elif kind == "hours":
            op, value, unit = OPS[m.group(1)], int(m.group(2)), ""
        else:
            op, value = OPS[m.group(1)], float(m.group(2).replace(",", ""))
            unit = m.group(3) or ""
        out.append({"snippet": snip.strip(), "kind": kind, "op": op, "value": value,
                    "unit": unit})  # fmt: skip
    return out


def _cmp(x, op, v) -> bool:
    return {">=": x >= v, ">": x > v, "<": x < v, "<=": x <= v}[op]


def sharc_solve(f: dict) -> str | None:
    if "rule" not in f or "x" not in f:
        return None
    return "yes" if _cmp(f["x"], f["op"], f["value"]) else "no"


SHARC_CASE = {
    "train": {"age": "The applicant is {x} years old.", "hours": "The applicant works {x} hours a week.",
              "money": "The amount concerned is £{x}{per}."},
    "heldout": {"age": "Their age: {x}.", "hours": "Weekly working time: {x} hours.",
                "money": "Figure to check: £{x}{per}."},
}  # fmt: skip
SHARC_Q = {
    "train": {"age": "Does the applicant meet the age requirement in this guidance?",
              "hours": "Does the applicant meet the weekly-hours condition in this guidance?",
              "money": "Does the amount meet the money condition in this guidance?"},
    "heldout": {"age": "Is the age condition of the guidance satisfied?",
                "hours": "Is the hours condition of the guidance satisfied?",
                "money": "Is the amount condition of the guidance satisfied?"},
}  # fmt: skip


def sharc_threshold(rng, lang, split, srng) -> Group | None:
    seeds = sharc_seeds(_s(split))
    if not seeds:
        raise SeedsMissingError("ShARC: no usable snippet")
    seed = seeds[int(rng.integers(len(seeds)))]
    s = _s(split)
    op, v, kind = seed["op"], seed["value"], seed["kind"]
    yes = rng.random() < 0.5

    def value(want_yes: bool):
        for _ in range(100):
            if kind == "money":
                step = max(1.0, round(v * float(rng.choice([0.01, 0.05, 0.2])), 2))
                x = round(v + float(rng.integers(-5, 6)) * step, 2)
            else:
                x = int(v + int(rng.integers(-8, 9)))
            if x > 0 and _cmp(x, op, v) == want_yes:
                return x
        return None

    x, x2 = value(yes), value(not yes)
    if x is None or x2 is None:
        return None
    per = f" a {seed['unit']}" if seed["unit"] else ""

    def case_line(xv):
        shown = f"{xv:,.2f}" if kind == "money" else xv
        text = SHARC_CASE[s][kind].format(x=shown, per=per)
        return Line(text, "case", ("x",), (text,))

    facts = {"rule": seed["kind"], "op": op, "value": v, "x": x}
    snippet = Line(seed["snippet"], "rule", ("rule", "op", "value"), (seed["snippet"],))
    lines = [snippet, case_line(x)]
    q = {"type": "noul", "instructions": SHARC_Q[s][kind], "criteria": YES_NO["en"]}
    a = Member(lines, facts, "yes" if yes else "no")
    f2 = {**facts, "x": x2}
    b = Member([snippet, case_line(x2)], f2, sharc_solve(f2))
    return Group("rt_sharc", f"rt_sharc.{s}.en.{kind}", "en", q, a, b, ("rule", "case"),
                 {t: {"target": {"yes": 0.5, "no": 0.5}} for t in ("rule", "case")},
                 license_use=LIC_SHARC, extra={"twin": "rule" if rng.random() < 0.5 else "case"})  # fmt: skip


# ---------------------------------------------------------------------------------------
# French legal text: délais (KALI, COLD) and periods by category (KALI)

NUM_WORDS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6,
             "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "quinze": 15, "trente": 30}  # fmt: skip
NUM = r"(\d{1,3}|un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|quinze|trente)"
DELAY_RE = re.compile(
    r"dans (?:un|le) délai (?:maximum |maximal |de rigueur )?(?:de |d')"
    + NUM
    + r" (jours|mois)(?! ouvr| franc| calendaires| pleins)(?: au plus tard)?"
    r"( (?:à compter|à partir) (?:de la|de l'|du|des|de) [^.;:,()\d]{4,80}"
    r"| suivant (?:la|le|l') [^.;:,()\d]{4,80})"
)
BAD_ANCHOR = re.compile(r"\b(?:sa|son|ses|leur|leurs|celui|celle|cet|cette|ce|dernier|"
                        r"présent|présente|laquelle|lequel|dont)\b")  # fmt: skip
ANCHOR_END = re.compile(
    r"\s(?:vaut|est|sont|doit|doivent|peut|peuvent|ne|n'|devant|auprès|"
    r"sous peine|selon|et|ou)\b"
)
NOT_CATEGORY = re.compile(r"essai|préavis|période|durée|mois|jours|semaine|ancienneté|contrat|"
                          r"\d")  # fmt: skip
PAIR_A = re.compile(r"pour les ([a-zéèêàâçôîû' -]{3,60}?) ?: ?" + NUM + r" (mois|semaines|jours)")
PAIR_B = re.compile(NUM + r" (mois|semaines|jours) pour les ([a-zéèêàâçôîû' -]{3,60}?)(?=[;,.)])")
KIND_WORDS = re.compile(r"période d'essai|préavis|délai-congé|délai de prévenance")


def _num(token: str) -> int:
    return int(token) if token.isdigit() else NUM_WORDS[token]


SENT_END = re.compile(r"(?<!\b[A-Z])(?<!\bart)(?<!\bArt)(?<!\bn°)\. (?=[A-ZÀ-Ý«(\-–])")


def _sentence_window(text: str, start: int, end: int, limit: int = 900) -> str:
    """The sentences of ``text`` around [start, end), at most ~``limit`` characters
    (sentence ends: ". " before a capital, not after "L." / "art." abbreviations)."""
    ends = [m.end() for m in SENT_END.finditer(text)]
    lo = max([e for e in ends if e <= start], default=0)
    hi = min([e - 1 for e in ends if e - 1 > end], default=len(text))
    for e in ends:  # drop leading sentences while the window is too long
        if hi - lo <= limit or e > start:
            break
        if e > lo:
            lo = e
    return re.sub(r"\s+", " ", text[lo:hi]).strip()


def extract_delay(text: str) -> dict | None:
    """The one délai of an article (``None`` unless exactly one clean match)."""
    ms = list(DELAY_RE.finditer(text))
    if len(ms) != 1:
        return None
    m = ms[0]
    anchor = re.sub(r"\s+", " ", m.group(3)).strip()
    anchor = ANCHOR_END.split(anchor)[0].strip()  # the noun phrase, not the verb after it
    if BAD_ANCHOR.search(anchor) or len(anchor) < 12:
        return None
    n = _num(m.group(1))
    if not 1 <= n <= 365:
        return None
    excerpt = _sentence_window(text, m.start(), m.end())
    if (
        m.group(0) not in excerpt.replace("\n", " ")
        and re.sub(r"\s+", " ", m.group(0)) not in excerpt
    ):
        return None
    return {"kind": "delay", "n": n, "unit": m.group(2), "anchor": anchor, "excerpt": excerpt}


def extract_periods(text: str) -> dict | None:
    """Trial (période d'essai) or notice (préavis) durations by category: ≥ 2 categories,
    each once, no renewal / seniority / exception wording in the window. The kind is the
    last keyword before the list, at most 200 characters before its first duration."""
    for m in KIND_WORDS.finditer(text):
        word = m.group(0)
        if word not in ("période d'essai", "préavis"):
            continue
        i = m.start()
        window = text[i : i + 500]
        found = [
            (mm.start(), mm.group(1).strip(" -"), _num(mm.group(2)), mm.group(3))
            for mm in PAIR_A.finditer(window)
        ]
        found += [(mm.start(), mm.group(3).strip(" -"), _num(mm.group(1)), mm.group(2))
                  for mm in PAIR_B.finditer(window)]  # fmt: skip
        if len(found) < 2:
            continue
        first = min(f[0] for f in found)
        between = window[len(word) : first]
        if first > 200 or KIND_WORDS.search(between):  # another keyword is closer to the list
            continue
        pairs = [(c, n, u) for _, c, n, u in sorted(found)]
        cats = [c for c, _, _ in pairs]
        if len(set(cats)) != len(cats) or len({(n, u) for _, n, u in pairs}) < 2:
            continue
        if re.search(r"renouvel|ancienneté|toutefois|sauf|exception|prolong", window):
            continue
        if any(len(c) < 4 or not re.search(r"[a-z]{4}", c) or NOT_CATEGORY.search(c) for c in cats):
            continue
        end = i + max(f[0] for f in found) + 80
        excerpt = _sentence_window(text, i, min(end, len(text)))
        if not all(c in excerpt for c in cats):
            continue
        return {
            "kind": "essai" if word == "période d'essai" else "preavis",
            "pairs": pairs,
            "excerpt": excerpt,
        }
    return None


DELAY_RULE = (
    "Mode de calcul retenu ici : le jour du point de départ ne compte pas ; un délai "
    "en jours expire à la fin du dernier jour ; un délai en mois expire le même "
    "quantième du dernier mois (ou le dernier jour de ce mois s'il est plus court) ; "
    "aucun report au jour ouvrable suivant."
)
PERIOD_RULE = (
    "Mode de calcul retenu ici : la période commence le jour indiqué ; une durée de "
    "N mois s'achève la veille du même quantième N mois plus tard (ou le dernier jour "
    "du mois s'il est plus court, moins un jour) ; une durée de N semaines ou de N "
    "jours s'achève la veille du jour de début + 7N ou + N."
)
DELAY_CASE = {
    "train": [
        "Point de départ du délai (« {anchor} ») : {d1}. Démarche accomplie le {d2}.",
        (
            "Le point de départ (« {anchor} ») est intervenu le {d1} ; la démarche a été "
            "faite le {d2}."
        ),
    ],
    "heldout": ["Événement de départ (« {anchor} ») : {d1} — acte effectué le {d2}."],
}
DELAY_Q = {"train": "D'après cet article et le mode de calcul indiqué, la démarche a-t-elle "
                    "été accomplie dans le délai ?",
           "heldout": "L'acte respecte-t-il le délai fixé par le texte, selon le mode de "
                      "calcul indiqué ?"}  # fmt: skip


def delay_solve(f: dict) -> str | None:
    if "n" not in f or "d1" not in f:
        return None
    last = delay_deadline(dt.date.fromisoformat(f["d1"]), f["n"], f["unit"])
    return "yes" if dt.date.fromisoformat(f["d2"]) <= last else "no"


def _seed_file(name: str) -> dict[str, list[dict]]:
    return _read_seeds(seed_dir(), name)


@functools.cache
def _read_seeds(root: Path, name: str) -> dict[str, list[dict]]:
    path = root / name
    if not path.exists():
        raise SeedsMissingError(
            f"{name}: {path} (python -m tjev.data.sources.gen_v7.realtext build-seeds, on Colab)"
        )
    out: dict[str, list[dict]] = {"train": [], "heldout": []}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[row["split"]].append(row)
    return out


def _legal_delay(rng, split, srng, file: str, base: str, licence: str, title_of) -> Group | None:
    seeds = _seed_file(file)
    s = _s(split)
    pool = [r for r in seeds.get(s, []) if r["kind"] == "delay"]
    if not pool:
        return None
    seed = pool[int(rng.integers(len(pool)))]
    n, unit = seed["n"], seed["unit"]
    d1 = _rand_date(rng)
    last = delay_deadline(d1, n, unit)
    yes = rng.random() < 0.5

    def d2_for(want_yes):
        if want_yes:
            span = (last - d1).days
            back = (
                int(rng.integers(0, min(span, 10) + 1))
                if rng.random() < 0.7
                else int(rng.integers(0, span))
            )
            return last - dt.timedelta(days=back)
        return last + dt.timedelta(days=int(rng.integers(1, 21)))

    d2, d2b = d2_for(yes), d2_for(not yes)
    style = int(srng.choice([0, 2]))
    tpl = DELAY_CASE[s][int(srng.integers(len(DELAY_CASE[s])))]

    def case_line(dd):
        text = tpl.format(
            anchor=seed["anchor"], d1=_date(d1, "fr", style), d2=_date(dd, "fr", style)
        )
        return Line(text, "case", ("d1", "d2"), (text,))

    facts = {"n": n, "unit": unit, "d1": d1.isoformat(), "d2": d2.isoformat()}
    head = title_of(seed)
    rule = Line(seed["excerpt"], "rule", ("n", "unit"), (seed["excerpt"],))
    lines = ([Line(head, "head")] if head else []) + [rule, Line(DELAY_RULE, "convention"),
                                                     case_line(d2)]  # fmt: skip
    f2 = {**facts, "d2": d2b.isoformat()}
    q = {"type": "noul", "instructions": DELAY_Q[s], "criteria": YES_NO["fr"]}
    a = Member(lines, facts, "yes" if yes else "no")
    b = Member([*lines[:-1], case_line(d2b)], f2, delay_solve(f2))
    return Group(base, f"{base}.{s}.fr.{DELAY_CASE[s].index(tpl)}", "fr", q, a, b,
                 ("rule", "case"), {t: {"target": {"yes": 0.5, "no": 0.5}} for t in ("rule", "case")},
                 license_use=licence,
                 extra={"twin": "rule" if rng.random() < 0.5 else "case", "seed": seed.get("id")})  # fmt: skip


def kali_delay(rng, lang, split, srng):
    return _legal_delay(
        rng,
        split,
        srng,
        "kali.jsonl",
        "rt_kali_delay",
        LIC_KALI,
        lambda s: "Extrait d'une convention collective (KALI) :",
    )


def cold_delay(rng, lang, split, srng):
    def title(seed):
        num = f", article {seed['article_num']}" if seed.get("article_num") else ""
        return (
            f"{seed['titre']}{num} (version en vigueur depuis le "
            f"{_date(dt.date.fromisoformat(seed['debut']), 'fr', 0)}) :"
        )

    return _legal_delay(rng, split, srng, "cold.jsonl", "rt_cold_delay", LIC_COLD, title)


PERIOD_CASE = {
    "essai": {"train": "Salarié de la catégorie « {cat} », embauché le {d1} ; date considérée : {d2}.",
              "heldout": "Embauche le {d1} dans la catégorie « {cat} » — situation examinée au {d2}."},
    "preavis": {"train": "Salarié de la catégorie « {cat} » ; préavis commencé le {d1} ; date "
                         "considérée : {d2}.",
                "heldout": "Préavis débuté le {d1} (catégorie « {cat} ») — situation examinée au {d2}."},
}  # fmt: skip
PERIOD_Q = {
    "essai": {"train": "Selon cet extrait et le mode de calcul indiqué, le salarié est-il encore "
                       "en période d'essai à la date considérée ?",
              "heldout": "La période d'essai court-elle encore à la date examinée ?"},
    "preavis": {"train": "Selon cet extrait et le mode de calcul indiqué, le préavis est-il encore "
                         "en cours à la date considérée ?",
                "heldout": "Le préavis court-il encore à la date examinée ?"},
}  # fmt: skip


def period_solve(f: dict) -> str | None:
    if "pairs" not in f or "cat" not in f:
        return None
    n, unit = next((n, u) for c, n, u in f["pairs"] if c == f["cat"])
    start, day = dt.date.fromisoformat(f["d1"]), dt.date.fromisoformat(f["d2"])
    return "yes" if start <= day <= period_end(start, n, unit) else "no"


def kali_period(rng, lang, split, srng) -> Group | None:
    seeds = _seed_file("kali.jsonl")
    s = _s(split)
    pool = [r for r in seeds.get(s, []) if r["kind"] in ("essai", "preavis")]
    if not pool:
        return None
    seed = pool[int(rng.integers(len(pool)))]
    kind, pairs = seed["kind"], [tuple(p) for p in seed["pairs"]]
    cat = pairs[int(rng.integers(len(pairs)))][0]
    d1 = _rand_date(rng)
    d1 = d1.replace(day=min(d1.day, 28))
    yes = rng.random() < 0.5
    style = int(srng.choice([0, 2]))
    facts = {"pairs": [list(p) for p in pairs], "cat": cat, "d1": d1.isoformat()}

    def d2_for(c, want_yes):
        n, unit = next((n, u) for cc, n, u in pairs if cc == c)
        end = period_end(d1, n, unit)
        if want_yes:
            back = int(rng.integers(0, min((end - d1).days, 10) + 1))
            return end - dt.timedelta(days=back)
        return end + dt.timedelta(days=int(rng.integers(1, 21)))

    d2 = d2_for(cat, yes)
    facts["d2"] = d2.isoformat()
    tpl = PERIOD_CASE[kind][s]

    def case_line(f):
        text = tpl.format(
            cat=f["cat"],
            d1=_date(d1, "fr", style),
            d2=_date(dt.date.fromisoformat(f["d2"]), "fr", style),
        )
        return Line(text, "case", ("cat", "d1", "d2"), (text,))

    f2 = None
    others = [c for c, _, _ in pairs if c != cat]
    if others and rng.random() < 0.4:  # another category, same dates: flips when its period differs
        c2 = others[int(rng.integers(len(others)))]
        cand = {**facts, "cat": c2}
        if (
            period_solve(cand) != period_solve(facts)
            and edit_words(case_line(facts).text, case_line(cand).text) <= 8
        ):
            f2 = cand
    if f2 is None:
        f2 = {**facts, "d2": d2_for(cat, not yes).isoformat()}
    rule = Line(seed["excerpt"], "rule", ("pairs",), (seed["excerpt"],))
    lines = [Line("Extrait d'une convention collective (KALI) :", "head"), rule,
             Line(PERIOD_RULE, "convention"), case_line(facts)]  # fmt: skip
    q = {"type": "noul", "instructions": PERIOD_Q[kind][s], "criteria": YES_NO["fr"]}
    a = Member(lines, facts, period_solve(facts))
    b = Member([*lines[:-1], case_line(f2)], f2, period_solve(f2))
    if a.expected == b.expected or a.expected != ("yes" if yes else "no"):
        return None
    return Group("rt_kali_period", f"rt_kali_period.{s}.fr.{kind}", "fr", q, a, b,
                 ("rule", "case"), {t: {"target": {"yes": 0.5, "no": 0.5}} for t in ("rule", "case")},
                 license_use=LIC_KALI,
                 extra={"twin": "rule" if rng.random() < 0.5 else "case", "seed": seed.get("id")})  # fmt: skip


# ---------------------------------------------------------------------------------------
# Seed extraction from the full datasets (run on Colab: large downloads)


def kali_split(excerpt: str) -> str:
    return "heldout" if _hash_frac(re.sub(r"\W+", " ", excerpt.lower())) < 100 else "train"


def cold_split(titre: str) -> str:
    """cold_law_fr's own group split (tjev.data.mix.group_split on texte_titre)."""
    return "heldout" if _hash_frac(titre) < 100 else "train"


def build_kali_seeds(out: Path, cap: int = 30000) -> dict:
    """Délais and category periods of the KALI articles → ``out`` (JSONL, deduplicated)."""
    from datasets import load_dataset

    ds = load_dataset("Nicolas-BZRD/KALI_opendata", split="train")
    seen, rows, stats = set(), [], {"articles": len(ds), "delay": 0, "essai": 0, "preavis": 0}
    for r in ds:
        text = r["text"] or ""
        for got in (extract_periods(text), extract_delay(text)):
            if got is None:
                continue
            key = re.sub(r"\W+", " ", got["excerpt"].lower())
            if key in seen:
                continue
            seen.add(key)
            got.update(id=r["id"], split=kali_split(got["excerpt"]))
            rows.append(got)
            stats[got["kind"]] += 1
    rows.sort(key=lambda g: hashlib.sha256(g["excerpt"].encode()).hexdigest())
    delays = [g for g in rows if g["kind"] == "delay"][:cap]
    rows = [g for g in rows if g["kind"] != "delay"] + delays
    _write(out, rows)
    stats["kept"] = len(rows)
    return stats


def build_cold_seeds(out: Path, cap: int = 30000) -> dict:
    """Délais of the COLD French Law articles (full CSV, chunked) → ``out``."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    from tjev.data.sources.targeted import COLD_REPO, NO_END, _iso

    path = hf_hub_download(COLD_REPO, "cold-french-law.csv", repo_type="dataset")
    cols = ["article_identifier", "article_num", "article_date_debut", "texte_titre",
            "article_contenu_text"]  # fmt: skip
    seen, rows, stats = set(), [], {"articles": 0, "delay": 0}
    for chunk in pd.read_csv(path, usecols=cols, dtype=str, chunksize=50000):
        chunk = chunk.fillna("")
        stats["articles"] += len(chunk)
        for r in chunk.itertuples(index=False):
            got = extract_delay(r.article_contenu_text)
            debut = _iso(r.article_date_debut)
            if got is None or debut is None or debut.year < 1900 or debut >= NO_END:
                continue
            key = re.sub(r"\W+", " ", got["excerpt"].lower())
            if key in seen or not r.texte_titre.strip():
                continue
            seen.add(key)
            got.update(id=r.article_identifier, titre=r.texte_titre.strip()[:200],
                       article_num=r.article_num.strip(), debut=debut.isoformat(),
                       split=cold_split(r.texte_titre))  # fmt: skip
            rows.append(got)
            stats["delay"] += 1
    rows.sort(key=lambda g: hashlib.sha256(g["excerpt"].encode()).hexdigest())
    _write(out, rows[:cap])
    stats["kept"] = min(cap, len(rows))
    return stats


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------------------------------
# Registry (merged into V7_GENERATORS by the package)

BASES = {
    "rt_tau_bags": tau_bags,
    "rt_tau_cancel": tau_cancel,
    "rt_tau_status": tau_status,
    "rt_tau_comp": tau_compensation,
    "rt_tau_gift": tau_giftcard,
    "rt_sharc": sharc_threshold,
    "rt_kali_delay": kali_delay,
    "rt_kali_period": kali_period,
    "rt_cold_delay": cold_delay,
}
SOLVERS = {
    "rt_tau_bags": tau_bags_solve,
    "rt_tau_cancel": tau_cancel_solve,
    "rt_tau_status": tau_status_solve,
    "rt_tau_comp": tau_comp_solve,
    "rt_tau_gift": tau_gift_solve,
    "rt_sharc": sharc_solve,
    "rt_kali_delay": delay_solve,
    "rt_kali_period": period_solve,
    "rt_cold_delay": delay_solve,
}
FAMILY = {
    "rt_tau_bags": "policy",
    "rt_tau_cancel": "policy",
    "rt_tau_status": "policy",
    "rt_tau_comp": "policy",
    "rt_tau_gift": "policy",
    "rt_sharc": "policy",
    "rt_kali_delay": "temporal",
    "rt_kali_period": "temporal",
    "rt_cold_delay": "temporal",
}
# a group's rows: both members of the pair and the unknowable twin (the case line dropped)
ROLES_RT = ("a", "b", "unk")
SOURCES = {
    "gen7_rt_tau_pairs": V7Source(
        {
            "rt_tau_bags": 1.0,
            "rt_tau_cancel": 1.0,
            "rt_tau_status": 1.0,
            "rt_tau_comp": 1.0,
            "rt_tau_gift": 1.0,
        },
        ROLES_RT,
        "policy",
        (
            "MIT",
            "yes",
            (
                "sierra-research/tau-bench LICENSE @59a200c; policy excerpts from "
                "tau_bench/envs/{airline,retail}/wiki.md; no word 8-gram shared with JevBench public"
            ),
        ),
    ),
    "gen7_rt_sharc_pairs": V7Source(
        {"rt_sharc": 1.0},
        ROLES_RT,
        "policy",
        (
            "CC BY-SA 3.0",
            "share-alike",
            (
                "UCLNLP/sharc card tag (snippets via the "
                "tasksource/sharc parquet mirror); train snippets only for training, dev held out"
            ),
        ),
    ),
    "gen7_rt_kali_fr_pairs": V7Source(
        {"rt_kali_delay": 1.0, "rt_kali_period": 1.0},
        ROLES_RT,
        "temporal",
        (
            "odc-by",
            "yes",
            ("Nicolas-BZRD/KALI_opendata card tag (DILA KALI open data); attribution required"),
        ),
    ),
    "gen7_rt_cold_fr_pairs": V7Source(
        {"rt_cold_delay": 1.0},
        ROLES_RT,
        "temporal",
        ("CC BY 4.0", "yes", "harvard-lil/cold-french-law card tag (LEGI: Licence Ouverte)"),
    ),
}


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Extract the KALI / COLD seeds (large downloads)")
    p.add_argument("cmd", choices=["build-seeds"])
    p.add_argument("--out", default=str(SEEDS))
    p.add_argument("--only", choices=["kali", "cold"], default=None)
    args = p.parse_args(argv)
    out = Path(args.out)
    if args.only in (None, "kali"):
        print("kali", build_kali_seeds(out / "kali.jsonl"), flush=True)
    if args.only in (None, "cold"):
        print("cold", build_cold_seeds(out / "cold.jsonl"), flush=True)


if __name__ == "__main__":
    main()
