"""Contrastive one-fact groups (mix-v7): return window, refund policy (short and long),
schedule conflict, invoice budget, weekday.

Every group has a state ``a`` whose label needs exactly two evidence lines (a rule and a
fact), a twin ``b`` that changes one of them by a minimal edit (≤ 8 words) and flips the
label, the unknowable twin (``a`` without one evidence line) and an auxiliary question
about an intermediate quantity (elapsed days, last day, total, overlap, refund period,
date). The surface text comes from *template families*: train and held-out families are
disjoint tables, so the held-out split is held out by family, not by row.

The decoy lines (a date that is not the rule's anchor, another meeting, another
category's window) carry no fact the label needs.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import replace as dc_replace
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from tjev.data.sources.gen_v7.core import (
    ABSTAIN_KEY,
    YES_NO,
    Group,
    Line,
    Member,
    abstain_key,
    with_abstain,
)
from tjev.data.sources.generators import (
    COMMON_WINDOWS,
    HELDOUT_WINDOWS,
    POLICY_DOMAINS,
    TRAIN_WINDOWS,
    WEEKDAYS,
    _cap,
    _clock,
    _date,
    _long_policy,
    _money,
    _pick,
)

SPLITS = ("train", "heldout")


def _family(srng, table, split, lang, base):
    fams = table[("train" if split == "train" else "heldout"), lang]
    i = int(srng.integers(len(fams)))
    return f"{base}.{'train' if split == 'train' else 'heldout'}.{lang}.{i}", fams[i]


def _iso(d: dt.date) -> str:
    return d.isoformat()


def _d(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _rand_date(rng, start=dt.date(2023, 1, 1), span=1200) -> dt.date:
    return start + dt.timedelta(days=int(rng.integers(span)))


NAMES = {
    ("train", "en"): ["Ms Patel", "Mr Okafor", "Ms Lindqvist", "Mr Romero", "Ms Haddad",
                      "Mr Novak", "Ms Tanaka", "Mr Mensah"],
    ("train", "fr"): ["Mme Lefèvre", "M. Benali", "Mme Rousseau", "M. Moreau", "Mme Nguyen",
                      "M. Girard", "Mme Diallo", "M. Fontaine"],
    ("heldout", "en"): ["Ms Albescu", "Mr Kowalczyk", "Ms Oyelaran", "Mr Haugland"],
    ("heldout", "fr"): ["Mme Peyrac", "M. Ouattara", "Mme Lemaire", "M. Castagne"],
}  # fmt: skip
REF_LINE = {
    ("train", "en"): ["Reference {r}.", "File no. {r}.", "Case {r} — {n}."],
    ("train", "fr"): ["Référence {r}.", "Dossier n° {r}.", "Dossier {r} — {n}."],
    ("heldout", "en"): ["Ticket #{r} ({n})."],
    ("heldout", "fr"): ["Ticket n° {r} ({n})."],
}


def _filler(rng, split, lang) -> Line:
    s = "train" if split == "train" else "heldout"
    text = _pick(rng, REF_LINE[s, lang]).format(
        r=int(rng.integers(10**4, 10**6)), n=_pick(rng, NAMES[s, lang])
    )
    return Line(text, "filler")


def _arrange(rng, evidence: list[Line], extra: list[Line]) -> list[Line]:
    """Evidence lines in their order (rule first 70% of the time), extra lines anywhere."""
    lines = list(evidence) if rng.random() < 0.7 else list(reversed(evidence))
    for line in extra:
        lines.insert(int(rng.integers(len(lines) + 1)), line)
    return lines


def _replace(lines: list[Line], tag: str, new: Line) -> list[Line]:
    """``lines`` with the line tagged ``tag`` replaced (its separator kept)."""
    return [dc_replace(new, pre=line.pre) if line.tag == tag else line for line in lines]


def _tail(template: str, key: str, **fmt) -> str:
    """The template from the word before the ``{key}`` placeholder to its end, formatted:
    the words that state the value in this line ("within 30 days of purchase.")."""
    i = template.index("{" + key + "}")
    j = template.rfind(" ", 0, max(0, i - 1)) + 1
    return template[j:].format(**fmt)


def _span(template: str, keys: tuple[str, ...], **fmt) -> str:
    """The template from its first to its last placeholder among ``keys``, formatted."""
    starts = [template.index("{" + k + "}") for k in keys if "{" + k + "}" in template]
    ends = [template.index("{" + k + "}") + len(k) + 2 for k in keys if "{" + k + "}" in template]
    return template[min(starts) : max(ends)].format(**fmt)


def _numbers_choice(values: list[int], unit: dict, lang: str) -> dict:
    return {str(v): unit[lang].format(v=v) for v in values}


def _distinct(true, candidates, k=4):
    out = [true]
    for c in candidates:
        if c not in out and len(out) < k:
            out.append(c)
    return out


# ---------------------------------------------------------------------------------------
# Return window (noul): a time limit counted from an anchor event, and the two dates

# family: rule ({w}), dates ({b}, {t}) or elapsed ({e}), questions, decoys ({x}),
# aux questions (elapsed, last day)
RW: dict[tuple[str, str], list[dict]] = {
    ("train", "en"): [
        {"rule": ["Items may be returned within {w} days of purchase.",
                  "Our return period is {w} days from the date of purchase.",
                  "You can bring an item back up to {w} days after buying it."],
         "dates": ["Purchased on {b}; returned on {t}.",
                   "The receipt is dated {b} and the customer came back on {t}.",
                   "Bought {b}. Return attempted {t}."],
         "elapsed": "The item is brought back {e} days after it was bought.",
         "q": ["Is the return within the return period?",
               "Can the store accept this return under its time limit?"],
         "decoys": ["The loyalty card was renewed on {x}.", "A price drop was announced on {x}."],
         "aux_e": "How many days after the purchase was the return made?",
         "aux_d": "What was the last day to return the item?"},
        {"rule": ["Subscribers can cancel free of charge within {w} days of signing up.",
                  "The free cancellation period lasts {w} days from sign-up."],
         "dates": ["Signed up on {b}, cancelled on {t}.",
                   "Sign-up date: {b}. Cancellation date: {t}."],
         "elapsed": "The cancellation comes {e} days after sign-up.",
         "q": ["Is the cancellation free of charge?",
               "Does the cancellation fall inside the free period?"],
         "decoys": ["The first invoice was sent on {x}.", "A reminder e-mail went out on {x}."],
         "aux_e": "How many days after signing up did the customer cancel?",
         "aux_d": "What was the last day of the free cancellation period?"},
        {"rule": ["Expense claims must be submitted within {w} days of the expense.",
                  ("Claims are reimbursed only if filed no later than {w} days after the "
                  "expense date.")],
         "dates": ["Expense date: {b}. Claim submitted: {t}.",
                   "The taxi ride was on {b}; the claim reached finance on {t}."],
         "elapsed": "The claim was filed {e} days after the expense.",
         "q": ["Is the claim on time?", "Will the claim be accepted under the time limit?"],
         "decoys": ["The card statement was issued on {x}.",
                    "The trip had been approved on {x}."],
         "aux_e": "How many days passed between the expense and the claim?",
         "aux_d": "What was the last day to submit the claim?"},
        {"rule": ["Loans last {w} days from checkout.",
                  "A borrowed book must be back within {w} days of checkout."],
         "dates": ["Checked out {b}; returned {t}.", "Borrowed on {b} and brought back on {t}."],
         "elapsed": "The book came back {e} days after checkout.",
         "q": ["Was the book returned on time?", "Is the return free of a late fee?"],
         "decoys": ["The library newsletter went out on {x}.",
                    "Another reader reserved the book on {x}."],
         "aux_e": "For how many days was the book out?",
         "aux_d": "What was the due date?"},
        {"rule": ["An appeal must be lodged within {w} days of the decision.",
                  "The deadline for an appeal is {w} days from the date of the decision."],
         "dates": ["Decision dated {b}; appeal lodged on {t}.",
                   "The decision was issued on {b} and the appeal arrived on {t}."],
         "elapsed": "The appeal arrived {e} days after the decision.",
         "q": ["Was the appeal lodged in time?", "Is the appeal within the deadline?"],
         "decoys": ["The hearing had taken place on {x}.",
                    "The applicant asked for a copy of the file on {x}."],
         "aux_e": "How many days after the decision was the appeal lodged?",
         "aux_d": "What was the last day to appeal?"},
        {"rule": ["Tickets can be exchanged up to {w} days after booking.",
                  "Exchanges are allowed for {w} days from the booking date."],
         "dates": ["Booked on {b}; exchange requested on {t}.",
                   "Booking made {b}. Exchange request: {t}."],
         "elapsed": "The exchange is requested {e} days after booking.",
         "q": ["Can the ticket still be exchanged?", "Is the exchange request within the limit?"],
         "decoys": ["The venue confirmed the line-up on {x}.",
                    "A newsletter about the event was sent on {x}."],
         "aux_e": "How many days after booking was the exchange requested?",
         "aux_d": "What was the last day to exchange the ticket?"},
    ],
    ("train", "fr"): [
        {"rule": ["Les articles peuvent être rapportés dans les {w} jours suivant l'achat.",
                  "Notre délai de retour est de {w} jours à compter de la date d'achat.",
                  "Vous pouvez rapporter un article jusqu'à {w} jours après l'avoir acheté."],
         "dates": ["Acheté le {b} ; rapporté le {t}.",
                   "Le ticket est daté du {b} et le client est revenu le {t}.",
                   "Achat : {b}. Retour tenté : {t}."],
         "elapsed": "L'article est rapporté {e} jours après l'achat.",
         "q": ["Le retour est-il dans le délai ?",
               "Le magasin peut-il accepter ce retour au regard du délai ?"],
         "decoys": ["La carte de fidélité a été renouvelée le {x}.",
                    "Une baisse de prix a été annoncée le {x}."],
         "aux_e": "Combien de jours après l'achat le retour a-t-il eu lieu ?",
         "aux_d": "Quel était le dernier jour pour rapporter l'article ?"},
        {"rule": [("Les abonnés peuvent résilier sans frais dans les {w} jours suivant "
                  "l'inscription."),
                  ("La période de résiliation gratuite dure {w} jours à compter de "
                  "l'inscription.")],
         "dates": ["Inscription le {b}, résiliation le {t}.",
                   "Date d'inscription : {b}. Date de résiliation : {t}."],
         "elapsed": "La résiliation intervient {e} jours après l'inscription.",
         "q": ["La résiliation est-elle gratuite ?",
               "La résiliation tombe-t-elle dans la période gratuite ?"],
         "decoys": ["La première facture a été envoyée le {x}.",
                    "Un e-mail de rappel est parti le {x}."],
         "aux_e": "Combien de jours après l'inscription le client a-t-il résilié ?",
         "aux_d": "Quel était le dernier jour de la période de résiliation gratuite ?"},
        {"rule": [("Les notes de frais doivent être déposées dans les {w} jours suivant la "
                  "dépense."),
                  ("Une note de frais n'est remboursée que si elle est déposée au plus tard "
                  "{w} jours après la date de la dépense.")],
         "dates": ["Date de la dépense : {b}. Note déposée : {t}.",
                   ("Le trajet en taxi a eu lieu le {b} ; la note est arrivée à la "
                   "comptabilité le {t}.")],
         "elapsed": "La note a été déposée {e} jours après la dépense.",
         "q": ["La note de frais est-elle dans les temps ?",
               "La note sera-t-elle acceptée au regard du délai ?"],
         "decoys": ["Le relevé de carte a été émis le {x}.",
                    "Le déplacement avait été validé le {x}."],
         "aux_e": "Combien de jours se sont écoulés entre la dépense et la note ?",
         "aux_d": "Quel était le dernier jour pour déposer la note ?"},
        {"rule": ["Un prêt dure {w} jours à compter de l'emprunt.",
                  "Un livre emprunté doit être rendu dans les {w} jours suivant l'emprunt."],
         "dates": ["Emprunté le {b} ; rendu le {t}.",
                   "Emprunt le {b}, retour le {t}."],
         "elapsed": "Le livre est revenu {e} jours après l'emprunt.",
         "q": ["Le livre a-t-il été rendu à temps ?",
               "Le retour échappe-t-il aux pénalités de retard ?"],
         "decoys": ["La lettre d'information de la médiathèque est partie le {x}.",
                    "Un autre lecteur a réservé le livre le {x}."],
         "aux_e": "Pendant combien de jours le livre a-t-il été emprunté ?",
         "aux_d": "Quelle était la date de retour prévue ?"},
        {"rule": ["Un recours doit être formé dans les {w} jours suivant la décision.",
                  "Le délai de recours est de {w} jours à compter de la date de la décision."],
         "dates": ["Décision du {b} ; recours formé le {t}.",
                   "La décision a été rendue le {b} et le recours est arrivé le {t}."],
         "elapsed": "Le recours est arrivé {e} jours après la décision.",
         "q": ["Le recours a-t-il été formé à temps ?", "Le recours est-il dans le délai ?"],
         "decoys": ["L'audience avait eu lieu le {x}.",
                    "Le demandeur a demandé une copie du dossier le {x}."],
         "aux_e": "Combien de jours après la décision le recours a-t-il été formé ?",
         "aux_d": "Quel était le dernier jour pour former un recours ?"},
        {"rule": ["Les billets sont échangeables jusqu'à {w} jours après la réservation.",
                  "Les échanges sont possibles pendant {w} jours à compter de la réservation."],
         "dates": ["Réservé le {b} ; échange demandé le {t}.",
                   "Réservation effectuée le {b}. Demande d'échange : {t}."],
         "elapsed": "L'échange est demandé {e} jours après la réservation.",
         "q": ["Le billet peut-il encore être échangé ?",
               "La demande d'échange respecte-t-elle le délai ?"],
         "decoys": ["La salle a confirmé la programmation le {x}.",
                    "Une lettre d'information sur l'événement a été envoyée le {x}."],
         "aux_e": "Combien de jours après la réservation l'échange a-t-il été demandé ?",
         "aux_d": "Quel était le dernier jour pour échanger le billet ?"},
    ],
    ("heldout", "en"): [
        {"rule": ["Faults must be reported within {w} days of installation.",
                  "The installer accepts fault reports for {w} days after fitting."],
         "dates": ["Installed {b}. Fault reported {t}.",
                   "Fitting completed on {b}; the fault was reported on {t}."],
         "elapsed": "The fault was reported {e} days after installation.",
         "q": ["Was the fault reported inside the installer's period?"],
         "decoys": ["The installer's invoice was paid on {x}."],
         "aux_e": "How many days after installation was the fault reported?",
         "aux_d": "What was the last day to report a fault?"},
        {"rule": ["Incidents must be declared to the insurer within {w} days of occurring.",
                  "Declarations are accepted up to {w} days after the incident."],
         "dates": ["Incident on {b}; declared on {t}.",
                   "The incident happened on {b} and was declared on {t}."],
         "elapsed": "The declaration was made {e} days after the incident.",
         "q": ["Was the declaration made in time?"],
         "decoys": ["The policy had been renewed on {x}."],
         "aux_e": "How many days after the incident was it declared?",
         "aux_d": "What was the last day to declare the incident?"},
    ],
    ("heldout", "fr"): [
        {"rule": ["Les pannes doivent être signalées dans les {w} jours suivant la pose.",
                  ("L'installateur accepte les signalements de panne pendant {w} jours après "
                  "la pose.")],
         "dates": ["Pose le {b}. Panne signalée le {t}.",
                   "Pose terminée le {b} ; la panne a été signalée le {t}."],
         "elapsed": "La panne a été signalée {e} jours après la pose.",
         "q": ["La panne a-t-elle été signalée dans la période de l'installateur ?"],
         "decoys": ["La facture de l'installateur a été réglée le {x}."],
         "aux_e": "Combien de jours après la pose la panne a-t-elle été signalée ?",
         "aux_d": "Quel était le dernier jour pour signaler une panne ?"},
        {"rule": ["Tout sinistre doit être déclaré à l'assureur dans les {w} jours.",
                  "Les déclarations sont acceptées jusqu'à {w} jours après le sinistre."],
         "dates": ["Sinistre le {b} ; déclaré le {t}.",
                   "Le sinistre a eu lieu le {b} et a été déclaré le {t}."],
         "elapsed": "La déclaration a été faite {e} jours après le sinistre.",
         "q": ["La déclaration a-t-elle été faite à temps ?"],
         "decoys": ["Le contrat avait été renouvelé le {x}."],
         "aux_e": "Combien de jours après le sinistre la déclaration a-t-elle été faite ?",
         "aux_d": "Quel était le dernier jour pour déclarer le sinistre ?"},
    ],
}  # fmt: skip
DAYS_UNIT = {"en": "{v} days", "fr": "{v} jours"}


def rw_window_prior(split: str) -> dict[int, float]:
    if split != "train":
        return {w: 1 / len(HELDOUT_WINDOWS) for w in HELDOUT_WINDOWS}
    prior = {w: 0.6 / len(TRAIN_WINDOWS) for w in TRAIN_WINDOWS}
    for w in COMMON_WINDOWS:
        prior[w] += 0.4 / len(COMMON_WINDOWS)
    return prior


def rw_base_rate(split: str) -> float:
    """P(in time) over the family: the window W from its prior, a fair coin for the label of
    ``a``, elapsed − W uniform on −10..0 (in time) or 1..11 (late), all three redrawn when
    elapsed < 1 (which removes in-time offsets of windows under 11 days)."""
    prior = rw_window_prior(split)
    yes = sum(p * 0.5 * min(11, w) / 11 for w, p in prior.items())
    return yes / (yes + 0.5)


def rw_solve(f: dict) -> str | None:
    if "w" not in f:
        return None
    if "e" in f:
        e = f["e"]
    elif "b" in f and "t" in f:
        e = (_d(f["t"]) - _d(f["b"])).days
    else:
        return None
    return "yes" if e <= f["w"] else "no"


def _rw_window(rng, split) -> int:
    if split != "train":
        return int(rng.choice(list(HELDOUT_WINDOWS)))
    if rng.random() < 0.4:
        return int(rng.choice(list(COMMON_WINDOWS)))
    return int(rng.choice(list(TRAIN_WINDOWS)))


def return_window(rng, lang, split, srng) -> Group:
    tid, fam = _family(srng, RW, split, lang, "rw")
    while True:  # rejection on elapsed < 1 (see rw_base_rate)
        w = _rw_window(rng, split)
        yes = rng.random() < 0.5
        d = int(rng.integers(-10, 1)) if yes else int(rng.integers(1, 12))
        if w + d >= 1:
            break
    b = _rand_date(rng)
    e = w + d
    t = b + dt.timedelta(days=e)
    style = int(srng.integers(4))
    use_e = rng.random() < 0.2
    rule_tpl, dates_tpl = _pick(srng, fam["rule"]), _pick(srng, fam["dates"])

    def rule_line(win):
        return Line(rule_tpl.format(w=win), "rule", ("w",), (_tail(rule_tpl, "w", w=win),))

    def dates_line(bd, td):
        if use_e:
            el = (td - bd).days
            return Line(fam["elapsed"].format(e=el), "dates", ("e",),
                        (_tail(fam["elapsed"], "e", e=el),))  # fmt: skip
        bs, ts = _date(bd, lang, style), _date(td, lang, style)
        return Line(dates_tpl.format(b=bs, t=ts), "dates", ("b", "t"), (bs, ts))

    extra = [_filler(rng, split, lang)] if rng.random() < 0.6 else []
    if not use_e and rng.random() < 0.5:
        for _ in range(20):
            x = b + dt.timedelta(days=int(rng.integers(-30, e + 30)))
            if x not in (b, t) and _date(x, lang, style) not in (
                _date(b, lang, style),
                _date(t, lang, style),
            ):
                extra.append(Line(_pick(rng, fam["decoys"]).format(x=_date(x, lang, style)),
                                  "decoy"))  # fmt: skip
                break
    lines = _arrange(rng, [rule_line(w), dates_line(b, t)], extra)
    facts: dict[str, Any] = {"w": w, "e": e} if use_e else {"w": w, "b": _iso(b), "t": _iso(t)}
    a = Member(lines, facts, "yes" if yes else "no")
    # b: one fact moved across the boundary: the end date (in training also the window)
    w2 = None
    if split == "train" and rng.random() < 0.5:
        for _ in range(50):
            cand = e - int(rng.integers(1, 11)) if yes else e + int(rng.integers(0, 11))
            if cand >= 1 and cand not in HELDOUT_WINDOWS and cand != w:
                w2 = cand
                break
    if w2 is not None:
        lines_b = _replace(lines, "rule", rule_line(w2))
        facts_b = {**facts, "w": w2}
    else:
        d2 = int(rng.integers(1, 12)) if yes else int(rng.integers(max(-10, 1 - w), 1))
        t2 = b + dt.timedelta(days=w + d2)
        lines_b = _replace(lines, "dates", dates_line(b, t2))
        facts_b = {**facts, "e": w + d2} if use_e else {**facts, "t": _iso(t2)}
    member_b = Member(lines_b, facts_b, rw_solve(facts_b))
    q = {"type": "noul", "instructions": _pick(srng, fam["q"]), "criteria": YES_NO[lang]}
    # unknowable twins: the family's base rate (not the sampler's conditional P(yes | the
    # facts left), which would reward learning its priors: elapsed ≤ 2 is always in time
    # because no window is under 3 days)
    rate = rw_base_rate(split)
    unknown = {tag: {"target": {"yes": rate, "no": 1 - rate}} for tag in ("rule", "dates")}
    aux = []
    if not use_e:
        if rng.random() < 0.5:
            shortcut = (t.year - b.year) * 360 + (t.month - b.month) * 30 + (t.day - b.day)
            vals = _distinct(e, [v for v in (shortcut, e + 1, e - 1, e + 7, e - 7) if v > 0])
            vals = [vals[i] for i in rng.permutation(len(vals))]
            aux.append({"type": "choice", "instructions": fam["aux_e"],
                        "criteria": _numbers_choice(vals, DAYS_UNIT, lang),
                        "expected": str(e), "quantity": "elapsed"})  # fmt: skip
        else:
            last = b + dt.timedelta(days=w)
            vals = _distinct(last, [last + dt.timedelta(days=k) for k in (-1, 1, 30, -30, 7)])
            vals = [vals[i] for i in rng.permutation(len(vals))]
            aux.append({"type": "choice", "instructions": fam["aux_d"],
                        "criteria": {_iso(v): _date(v, lang, style) for v in vals},
                        "expected": _iso(last), "quantity": "last_day"})  # fmt: skip
    g = Group("rw", tid, lang, q, a, member_b, ("rule", "dates"), unknown, aux)
    g.extra["twin"] = "rule" if rng.random() < 0.7 else "dates"
    return g


# ---------------------------------------------------------------------------------------
# Invoice against a budget (noul): an amount limit and the order lines

INV: dict[tuple[str, str], list[dict]] = {
    ("train", "en"): [
        {"rule": ["The purchase was approved for up to {B}.", "Approved budget for this order: {B}."],
         "q": ["Is the order total within the approved budget?", "Does the order fit the budget?"],
         "dir": "le"},
        {"rule": ["The company card has a single-purchase limit of {B}.",
                  "Card purchases are capped at {B} per transaction."],
         "q": ["Is the purchase within the card limit?", "Can the order go on the company card?"],
         "dir": "le"},
        {"rule": ["The grant covers equipment costs up to {B}.",
                  "Equipment is funded by the grant up to a ceiling of {B}."],
         "q": ["Does the grant cover the whole equipment order?"], "dir": "le"},
        {"rule": ["The gift card holds a balance of {B}.", "Remaining gift card balance: {B}."],
         "q": ["Is the gift card balance enough to pay for the order?"], "dir": "le"},
        {"rule": ["Free delivery applies to orders of at least {B}.",
                  "Orders worth {B} or more ship free of charge."],
         "q": ["Does the order qualify for free delivery?"], "dir": "ge"},
        {"rule": ["A volume rebate is granted on orders totalling {B} or more.",
                  "Customers who spend at least {B} in one order earn the rebate voucher."],
         "q": ["Does this order earn the rebate voucher?"], "dir": "ge"},
    ],
    ("train", "fr"): [
        {"rule": ["L'achat a été validé pour un montant maximal de {B}.",
                  "Budget validé pour cette commande : {B}."],
         "q": ["Le total de la commande respecte-t-il le budget validé ?",
               "La commande tient-elle dans le budget ?"], "dir": "le"},
        {"rule": ["La carte d'entreprise est plafonnée à {B} par achat.",
                  "Les paiements par carte sont limités à {B} par transaction."],
         "q": ["L'achat respecte-t-il le plafond de la carte ?",
               "La commande peut-elle être réglée avec la carte d'entreprise ?"], "dir": "le"},
        {"rule": ["La subvention couvre les dépenses d'équipement jusqu'à {B}.",
                  "L'équipement est financé par la subvention dans la limite de {B}."],
         "q": ["La subvention couvre-t-elle toute la commande d'équipement ?"], "dir": "le"},
        {"rule": ["La carte cadeau présente un solde de {B}.", "Solde restant de la carte cadeau : {B}."],
         "q": ["Le solde de la carte cadeau suffit-il à régler la commande ?"], "dir": "le"},
        {"rule": ["La livraison est offerte à partir de {B} d'achat.",
                  "Les commandes d'au moins {B} sont livrées gratuitement."],
         "q": ["La commande bénéficie-t-elle de la livraison gratuite ?"], "dir": "ge"},
        {"rule": ["Une remise de volume est accordée aux commandes d'au moins {B}.",
                  "Les clients qui dépensent au moins {B} en une commande reçoivent un bon d'achat."],
         "q": ["Cette commande donne-t-elle droit au bon d'achat ?"], "dir": "ge"},
    ],
    ("heldout", "en"): [
        {"rule": ["The reimbursement ceiling for this trip is {B}."],
         "q": ["Is the cost fully reimbursable under the ceiling?"], "dir": "le"},
        {"rule": ["Supplier orders of {B} or more need a second signature."],
         "q": ["Does this supplier order need a second signature?"], "dir": "ge"},
    ],
    ("heldout", "fr"): [
        {"rule": ["Le plafond de remboursement de ce déplacement est de {B}."],
         "q": ["Le coût est-il entièrement remboursable sous ce plafond ?"], "dir": "le"},
        {"rule": ["Les commandes fournisseur de {B} ou plus exigent une double signature."],
         "q": ["Cette commande fournisseur exige-t-elle une double signature ?"], "dir": "ge"},
    ],
}  # fmt: skip
ORDER_LINE = {
    ("train", "en"): [
        "Order: {items}{disc}.",
        "The basket holds {items}{disc}.",
        "Items ordered: {items}{disc}.",
    ],
    ("train", "fr"): [
        "Commande : {items}{disc}.",
        "Le panier contient {items}{disc}.",
        "Articles commandés : {items}{disc}.",
    ],
    ("heldout", "en"): ["Lines on the purchase request: {items}{disc}."],
    ("heldout", "fr"): ["Lignes de la demande d'achat : {items}{disc}."],
}
ITEM_TEXT = {"en": "{q} {n} at {p} each", "fr": "{q} {n} à {p} l'unité"}
AND = {"en": " and ", "fr": " et "}
DISC = {"en": (", at full price", ", with a {d}% discount on the total"),
        "fr": (", au prix plein", ", avec une remise de {d} % sur le total")}  # fmt: skip
GOODS = {
    ("train", "en"): ["chairs", "desks", "monitors", "keyboards", "lamps", "headsets",
                      "printers", "shelves"],
    ("train", "fr"): ["chaises", "bureaux", "écrans", "claviers", "lampes", "casques audio",
                      "imprimantes", "étagères"],
    ("heldout", "en"): ["drills", "ladders", "toolboxes", "workbenches"],
    ("heldout", "fr"): ["perceuses", "échelles", "caisses à outils", "établis"],
}  # fmt: skip


def _cents_total(items, disc: int) -> int:
    gross = sum(q * p for _, q, p in items)
    net = (Decimal(gross) * Decimal(100 - disc) / Decimal(100)).quantize(
        Decimal(1), rounding=ROUND_HALF_UP
    )
    return int(net)


def inv_solve(f: dict) -> str | None:
    if "B" not in f or "items" not in f:
        return None
    total = _cents_total(f["items"], f["disc"])
    ok = total <= f["B"] if f["dir"] == "le" else total >= f["B"]
    return "yes" if ok else "no"


def _round_budget(rng, total: int, side: str) -> int:
    """A round amount (cents) against ``total``: side ge (≥ total), lt (<), le (≤), gt (>)."""
    step = 100 * int(rng.choice([1, 5, 10, 50]))
    k = int(rng.integers(0, 3))
    if side == "ge":
        return -(-total // step) * step + k * step
    if side == "gt":
        return (total // step) * step + (k + 1) * step
    top = (total - 1) // step if side == "lt" else total // step
    out = (top - k) * step
    if out <= 0:
        out = total - 1 if side == "lt" else total
    return out


def invoice_budget(rng, lang, split, srng) -> Group:
    s = "train" if split == "train" else "heldout"
    tid, fam = _family(srng, INV, split, lang, "inv")
    goods = list(rng.choice(GOODS[s, lang], size=int(rng.integers(1, 4)), replace=False))
    items = [[str(n), int(rng.integers(2, 13)),  # ≥ 2: the goods are plural nouns
              int(rng.integers(5, 400)) * 100 + int(rng.choice([0, 50, 99, 25]))]
             for n in goods]  # fmt: skip
    disc = int(rng.choice([0, 0, 5, 10, 15, 20]))
    total = _cents_total(items, disc)
    yes = rng.random() < 0.5
    le = fam["dir"] == "le"
    side = {(True, True): "ge", (True, False): "lt", (False, True): "le", (False, False): "gt"}
    budget = _round_budget(rng, total, side[le, yes])
    rule_tpl = _pick(srng, fam["rule"])
    order_tpl = _pick(srng, ORDER_LINE[s, lang])

    def money(c):
        return _money(c / 100, lang)

    def rule_line(B):
        return Line(rule_tpl.format(B=money(B)), "rule", ("B", "dir"),
                    (_tail(rule_tpl, "B", B=money(B)),))  # fmt: skip

    def order_line(its):
        phrases = [ITEM_TEXT[lang].format(q=q, n=n, p=money(p)) for n, q, p in its]
        dd = DISC[lang][1].format(d=disc) if disc else DISC[lang][0]
        return Line(order_tpl.format(items=AND[lang].join(phrases), disc=dd), "order",
                    ("items", "disc"), tuple(phrases))  # fmt: skip

    extra = [_filler(rng, split, lang)] if rng.random() < 0.6 else []
    lines = _arrange(rng, [rule_line(budget), order_line(items)], extra)
    facts = {"B": budget, "dir": fam["dir"], "items": items, "disc": disc}
    a = Member(lines, facts, inv_solve(facts))
    assert a.expected == ("yes" if yes else "no"), (facts, yes)
    member_b = None
    if rng.random() < 0.5:  # change one quantity, if a change flips the label
        for _ in range(30):
            i = int(rng.integers(len(items)))
            q2 = int(rng.integers(2, 13))
            its = [[n, q2 if j == i else q, p] for j, (n, q, p) in enumerate(items)]
            f2 = {**facts, "items": its}
            if q2 != items[i][1] and inv_solve(f2) != a.expected:
                member_b = Member(_replace(lines, "order", order_line(its)), f2, inv_solve(f2))
                break
    if member_b is None:
        b2 = _round_budget(rng, total, side[le, not yes])
        f2 = {**facts, "B": b2}
        member_b = Member(_replace(lines, "rule", rule_line(b2)), f2, inv_solve(f2))
    q = {"type": "noul", "instructions": _pick(srng, fam["q"]), "criteria": YES_NO[lang]}
    gross = sum(q_ * p for _, q_, p in items)
    first_only = gross - items[0][1] * items[0][2] + _cents_total(items[:1], disc)
    cands = [gross, first_only, total + 100, total - 100, _cents_total(items, min(2 * disc, 90))]
    vals = _distinct(total, [c for c in cands if c > 0])
    vals = [vals[i] for i in rng.permutation(len(vals))]
    aux = [{"type": "choice", "instructions": AUX_TOTAL[s, lang],
            "criteria": {money(v): money(v) for v in vals}, "expected": money(total),
            "quantity": "total"}]  # fmt: skip
    # the order line alone: a's label is a fair coin drawn independently of the order
    unknown = {"rule": {"target": {"yes": 0.5, "no": 0.5}}}
    g = Group("inv", tid, lang, q, a, member_b, ("rule", "order"), unknown, aux)
    g.extra["twin"] = "rule"
    return g


AUX_TOTAL = {
    ("train", "en"): "What does the order come to after any discount?",
    ("train", "fr"): "À combien s'élève la commande après remise éventuelle ?",
    ("heldout", "en"): "What is the net amount of the request?",
    ("heldout", "fr"): "Quel est le montant net de la demande ?",
}


# ---------------------------------------------------------------------------------------
# Schedule conflict (noul): one named booking and the proposal; other bookings are decoys

SC: dict[tuple[str, str], list[dict]] = {
    ("train", "en"): [
        {"event": "The {E} is booked from {s} to {e}.",
         "prop": ["A {P} is proposed from {ps} to {pe}.",
                  "Someone wants a {lp}-minute {P} starting at {ps}."],
         "q": "Does the proposed {P} overlap the {E}?",
         "E": ["budget review", "team stand-up", "client call", "board meeting"],
         "P": ["call", "interview", "demo"]},
        {"event": "Dr Ibsen's {E} runs from {s} to {e}.",
         "prop": ["A patient asks for an appointment from {ps} to {pe}.",
                  "A patient asks for a {lp}-minute appointment at {ps}."],
         "q": "Would the appointment clash with Dr Ibsen's {E}?",
         "E": ["surgery block", "ward round", "home visit"], "P": ["appointment"]},
        {"event": "The pool van is out for the {E} from {s} to {e}.",
         "prop": ["A colleague wants the van from {ps} to {pe}.",
                  "A colleague wants the van for {lp} minutes from {ps}."],
         "q": "Is the van still out for the {E} during the requested slot?",
         "E": ["warehouse run", "site visit", "airport pickup"], "P": ["loan"]},
        {"event": "Court 2 is reserved for the {E} from {s} to {e}.",
         "prop": ["A club member requests court 2 from {ps} to {pe}.",
                  "A club member requests court 2 for {lp} minutes starting {ps}."],
         "q": "Does the request run into the {E}?",
         "E": ["junior lesson", "league match", "coaching session"], "P": ["booking"]},
        {"event": "The microscope is reserved for the {E} from {s} until {e}.",
         "prop": ["A student asks for it from {ps} until {pe}.",
                  "A student asks for it for {lp} minutes from {ps}."],
         "q": "Would the student's slot overlap the {E}?",
         "E": ["calibration run", "imaging session", "maintenance check"], "P": ["slot"]},
    ],
    ("train", "fr"): [
        {"event": "La {E} est prévue de {s} à {e}.",
         "prop": ["Un {P} est proposé de {ps} à {pe}.",
                  "On propose un {P} de {lp} minutes à partir de {ps}."],
         "q": "{Pq} proposé chevauche-t-il la {E} ?",
         "E": ["revue budgétaire", "réunion d'équipe", "réunion du conseil"],
         "P": ["appel", "entretien"], "Pq": {"appel": "L'appel", "entretien": "L'entretien"}},
        {"event": "Le {E} du docteur Ibsen a lieu de {s} à {e}.",
         "prop": ["Un patient demande un rendez-vous de {ps} à {pe}.",
                  "Un patient demande un rendez-vous de {lp} minutes à {ps}."],
         "q": "Le rendez-vous empiéterait-il sur le {E} du docteur Ibsen ?",
         "E": ["bloc opératoire", "tour de service", "créneau de visites"], "P": ["rendez-vous"]},
        {"event": "La camionnette de service est sortie pour la {E} de {s} à {e}.",
         "prop": ["Un collègue veut la camionnette de {ps} à {pe}.",
                  "Un collègue veut la camionnette pendant {lp} minutes à partir de {ps}."],
         "q": "La camionnette est-elle encore prise par la {E} sur le créneau demandé ?",
         "E": ["tournée d'entrepôt", "visite de chantier", "navette aéroport"], "P": ["emprunt"]},
        {"event": "Le court 2 est réservé pour le {E} de {s} à {e}.",
         "prop": ["Un adhérent demande le court 2 de {ps} à {pe}.",
                  "Un adhérent demande le court 2 pendant {lp} minutes dès {ps}."],
         "q": "La demande empiète-t-elle sur le {E} ?",
         "E": ["cours des jeunes", "match de championnat", "stage de tennis"], "P": ["créneau"]},
        {"event": "Le microscope est réservé pour la {E} de {s} jusqu'à {e}.",
         "prop": ["Une étudiante le demande de {ps} jusqu'à {pe}.",
                  "Une étudiante le demande pendant {lp} minutes à partir de {ps}."],
         "q": "Le créneau de l'étudiante chevaucherait-il la {E} ?",
         "E": ["séance d'étalonnage", "séance d'imagerie", "vérification technique"], "P": ["créneau"]},
    ],
    ("heldout", "en"): [
        {"event": "The tower crane is assigned to the {E} between {s} and {e}.",
         "prop": ["The roofing crew wants the crane between {ps} and {pe}.",
                  "The roofing crew needs the crane for {lp} minutes beginning {ps}."],
         "q": "Would the roofing crew's lift collide with the {E}?",
         "E": ["steel delivery", "facade lift"], "P": ["lift"]},
        {"event": "Studio A is booked for the {E} between {s} and {e}.",
         "prop": ["A podcast team asks for studio A between {ps} and {pe}.",
                  "A podcast team asks for studio A for {lp} minutes beginning {ps}."],
         "q": "Does the podcast request collide with the {E}?",
         "E": ["radio broadcast", "voice-over session"], "P": ["request"]},
    ],
    ("heldout", "fr"): [
        {"event": "La grue est affectée à la {E} entre {s} et {e}.",
         "prop": ["L'équipe de couverture veut la grue entre {ps} et {pe}.",
                  "L'équipe de couverture a besoin de la grue pendant {lp} minutes dès {ps}."],
         "q": "Le levage de l'équipe de couverture entrerait-il en collision avec la {E} ?",
         "E": ["livraison d'acier", "pose de façade"], "P": ["levage"]},
        {"event": "Le studio A est réservé pour l'{E} entre {s} et {e}.",
         "prop": ["Une équipe de podcast demande le studio A entre {ps} et {pe}.",
                  "Une équipe de podcast demande le studio A pendant {lp} minutes dès {ps}."],
         "q": "La demande du podcast entre-t-elle en collision avec l'{E} ?",
         "E": ["émission de radio", "enregistrement de voix off"], "P": ["demande"]},
    ],
}  # fmt: skip
SC_OTHER = {
    ("train", "en"): "Also on the schedule: {o} from {s} to {e}.",
    ("train", "fr"): "Également au planning : {o} de {s} à {e}.",
    ("heldout", "en"): "Elsewhere on the rota: {o} between {s} and {e}.",
    ("heldout", "fr"): "Ailleurs au planning : {o} entre {s} et {e}.",
}
SC_OTHER_NAMES = {
    ("train", "en"): ["a supplier visit", "a fire drill", "a lunch talk", "a cleaning slot"],
    ("train", "fr"): ["une visite fournisseur", "un exercice incendie", "une conférence midi",
                      "un créneau de ménage"],
    ("heldout", "en"): ["a safety briefing", "a delivery window"],
    ("heldout", "fr"): ["un briefing sécurité", "une fenêtre de livraison"],
}  # fmt: skip
AUX_OVERLAP = {
    ("train", "en"): "For how many minutes do the two overlap?",
    ("train", "fr"): "Pendant combien de minutes les deux se chevauchent-ils ?",
    ("heldout", "en"): "How many minutes of overlap are there between the two?",
    ("heldout", "fr"): "Combien de minutes de chevauchement y a-t-il entre les deux ?",
}
MIN_UNIT = {"en": "{v} minutes", "fr": "{v} minutes"}


def sc_solve(f: dict) -> str | None:
    if "E" not in f or "P" not in f:
        return None
    (a, b), (c, d) = f["E"], f["P"]
    return "yes" if a < d and c < b else "no"


def schedule_conflict(rng, lang, split, srng) -> Group | None:
    s = "train" if split == "train" else "heldout"
    tid, fam = _family(srng, SC, split, lang, "sc")
    grid = 15 if s == "train" else 5
    lengths = [30, 45, 60, 90, 120] if s == "train" else [50, 75, 100]
    plens = [30, 45, 60, 90] if s == "train" else [20, 40, 80]
    start = int(rng.integers(8 * 60 // grid, 16 * 60 // grid + 1)) * grid
    ev = (start, start + int(rng.choice(lengths)))
    lp = int(rng.choice(plens))
    yes = rng.random() < 0.5

    def proposal(overlap: bool):
        for _ in range(100):  # never the booking's own slot (it would restate the booking)
            if overlap:
                lo, hi = ev[0] - lp + grid, ev[1] - grid
                ps = int(rng.integers(lo // grid, hi // grid + 1)) * grid
            else:
                gap = int(rng.choice([0, grid, 2 * grid, 60]))
                ps = ev[0] - lp - gap if rng.random() < 0.5 else ev[1] + gap
            if (ps, ps + lp) != ev:
                break
        return ps, ps + lp

    pr = proposal(yes)
    style = int(srng.integers(3))
    E, P = _pick(srng, fam["E"]), _pick(srng, fam["P"])
    form = int(srng.integers(len(fam["prop"])))
    prop_tpl = fam["prop"][form]

    def hm(m):
        return _clock(m, lang, style)

    def event_line(e):
        kw = {"E": E, "s": hm(e[0]), "e": hm(e[1])}
        return Line(fam["event"].format(**kw), "event", ("E",),
                    (_span(fam["event"], ("s", "e"), **kw),))  # fmt: skip

    def prop_line(p):
        kw = {"P": P, "ps": hm(p[0]), "pe": hm(p[1]), "lp": p[1] - p[0]}
        return Line(prop_tpl.format(**kw), "proposal", ("P",),
                    (_span(prop_tpl, ("ps", "pe", "lp"), **kw),))  # fmt: skip

    used = {hm(x) for x in (*ev, *pr)}
    extra = []
    for _ in range(int(rng.integers(0, 3))):  # other bookings: they may overlap the proposal
        for _ in range(20):
            o0 = int(rng.integers(8 * 60 // grid, 17 * 60 // grid)) * grid
            other = (o0, o0 + int(rng.choice(lengths)))
            if not {hm(other[0]), hm(other[1])} & used and not (
                other[0] < ev[1] and ev[0] < other[1]
            ):
                used |= {hm(other[0]), hm(other[1])}
                name = _pick(rng, SC_OTHER_NAMES[s, lang])
                extra.append(Line(SC_OTHER[s, lang].format(o=name, s=hm(other[0]),
                                                           e=hm(other[1])), "decoy"))  # fmt: skip
                break
    lines = _arrange(rng, [event_line(ev), prop_line(pr)], extra)
    facts = {"E": list(ev), "P": list(pr)}
    a = Member(lines, facts, "yes" if yes else "no")
    member_b = None
    decoy_times = used - {hm(x) for x in (*ev, *pr)}
    for _ in range(50):
        if rng.random() < 0.6:
            pr2 = proposal(not yes)
            if {hm(pr2[0]), hm(pr2[1])} & decoy_times:
                continue
            f2 = {**facts, "P": list(pr2)}
            member_b = Member(_replace(lines, "proposal", prop_line(pr2)), f2, sc_solve(f2))
        else:  # move the booking's end across the proposal's start (or onto it)
            k = int(rng.integers(1, 4))
            end2 = pr[0] + grid * k if not yes else pr[0] - grid * (k - 1)
            if end2 <= ev[0] or end2 == ev[1] or hm(end2) in decoy_times:
                continue
            f2 = {**facts, "E": [ev[0], end2]}
            member_b = Member(_replace(lines, "event", event_line((ev[0], end2))), f2,
                              sc_solve(f2))  # fmt: skip
        if member_b.expected != a.expected:
            break
    if member_b is None or member_b.expected == a.expected:
        return None  # no flipping edit found (rare): the caller draws again
    pq = fam.get("Pq", {}).get(P, P)
    q = {"type": "noul", "instructions": fam["q"].format(E=E, P=P, Pq=pq),
         "criteria": YES_NO[lang]}  # fmt: skip
    ov = max(0, min(ev[1], pr[1]) - max(ev[0], pr[0]))
    vals = _distinct(ov, [ov + grid, abs(ov - grid), lp, ev[1] - ev[0], 0, 2 * grid])
    vals = [vals[i] for i in rng.permutation(len(vals))]
    aux = [{"type": "choice", "instructions": AUX_OVERLAP[s, lang],
            "criteria": _numbers_choice(vals, MIN_UNIT, lang), "expected": str(ov),
            "quantity": "overlap"}]  # fmt: skip
    unknown = {"proposal": {"target": {"yes": 0.5, "no": 0.5}}}  # a fair coin, independent of E
    g = Group("sc", tid, lang, q, a, member_b, ("event", "proposal"), unknown, aux)
    g.extra["twin"] = "proposal"
    return g


# ---------------------------------------------------------------------------------------
# Weekday (choice): a dated anchor and an offset

WD: dict[tuple[str, str], list[dict]] = {
    ("train", "en"): [
        {"start": "The site survey took place on {wd}, {d}.", "offset": "The report is due {o} days later.",
         "q": "On which weekday is the report due?", "aux": "On what date is the report due?"},
        {"start": "The patient was discharged on {wd}, {d}.",
         "offset": "The follow-up visit is {o} days after discharge.",
         "q": "On which weekday is the follow-up visit?", "aux": "On what date is the follow-up visit?"},
        {"start": "The shipment left the port on {wd}, {d}.", "offset": "It reaches the depot {o} days later.",
         "q": "On which weekday does the shipment reach the depot?",
         "aux": "On what date does the shipment reach the depot?"},
        {"start": "The tenancy started on {wd}, {d}.", "offset": "The first rent review is {o} days in.",
         "q": "On which weekday is the first rent review?", "aux": "On what date is the first rent review?"},
        {"start": "Sowing was done on {wd}, {d}.", "offset": "The crop is ready {o} days after sowing.",
         "q": "On which weekday is the crop ready?", "aux": "On what date is the crop ready?"},
        {"start": "The parcel was dispatched on {wd}, {d}.", "offset": "Customs clearance is completed {o} days after dispatch.",
         "q": "On which weekday is customs clearance completed?", "aux": "On what date is customs clearance completed?"},
        {"start": "The first dose was given on {wd}, {d}.", "offset": "The booster is scheduled {o} days after the first dose.",
         "q": "On which weekday is the booster scheduled?", "aux": "On what date is the booster scheduled?"},
    ],
    ("train", "fr"): [
        {"start": "Le relevé de terrain a eu lieu le {wd} {d}.", "offset": "Le rapport est dû {o} jours plus tard.",
         "q": "Quel jour de la semaine le rapport est-il dû ?", "aux": "À quelle date le rapport est-il dû ?"},
        {"start": "Le patient est sorti de l'hôpital le {wd} {d}.",
         "offset": "La visite de contrôle a lieu {o} jours après la sortie.",
         "q": "Quel jour de la semaine a lieu la visite de contrôle ?",
         "aux": "À quelle date a lieu la visite de contrôle ?"},
        {"start": "La cargaison a quitté le port le {wd} {d}.", "offset": "Elle arrive au dépôt {o} jours plus tard.",
         "q": "Quel jour de la semaine la cargaison arrive-t-elle au dépôt ?",
         "aux": "À quelle date la cargaison arrive-t-elle au dépôt ?"},
        {"start": "Le bail a commencé le {wd} {d}.", "offset": "La première révision du loyer a lieu {o} jours après.",
         "q": "Quel jour de la semaine a lieu la première révision du loyer ?",
         "aux": "À quelle date a lieu la première révision du loyer ?"},
        {"start": "Les semis ont été faits le {wd} {d}.", "offset": "La récolte est prête {o} jours après les semis.",
         "q": "Quel jour de la semaine la récolte est-elle prête ?",
         "aux": "À quelle date la récolte est-elle prête ?"},
        {"start": "Le colis a été expédié le {wd} {d}.", "offset": "Le dédouanement est terminé {o} jours après l'expédition.",
         "q": "Quel jour de la semaine le dédouanement est-il terminé ?", "aux": "À quelle date le dédouanement est-il terminé ?"},
        {"start": "La première dose a été administrée le {wd} {d}.", "offset": "Le rappel est prévu {o} jours après la première dose.",
         "q": "Quel jour de la semaine le rappel est-il prévu ?", "aux": "À quelle date le rappel est-il prévu ?"},
    ],
    ("heldout", "en"): [
        {"start": "The software freeze began on {wd}, {d}.", "offset": "The release ships {o} days after the freeze.",
         "q": "Which weekday does the release ship on?", "aux": "Which date does the release ship on?"},
        {"start": "The vessel was dry-docked on {wd}, {d}.", "offset": "It is refloated {o} days afterwards.",
         "q": "Which weekday is the vessel refloated on?", "aux": "Which date is the vessel refloated on?"},
    ],
    ("heldout", "fr"): [
        {"start": "Le gel du code a commencé le {wd} {d}.", "offset": "La version sort {o} jours après le gel.",
         "q": "Quel jour de la semaine sort la version ?", "aux": "À quelle date sort la version ?"},
        {"start": "Le navire est entré en cale sèche le {wd} {d}.", "offset": "Il est remis à flot {o} jours après.",
         "q": "Quel jour de la semaine le navire est-il remis à flot ?",
         "aux": "À quelle date le navire est-il remis à flot ?"},
    ],
}  # fmt: skip


def wd_solve(f: dict) -> str | None:
    if "d" not in f or "o" not in f:
        return None
    return WEEKDAYS[f["lang"]][(_d(f["d"]) + dt.timedelta(days=f["o"])).weekday()]


def weekday(rng, lang, split, srng) -> Group:
    s = "train" if split == "train" else "heldout"
    tid, fam = _family(srng, WD, split, lang, "wd")
    d0 = _rand_date(rng, dt.date(2022, 1, 1), 2500)
    if s == "train":
        o = int(rng.integers(1, 181))
    else:  # half the training range, half beyond it (as gen_weekday)
        o = int(rng.integers(1, 181) if rng.random() < 0.5 else rng.integers(181, 541))
    style = int(srng.choice([0, 3]))
    names = WEEKDAYS[lang]
    ds = _date(d0, lang, style)

    def offset_line(off):
        return Line(fam["offset"].format(o=off), "offset", ("o",),
                    (_tail(fam["offset"], "o", o=off),))  # fmt: skip

    start = Line(fam["start"].format(wd=names[d0.weekday()], d=ds), "start", ("d",), (ds,))
    extra = [_filler(rng, split, lang)] if rng.random() < 0.5 else []
    lines = _arrange(rng, [start, offset_line(o)], extra)
    facts = {"d": _iso(d0), "o": o, "lang": lang}
    crit = {n: n for n in names}
    if rng.random() < 0.5:
        crit = with_abstain(crit, lang, ABSTAIN_KEY[s])
    a = Member(lines, facts, wd_solve(facts))
    k = int(rng.choice([1, 2, 3, 4, 5, 6, 8, 9, 10]))
    o2 = o + k if rng.random() < 0.5 or o - k < 1 else o - k
    f2 = {**facts, "o": o2}
    member_b = Member(_replace(lines, "offset", offset_line(o2)), f2, wd_solve(f2))
    q = {"type": "choice", "instructions": fam["q"], "criteria": crit}
    target = d0 + dt.timedelta(days=o)
    vals = _distinct(target, [target + dt.timedelta(days=x) for x in (1, -1, 7, -7, 2)])
    vals = [vals[i] for i in rng.permutation(len(vals))]
    aux = [{"type": "choice", "instructions": fam["aux"],
            "criteria": {_iso(v): _date(v, lang, style) for v in vals},
            "expected": _iso(target), "quantity": "date"}]  # fmt: skip
    key = abstain_key(crit) or ABSTAIN_KEY[s]
    unknown = {"start": {"expected": key}, "offset": {"expected": key}}
    g = Group("wd", tid, lang, q, a, member_b, ("start", "offset"), unknown, aux)
    g.extra["twin"] = "offset" if rng.random() < 0.5 else "start"
    return g


# ---------------------------------------------------------------------------------------
# Refund policy (choice), short and long: the category's refund period and the case's
# delivery age decide; the other rules (member extension, escalation limit, final sale)
# are present and set so they do not decide the pair.

RP: dict[tuple[str, str], list[dict]] = {
    ("train", "en"): [
        {"head": "Refund policy:", "window": "Refunds for {d} are possible within {w} days of delivery.",
         "member": "Members get {b} extra days.", "limit": "Refunds over {l} need a supervisor.",
         "final": "Final-sale items are never refunded.",
         "days": "The order was delivered {n} days ago.", "is_member": ["The customer is a member.",
                                                                        "The customer is not a member."],
         "amount": "The refund requested is {a}.", "cat": "The item is in the {d} category.",
         "q": "Applying the refund period for this category, what should the agent do?",
         "crit": {"approve": "Refund now", "deny": "Refuse the refund", "escalate": "Pass to a supervisor"},
         "aux": "How many days does this customer have to ask for a refund?"},
        {"head": "Returns desk rules", "window": "The {d} refund period is {w} days, counted from delivery.",
         "member": "Loyalty members have {b} more days.", "limit": "A supervisor signs off any refund above {l}.",
         "final": "Nothing sold as final sale can be refunded.",
         "days": "Delivery took place {n} days ago.", "is_member": ["They hold a loyalty membership.",
                                                                   "They have no loyalty membership."],
         "amount": "They ask for {a} back.", "cat": "Product category: {d}.",
         "q": "Given the category's refund period, how should the desk handle the request?",
         "crit": {"approve": "Pay the refund", "deny": "Decline", "escalate": "Escalate to a supervisor"},
         "aux": "How long is this customer's refund period, in days?"},
        {"head": "Internal guidance — refunds", "window": "For {d}, customers may claim a refund up to {w} days after delivery.",
         "member": "Club members receive a {b}-day extension.", "limit": "Claims larger than {l} go to a supervisor.",
         "final": "Final-sale goods are excluded.",
         "days": "It has been {n} days since the parcel arrived.", "is_member": ["The buyer is a club member.",
                                                                                "The buyer is not a club member."],
         "amount": "Amount claimed: {a}.", "cat": "The goods are {d}.",
         "q": "Under this category's refund period, what is the right outcome?",
         "crit": {"approve": "Grant the refund", "deny": "Turn it down", "escalate": "Refer to a supervisor"},
         "aux": "How many days in total does this buyer have to claim?"},
        {"head": "Policy summary", "window": "{D}: refundable for {w} days from delivery.",
         "member": "Membership: +{b} days.", "limit": "Above {l}: supervisor approval.",
         "final": "Final sale: no refund.",
         "days": "Days since delivery: {n}.", "is_member": ["Member: yes.", "Member: no."],
         "amount": "Requested: {a}.", "cat": "Category: {d}.",
         "q": "Taking the category's refund period into account, decide the request.",
         "crit": {"approve": "Approve", "deny": "Deny", "escalate": "Escalate"},
         "aux": "What is this customer's refund period in days?"},
        {"head": "Store credit and refunds", "window": "Customers buying {d} have {w} days after delivery to ask for their money back.",
         "member": "Card holders are given an extra {b} days.", "limit": "Any payout above {l} is reviewed by a supervisor first.",
         "final": "Items tagged final sale stay non-refundable.",
         "days": "The goods arrived {n} days ago.", "is_member": ["The customer holds the store card.",
                                                                "The customer does not hold the store card."],
         "amount": "The customer wants {a} refunded.", "cat": "They bought {d}.",
         "q": "Considering the refund deadline for this category, what should happen?",
         "crit": {"approve": "Refund the customer", "deny": "Say no to the refund", "escalate": "Ask a supervisor to review"},
         "aux": "How many days after delivery can this customer still ask for a refund?"},
        {"head": "FAQ — can I get a refund?", "window": "How long do I have? For {d}, {w} days from the day your order is delivered.",
         "member": "Do members get longer? Yes, {b} more days.", "limit": "Is there a cap? Refunds above {l} are checked by a supervisor.",
         "final": "What about final sale? Final-sale items are not refundable.",
         "days": "My order was delivered {n} days ago.", "is_member": ["I am a member.", "I am not a member."],
         "amount": "I would like {a} back.", "cat": "I bought {d}.",
         "q": "Based on the FAQ and the refund period for this category, how should the agent answer the customer?",
         "crit": {"approve": "Confirm the refund", "deny": "Explain that no refund is possible", "escalate": "Forward to a supervisor"},
         "aux": "How many days does this customer have to ask for a refund?"},
    ],
    ("train", "fr"): [
        {"head": "Politique de remboursement :", "window": "Les remboursements pour la catégorie {d} sont possibles dans les {w} jours suivant la livraison.",
         "member": "Les membres ont {b} jours de plus.", "limit": "Tout remboursement de plus de {l} passe par un superviseur.",
         "final": "Les articles en vente finale ne sont jamais remboursés.",
         "days": "La commande a été livrée il y a {n} jours.", "is_member": ["Le client est membre.",
                                                                            "Le client n'est pas membre."],
         "amount": "Le remboursement demandé est de {a}.", "cat": "L'article relève de la catégorie {d}.",
         "q": "En appliquant le délai de remboursement de cette catégorie, que doit faire l'agent ?",
         "crit": {"approve": "Rembourser maintenant", "deny": "Refuser le remboursement", "escalate": "Transmettre à un superviseur"},
         "aux": "De combien de jours ce client dispose-t-il pour demander un remboursement ?"},
        {"head": "Règles du service retours", "window": "Le délai de remboursement de la catégorie {d} est de {w} jours à compter de la livraison.",
         "member": "Les adhérents du programme de fidélité ont {b} jours de plus.", "limit": "Un superviseur valide tout remboursement supérieur à {l}.",
         "final": "Rien de ce qui est vendu en vente finale n'est remboursable.",
         "days": "La livraison a eu lieu il y a {n} jours.", "is_member": ["Il est adhérent du programme de fidélité.",
                                                                          "Il n'est pas adhérent du programme de fidélité."],
         "amount": "Il demande {a}.", "cat": "Catégorie du produit : {d}.",
         "q": "Compte tenu du délai de cette catégorie, comment le service doit-il traiter la demande ?",
         "crit": {"approve": "Verser le remboursement", "deny": "Refuser", "escalate": "Faire remonter à un superviseur"},
         "aux": "Quelle est la durée du délai de remboursement de ce client, en jours ?"},
        {"head": "Consignes internes — remboursements", "window": "Pour la catégorie {d}, le client peut demander un remboursement jusqu'à {w} jours après la livraison.",
         "member": "Les membres du club bénéficient d'une prolongation de {b} jours.", "limit": "Les demandes supérieures à {l} vont à un superviseur.",
         "final": "Les produits en vente finale sont exclus.",
         "days": "Le colis est arrivé il y a {n} jours.", "is_member": ["L'acheteur est membre du club.",
                                                                       "L'acheteur n'est pas membre du club."],
         "amount": "Montant demandé : {a}.", "cat": "Le produit relève de la catégorie {d}.",
         "q": "Selon le délai de remboursement de cette catégorie, quelle est la bonne issue ?",
         "crit": {"approve": "Accorder le remboursement", "deny": "Ne pas donner suite", "escalate": "Renvoyer vers un superviseur"},
         "aux": "De combien de jours au total cet acheteur dispose-t-il ?"},
        {"head": "Politique en bref", "window": "Catégorie {d} : remboursable {w} jours après la livraison.",
         "member": "Adhésion : +{b} jours.", "limit": "Au-delà de {l} : accord d'un superviseur.",
         "final": "Vente finale : aucun remboursement.",
         "days": "Jours depuis la livraison : {n}.", "is_member": ["Membre : oui.", "Membre : non."],
         "amount": "Demandé : {a}.", "cat": "Catégorie : {d}.",
         "q": "En tenant compte du délai de la catégorie, tranchez la demande.",
         "crit": {"approve": "Accepter", "deny": "Refuser", "escalate": "Faire valider"},
         "aux": "Quel est le délai de remboursement de ce client, en jours ?"},
        {"head": "Avoirs et remboursements", "window": "Les clients qui achètent dans la catégorie {d} ont {w} jours après la livraison pour se faire rembourser.",
         "member": "Les titulaires de la carte du magasin ont {b} jours de plus.", "limit": "Tout versement supérieur à {l} est d'abord examiné par un superviseur.",
         "final": "Les articles étiquetés vente finale restent non remboursables.",
         "days": "La marchandise est arrivée il y a {n} jours.", "is_member": ["Le client a la carte du magasin.",
                                                                            "Le client n'a pas la carte du magasin."],
         "amount": "Le client veut être remboursé de {a}.", "cat": "Il a acheté dans la catégorie {d}.",
         "q": "Compte tenu de la date limite de remboursement de cette catégorie, que faut-il faire ?",
         "crit": {"approve": "Rembourser le client", "deny": "Refuser le remboursement", "escalate": "Demander l'avis d'un superviseur"},
         "aux": "Combien de jours après la livraison ce client peut-il encore demander un remboursement ?"},
        {"head": "FAQ — puis-je être remboursé ?", "window": "De combien de temps est-ce que je dispose ? Pour la catégorie {d}, {w} jours à partir du jour de livraison.",
         "member": "Les membres ont-ils plus de temps ? Oui, {b} jours de plus.", "limit": "Y a-t-il un plafond ? Les remboursements de plus de {l} sont vérifiés par un superviseur.",
         "final": "Et la vente finale ? Les articles en vente finale ne sont pas remboursables.",
         "days": "Ma commande a été livrée il y a {n} jours.", "is_member": ["Je suis membre.", "Je ne suis pas membre."],
         "amount": "Je voudrais récupérer {a}.", "cat": "J'ai acheté dans la catégorie {d}.",
         "q": "D'après la FAQ et le délai de remboursement de cette catégorie, que doit répondre l'agent au client ?",
         "crit": {"approve": "Confirmer le remboursement", "deny": "Expliquer qu'aucun remboursement n'est possible", "escalate": "Transférer à un superviseur"},
         "aux": "De combien de jours ce client dispose-t-il pour demander un remboursement ?"},
    ],
    ("heldout", "en"): [
        {"head": "Aftersales terms", "window": "Goods in the {d} range can be handed back for a refund within {w} days of handover.",
         "member": "Subscribers to the loyalty scheme enjoy {b} additional days.", "limit": "Claims worth more than {l} require a manager.",
         "final": "Clearance lines sold as final are not covered.",
         "days": "Handover was {n} days ago.", "is_member": ["The purchaser subscribes to the loyalty scheme.",
                                                            "The purchaser does not subscribe to the loyalty scheme."],
         "amount": "The claim is for {a}.", "cat": "The goods belong to the {d} range.",
         "q": "In light of the refund window for this range, what is the appropriate response?",
         "crit": {"approve": "Honour the claim", "deny": "Reject the claim", "escalate": "Refer to a manager"},
         "aux": "How many days does this purchaser have for a refund claim?"},
        {"head": "Merchant returns charter", "window": "Purchases from the {d} department may be refunded if claimed no later than {w} days after receipt.",
         "member": "Holders of a premium account are allowed {b} more days.", "limit": "A claim exceeding {l} must be countersigned by a manager.",
         "final": "Goods sold as seen are never refunded.",
         "days": "Receipt of the goods: {n} days ago.", "is_member": ["The client has a premium account.",
                                                                    "The client has no premium account."],
         "amount": "Sum claimed: {a}.", "cat": "Department: {d}.",
         "q": "Given the refund deadline of this department, what should be done?",
         "crit": {"approve": "Pay out the claim", "deny": "Refuse the claim", "escalate": "Get a manager's countersignature"},
         "aux": "Within how many days of receipt may this client claim?"},
    ],
    ("heldout", "fr"): [
        {"head": "Conditions après-vente", "window": "Les produits de la gamme {d} peuvent être repris contre remboursement dans les {w} jours suivant la remise.",
         "member": "Les abonnés au programme de fidélité disposent de {b} jours supplémentaires.", "limit": "Les demandes d'un montant supérieur à {l} exigent un responsable.",
         "final": "Les lots de déstockage vendus comme définitifs ne sont pas couverts.",
         "days": "La remise date d'il y a {n} jours.", "is_member": ["L'acheteur est abonné au programme de fidélité.",
                                                                    "L'acheteur n'est pas abonné au programme de fidélité."],
         "amount": "La demande porte sur {a}.", "cat": "Les produits relèvent de la gamme {d}.",
         "q": "Au vu du délai de remboursement de cette gamme, quelle suite donner ?",
         "crit": {"approve": "Honorer la demande", "deny": "Rejeter la demande", "escalate": "Soumettre à un responsable"},
         "aux": "De combien de jours cet acheteur dispose-t-il pour demander un remboursement ?"},
        {"head": "Charte des retours marchands", "window": "Les achats du rayon {d} peuvent être remboursés s'ils sont réclamés au plus tard {w} jours après réception.",
         "member": "Les titulaires d'un compte premium disposent de {b} jours de plus.", "limit": "Une réclamation de plus de {l} doit être contresignée par un responsable.",
         "final": "Les produits vendus en l'état ne sont jamais remboursés.",
         "days": "Réception des produits : il y a {n} jours.", "is_member": ["Le client a un compte premium.",
                                                                          "Le client n'a pas de compte premium."],
         "amount": "Somme réclamée : {a}.", "cat": "Rayon : {d}.",
         "q": "Au vu du délai de remboursement de ce rayon, que faut-il faire ?",
         "crit": {"approve": "Verser la somme réclamée", "deny": "Refuser la réclamation", "escalate": "Obtenir le contreseing d'un responsable"},
         "aux": "Dans combien de jours après réception ce client peut-il réclamer ?"},
    ],
}  # fmt: skip
RP_WINDOWS = {"train": [10, 14, 15, 28, 30, 31, 40, 45, 50, 90], "heldout": [21, 35, 60]}
RP_BONUS = {"train": [7, 10, 15, 30], "heldout": [14, 20]}
RP_LIMIT = {"train": [200, 300, 500, 1000], "heldout": [250, 750]}


def rp_solve(f: dict) -> str | None:
    need = ("w", "n", "member", "bonus", "limit", "amount")
    if any(k not in f for k in need):
        return None
    allowed = f["w"] + (f["bonus"] if f["member"] else 0)
    if f["n"] > allowed:
        return "deny"
    return "escalate" if f["amount"] > f["limit"] else "approve"


def refund_policy(rng, lang, split, srng, long: bool = False) -> Group:
    s = "train" if split == "train" else "heldout"
    base = "lrp" if long else "rp"
    tid, fam = _family(srng, RP, split, lang, base)
    li = 0 if lang == "en" else 1
    domain = POLICY_DOMAINS[s][int(rng.integers(len(POLICY_DOMAINS[s])))][li]
    w = int(rng.choice(RP_WINDOWS[s]))
    bonus = int(rng.choice(RP_BONUS[s]))
    limit = int(rng.choice(RP_LIMIT[s]))
    member = bool(rng.integers(2))
    extension = bonus if member else 0
    allowed = w + extension
    inside = rng.random() < 0.5

    def days_for(inside_):
        if inside_:
            return int(rng.integers(max(1, allowed - 12), allowed + 1))
        return int(rng.integers(allowed + 1, allowed + 14))

    n = days_for(inside)
    if rng.random() < 0.7:
        amount = float(rng.integers(20, int(limit * 0.9)))
    else:
        amount = float(rng.integers(int(limit * 1.1), 2 * limit))
    window_tpl = fam["window"]

    def window_line(win):
        text = window_tpl.format(d=domain, D=_cap(domain), w=win)
        surface = text if long else _tail(window_tpl, "w", w=win, d=domain, D=_cap(domain))
        return Line(text, "rule", ("w",), (surface,))

    def days_line(nd):
        text = fam["days"].format(n=nd)
        return Line(text, "days", ("n",), (text,))

    rules = [Line(fam["member"].format(b=bonus), "member_rule", ("bonus",)),
             Line(fam["limit"].format(l=_money(limit, lang)), "limit_rule", ("limit",)),
             Line(fam["final"], "final_rule")]  # fmt: skip
    case = [Line(fam["cat"].format(d=domain), "category"),
            Line(fam["is_member"][0 if member else 1], "membership", ("member",)),
            Line(fam["amount"].format(a=_money(amount, lang)), "amount", ("amount",))]  # fmt: skip
    case = [case[i] for i in rng.permutation(len(case))]
    case.insert(int(rng.integers(len(case) + 1)), days_line(n))
    head = Line(fam["head"], "head")
    if long:  # the window sentence buried in a long numbered document (gen_long_refund_policy's)
        marker = "\x00RULE\x00"
        policy = {"window": window_tpl.replace("{D}", "{d}")}
        doc = _long_policy(rng, lang, s, domain, marker, [r.text for r in rules], policy)
        before, after = doc.split(marker)
        case = [dc_replace(c, pre="\n\n" if i == 0 else "\n") for i, c in enumerate(case)]
        lines = [head, dc_replace(Line(before, "doc"), pre="\n"),
                 dc_replace(window_line(w), pre=""), dc_replace(Line(after, "doc2"), pre=""),
                 *case]  # fmt: skip
    else:
        pol = [window_line(w), *rules]
        lines = [head, *(pol[i] for i in rng.permutation(len(pol))), *case]
    facts = {"w": w, "n": n, "member": member, "bonus": bonus, "limit": limit, "amount": amount}
    crit = dict(fam["crit"])
    if rng.random() < 0.5:
        crit = with_abstain(crit, lang, ABSTAIN_KEY[s])
    a = Member(lines, facts, rp_solve(facts))
    member_b = None
    if s == "train" and rng.random() < 0.4:  # move the category's window across the case
        if inside:
            w2 = n - extension - int(rng.integers(1, 6))
        else:
            w2 = n - extension + int(rng.integers(0, 6))
        if w2 >= 3 and w2 != w and w2 not in RP_WINDOWS["heldout"]:
            f2 = {**facts, "w": w2}
            member_b = Member(_replace(lines, "rule", window_line(w2)), f2, rp_solve(f2))
    if member_b is None:
        f2 = {**facts, "n": days_for(not inside)}
        member_b = Member(_replace(lines, "days", days_line(f2["n"])), f2, rp_solve(f2))
    q = {"type": "choice", "instructions": fam["q"], "criteria": crit}
    vals = _distinct(allowed, [w + bonus, w, w + 2 * bonus, allowed + 1, max(1, allowed - 1)])
    vals = [vals[i] for i in rng.permutation(len(vals))]
    aux = [{"type": "choice", "instructions": fam["aux"],
            "criteria": _numbers_choice(vals, DAYS_UNIT, lang), "expected": str(allowed),
            "quantity": "refund_period"}]  # fmt: skip
    key = abstain_key(crit) or ABSTAIN_KEY[s]
    unknown = {"rule": {"expected": key}, "days": {"expected": key}}
    g = Group(base, tid, lang, q, a, member_b, ("rule", "days"), unknown, aux)
    g.extra["twin"] = "rule" if rng.random() < 0.5 else "days"
    return g


def long_refund_policy(rng, lang, split, srng) -> Group:
    return refund_policy(rng, lang, split, srng, long=True)


SOLVERS = {
    "rw": rw_solve,
    "inv": inv_solve,
    "sc": sc_solve,
    "wd": wd_solve,
    "rp": rp_solve,
    "lrp": rp_solve,
}
BASES = {
    "rw": return_window,
    "inv": invoice_budget,
    "sc": schedule_conflict,
    "wd": weekday,
    "rp": refund_policy,
    "lrp": long_refund_policy,
}
