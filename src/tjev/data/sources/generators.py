"""Code-labelled decision generators (EN/FR). The label is computed; text is templated.

These follow the leaders' recipe (decider, spark, Hopper): a rule engine or arithmetic
decides the answer and only the surface text varies. Every generator is deterministic
given its ``rng`` and returns one item dict. ``GENERATORS`` maps name -> (fn, family).

Held-out validation (``split="heldout"``) uses different generator seeds *and*, in every
generator, different surface text: framings, questions, template sentences, vocabularies,
names and criteria wording that training never sees, so evaluation measures transfer rather
than memorised templates. Numeric ranges: ``weekday`` draws half its held-out offsets from
the training range (1–180 days) and half beyond it (181–540); ``return_window`` holds out
the window lengths (7/21/45/120 days); ``schedule_conflict`` a longer day and other slot
lengths. ``generate`` drops repeated (state, instructions) keys, normalised as
``build_mix`` does, so no item repeats within one call. mix-v2 changed the train output.
"""

from __future__ import annotations

import datetime as dt
import re
import zlib
from typing import Any

import numpy as np

WEEKDAYS = {
    "en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "fr": ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"],
}
MONTHS = {
    "en": [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ],
    "fr": [
        "janvier",
        "février",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "août",
        "septembre",
        "octobre",
        "novembre",
        "décembre",
    ],
}
YES_NO = {
    "en": {"false": "No", "true": "Yes"},
    "fr": {"false": "Non", "true": "Oui"},
}


def _date(d: dt.date, lang: str) -> str:
    m = MONTHS[lang][d.month - 1]
    return f"{m} {d.day}, {d.year}" if lang == "en" else f"{d.day} {m} {d.year}"


def _rand_date(rng, start=dt.date(2024, 1, 1), span=900) -> dt.date:
    return start + dt.timedelta(days=int(rng.integers(span)))


def _money(x: float, lang: str) -> str:
    return f"${x:,.2f}" if lang == "en" else f"{x:,.2f} €".replace(",", " ").replace(".", ",")


def _pct(x: float, digits: int, lang: str) -> str:
    return f"{x:.{digits}%}" if lang == "en" else f"{100 * x:.{digits}f} %".replace(".", ",")


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


# ---------------------------------------------------------------------------------------
# Temporal / numeric


def return_window(rng, lang, split):
    bought = _rand_date(rng)
    window = int(rng.choice([10, 14, 30, 60, 90, 180] if split == "train" else [7, 21, 45, 120]))
    elapsed = int(rng.integers(max(1, window - 10), window + 12))
    today = bought + dt.timedelta(days=elapsed)
    ok = elapsed <= window
    if split != "train":
        if lang == "en":
            state = (
                f"Warranty terms: claims are accepted up to {window} days after delivery.\n"
                f"Delivered on {_date(bought, lang)}; the claim is filed on {_date(today, lang)}."
            )
            q = "Is the claim filed in time?"
        else:
            state = (
                f"Garantie : les réclamations sont acceptées jusqu'à {window} jours après la "
                f"livraison.\nLivré le {_date(bought, lang)} ; réclamation déposée le "
                f"{_date(today, lang)}."
            )
            q = "La réclamation est-elle déposée à temps ?"
    elif lang == "en":
        state = (
            f"Store policy: items may be returned within {window} days of purchase.\n"
            f"Purchase date: {_date(bought, lang)}. Today: {_date(today, lang)}."
        )
        q = "Is the return still within the allowed window?"
    else:
        state = (
            f"Politique du magasin : retour possible dans les {window} jours suivant l'achat.\n"
            f"Date d'achat : {_date(bought, lang)}. Aujourd'hui : {_date(today, lang)}."
        )
        q = "Le retour est-il encore dans le délai autorisé ?"
    return {
        "state": state,
        "expected": "yes" if ok else "no",
        "question": {"type": "noul", "instructions": q, "criteria": YES_NO[lang]},
    }


def weekday(rng, lang, split):
    d = _rand_date(rng, span=3000)
    if split == "train":
        offset = int(rng.integers(1, 181))
    else:  # half "new surface, same range", half "new range"
        offset = int(rng.integers(1, 181) if rng.random() < 0.5 else rng.integers(181, 541))
    target = d + dt.timedelta(days=offset)
    names = WEEKDAYS[lang]
    if split != "train":
        if lang == "en":
            state = (
                f"The contract was signed on {names[d.weekday()]}, {_date(d, lang)}. "
                f"Payment is due {offset} days after signature."
            )
            q = "On which weekday is the payment due?"
        else:
            state = (
                f"Le contrat a été signé le {names[d.weekday()]} {_date(d, lang)}. "
                f"Le paiement est dû {offset} jours après la signature."
            )
            q = "Quel jour de la semaine le paiement est-il dû ?"
    elif lang == "en":
        state = f"The kickoff was on {names[d.weekday()]}, {_date(d, lang)}. The review is {offset} days later."
        q = "On which weekday is the review?"
    else:
        state = f"Le lancement a eu lieu le {names[d.weekday()]} {_date(d, lang)}. La revue a lieu {offset} jours plus tard."
        q = "Quel jour de la semaine a lieu la revue ?"
    return {
        "state": state,
        "expected": names[target.weekday()],
        "question": {"type": "choice", "instructions": q, "criteria": {n: n for n in names}},
    }


INVOICE_TEXT = {  # (header, discount line, total line, question)
    ("train", "en"): (
        "Invoice lines:",
        "Discount: {p}% on the whole order.",
        "Stated total: {t}",
        "Is the stated total correct (to the cent)?",
    ),
    ("train", "fr"): (
        "Lignes de facture :",
        "Remise : {p} % sur toute la commande.",
        "Total indiqué : {t}",
        "Le total indiqué est-il correct (au centime près) ?",
    ),
    ("heldout", "en"): (
        "Purchase order items:",
        "Rebate applied to the full order: {p}%.",
        "Amount billed: {t}",
        "Does the amount billed match the items and the rebate exactly?",
    ),
    ("heldout", "fr"): (
        "Articles du bon de commande :",
        "Rabais appliqué à l'ensemble : {p} %.",
        "Montant facturé : {t}",
        "Le montant facturé correspond-il exactement aux articles et au rabais ?",
    ),
}


def invoice_total(rng, lang, split):
    items = []
    total = 0.0
    goods = {
        ("train", "en"): ["cables", "chairs", "licences", "toner", "laptops", "desks", "badges"],
        ("train", "fr"): [
            "câbles", "chaises", "licences", "toner", "ordinateurs", "bureaux", "badges",
        ],
        ("heldout", "en"): [
            "sensors", "valves", "gloves", "helmets", "pallets", "filters", "batteries",
        ],
        ("heldout", "fr"): [
            "capteurs", "vannes", "gants", "casques", "palettes", "filtres", "batteries",
        ],
    }[split, lang]  # fmt: skip
    for name in rng.choice(goods, size=int(rng.integers(2, 5)), replace=False):
        qty = int(rng.integers(1, 12))
        price = float(rng.integers(5, 400)) + float(rng.choice([0, 0.5, 0.99]))
        items.append((name, qty, price))
        total += qty * price
    discount = int(rng.choice([0, 0, 5, 10, 15]))
    total = round(total * (1 - discount / 100), 2)
    correct = rng.random() < 0.5
    shown = (
        total
        if correct
        else round(
            total + float(rng.choice([-1, 1])) * float(rng.choice([1, 9.99, 10, 0.5 * total])), 2
        )
    )
    lines = "\n".join(f"- {n}: {q} × {_money(p, lang)}" for n, q, p in items)
    head, disc, stated, q = INVOICE_TEXT[split, lang]
    state = "\n".join([head, lines, disc.format(p=discount), stated.format(t=_money(shown, lang))])
    return {
        "state": state,
        "expected": "yes" if abs(shown - total) < 0.005 else "no",
        "question": {"type": "noul", "instructions": q, "criteria": YES_NO[lang]},
    }


BASE_RATE_TEXT = {  # (state template, question template); {p} prior, {s} sens, {f} fpr
    ("train", "en"): (
        (
            "Base rate: {p} of cases are positive. The detector flags {s} of positives "
            "and {f} of negatives. This case was flagged."
        ),
        "Is it true that the {t}?",
    ),
    ("train", "fr"): (
        (
            "Taux de base : {p} des cas sont positifs. Le détecteur signale {s} des "
            "positifs et {f} des négatifs. Ce cas a été signalé."
        ),
        "Est-il vrai que {t} ?",
    ),
    ("heldout", "en"): (
        (
            "Prevalence: {p} of all cases are positive. The screening test catches {s} of true "
            "positives but also raises an alert on {f} of negatives. An alert was raised for "
            "this case."
        ),
        "Given the alert, is it the case that the {t}?",
    ),
    ("heldout", "fr"): (
        (
            "Prévalence : {p} de l'ensemble des cas sont positifs. Le test de dépistage détecte "
            "{s} des vrais positifs mais déclenche aussi une alerte pour {f} des négatifs. Une "
            "alerte a été déclenchée pour ce cas."
        ),
        "Compte tenu de l'alerte, peut-on dire que {t} ?",
    ),
}


def base_rate(rng, lang, split):
    # Continuous parameters (rounded as displayed): no template collisions across splits.
    prior = round(float(10 ** rng.uniform(-3, -0.5)), 4)
    sens = round(float(rng.uniform(0.7, 0.995)), 3)
    fpr = round(float(10 ** rng.uniform(-2.5, -0.7)), 4)
    posterior = prior * sens / (prior * sens + (1 - prior) * fpr)
    thing = {
        ("train", "en"): [
            "transaction is fraudulent",
            "patient has the condition",
            "email is phishing",
            "machine will fail this month",
            "applicant misreported income",
        ],
        ("train", "fr"): [
            "la transaction est frauduleuse",
            "le patient a la maladie",
            "l'e-mail est une tentative d'hameçonnage",
            "la machine tombera en panne ce mois-ci",
            "le candidat a mal déclaré ses revenus",
        ],
        ("heldout", "en"): [
            "shipment arrived damaged",
            "account is compromised",
            "part is defective",
            "loan will default",
            "review is fake",
        ],
        ("heldout", "fr"): [
            "l'envoi est arrivé endommagé",
            "le compte est compromis",
            "la pièce est défectueuse",
            "le prêt fera défaut",
            "l'avis est faux",
        ],
    }[split, lang]
    i = int(rng.integers(len(thing)))
    template, question = BASE_RATE_TEXT[split, lang]
    state = template.format(p=_pct(prior, 2, lang), s=_pct(sens, 1, lang), f=_pct(fpr, 2, lang))
    return {
        "state": state,
        "target": {"yes": posterior, "no": 1 - posterior},
        "question": {
            "type": "noul",
            "instructions": question.format(t=thing[i]),
            "criteria": YES_NO[lang],
        },
    }


def schedule_conflict(rng, lang, split):
    # held-out: a longer day and other slot lengths (7:00-20:00, 20-120 min)
    lo, hi = (8 * 4, 17 * 4) if split == "train" else (7 * 4, 20 * 4)
    lengths = [30, 45, 60, 90] if split == "train" else [20, 50, 75, 120]
    proposed = [30, 60] if split == "train" else [45, 90]
    busy = []
    for _ in range(int(rng.integers(2, 6))):
        start = int(rng.integers(lo, hi)) * 15
        busy.append((start, start + int(rng.choice(lengths))))
    start = int(rng.integers(lo, hi)) * 15
    new = (start, start + int(rng.choice(proposed)))
    conflict = any(a < new[1] and new[0] < b for a, b in busy)

    def hm(m):
        return f"{m // 60:02d}:{m % 60:02d}" if lang == "en" else f"{m // 60}h{m % 60:02d}"

    listing = "\n".join(f"- {hm(a)}–{hm(b)}" for a, b in sorted(busy))
    if split != "train":
        if lang == "en":
            state = f"Room bookings today:\n{listing}\nNew request: {hm(new[0])}–{hm(new[1])}."
            q = "Does the new request clash with a booking?"
        else:
            state = (
                f"Réservations de la salle :\n{listing}\n"
                f"Nouvelle demande : {hm(new[0])}–{hm(new[1])}."
            )
            q = "La nouvelle demande entre-t-elle en conflit avec une réservation ?"
    elif lang == "en":
        state = f"Calendar (busy):\n{listing}\nProposed meeting: {hm(new[0])}–{hm(new[1])}."
        q = "Does the proposed meeting overlap an existing one?"
    else:
        state = f"Agenda (occupé) :\n{listing}\nRéunion proposée : {hm(new[0])}–{hm(new[1])}."
        q = "La réunion proposée chevauche-t-elle un créneau existant ?"
    return {
        "state": state,
        "expected": "yes" if conflict else "no",
        "question": {"type": "noul", "instructions": q, "criteria": YES_NO[lang]},
    }


def table_argmax(rng, lang, split):
    names = (
        ["Alice", "Bruno", "Chloé", "Dmitri", "Emma", "Farid", "Grace", "Hugo"]
        if split == "train"
        else ["Inès", "Jonas", "Kenji", "Léa", "Mateo", "Nadia", "Oscar", "Priya"]
    )
    picked = list(rng.choice(names, size=int(rng.integers(3, 7)), replace=False))
    values = rng.choice(np.arange(10, 500), size=len(picked), replace=False)
    metric = {
        ("train", "en"): ["sales", "tickets closed", "hours logged"],
        ("train", "fr"): ["ventes", "tickets clos", "heures saisies"],
        ("heldout", "en"): ["units shipped", "calls answered", "defects found"],
        ("heldout", "fr"): ["unités expédiées", "appels traités", "défauts trouvés"],
    }[split, lang]
    m = metric[int(rng.integers(len(metric)))]
    lowest = rng.random() < 0.3
    rows = "\n".join(f"| {n} | {int(v)} |" for n, v in zip(picked, values, strict=True))
    gold = picked[int(np.argmin(values) if lowest else np.argmax(values))]
    if lang == "en":
        state = f"| name | {m} |\n|---|---|\n{rows}"
        q = f"Who has the {'fewest' if lowest else 'most'} {m}?"
    else:
        state = f"| nom | {m} |\n|---|---|\n{rows}"
        de = "d'" if m[0] in "aeéèiouh" else "de "
        q = f"Qui a le {'moins' if lowest else 'plus'} {de}{m} ?"
    return {
        "state": state,
        "expected": gold,
        "question": {"type": "choice", "instructions": q, "criteria": {n: n for n in picked}},
    }


# ---------------------------------------------------------------------------------------
# Policy engine with exceptions (short and long documents)

POLICY_DOMAINS = {
    "train": [
        ("electronics", "électronique"),
        ("furniture", "mobilier"),
        ("apparel", "vêtements"),
        ("software", "logiciels"),
        ("groceries", "épicerie"),
    ],
    "heldout": [
        ("medical devices", "dispositifs médicaux"),
        ("industrial parts", "pièces industrielles"),
        ("event tickets", "billets d'événements"),
        ("musical instruments", "instruments de musique"),
        ("pet supplies", "animalerie"),
    ],
}
NO_OPEN = {  # the one domain whose opened items are never refunded
    "train": ("software", "logiciels"),
    "heldout": ("medical devices", "dispositifs médicaux"),
}
POLICY_TEXT: dict[
    tuple[str, str], dict[str, Any]
] = {  # {d} domain, {w} window, {b} member bonus, {l} limit
    ("train", "en"): {
        "head": "Policy:",
        "window": "Refunds for {d} are accepted within {w} days of delivery.",
        "member": "Members receive an extra {b} days.",
        "no_open": "Opened software cannot be refunded.",
        "limit": "Refunds above {l} must be escalated to a supervisor.",
        "final": "Items marked final sale are never refunded.",
        "case": "Request: {m} customer, {c}delivered {days} days ago, item {o}, amount {a}{f}.",
        "words": (("member", "non-member"), ("opened", "unopened"), ", marked final sale"),
        "category": "category {d}, ",
        "q": "What should the agent do with this refund request?",
        "crit": {
            "approve": "Refund it now",
            "deny": "Refuse the refund",
            "escalate": "Send to a supervisor",
        },
    },
    ("train", "fr"): {
        "head": "Politique :",
        "window": "Les remboursements pour la catégorie {d} sont acceptés dans les {w} jours "
        "suivant la livraison.",
        "member": "Les membres bénéficient de {b} jours supplémentaires.",
        "no_open": "Un logiciel ouvert ne peut pas être remboursé.",
        "limit": "Tout remboursement supérieur à {l} doit être transmis à un superviseur.",
        "final": "Les articles marqués « vente finale » ne sont jamais remboursés.",
        "case": "Demande : client {m}, {c}livré il y a {days} jours, article {o}, montant {a}{f}.",
        "words": (("membre", "non membre"), ("ouvert", "non ouvert"), ", marqué « vente finale »"),
        "category": "catégorie {d}, ",
        "q": "Que doit faire l'agent avec cette demande de remboursement ?",
        "crit": {
            "approve": "Rembourser maintenant",
            "deny": "Refuser le remboursement",
            "escalate": "Transmettre à un superviseur",
        },
    },
    ("heldout", "en"): {
        "head": "Terms:",
        "window": "Goods listed under {d} may be returned for a refund up to {w} days after "
        "delivery.",
        "member": "Loyalty members get {b} additional days to return goods.",
        "no_open": "Unsealed goods listed under medical devices are not eligible for a refund.",
        "limit": "Any refund of more than {l} requires a supervisor's approval.",
        "final": "Clearance goods sold as final are excluded from refunds.",
        "case": "Case: the purchaser is {m}; {c}delivery was {days} days ago; the goods are "
        "{o}; refund requested: {a}{f}.",
        "words": (
            ("a loyalty member", "not a loyalty member"),
            ("unsealed", "still sealed"),
            "; sold as final clearance",
        ),
        "category": "goods listed under {d}; ",
        "q": "How should this refund request be handled?",
        "crit": {
            "approve": "Issue the refund",
            "deny": "Decline the request",
            "escalate": "Refer it to a supervisor",
        },
    },
    ("heldout", "fr"): {
        "head": "Conditions :",
        "window": "Les produits du rayon {d} peuvent être retournés contre remboursement "
        "jusqu'à {w} jours après la livraison.",
        "member": "Les adhérents du programme de fidélité disposent de {b} jours de plus pour "
        "retourner un produit.",
        "no_open": "Les produits du rayon dispositifs médicaux dont l'emballage a été descellé "
        "ne sont pas remboursables.",
        "limit": "Au-delà de {l}, tout remboursement nécessite l'accord d'un superviseur.",
        "final": "Les produits de déstockage vendus comme définitifs sont exclus des "
        "remboursements.",
        "case": "Dossier : client {m} du programme de fidélité ; {c}livraison il y a {days} "
        "jours ; produit {o} ; remboursement demandé : {a}{f}.",
        "words": (
            ("adhérent", "non adhérent"),
            ("descellé", "encore scellé"),
            " ; déstockage vendu comme définitif",
        ),
        "category": "rayon {d} ; ",
        "q": "Comment traiter cette demande de remboursement ?",
        "crit": {
            "approve": "Effectuer le remboursement",
            "deny": "Décliner la demande",
            "escalate": "Soumettre à un superviseur",
        },
    },
}

# Long documents: numbered sections of boilerplate (general topics, per-category notes with
# windows for *other* categories, refund procedure) in which the five decisive rules are
# buried at random places. {k} is a small number, {c} a category.
LONG_POLICY: dict[tuple[str, str], dict[str, Any]] = {
    ("train", "en"): {
        "section": "Section",
        "category_title": "{C}",
        "extra_categories": [
            "books", "toys", "kitchenware", "garden equipment", "sports gear",
            "beauty products", "stationery", "luggage", "jewellery", "home appliances",
            "video games", "bedding", "craft supplies", "cameras",
        ],
        "refund_titles": ["Refunds", "Returns", "Eligibility", "Exceptions", "Approvals"],
        "refund_neutral": [
            "Approved refunds are paid to the original payment method within {k} business days.",
            "Agents must record the reason given by the customer in the case notes.",
            "Customers receive an e-mail confirmation once a request has been logged.",
            "Return labels are provided free of charge for eligible requests.",
            "Refunds are calculated in the currency of the original order.",
            "Customers can follow the status of a request from their account page.",
        ],
        "category_sentences": [
            "Items in the {c} category are dispatched from the main warehouse in protective packaging.",
            "Set-up or care instructions for {c} items are included with every order.",
            "Questions about {c} items can be sent to the dedicated product desk.",
            "Stock levels for the {c} category are updated on the website every hour.",
            "Gift wrapping is available for most {c} items.",
            "Bulk orders of {c} items may qualify for a trade discount.",
            "Product photos in the {c} category are for illustration only; colours may vary slightly.",
            "Safety information for {c} items is printed on the packaging.",
            "Some {c} items are sold exclusively online.",
            "Pre-orders in the {c} category are charged on dispatch.",
            "Seasonal promotions on {c} items are announced in the newsletter.",
            "Delivery of large {c} items may require an appointment.",
        ],
        "topics": [
            ("Scope", [
                "This policy applies to all orders placed through the online store, the mobile app and the telephone sales line.",
                "It does not apply to purchases made on third-party marketplaces, which follow their own terms.",
                "Where this policy conflicts with a product leaflet, this policy prevails.",
                "Headings are included for convenience only and do not affect interpretation.",
                "The version of this policy in force on the order date applies to the order.",
            ]),
            ("Definitions", [
                "\"Customer\" means the person who placed the order and whose name appears on the invoice.",
                "\"Delivery\" means the date on which the carrier records the parcel as handed over.",
                "\"Business day\" means any day other than a Saturday, a Sunday or a public holiday.",
                "\"Agent\" means a member of the customer-care team handling a request.",
                "\"Supervisor\" means a team lead with authority to approve exceptional payments.",
            ]),
            ("Communications", [
                "All notices must be sent in writing to the address on file.",
                "Customers can reach the care team by chat, e-mail or telephone during opening hours.",
                "Messages received outside opening hours are answered on the next business day.",
                "Agents use the customer's preferred language wherever it is supported.",
                "Telephone calls may be recorded for training and quality purposes.",
            ]),
            ("Data protection", [
                "Customer data is processed according to the privacy notice.",
                "Personal data collected during a request is kept only as long as necessary to handle it.",
                "Customers may ask for a copy of the data held about them at any time.",
                "Payment card details are never stored by the care team.",
                "Access to case notes is restricted to authorised staff.",
            ]),
            ("Shipping", [
                "Standard shipping takes three to seven business days depending on the region.",
                "Express shipping is available for an additional fee in most regions.",
                "Delivery estimates shown at checkout are indicative and not guaranteed.",
                "Parcels that cannot be delivered are held at the local depot for {k} business days.",
                "Customers are notified by e-mail when an order is dispatched.",
            ]),
            ("Pricing", [
                "Prices are displayed including all applicable taxes.",
                "Promotional prices are valid only for the period stated in the promotion.",
                "Price-match requests are reviewed by the pricing team, not by care agents.",
                "Obvious pricing errors may be corrected before an order is dispatched.",
                "Vouchers cannot be combined unless the voucher terms say otherwise.",
            ]),
            ("Loyalty points", [
                "Points expire twelve months after they are earned and have no cash value.",
                "Points are credited {k} days after the order is dispatched.",
                "Membership is free and can be cancelled at any time from the account page.",
                "Members receive advance notice of seasonal sales by e-mail.",
                "Points cannot be transferred between accounts.",
            ]),
            ("Warranty", [
                "Products are covered by the manufacturer's warranty for the period stated in the leaflet.",
                "Warranty claims are handled separately from refund requests.",
                "The warranty does not cover damage caused by misuse or accidents.",
                "Repairs under warranty are free of charge, including return shipping.",
                "Replacement parts may be new or refurbished.",
            ]),
            ("Payment", [
                "Orders can be paid by card, bank transfer or approved digital wallets.",
                "Card payments are authorised at checkout and captured on dispatch.",
                "Instalment plans are offered by a third-party lender under its own terms.",
                "Failed payments are retried once after {k} days.",
                "Invoices are available for download from the account page.",
            ]),
            ("Force majeure", [
                "Neither party is liable for delays caused by events beyond control.",
                "Such events include floods, strikes, pandemics and failures of public networks.",
                "Obligations are suspended for as long as the event continues.",
                "The seller informs customers of significant delays as soon as possible.",
                "If the event lasts more than {k} weeks, either party may cancel the affected order.",
            ]),
            ("Governing law", [
                "This policy is governed by the laws of the seller's jurisdiction.",
                "Disputes are first handled through the internal complaints procedure.",
                "Customers may also contact an approved mediation service free of charge.",
                "Nothing in this policy limits the customer's statutory rights.",
                "If a clause is found invalid, the remaining clauses stay in force.",
            ]),
            ("Complaints", [
                "Complaints about the service can be submitted through the contact form.",
                "Every complaint receives a reference number within one business day.",
                "The care team aims to resolve complaints within {k} business days.",
                "Unresolved complaints are reviewed by the quality team.",
                "Customers are informed of the outcome in writing.",
            ]),
        ],
    },
    ("train", "fr"): {
        "section": "Article",
        "category_title": "Catégorie {c}",
        "extra_categories": [
            "livres", "jouets", "ustensiles de cuisine", "jardinage", "articles de sport",
            "beauté", "papeterie", "bagagerie", "bijouterie", "électroménager", "jeux vidéo",
            "literie", "loisirs créatifs", "photographie",
        ],
        "refund_titles": ["Remboursements", "Retours", "Éligibilité", "Exceptions", "Validations"],
        "refund_neutral": [
            "Les remboursements acceptés sont versés sur le moyen de paiement d'origine sous {k} jours ouvrés.",
            "L'agent consigne dans le dossier le motif donné par le client.",
            "Le client reçoit un e-mail de confirmation dès l'enregistrement de sa demande.",
            "Une étiquette de retour gratuite est fournie pour les demandes éligibles.",
            "Les remboursements sont calculés dans la devise de la commande d'origine.",
            "Le client peut suivre l'état de sa demande depuis son espace personnel.",
        ],
        "category_sentences": [
            "Les articles de la catégorie {c} sont expédiés depuis l'entrepôt principal dans un emballage protecteur.",
            "Une notice d'utilisation ou d'entretien accompagne chaque article de la catégorie {c}.",
            "Les questions sur la catégorie {c} peuvent être adressées au service produit dédié.",
            "Les stocks de la catégorie {c} sont mis à jour sur le site toutes les heures.",
            "Un emballage cadeau est proposé pour la plupart des articles de la catégorie {c}.",
            "Les commandes en gros dans la catégorie {c} peuvent bénéficier d'une remise professionnelle.",
            "Les photos de la catégorie {c} sont non contractuelles ; les couleurs peuvent légèrement varier.",
            "Les consignes de sécurité des articles de la catégorie {c} figurent sur l'emballage.",
            "Certains articles de la catégorie {c} sont vendus exclusivement en ligne.",
            "Les précommandes dans la catégorie {c} sont débitées à l'expédition.",
            "Les promotions saisonnières de la catégorie {c} sont annoncées dans la lettre d'information.",
            "La livraison des articles volumineux de la catégorie {c} peut nécessiter un rendez-vous.",
        ],
        "topics": [
            ("Champ d'application", [
                "La présente politique s'applique à toutes les commandes passées sur la boutique en ligne, l'application mobile et la vente par téléphone.",
                "Elle ne s'applique pas aux achats effectués sur des places de marché tierces, qui suivent leurs propres conditions.",
                "En cas de contradiction avec une notice produit, la présente politique prévaut.",
                "Les titres sont donnés à titre indicatif et n'ont aucune portée d'interprétation.",
                "La version de la politique en vigueur à la date de la commande s'applique à celle-ci.",
            ]),
            ("Définitions", [
                "« Client » désigne la personne qui a passé la commande et dont le nom figure sur la facture.",
                "« Livraison » désigne la date à laquelle le transporteur enregistre la remise du colis.",
                "« Jour ouvré » désigne tout jour autre qu'un samedi, un dimanche ou un jour férié.",
                "« Agent » désigne un membre du service client chargé d'une demande.",
                "« Superviseur » désigne un responsable d'équipe habilité à valider les paiements exceptionnels.",
            ]),
            ("Communications", [
                "Toute notification doit être adressée par écrit à l'adresse enregistrée.",
                "Le service client est joignable par messagerie, e-mail ou téléphone aux heures d'ouverture.",
                "Les messages reçus en dehors des heures d'ouverture sont traités le jour ouvré suivant.",
                "L'agent utilise la langue préférée du client lorsqu'elle est prise en charge.",
                "Les appels peuvent être enregistrés à des fins de formation et de qualité.",
            ]),
            ("Données personnelles", [
                "Les données sont traitées conformément à la politique de confidentialité.",
                "Les données collectées lors d'une demande ne sont conservées que le temps nécessaire à son traitement.",
                "Le client peut demander à tout moment une copie des données le concernant.",
                "Les coordonnées bancaires ne sont jamais conservées par le service client.",
                "L'accès aux dossiers est réservé au personnel habilité.",
            ]),
            ("Livraison", [
                "La livraison standard prend de trois à sept jours ouvrés selon la région.",
                "Une livraison express est proposée moyennant un supplément dans la plupart des régions.",
                "Les délais affichés lors de la commande sont indicatifs et non garantis.",
                "Les colis non distribués sont conservés au dépôt local pendant {k} jours ouvrés.",
                "Le client est prévenu par e-mail de l'expédition de sa commande.",
            ]),
            ("Prix", [
                "Les prix sont affichés toutes taxes comprises.",
                "Les prix promotionnels ne sont valables que pendant la période indiquée.",
                "Les demandes d'alignement de prix sont examinées par l'équipe tarifaire et non par les agents.",
                "Une erreur de prix manifeste peut être corrigée avant l'expédition de la commande.",
                "Les bons d'achat ne sont pas cumulables, sauf mention contraire dans leurs conditions.",
            ]),
            ("Points de fidélité", [
                "Les points expirent douze mois après leur obtention et n'ont aucune valeur monétaire.",
                "Les points sont crédités {k} jours après l'expédition de la commande.",
                "L'adhésion est gratuite et peut être résiliée à tout moment depuis l'espace personnel.",
                "Les membres sont prévenus à l'avance des soldes saisonniers par e-mail.",
                "Les points ne peuvent pas être transférés d'un compte à un autre.",
            ]),
            ("Garantie", [
                "Les produits bénéficient de la garantie du fabricant pour la durée indiquée dans la notice.",
                "Les demandes de garantie sont traitées séparément des demandes de remboursement.",
                "La garantie ne couvre pas les dommages dus à une mauvaise utilisation ou à un accident.",
                "Les réparations sous garantie sont gratuites, frais de retour compris.",
                "Les pièces de rechange peuvent être neuves ou reconditionnées.",
            ]),
            ("Paiement", [
                "Les commandes peuvent être réglées par carte, par virement ou par portefeuille numérique agréé.",
                "Les paiements par carte sont autorisés à la commande et débités à l'expédition.",
                "Le paiement en plusieurs fois est proposé par un organisme de crédit selon ses propres conditions.",
                "Un paiement refusé est présenté à nouveau une seule fois après {k} jours.",
                "Les factures peuvent être téléchargées depuis l'espace personnel.",
            ]),
            ("Force majeure", [
                "Aucune partie n'est responsable des retards dus à des événements imprévisibles.",
                "Ces événements comprennent les inondations, les grèves, les pandémies et les pannes des réseaux publics.",
                "Les obligations sont suspendues tant que l'événement se poursuit.",
                "Le vendeur informe les clients des retards importants dans les meilleurs délais.",
                "Si l'événement dure plus de {k} semaines, chaque partie peut annuler la commande concernée.",
            ]),
            ("Droit applicable", [
                "La présente politique est régie par le droit du pays du vendeur.",
                "Les litiges sont d'abord traités par la procédure interne de réclamation.",
                "Le client peut aussi saisir gratuitement un médiateur agréé.",
                "Aucune disposition de la présente politique ne limite les droits légaux du client.",
                "Si une clause est jugée invalide, les autres clauses restent applicables.",
            ]),
            ("Réclamations", [
                "Les réclamations sur le service peuvent être déposées via le formulaire de contact.",
                "Chaque réclamation reçoit un numéro de référence sous un jour ouvré.",
                "Le service client s'efforce de traiter les réclamations sous {k} jours ouvrés.",
                "Les réclamations non résolues sont examinées par l'équipe qualité.",
                "Le client est informé de la décision par écrit.",
            ]),
        ],
    },
    ("heldout", "en"): {
        "section": "Clause",
        "category_title": "Product line: {c}",
        "extra_categories": [
            "office supplies", "bicycles", "lighting", "home textiles", "baby care",
            "camping gear", "board games", "wall paint", "car accessories", "watches",
            "flooring", "aquarium equipment",
        ],
        "refund_titles": ["Returns and refunds", "Refund eligibility", "Exclusions", "Authorisation"],
        "refund_neutral": [
            "Accepted refunds reach the purchaser's account within {k} working days.",
            "Advisers note the purchaser's stated reason on the return case.",
            "An acknowledgement e-mail is sent as soon as a return case is opened.",
            "Prepaid return labels are supplied for accepted returns.",
            "Refunds are paid in the currency used at checkout.",
            "The status of each return case is visible in the customer portal.",
        ],
        "category_sentences": [
            "Goods listed under {c} leave the regional depot in reinforced cartons.",
            "A user guide accompanies all goods listed under {c}.",
            "Enquiries relating to {c} are handled by a specialist adviser.",
            "Availability of goods listed under {c} is refreshed several times a day.",
            "Engraving or personalisation is offered on selected {c} goods.",
            "Business accounts ordering {c} goods in volume can request a quote.",
            "Images of {c} goods may differ slightly from the delivered product.",
            "Hazard labels for {c} goods follow the applicable regulations.",
            "Certain {c} goods are only available in selected regions.",
            "Back-ordered {c} goods are shipped as soon as stock arrives.",
            "Promotional codes rarely apply to {c} goods.",
            "Assembly services for {c} goods can be booked at checkout.",
            "Spare parts for {c} goods are stocked for at least five years.",
            "Delivery slots for bulky {c} goods are agreed by phone.",
        ],
        "topics": [
            ("Purpose", [
                "These terms explain how returns are handled for goods bought from the company.",
                "They apply equally to online, telephone and in-store purchases.",
                "Special terms agreed in writing with a business customer take precedence.",
                "Terms in bold type are explained in the glossary.",
                "The company reviews these terms at least once a year.",
            ]),
            ("Glossary", [
                "\"Purchaser\" refers to the individual or business named on the order confirmation.",
                "\"Handover\" refers to the moment the courier records the goods as received.",
                "\"Working day\" refers to Monday to Friday, excluding bank holidays.",
                "\"Adviser\" refers to the employee who handles a return case.",
                "\"Manager\" refers to the person authorised to sign off exceptional payments.",
            ]),
            ("Contacting us", [
                "Written correspondence should quote the order confirmation number.",
                "The helpdesk can be reached by chat, letter or phone on working days.",
                "Messages left after closing time are handled the following working day.",
                "Advisers respond in the language of the original message where possible.",
                "Calls are logged to improve the quality of the service.",
            ]),
            ("Confidentiality", [
                "Information supplied by the purchaser is handled under the company's privacy statement.",
                "Case records are deleted once the statutory retention period ends.",
                "The purchaser may request correction of inaccurate information.",
                "Advisers never ask for full card numbers by phone or e-mail.",
                "Only trained staff may consult return case files.",
            ]),
            ("Transport", [
                "Goods are dispatched from the nearest warehouse with available stock.",
                "Courier transit times average {k} working days.",
                "Deliveries to islands and remote areas may take longer.",
                "A signature on delivery may be required for high-value parcels.",
                "A tracking link is sent as soon as the parcel leaves the warehouse.",
            ]),
            ("Invoicing", [
                "The company accepts debit cards, credit cards and bank transfers.",
                "Business customers may be offered payment terms of thirty days.",
                "VAT invoices are issued automatically for every order.",
                "Currency conversion fees are set by the card issuer.",
                "Payment reminders are sent {k} days before an instalment is due.",
            ]),
            ("Unforeseeable events", [
                "The company is not responsible for failures caused by circumstances outside its reasonable control.",
                "Examples include severe weather, industrial action and cyber attacks on suppliers.",
                "Affected deadlines are extended for the duration of the event.",
                "Purchasers are told about major disruption through a banner on the website.",
                "Orders delayed for more than {k} weeks may be withdrawn by either side.",
            ]),
            ("Disputes", [
                "The company tries to settle disagreements amicably in the first instance.",
                "A purchaser who is not satisfied may refer the matter to an independent ombudsman.",
                "Statutory consumer rights are not affected by these terms.",
                "Any invalid provision is replaced by the closest valid one.",
                "The courts of the company's registered office have jurisdiction.",
            ]),
            ("Service quality", [
                "Feedback surveys are sent after each closed case.",
                "Survey answers are anonymised before analysis.",
                "Advisers receive refresher training every quarter.",
                "Recurring issues are reported to the product teams.",
                "Monthly service statistics are published on the website.",
            ]),
            ("Environment", [
                "Packaging is made from at least seventy per cent recycled material.",
                "Purchasers can drop used packaging at any collection point.",
                "Deliveries are grouped where possible to reduce emissions.",
                "The company publishes an annual sustainability report.",
                "Old appliances are collected free of charge when a replacement is delivered.",
            ]),
            ("Accessibility", [
                "The website follows recognised accessibility guidelines.",
                "Large-print copies of these terms are available on request.",
                "A text relay service is available for purchasers with hearing impairments.",
                "Advisers can arrange a call-back at a time chosen by the purchaser.",
                "Feedback on accessibility is reviewed every month.",
            ]),
            ("Business customers", [
                "Business accounts are opened after a credit check.",
                "Several users can share one business account.",
                "Consolidated monthly invoices are available to business accounts.",
                "Account managers are assigned to large business customers.",
                "Purchase order numbers can be printed on every invoice.",
            ]),
        ],
    },
    ("heldout", "fr"): {
        "section": "Clause",
        "category_title": "Rayon {c}",
        "extra_categories": [
            "fournitures de bureau", "cycles", "luminaires", "linge de maison", "puériculture",
            "camping", "jeux de société", "peinture murale", "accessoires auto", "montres",
            "revêtements de sol", "aquariophilie",
        ],
        "refund_titles": ["Retours et remboursements", "Conditions de remboursement", "Exclusions", "Autorisations"],
        "refund_neutral": [
            "Les remboursements accordés parviennent sur le compte de l'acheteur sous {k} jours ouvrables.",
            "Le conseiller note sur le dossier de retour le motif indiqué par l'acheteur.",
            "Un accusé de réception est envoyé par e-mail dès l'ouverture du dossier de retour.",
            "Des étiquettes de retour prépayées sont fournies pour les retours acceptés.",
            "Les remboursements sont versés dans la devise utilisée lors du paiement.",
            "L'avancement de chaque dossier de retour est visible dans le portail client.",
        ],
        "category_sentences": [
            "Les produits du rayon {c} quittent le dépôt régional dans des cartons renforcés.",
            "Un guide d'utilisation est fourni avec tous les produits du rayon {c}.",
            "Les demandes relatives au rayon {c} sont traitées par un conseiller spécialisé.",
            "La disponibilité des produits du rayon {c} est actualisée plusieurs fois par jour.",
            "Une gravure ou une personnalisation est proposée sur certains produits du rayon {c}.",
            "Les comptes professionnels commandant en volume au rayon {c} peuvent demander un devis.",
            "Les visuels des produits du rayon {c} peuvent différer légèrement du produit livré.",
            "L'étiquetage de danger des produits du rayon {c} suit la réglementation en vigueur.",
            "Certains produits du rayon {c} ne sont disponibles que dans certaines régions.",
            "Les produits du rayon {c} en rupture sont expédiés dès le réassort.",
            "Les codes promotionnels s'appliquent rarement aux produits du rayon {c}.",
            "Un service de montage peut être réservé à la commande pour les produits du rayon {c}.",
            "Les pièces détachées des produits du rayon {c} sont disponibles pendant au moins cinq ans.",
            "Les créneaux de livraison des produits volumineux du rayon {c} sont fixés par téléphone.",
        ],
        "topics": [
            ("Objet", [
                "Les présentes conditions expliquent le traitement des retours des produits achetés auprès de la société.",
                "Elles s'appliquent de la même façon aux achats en ligne, par téléphone et en magasin.",
                "Les conditions particulières convenues par écrit avec un client professionnel priment.",
                "Les termes en gras sont expliqués dans le glossaire.",
                "La société révise les présentes conditions au moins une fois par an.",
            ]),
            ("Glossaire", [
                "« Acheteur » s'entend de la personne ou de l'entreprise nommée sur la confirmation de commande.",
                "« Remise » s'entend du moment où le coursier enregistre la réception des produits.",
                "« Jour ouvrable » s'entend du lundi au vendredi, hors jours fériés.",
                "« Conseiller » s'entend de l'employé qui traite un dossier de retour.",
                "« Responsable » s'entend de la personne habilitée à signer les paiements exceptionnels.",
            ]),
            ("Nous contacter", [
                "Tout courrier doit rappeler le numéro de confirmation de commande.",
                "L'assistance est joignable par messagerie, courrier ou téléphone les jours ouvrables.",
                "Les messages laissés après la fermeture sont traités le jour ouvrable suivant.",
                "Les conseillers répondent si possible dans la langue du message initial.",
                "Les appels sont consignés afin d'améliorer la qualité du service.",
            ]),
            ("Confidentialité", [
                "Les informations fournies par l'acheteur sont traitées selon la déclaration de confidentialité de la société.",
                "Les dossiers sont effacés à la fin de la durée légale de conservation.",
                "L'acheteur peut demander la rectification d'informations inexactes.",
                "Les conseillers ne demandent jamais un numéro de carte complet par téléphone ou par e-mail.",
                "Seul le personnel formé peut consulter les dossiers de retour.",
            ]),
            ("Transport", [
                "Les produits partent de l'entrepôt le plus proche disposant du stock.",
                "Le transport par coursier prend en moyenne {k} jours ouvrables.",
                "Les livraisons vers les îles et les zones isolées peuvent prendre plus de temps.",
                "Une signature peut être exigée à la livraison pour les colis de grande valeur.",
                "Un lien de suivi est envoyé dès que le colis quitte l'entrepôt.",
            ]),
            ("Facturation", [
                "La société accepte les cartes de débit, les cartes de crédit et les virements.",
                "Des délais de paiement de trente jours peuvent être accordés aux professionnels.",
                "Une facture avec TVA est émise automatiquement pour chaque commande.",
                "Les frais de conversion de devises sont fixés par l'émetteur de la carte.",
                "Un rappel est envoyé {k} jours avant chaque échéance.",
            ]),
            ("Événements imprévus", [
                "La société n'est pas responsable des défaillances dues à des circonstances hors de son contrôle raisonnable.",
                "Il peut s'agir d'intempéries, de mouvements sociaux ou de cyberattaques visant des fournisseurs.",
                "Les délais concernés sont prolongés pendant la durée de l'événement.",
                "Les acheteurs sont informés des perturbations majeures par un bandeau sur le site.",
                "Une commande retardée de plus de {k} semaines peut être retirée par l'une ou l'autre partie.",
            ]),
            ("Différends", [
                "La société cherche d'abord à régler les désaccords à l'amiable.",
                "Un acheteur insatisfait peut saisir un médiateur indépendant.",
                "Les présentes conditions ne portent pas atteinte aux droits légaux des consommateurs.",
                "Toute stipulation invalide est remplacée par la stipulation valable la plus proche.",
                "Les tribunaux du siège social de la société sont compétents.",
            ]),
            ("Qualité de service", [
                "Un questionnaire de satisfaction est envoyé après chaque dossier clos.",
                "Les réponses aux questionnaires sont anonymisées avant analyse.",
                "Les conseillers suivent une formation de mise à niveau chaque trimestre.",
                "Les problèmes récurrents sont signalés aux équipes produit.",
                "Les statistiques mensuelles du service sont publiées sur le site.",
            ]),
            ("Environnement", [
                "Les emballages contiennent au moins soixante-dix pour cent de matière recyclée.",
                "Les acheteurs peuvent déposer les emballages usagés dans tout point de collecte.",
                "Les livraisons sont regroupées autant que possible pour réduire les émissions.",
                "La société publie chaque année un rapport de développement durable.",
                "Les anciens appareils sont repris gratuitement lors de la livraison d'un remplaçant.",
            ]),
            ("Accessibilité", [
                "Le site respecte les référentiels d'accessibilité reconnus.",
                "Une version en gros caractères des présentes conditions est disponible sur demande.",
                "Un service de relais texte est proposé aux acheteurs malentendants.",
                "Les conseillers peuvent rappeler l'acheteur à l'heure de son choix.",
                "Les retours sur l'accessibilité sont examinés chaque mois.",
            ]),
            ("Clients professionnels", [
                "Les comptes professionnels sont ouverts après une vérification de solvabilité.",
                "Plusieurs utilisateurs peuvent partager un même compte professionnel.",
                "Une facture mensuelle récapitulative est proposée aux comptes professionnels.",
                "Un chargé de compte est attribué aux grands clients professionnels.",
                "Le numéro de bon de commande peut figurer sur chaque facture.",
            ]),
        ],
    },
}  # fmt: skip
LONG_BUDGET = {"en": (10500, 15500), "fr": (9500, 14500)}  # characters of boilerplate


def _long_policy(rng, lang, split, domain, window_rule, rules):
    """A long numbered document; the domain's window and the other rules sit at random places."""
    t = LONG_POLICY[split, lang]
    li = 0 if lang == "en" else 1
    others = [d[li] for d in POLICY_DOMAINS[split] if d[li] != domain] + t["extra_categories"]
    windows = [7, 10, 14, 21, 30, 45, 60, 90]
    policy = POLICY_TEXT[split, lang]

    def fill(s, c=""):
        return s.format(k=int(rng.integers(2, 10)), c=c)

    def category(c):
        idx = rng.choice(
            len(t["category_sentences"]),
            size=int(rng.integers(5, len(t["category_sentences"]) + 1)),
            replace=False,
        )
        return t["category_title"].format(C=_cap(c), c=c), [
            fill(t["category_sentences"][i], c) for i in idx
        ]

    blocks = []
    for title, sentences in t["topics"]:
        idx = rng.choice(len(sentences), size=int(rng.integers(3, 6)), replace=False)
        blocks.append((title, [fill(sentences[i]) for i in idx]))
    for c in others:
        title, body = category(c)
        if rng.random() < 0.6:  # windows for *other* categories: the reader must match the one
            rule = policy["window"].format(d=c, w=int(rng.choice(windows)))
            body.insert(int(rng.integers(len(body) + 1)), rule)
        blocks.append((title, body))
    blocks = [blocks[i] for i in rng.permutation(len(blocks))]
    budget = int(rng.integers(*LONG_BUDGET[lang]))
    kept, size = [], 0
    for block in blocks:
        if size >= budget:
            break
        kept.append(block)
        size += len(block[0]) + sum(len(s) + 1 for s in block[1])
    title, body = category(domain)
    body.insert(int(rng.integers(len(body) + 1)), window_rule)
    kept.insert(int(rng.integers(len(kept) + 1)), (title, body))
    for _ in range(int(rng.integers(1, 3))):
        idx = rng.choice(len(t["refund_neutral"]), size=int(rng.integers(2, 5)), replace=False)
        refund = (_pick(rng, t["refund_titles"]), [fill(t["refund_neutral"][i]) for i in idx])
        kept.insert(int(rng.integers(len(kept) + 1)), refund)
    for rule in rules:
        body = kept[int(rng.integers(len(kept)))][1]
        body.insert(int(rng.integers(len(body) + 1)), rule)
    return "\n".join(
        f"{t['section']} {n}. {title}. " + " ".join(body) for n, (title, body) in enumerate(kept, 1)
    )


def refund_policy(rng, lang, split, long=False):
    domain = POLICY_DOMAINS[split][int(rng.integers(len(POLICY_DOMAINS[split])))][
        0 if lang == "en" else 1
    ]
    window = int(rng.choice([14, 30, 45]))
    member_bonus = int(rng.choice([15, 30]))
    limit = int(rng.choice([200, 500, 1000]))
    days = int(rng.integers(1, window + member_bonus + 15))
    member = bool(rng.integers(2))
    opened = bool(rng.integers(2))
    amount = float(rng.integers(20, 2 * limit))
    final_sale = rng.random() < 0.15
    allowed = window + (member_bonus if member else 0)
    if final_sale or days > allowed or (opened and domain in NO_OPEN[split]):
        decision = "deny"
    elif amount > limit:
        decision = "escalate"
    else:
        decision = "approve"
    t = POLICY_TEXT[split, lang]
    rules = [
        t["window"].format(d=domain, w=window),
        t["member"].format(b=member_bonus),
        t["no_open"],
        t["limit"].format(l=_money(limit, lang)),
        t["final"],
    ]
    (member_w, opened_w, final_w) = t["words"]
    case = t["case"].format(
        m=member_w[0] if member else member_w[1],
        c=t["category"].format(d=domain) if long else "",
        days=days,
        o=opened_w[0] if opened else opened_w[1],
        a=_money(amount, lang),
        f=final_w if final_sale else "",
    )
    order = list(rng.permutation(len(rules)))
    sections = [rules[i] for i in order]
    if long:  # bury the rules in a long policy document (2–4k tokens)
        policy = _long_policy(
            rng, lang, split, domain, rules[0], [r for r in sections if r != rules[0]]
        )
    else:
        policy = "\n".join(f"- {r}" for r in sections)
    return {
        "state": f"{t['head']}\n{policy}\n\n{case}",
        "expected": decision,
        "question": {"type": "choice", "instructions": t["q"], "criteria": t["crit"]},
    }


def long_refund_policy(rng, lang, split):
    return refund_policy(rng, lang, split, long=True)


# ---------------------------------------------------------------------------------------
# Support triage with abstention

# queue -> (EN sentences, FR sentences); {x} product (with its article), {X} capitalised.
TRIAGE = {
    "train": {
        "billing": (
            [
                "I was charged twice for {x}.",
                "My invoice for {x} shows the wrong amount.",
                "I was billed for {x} after I had already paid.",
                "Why did the price for {x} go up on my last bill?",
                "I need a copy of the receipt for {x}.",
            ],
            [
                "J'ai été débité deux fois pour {x}.",
                "Ma facture pour {x} indique un mauvais montant.",
                "On m'a facturé {x} alors que j'avais déjà payé.",
                "Pourquoi le prix pour {x} a-t-il augmenté sur ma dernière facture ?",
                "J'ai besoin d'un justificatif de paiement pour {x}.",
            ],
        ),
        "technical": (
            [
                "{X} crashes every time I open it.",
                "I get an error code when using {x}.",
                "{X} stopped working after the last update.",
                "Everything is extremely slow with {x} since yesterday.",
                "{X} keeps freezing and I have to restart it.",
            ],
            [
                "{X} plante à chaque ouverture.",
                "J'obtiens un code d'erreur en utilisant {x}.",
                "{X} ne fonctionne plus depuis la dernière mise à jour.",
                "Tout est extrêmement lent avec {x} depuis hier.",
                "{X} se fige et je dois tout redémarrer.",
            ],
        ),
        "account": (
            [
                "I can't log in to my account for {x}.",
                "Please change the email address on my account for {x}.",
                "I forgot my password for {x}.",
                "My account for {x} has been locked.",
                "How do I add a second user to my account for {x}?",
            ],
            [
                "Je n'arrive pas à me connecter à mon compte pour {x}.",
                "Merci de changer l'adresse e-mail de mon compte pour {x}.",
                "J'ai oublié mon mot de passe pour {x}.",
                "Mon compte pour {x} a été bloqué.",
                "Comment ajouter un second utilisateur sur mon compte pour {x} ?",
            ],
        ),
        "shipping": (
            [
                "My order for {x} hasn't arrived yet.",
                "{X} was delivered to the wrong address.",
                "The box with {x} arrived damaged.",
                "The tracking number for {x} doesn't work.",
                "I received a different model instead of {x}.",
            ],
            [
                "J'ai commandé {x} il y a deux semaines et rien n'est arrivé.",
                "Le colis contenant {x} a été livré à la mauvaise adresse.",
                "Le carton contenant {x} est arrivé abîmé.",
                "Le numéro de suivi du colis pour {x} ne fonctionne pas.",
                "Le modèle reçu ne correspond pas à ma commande pour {x}.",
            ],
        ),
    },
    "heldout": {
        "billing": (
            [
                "There is an unexpected charge for {x} on my card statement.",
                "The amount taken for {x} doesn't match the quote I was given.",
                "Could you reverse the duplicate payment for {x}?",
            ],
            [
                "Un prélèvement inattendu pour {x} apparaît sur mon relevé.",
                "Le montant prélevé pour {x} ne correspond pas au devis reçu.",
                "Pouvez-vous annuler le double paiement pour {x} ?",
            ],
        ),
        "technical": (
            [
                "{X} shows a blank screen whenever I try to use it.",
                "Since this morning {x} refuses to connect.",
                "{X} keeps rebooting on its own.",
            ],
            [
                "{X} affiche un écran noir dès que je veux l'utiliser.",
                "Depuis ce matin, impossible de se connecter avec {x}.",
                "{X} redémarre en boucle sans raison.",
            ],
        ),
        "account": (
            [
                "The two-factor code for {x} never arrives, so I'm locked out.",
                "I'd like to rename the profile attached to {x}.",
                "Someone else seems to be signed in to {x} with my details.",
            ],
            [
                "Le code de double authentification pour {x} n'arrive jamais, je suis bloqué.",
                "Je voudrais renommer le profil associé à mon compte pour {x}.",
                "Quelqu'un d'autre semble connecté avec mes identifiants pour {x}.",
            ],
        ),
        "shipping": (
            [
                "The courier says {x} was delivered, but nothing came.",
                "According to tracking, {x} is stuck at the depot.",
                "Parts of {x} are missing from the parcel I received.",
            ],
            [
                "Le transporteur indique que le colis pour {x} est livré, mais je n'ai rien reçu.",
                "D'après le suivi, le colis pour {x} est bloqué au dépôt.",
                "Il manque des pièces dans le colis contenant {x}.",
            ],
        ),
        "privacy": (
            [
                "Delete all the data you hold about me for {x}.",
                "Please send me a copy of the personal data stored for {x}.",
                "Stop using my details from {x} for marketing.",
            ],
            [
                "Supprimez toutes les données que vous avez sur moi pour {x}.",
                "Merci de m'envoyer une copie des données personnelles conservées pour {x}.",
                "Arrêtez d'utiliser mes coordonnées pour la prospection sur {x}.",
            ],
        ),
        "cancellation": (
            [
                "I want to end my subscription to {x}.",
                "Please close my plan for {x} at the end of the month.",
                "How do I stop the automatic renewal for {x}?",
            ],
            [
                "Je veux résilier mon abonnement pour {x}.",
                "Merci de clôturer ma formule pour {x} à la fin du mois.",
                "Comment arrêter le renouvellement automatique pour {x} ?",
            ],
        ),
    },
}
QUEUE_KIND = {
    "account": "service",
    "shipping": "device",
    "privacy": "service",
    "cancellation": "service",
}
PRODUCTS = {  # (name, kind); brand names are the same in both languages
    ("train", "en"): [
        ("the router", "device"), ("the smart watch", "device"), ("the printer", "device"),
        ("the wireless headphones", "device"), ("the tablet", "device"),
        ("the mobile app", "service"), ("Premium", "service"), ("CloudDrive", "service"),
        ("the desktop client", "service"), ("the family plan", "service"),
        ("PhotoBox", "service"), ("the web portal", "service"),
    ],
    ("train", "fr"): [
        ("le routeur", "device"), ("la montre connectée", "device"), ("l'imprimante", "device"),
        ("le casque sans fil", "device"), ("la tablette", "device"),
        ("l'application mobile", "service"), ("Premium", "service"), ("CloudDrive", "service"),
        ("le logiciel de bureau", "service"), ("la formule famille", "service"),
        ("PhotoBox", "service"), ("le portail web", "service"),
    ],
    ("heldout", "en"): [
        ("the thermostat", "device"), ("the e-bike", "device"), ("the security camera", "device"),
        ("StreamMax", "service"), ("the business plan", "service"), ("MailPro", "service"),
        ("the online banking app", "service"),
    ],
    ("heldout", "fr"): [
        ("le thermostat", "device"), ("le vélo électrique", "device"),
        ("la caméra de surveillance", "device"),
        ("StreamMax", "service"), ("la formule entreprise", "service"), ("MailPro", "service"),
        ("l'application bancaire", "service"),
    ],
}  # fmt: skip
QUEUE_TEXT = {  # (queue descriptions, "other" description, question)
    ("train", "en"): (
        {
            "billing": "Requests about billing and payments",
            "technical": "Requests about technical problems",
            "account": "Requests about account access and settings",
            "shipping": "Requests about deliveries",
        },
        "Not a support request for these queues",
        "Which support queue should handle this message?",
    ),
    ("train", "fr"): (
        {
            "billing": "Demandes liées à la facturation et aux paiements",
            "technical": "Demandes liées aux problèmes techniques",
            "account": "Demandes liées à l'accès et aux réglages du compte",
            "shipping": "Demandes liées aux livraisons",
        },
        "Pas une demande pour ces files",
        "Quelle file de support doit traiter ce message ?",
    ),
    ("heldout", "en"): (
        {
            "billing": "Payments, charges and invoices",
            "technical": "Faults, errors and bugs",
            "account": "Sign-in, passwords and profiles",
            "shipping": "Parcels, couriers and delivery issues",
            "privacy": "Personal data and privacy",
            "cancellation": "Ending or not renewing a subscription",
        },
        "None of these teams: not a customer-support matter",
        "Which team should this message be routed to?",
    ),
    ("heldout", "fr"): (
        {
            "billing": "Paiements, prélèvements et factures",
            "technical": "Pannes, erreurs et bogues",
            "account": "Connexion, mots de passe et profils",
            "shipping": "Colis, transporteurs et problèmes de livraison",
            "privacy": "Données personnelles et confidentialité",
            "cancellation": "Résiliation ou non-renouvellement d'un abonnement",
        },
        "Aucune de ces équipes : pas une demande de support",
        "À quelle équipe faut-il transmettre ce message ?",
    ),
}
OFF_TOPIC = {
    ("train", "en"): [
        "What's a good recipe for lasagna?",
        "Who won the match last night?",
        "Can you recommend a novel for the holidays?",
        "How tall is the Eiffel Tower?",
        "What time does the sun set today?",
        "Can you help me with my maths homework?",
        "What's the best way to learn the guitar?",
        "Which films are showing this weekend?",
        "How long should I boil an egg?",
        "Do you know a good hotel in Rome?",
        "What's the difference between a crocodile and an alligator?",
        "Can you translate 'good luck' into Japanese?",
    ],
    ("train", "fr"): [
        "Quelle est une bonne recette de lasagnes ?",
        "Qui a gagné le match hier soir ?",
        "Tu peux me conseiller un roman pour les vacances ?",
        "Quelle est la hauteur de la tour Eiffel ?",
        "À quelle heure le soleil se couche-t-il aujourd'hui ?",
        "Tu peux m'aider pour mes devoirs de maths ?",
        "Quelle est la meilleure façon d'apprendre la guitare ?",
        "Quels films passent au cinéma ce week-end ?",
        "Combien de temps faut-il faire cuire un œuf ?",
        "Tu connais un bon hôtel à Rome ?",
        "Quelle est la différence entre un crocodile et un alligator ?",
        "Comment dit-on « bonne chance » en japonais ?",
    ],
    ("heldout", "en"): [
        "Is it going to rain in Lyon tomorrow?",
        "Write me a poem about autumn.",
        "What's the capital of Australia?",
        "Any tips for growing tomatoes on a balcony?",
    ],
    ("heldout", "fr"): [
        "Va-t-il pleuvoir à Lyon demain ?",
        "Écris-moi un poème sur l'automne.",
        "Quelle est la capitale de l'Australie ?",
        "Des conseils pour faire pousser des tomates sur un balcon ?",
    ],
}
GREETINGS = {
    ("train", "en"): ["", "Hello, ", "Hi there. ", "Good morning, ", "Hey — "],
    ("train", "fr"): ["", "Bonjour, ", "Salut. ", "Bonsoir, ", "Madame, Monsieur, "],
    ("heldout", "en"): ["", "Dear support, ", "Hi team — ", "Evening. "],
    ("heldout", "fr"): ["", "Chère équipe, ", "Rebonjour. ", "Allô ? "],
}
CLOSINGS = {
    ("train", "en"): ["", " Thanks.", " Please help asap.", " Order #{n}.", " Ref {n}."],
    ("train", "fr"): ["", " Merci.", " Merci de faire vite.", " Commande n°{n}.", " Réf. {n}."],
    ("heldout", "en"): ["", " Cheers.", " Urgent please.", " Customer no. {n}.", " Case {n}."],
    ("heldout", "fr"): ["", " Cordialement.", " C'est urgent.", " Client n°{n}.", " Dossier {n}."],
}


def _after_greeting(template: str, x: str, greeting: str) -> str:
    """Fill {x}/{X}; after a comma greeting the sentence continues in lower case."""
    mid = greeting.endswith(", ")
    text = template.format(x=x, X=x if mid else _cap(x))
    if mid and not template.startswith("{X}") and not text.startswith(("I ", "I'")):
        text = text[0].lower() + text[1:]
    return text


def ticket_triage(rng, lang, split):
    table = TRIAGE[split]
    li = 0 if lang == "en" else 1
    queues = list(table)
    abstain = rng.random() < 0.15
    greeting = _pick(rng, GREETINGS[split, lang])
    closing = _pick(rng, CLOSINGS[split, lang]).format(n=int(rng.integers(10**5, 10**6)))
    if abstain:
        text, gold = _pick(rng, OFF_TOPIC[split, lang]), "other"
        state = greeting + _after_greeting(text, "", greeting) + closing
    else:
        gold = queues[int(rng.integers(len(queues)))]
        kind = QUEUE_KIND.get(gold)
        products = [p for p, k in PRODUCTS[split, lang] if kind in (None, k)]
        x = _pick(rng, products)
        state = greeting + _after_greeting(_pick(rng, table[gold][li]), x, greeting) + closing
    names, other, q = QUEUE_TEXT[split, lang]
    crit = {k: names[k] for k in queues}
    crit["other"] = other
    return {
        "state": state,
        "expected": gold,
        "question": {"type": "choice", "instructions": q, "criteria": crit},
    }


# ---------------------------------------------------------------------------------------
# Writing decisions (EN/FR): error injection, register

NAMES = {
    "train": ["Nadia", "Tom", "Léa", "Marc", "Sofia", "Paul", "Inès", "Karim"],
    "heldout": ["Yann", "Olga", "Rémi", "Ada"],
}
DOCS = {  # FR held-out documents are masculine and start with a consonant ("du {d}")
    ("train", "en"): ["report", "invoice", "proposal", "budget", "minutes"],
    ("train", "fr"): ["rapport", "devis", "budget", "compte rendu", "dossier"],
    ("heldout", "en"): ["contract", "timeline", "forecast", "brief", "spreadsheet"],
    ("heldout", "fr"): ["planning", "contrat", "bilan", "cahier des charges", "procès-verbal"],
}
# Slot templates: {n} name, {w} weekday, {d} document. Each (template, {word: misspelling});
# every misspelling is a real spelling or agreement error in context.
WRITING_TEMPLATES = {
    "en": {
        "train": [
            (
                "{n} will send the revised contract to the team before {w}.",
                {"revised": "revized", "contract": "contrat", "before": "befor"},
            ),
            (
                "The results of the survey were better than {n} expected.",
                {"results": "resluts", "survey": "survay", "expected": "expcted"},
            ),
            (
                "Please tell {n} whether the new schedule works on {w}.",
                {"schedule": "shedule", "whether": "wether", "Please": "Pleese"},
            ),
            (
                "The office will be closed on {w} because of the holiday.",
                {"because": "becuase", "closed": "closd", "holiday": "holliday"},
            ),
            (
                "{n} would appreciate it if you could review the {d}.",
                {"appreciate": "apreciate", "review": "reveiw", "could": "cuold"},
            ),
            (
                "I have received the {d} and will answer {n} on {w}.",
                {"received": "recieved", "answer": "anser", "will": "wil"},
            ),
            (
                "The client asked {n} to confirm the delivery address by {w}.",
                {"address": "adress", "delivery": "delivary", "client": "cleint"},
            ),
            (
                "We are planning to finish the {d} before the end of the month.",
                {"planning": "planing", "month": "mounth", "finish": "finsh"},
            ),
            (
                "{n} said the new process is much simpler than the old one.",
                {"process": "proccess", "simpler": "simplier", "said": "sayed"},
            ),
            (
                "It is necessary to update the {d} every {w}.",
                {"necessary": "neccessary", "update": "updat", "every": "evry"},
            ),
            (
                "{n} and the team were surprised by the positive feedback.",
                {"surprised": "suprised", "positive": "possitive", "were": "was"},
            ),
            (
                "The manager recommends that {n} attend the training on {w}.",
                {"recommends": "recomends", "attend": "atend", "training": "trainning"},
            ),
        ],
        "heldout": [
            (
                "{n} definitely needs the {d} by {w}.",
                {"definitely": "definately", "needs": "neads"},
            ),
            (
                "The committee agreed to postpone the {d} until {w}.",
                {"committee": "comittee", "agreed": "agred", "until": "untill"},
            ),
            (
                "Our colleague {n} occasionally works from home on {w}.",
                {"colleague": "collegue", "occasionally": "occassionally", "works": "work"},
            ),
            (
                "The {d} was accidentally deleted, so {n} restored it.",
                {"accidentally": "accidently", "deleted": "deletted", "restored": "restord"},
            ),
            (
                "{n} believes the new supplier is more reliable.",
                {"believes": "beleives", "supplier": "suplier", "reliable": "reliabel"},
            ),
            (
                "There is a separate budget for the equipment {n} ordered.",
                {"separate": "seperate", "equipment": "equipement", "is": "are"},
            ),
            (
                "Please acknowledge receipt of the {d} before {w}.",
                {"acknowledge": "acknowlege", "receipt": "reciept", "Please": "Plese"},
            ),
        ],
    },
    "fr": {
        "train": [
            (
                "{n} enverra le contrat révisé à l'équipe avant {w}.",
                {"révisé": "révisée", "enverra": "envera", "équipe": "équippe"},
            ),
            (
                "Les résultats de l'enquête étaient meilleurs que prévu, selon {n}.",
                {"résultats": "résultat", "meilleurs": "meilleur", "prévu": "prévue"},
            ),
            (
                "Merci de dire à {n} si le nouveau planning convient pour {w}.",
                {"planning": "planing", "nouveau": "nouvau", "convient": "conviens"},
            ),
            (
                "Les bureaux seront fermés {w} en raison du jour férié.",
                {"fermés": "fermé", "raison": "raisson", "férié": "férier"},
            ),
            (
                "{n} vous serait reconnaissant de relire le {d}.",
                {"reconnaissant": "reconaissant", "relire": "relir", "serait": "serai"},
            ),
            (
                "J'ai bien reçu le {d} et je répondrai à {n} {w}.",
                {"reçu": "reçue", "répondrai": "répondrait", "bien": "bein"},
            ),
            (
                "{n} a demandé de confirmer l'adresse de livraison avant {w}.",
                {"adresse": "addresse", "demandé": "demander", "livraison": "livraision"},
            ),
            (
                "Nous prévoyons de terminer le {d} avant la fin du mois.",
                {"prévoyons": "prévoyont", "terminer": "terminé", "mois": "moi"},
            ),
            (
                "Selon {n}, la nouvelle procédure est beaucoup plus simple.",
                {"nouvelle": "nouvel", "procédure": "procédur", "beaucoup": "beaucoups"},
            ),
            (
                "Il est nécessaire de mettre à jour le {d} chaque {w}.",
                {"nécessaire": "nécéssaire", "chaque": "chaques", "mettre": "mètre"},
            ),
            (
                "{n} et son équipe ont été surpris par les retours positifs.",
                {"surpris": "surprit", "retours": "retour", "positifs": "positif"},
            ),
            (
                "La responsable recommande que {n} assiste à la formation {w}.",
                {"recommande": "recommende", "assiste": "assistes", "formation": "formations"},
            ),
        ],
        "heldout": [
            (
                "{n} a absolument besoin du {d} pour {w}.",
                {"absolument": "absolumment", "besoin": "besion"},
            ),
            (
                "Le comité a accepté de reporter le {d} à {w}.",
                {"comité": "commité", "accepté": "acceptée", "reporter": "reportter"},
            ),
            (
                "Notre collègue {n} travaille parfois à distance le {w}.",
                {"collègue": "colègue", "travaille": "travailles", "parfois": "parfoix"},
            ),
            (
                "Le {d} a été supprimé par erreur, puis {n} l'a restauré.",
                {"supprimé": "suprimé", "erreur": "éreur", "restauré": "restaurer"},
            ),
            (
                "{n} pense que le nouveau fournisseur est plus fiable.",
                {"pense": "penses", "fournisseur": "fournisseurr", "fiable": "fiabble"},
            ),
            (
                "Un budget séparé est prévu pour le matériel commandé par {n}.",
                {"séparé": "séparée", "matériel": "matérielle", "commandé": "commander"},
            ),
            (
                "Merci d'accuser réception du {d} avant {w}.",
                {"accuser": "accusé", "réception": "récéption", "Merci": "Mercie"},
            ),
        ],
    },
}


def spelling_error(rng, lang, split):
    templates = WRITING_TEMPLATES[lang][split]
    names = NAMES[split]
    docs = DOCS[split, lang]
    picked = rng.choice(len(templates), size=1 + int(rng.random() < 0.5), replace=False)
    sentences = [
        templates[i][0].format(
            n=names[int(rng.integers(len(names)))],
            w=WEEKDAYS[lang][int(rng.integers(5))],
            d=docs[int(rng.integers(len(docs)))],
        )
        for i in picked
    ]
    has_error = rng.random() < 0.5
    if has_error:  # one error, in one of the sentences
        j = int(rng.integers(len(picked)))
        errors = templates[picked[j]][1]
        word = list(errors)[int(rng.integers(len(errors)))]
        sentences[j] = re.sub(rf"\b{re.escape(word)}\b", errors[word], sentences[j], count=1)
    q = {
        ("train", "en"): "Does the text contain a spelling or agreement error?",
        ("train", "fr"): "Le texte contient-il une faute d'orthographe ou d'accord ?",
        ("heldout", "en"): "Is there a spelling or grammar mistake in this text?",
        ("heldout", "fr"): "Y a-t-il une faute d'orthographe ou de grammaire dans ce texte ?",
    }[split, lang]
    return {
        "state": " ".join(sentences),
        "expected": "yes" if has_error else "no",
        "question": {"type": "noul", "instructions": q, "criteria": YES_NO[lang]},
    }


# (split, lang) -> level -> (openings, bodies, closings); {n} name, {w} weekday, {d} document
REGISTER = {
    ("train", "en"): {
        "formal": (
            ["Dear {n},", "Dear Sir or Madam,", "Good afternoon {n},", "To whom it may concern,"],
            [
                "I am writing to confirm our appointment on {w}.",
                "Please find attached the {d} you requested.",
                "I would be grateful if you could review the {d} at your earliest convenience.",
                "We would like to inform you that the {d} will be circulated on {w}.",
                "Kindly let us know whether {w} would be convenient for you.",
            ],
            ["Yours sincerely.", "Kind regards.", "With best regards.", "Yours faithfully."],
        ),
        "neutral": (
            ["Hi {n},", "Hello team,", "Hi all,", "Hello {n},"],
            [
                "the meeting is moved to {w}.",
                "here is the {d} you asked for.",
                "can you check the {d} before {w}?",
                "just a reminder that the {d} is due on {w}.",
                "I've updated the {d}, let me know if anything is missing.",
            ],
            ["Thanks!", "Best,", "Cheers.", "Thanks in advance."],
        ),
        "informal": (
            ["hey {n}", "yo", "hiya", "heyyy"],
            [
                "{w} works for me lol",
                "sent u the {d}, lmk",
                "can u look at the {d} b4 {w}??",
                "omg the {d} is sooo long",
                "gonna finish the {d} on {w} prob",
            ],
            [":)", "cya", "thx!!", "ttyl"],
        ),
    },
    ("train", "fr"): {
        "formal": (
            [
                "Madame, Monsieur,",
                "Chère Madame {n},",
                "Cher Monsieur {n},",
                "Monsieur le Directeur,",
            ],
            [
                "je vous confirme notre rendez-vous de {w}.",
                "veuillez trouver ci-joint le {d} demandé.",
                "je vous serais reconnaissant de bien vouloir relire le {d}.",
                "nous avons l'honneur de vous informer que le {d} sera diffusé {w}.",
                "je vous saurais gré de me faire savoir si {w} vous convient.",
            ],
            [
                "Je vous prie d'agréer mes salutations distinguées.",
                "Bien cordialement.",
                "Respectueusement.",
                "Veuillez agréer l'expression de ma considération distinguée.",
            ],
        ),
        "neutral": (
            ["Bonjour {n},", "Bonjour à tous,", "Bonjour l'équipe,", "Bonjour tout le monde,"],
            [
                "la réunion est déplacée à {w}.",
                "voici le {d} que tu m'as demandé.",
                "peux-tu vérifier le {d} avant {w} ?",
                "petit rappel : le {d} est à rendre pour {w}.",
                "j'ai mis à jour le {d}, dis-moi s'il manque quelque chose.",
            ],
            ["Merci !", "Bonne journée.", "À bientôt.", "Merci d'avance."],
        ),
        "informal": (
            ["salut {n}", "coucou", "yo", "slt {n}"],
            [
                "{w} ça me va mdr",
                "je t'ai envoyé le {d}, tu me dis",
                "tu peux checker le {d} avant {w} ??",
                "le {d} est trop long jpp",
                "je finis le {d} {w} normalement",
            ],
            [":)", "à plus", "bisous", "a+"],
        ),
    },
    ("heldout", "en"): {
        "formal": (
            ["Esteemed colleagues,", "Dear Dr {n},", "Dear Members of the Board,"],
            [
                "Please accept our apologies for the delay concerning the {d}.",
                "We should be most obliged if the {d} could be returned by {w}.",
                "Allow me to express my gratitude for your assistance with the {d}.",
            ],
            ["Respectfully yours.", "I remain at your disposal.", "With sincere appreciation."],
        ),
        "neutral": (
            ["Morning {n},", "Hey everyone,", "Good morning all,"],
            [
                "the {d} is ready for review.",
                "could we move our call to {w}?",
                "I added my comments to the {d}.",
            ],
            ["Talk soon.", "Many thanks,", "See you {w}."],
        ),
        "informal": (
            ["sup {n}", "ayy", "oi {n}"],
            [
                "the {d}?? done lol",
                "{w} is gonna be wild haha",
                "no way im reading that {d} rn",
            ],
            ["peace", "xx", "lmao ok"],
        ),
    },
    ("heldout", "fr"): {
        "formal": (
            ["Maître {n},", "Madame la Directrice,", "Mesdames, Messieurs,"],
            [
                "nous vous prions de bien vouloir excuser le retard concernant le {d}.",
                "je vous serais obligé de nous retourner le {d} avant {w}.",
                "permettez-moi de vous exprimer ma gratitude pour votre aide sur le {d}.",
            ],
            [
                "Veuillez recevoir mes respectueuses salutations.",
                "Je reste à votre entière disposition.",
                "Avec toute ma considération.",
            ],
        ),
        "neutral": (
            ["Rebonjour {n},", "Bonjour à toutes et à tous,", "Bonjour l'équipe projet,"],
            [
                "le {d} est prêt pour relecture.",
                "on peut décaler notre appel à {w} ?",
                "j'ai ajouté mes commentaires dans le {d}.",
            ],
            ["À plus tard.", "Merci beaucoup,", "À {w}."],
        ),
        "informal": (
            ["wesh {n}", "cc", "yo la team"],
            [
                "le {d} ?? fini tkt",
                "{w} ça va être chaud mdrr",
                "jsuis pas chaud pour lire ce {d} là",
            ],
            ["biz", "+++", "ciao"],
        ),
    },
}
REGISTER_CRITERIA = {
    ("train", "en"): ["informal / casual", "neutral / everyday professional", "formal"],
    ("train", "fr"): ["familier", "courant", "soutenu"],
    ("heldout", "en"): ["casual / chatty", "standard workplace tone", "very formal / official"],
    ("heldout", "fr"): ["relâché / très familier", "standard", "très soutenu / officiel"],
}


def register(rng, lang, split):
    levels = ["informal", "neutral", "formal"]
    gold = levels[int(rng.integers(3))]
    openings, bodies, closings = REGISTER[split, lang][gold]
    names = NAMES[split]
    docs = DOCS[split, lang]
    fill = {
        "n": names[int(rng.integers(len(names)))],
        "w": WEEKDAYS[lang][int(rng.integers(5))],
        "d": docs[int(rng.integers(len(docs)))],
    }
    parts = [
        openings[int(rng.integers(len(openings)))],
        bodies[int(rng.integers(len(bodies)))],
        closings[int(rng.integers(len(closings)))],
    ]
    text = " ".join(p.format(**fill) for p in parts)
    q = {
        ("train", "en"): "Rate the formality of the text.",
        ("train", "fr"): "Évaluez le niveau de formalité du texte.",
        ("heldout", "en"): "How formal is this message?",
        ("heldout", "fr"): "Quel est le degré de formalité de ce message ?",
    }[split, lang]
    return {
        "state": text,
        "expected": str(levels.index(gold)),
        "question": {
            "type": "score",
            "instructions": q,
            "criteria": REGISTER_CRITERIA[split, lang],
        },
    }


GENERATORS = {
    "gen_return_window": (return_window, "temporal"),
    "gen_weekday": (weekday, "temporal"),
    "gen_invoice_total": (invoice_total, "math"),
    "gen_base_rate": (base_rate, "probability"),
    "gen_schedule_conflict": (schedule_conflict, "temporal"),
    "gen_table_argmax": (table_argmax, "tables"),
    "gen_refund_policy": (refund_policy, "policy"),
    "gen_long_refund_policy": (long_refund_policy, "long_policy"),
    "gen_ticket_triage": (ticket_triage, "routing"),
    "gen_spelling_error": (spelling_error, "writing"),
    "gen_register": (register, "writing"),
}


def _normalized(text: str) -> str:  # as tjev.data.mix.normalized
    return re.sub(r"\W+", " ", text.lower()).strip()


def item_key(item: dict) -> str:
    return _normalized(item["state"]) + "\x1f" + _normalized(item["question"]["instructions"])


def generate(
    name: str, n: int, seed: int, *, fr_share: float = 0.3, split: str = "train"
) -> list[dict]:
    """``n`` items with distinct keys (fewer if the generator's space runs out)."""
    fn, family = GENERATORS[name]
    rng = np.random.default_rng([seed, zlib.crc32(name.encode()), zlib.crc32(split.encode())])
    out, seen = [], set()
    for _ in range(20 * n + 100):
        if len(out) == n:
            break
        lang = "fr" if rng.random() < fr_share else "en"
        item = fn(rng, lang, split)
        key = item_key(item)
        if key in seen:
            continue
        seen.add(key)
        item.update(id=f"{name}-{split}-{seed}-{len(out)}", source=name, family=family, lang=lang)
        out.append(item)
    return out
