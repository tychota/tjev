"""Exact-posterior worlds (mix-v7): states whose answer is a known probability.

Each world states every number the posterior needs, and the soft target is that posterior,
computed exactly (fractions, binomials, Bayes): two noisy signals, sampling a lot without
replacement, outcome frequencies of the matching past cases, a delay distribution against
a cut-off, and a rule with one unrecorded condition of known frequency. Most noul targets
fall inside the JevBench abstention band (0.2–0.8), where a one-hot label would teach
over-confidence; the choice targets are distributions over 3 outcomes.
"""

from __future__ import annotations

from math import comb

from tjev.data.sources.gen_v7.core import YES_NO, Group, Line, Member
from tjev.data.sources.generators import _pct

FAMILY = {f"pw_{k}": "probability" for k in ("signals", "sampling", "table", "delay", "policy")}


def _fam(srng, table, split, lang, base):
    s = "train" if split == "train" else "heldout"
    fams = table[s, lang]
    i = int(srng.integers(len(fams)))
    return f"{base}.{s}.{lang}.{i}", fams[i]


def _group(base, tid, lang, state_lines, question, target, **facts) -> Group:
    a = Member([Line(t) for t in state_lines], facts, target=target)
    return Group(base, tid, lang, question, a, None, (), {})


def _noul(instr, lang):
    return {"type": "noul", "instructions": instr, "criteria": YES_NO[lang]}


def _clip(p: float) -> float:
    return min(max(p, 0.0), 1.0)


# ---------------------------------------------------------------------------------------
# Two conditionally independent signals

SIGNALS = {
    ("train", "en"): [
        {"intro": "Among {pop}, {p} {are}.",
         "sig": "{S} is positive for {s} of {pos} and for {f} of the others.",
         "res": "For this {unit}: {S1} {r1}, {S2} {r2}.", "q": "Is this {unit} {thing}?",
         "pop": "machines of this model", "are": "have a cracked bearing", "unit": "machine",
         "thing": "fitted with a cracked bearing", "pos": "cracked bearings",
         "S": ["the vibration check", "the acoustic test"]},
        {"intro": "{p} of {pop} {are}.",
         "sig": "{S} flags {s} of {pos} and {f} of the rest.",
         "res": "This {unit} got: {S1} {r1}; {S2} {r2}.", "q": "Is the {unit} {thing}?",
         "pop": "incoming invoices", "are": "are duplicates", "unit": "invoice",
         "thing": "a duplicate", "pos": "duplicates",
         "S": ["the amount matcher", "the supplier-reference check"]},
        {"intro": "Historically {p} of {pop} {are}.",
         "sig": "{S} detects {s} of {pos}; it also fires on {f} of the others.",
         "res": "Results here: {S1} {r1} and {S2} {r2}.", "q": "Is this {unit} {thing}?",
         "pop": "new seller accounts", "are": "are fraudulent", "unit": "account",
         "thing": "fraudulent", "pos": "fraudulent accounts",
         "S": ["the device fingerprint", "the address check"]},
    ],
    ("train", "fr"): [
        {"intro": "Parmi les {pop}, {p} {are}.",
         "sig": "{S} est positif pour {s} des {pos} et pour {f} des autres.",
         "res": "Pour cette {unit} : {S1} {r1}, {S2} {r2}.", "q": "Cette {unit} {thing} ?",
         "pop": "machines de ce modèle", "are": "ont un roulement fissuré", "unit": "machine",
         "thing": "a-t-elle un roulement fissuré", "pos": "roulements fissurés",
         "S": ["le contrôle vibratoire", "le test acoustique"]},
        {"intro": "{p} des {pop} {are}.",
         "sig": "{S} signale {s} des {pos} et {f} des autres.",
         "res": "Cette {unit} a obtenu : {S1} {r1} ; {S2} {r2}.", "q": "La {unit} {thing} ?",
         "pop": "factures reçues", "are": "sont des doublons", "unit": "facture",
         "thing": "est-elle un doublon", "pos": "doublons",
         "S": ["le rapprochement des montants", "le contrôle de la référence fournisseur"]},
        {"intro": "Historiquement, {p} des {pop} {are}.",
         "sig": "{S} détecte {s} des {pos} ; il se déclenche aussi pour {f} des autres.",
         "res": "Résultats : {S1} {r1} et {S2} {r2}.", "q": "Ce {unit} {thing} ?",
         "pop": "nouveaux comptes vendeurs", "are": "sont frauduleux", "unit": "compte",
         "thing": "est-il frauduleux", "pos": "comptes frauduleux",
         "S": ["l'empreinte de l'appareil", "le contrôle d'adresse"]},
    ],
    ("heldout", "en"): [
        {"intro": "Of all {pop}, {p} {are}.",
         "sig": "{S} is positive on {s} of {pos} and on {f} of sound ones.",
         "res": "Readings for this {unit}: {S1} {r1}, {S2} {r2}.", "q": "Is this {unit} {thing}?",
         "pop": "bridge cables inspected", "are": "are corroded inside", "unit": "cable",
         "thing": "corroded inside", "pos": "corroded cables",
         "S": ["the magnetic scan", "the ultrasound probe"]},
    ],
    ("heldout", "fr"): [
        {"intro": "Sur l'ensemble des {pop}, {p} {are}.",
         "sig": "{S} est positif sur {s} des {pos} et sur {f} des câbles sains.",
         "res": "Mesures pour ce {unit} : {S1} {r1}, {S2} {r2}.", "q": "Ce {unit} {thing} ?",
         "pop": "câbles de pont inspectés", "are": "sont corrodés à cœur", "unit": "câble",
         "thing": "est-il corrodé à cœur", "pos": "câbles corrodés",
         "S": ["le scan magnétique", "la sonde à ultrasons"]},
    ],
}  # fmt: skip
POSNEG = {"en": ("positive", "negative"), "fr": ("positif", "négatif")}


def signals(rng, lang, split, srng) -> Group:
    tid, f = _fam(srng, SIGNALS, split, lang, "pw_signals")
    prior = int(rng.integers(5, 61)) / 100
    sens = [int(rng.integers(60, 99)) / 100 for _ in range(2)]
    fpr = [int(rng.integers(2, 35)) / 100 for _ in range(2)]
    res = [bool(rng.integers(2)) for _ in range(2)]
    like_pos, like_neg = prior, 1 - prior
    for s, fp, r in zip(sens, fpr, res, strict=True):
        like_pos *= s if r else 1 - s
        like_neg *= fp if r else 1 - fp
    post = like_pos / (like_pos + like_neg)
    S = f["S"]

    def pct(x):
        return _pct(x, 0, lang)

    lines = [f["intro"].format(p=pct(prior), pop=f["pop"], are=f["are"])]
    lines += [_cap1(f["sig"].format(S=S[i], s=pct(sens[i]), f=pct(fpr[i]), pos=f["pos"]))
              for i in range(2)]  # fmt: skip
    pn = POSNEG[lang]
    lines.append(f["res"].format(unit=f["unit"], S1=S[0], S2=S[1], r1=pn[0] if res[0] else pn[1],
                                 r2=pn[0] if res[1] else pn[1]))  # fmt: skip
    q = _noul(f["q"].format(unit=f["unit"], thing=f["thing"]), lang)
    return _group(
        "pw_signals",
        tid,
        lang,
        lines,
        q,
        {"yes": post, "no": 1 - post},
        prior=prior,
        sens=sens,
        fpr=fpr,
        res=res,
    )


def _cap1(s: str) -> str:
    return s[:1].upper() + s[1:]


# ---------------------------------------------------------------------------------------
# Sampling a lot without replacement

SAMPLING = {
    ("train", "en"): [
        ("A crate holds {N} {items}; exactly {K} of them are {bad}, and nobody knows which.",
         ("The inspector picks {m} at random, without replacement, and rejects the crate if at "
         "least {r} of the picked {items} {isbad}."), "Will the crate be rejected?",
         ("bottles", "cracked", "is cracked", "are cracked")),
        ("Of the {N} {items} in this batch, {K} are {bad} (the count is certain).",
         ("Quality control draws {m} {items} at random without putting any back; the batch "
         "fails if {r} or more of them {isbad}."), "Does the batch fail?",
         ("circuit boards", "faulty", "is faulty", "are faulty")),
        ("The box contains {N} {items}, {K} of which are {bad}.",
         ("A clerk takes {m} {items} at random (no replacement) for an audit and escalates if at "
         "least {r} {isbad}."), "Will the audit be escalated?",
         ("expense files", "incomplete", "is incomplete", "are incomplete")),
    ],
    ("train", "fr"): [
        ("Une caisse contient {N} {items} ; exactement {K} sont {bad}, sans qu'on sache lesquels.",
         ("L'inspecteur en prélève {m} au hasard, sans remise, et refuse la caisse si au moins {r} "
         "des {items} prélevés {isbad}."), "La caisse sera-t-elle refusée ?",
         ("bouteilles", "fêlées", "est fêlée", "sont fêlées")),
        ("Sur les {N} {items} de ce lot, {K} sont {bad} (le compte est certain).",
         ("Le contrôle qualité en tire {m} au hasard sans remise ; le lot est rejeté si {r} ou "
         "plus {isbad}."), "Le lot est-il rejeté ?",
         ("cartes électroniques", "défectueuses", "est défectueuse", "sont défectueuses")),
        ("La boîte contient {N} {items}, dont {K} sont {bad}.",
         ("Un agent en prend {m} au hasard (sans remise) pour un audit et alerte si au moins {r} "
         "{isbad}."), "L'audit donnera-t-il lieu à une alerte ?",
         ("dossiers de frais", "incomplets", "est incomplet", "sont incomplets")),
    ],
    ("heldout", "en"): [
        ("The pallet carries {N} {items}; {K} of them are {bad}.",
         ("The receiving dock opens {m} {items} chosen at random, never the same one twice, and "
         "refuses the pallet when {r} or more {isbad}."), "Will the pallet be refused?",
         ("cartons", "water-damaged", "is water-damaged", "are water-damaged")),
    ],
    ("heldout", "fr"): [
        ("La palette porte {N} {items} ; {K} d'entre eux sont {bad}.",
         ("Le quai de réception ouvre {m} {items} tirés au hasard, jamais deux fois le même, et "
         "refuse la palette si {r} ou plus {isbad}."), "La palette sera-t-elle refusée ?",
         ("cartons", "mouillés", "est mouillé", "sont mouillés")),
    ],
}  # fmt: skip


def p_at_least(N: int, K: int, m: int, r: int) -> float:
    """P(at least r bad among m drawn without replacement from N with K bad)."""
    return sum(comb(K, k) * comb(N - K, m - k) for k in range(r, min(K, m) + 1)) / comb(N, m)


def sampling(rng, lang, split, srng) -> Group:
    tid, (intro, rule, q, (items, bad, is1, are)) = _fam(srng, SAMPLING, split, lang,
                                                          "pw_sampling")  # fmt: skip
    N = int(rng.integers(8, 41))
    K = int(rng.integers(2, max(3, N // 3)))  # ≥ 2: "K of them are" (plural)
    m = int(rng.integers(2, min(N - 1, 8) + 1))
    r = 1 if rng.random() < 0.7 or K < 2 or m < 2 else 2
    p = p_at_least(N, K, m, r)
    isbad = is1 if r == 1 else are
    lines = [intro.format(N=N, K=K, items=items, bad=bad),
             rule.format(m=m, r=r, items=items, isbad=isbad)]  # fmt: skip
    return _group(
        "pw_sampling", tid, lang, lines, _noul(q, lang), {"yes": p, "no": 1 - p}, N=N, K=K, m=m, r=r
    )


# ---------------------------------------------------------------------------------------
# Outcome frequencies of the matching past cases (choice)

TABLE = {
    ("train", "en"): [
        {"intro": "Past {cases} by {seg} (counts):", "row": "- {s}: {counts}",
         "case": "A new {noun} comes in: {segw} {s}.",
         "q": "How will this {noun} end? Use the share of each outcome among past {cases} of "
              "the same {seg}.",
         "cases": "support tickets", "noun": "ticket", "seg": "product tier", "segw": "tier",
         "segs": ["Basic", "Plus", "Enterprise", "Starter"],
         "out": {"solved": "Solved at first contact", "escalated": "Escalated",
                 "reopened": "Reopened later"}},
        {"intro": "Deliveries last quarter, by carrier:", "row": "{s} — {counts}",
         "case": "Today's parcel goes with {s}.",
         "q": "Estimate the parcel's outcome from that carrier's record last quarter.",
         "cases": "deliveries", "noun": "parcel", "seg": "carrier", "segw": "",
         "segs": ["Rapido", "Swiftline", "Parcelwise", "Northpost"],
         "out": {"early": "Early", "on_time": "On the promised day", "late": "Late"}},
        {"intro": "Loan applications reviewed this year, by income band:", "row": "* {s}: {counts}",
         "case": "The next applicant is in the {s} band.",
         "q": "What decision should be expected? Base it on applicants of the same band.",
         "cases": "applications", "noun": "application", "seg": "band", "segw": "",
         "segs": ["low", "middle", "high", "very high"],
         "out": {"approved": "Approved", "declined": "Declined", "referred": "Referred"}},
    ],
    ("train", "fr"): [
        {"intro": "Tickets passés par {seg} (effectifs) :", "row": "- {s} : {counts}",
         "case": "Un nouveau {noun} arrive : {segw} {s}.",
         "q": "Comment ce {noun} va-t-il se terminer ? Utilisez la part de chaque issue parmi "
              "les tickets passés de la même {seg}.",
         "cases": "tickets", "noun": "ticket", "seg": "offre", "segw": "offre",
         "segs": ["Basique", "Plus", "Entreprise", "Découverte"],
         "out": {"solved": "Résolu au premier contact", "escalated": "Escaladé",
                 "reopened": "Rouvert plus tard"}},
        {"intro": "Livraisons du dernier trimestre, par transporteur :", "row": "{s} — {counts}",
         "case": "Le colis du jour part avec {s}.",
         "q": "Estimez l'issue du colis d'après les résultats de ce transporteur au dernier "
              "trimestre.",
         "cases": "livraisons", "noun": "colis", "seg": "transporteur", "segw": "",
         "segs": ["Rapido", "Swiftline", "Parcelwise", "Northpost"],
         "out": {"early": "En avance", "on_time": "Le jour promis", "late": "En retard"}},
        {"intro": "Demandes de prêt examinées cette année, par tranche de revenus :",
         "row": "* {s} : {counts}", "case": "Le prochain demandeur est dans la tranche {s}.",
         "q": "Quelle décision faut-il attendre ? Fondez-vous sur les demandeurs de la même tranche.",
         "cases": "demandes", "noun": "demande", "seg": "tranche", "segw": "",
         "segs": ["basse", "moyenne", "haute", "très haute"],
         "out": {"approved": "Acceptée", "declined": "Refusée", "referred": "Transmise"}},
    ],
    ("heldout", "en"): [
        {"intro": "Insurance claims closed in 2025, by region:", "row": "{s}: {counts}",
         "case": "A claim has just been filed in {s}.",
         "q": "Forecast the claim's settlement using the 2025 claims of that region.",
         "cases": "claims", "noun": "claim", "seg": "region", "segw": "",
         "segs": ["North", "Coast", "Valley", "Capital"],
         "out": {"paid_full": "Paid in full", "paid_part": "Partly paid", "rejected": "Rejected"}},
    ],
    ("heldout", "fr"): [
        {"intro": "Sinistres clos en 2025, par région :", "row": "{s} : {counts}",
         "case": "Un sinistre vient d'être déclaré en région {s}.",
         "q": "Prévoyez le règlement du sinistre d'après les sinistres 2025 de cette région.",
         "cases": "sinistres", "noun": "sinistre", "seg": "région", "segw": "",
         "segs": ["Nord", "Littoral", "Vallée", "Capitale"],
         "out": {"paid_full": "Payé en totalité", "paid_part": "Payé en partie", "rejected": "Refusé"}},
    ],
}  # fmt: skip


def table(rng, lang, split, srng) -> Group:
    tid, f = _fam(srng, TABLE, split, lang, "pw_table")
    segs = list(rng.choice(f["segs"], size=int(rng.integers(2, 5)), replace=False))
    outs = list(f["out"])
    counts = {s: [int(rng.integers(0, 60)) for _ in outs] for s in segs}
    for s in segs:
        if sum(counts[s]) == 0:
            counts[s][0] = 1
    pick = segs[int(rng.integers(len(segs)))]
    sep = " ; " if lang == "fr" else "; "
    colon = " : " if lang == "fr" else ": "
    lines = [f["intro"].format(seg=f["seg"], cases=f["cases"])]
    for s in segs:
        text = sep.join(f"{f['out'][o].lower()}{colon}{c}" for o, c in zip(outs, counts[s],
                                                                          strict=True))  # fmt: skip
        lines.append(f["row"].format(s=s, counts=text))
    lines.append(f["case"].format(noun=f["noun"], segw=f["segw"], s=pick).replace("  ", " "))
    total = sum(counts[pick])
    target = {o: c / total for o, c in zip(outs, counts[pick], strict=True)}
    q = {"type": "choice", "instructions": f["q"].format(noun=f["noun"], cases=f["cases"],
                                                         seg=f["seg"]),
         "criteria": dict(f["out"])}  # fmt: skip
    return _group("pw_table", tid, lang, lines, q, target, counts=counts, pick=pick, outs=outs)


# ---------------------------------------------------------------------------------------
# A delay distribution against a cut-off (noul)

DELAY = {
    ("train", "en"): [
        ("The courier's arrival delay on this route has been: {dist}.",
         "The shop closes {slack} minutes after the scheduled arrival.",
         "Will the courier arrive before the shop closes?", "on time"),
        ("Nightly batch overruns (minutes past the scheduled end) were: {dist}.",
         "The reports must be out within {slack} minutes of the scheduled end.",
         "Will tonight's reports be out in time?", "no overrun"),
        ("Recorded lateness of the 7:42 train: {dist}.",
         "The connecting bus waits {slack} minutes after the train's scheduled arrival.",
         "Will a passenger on the 7:42 catch the bus?", "on time"),
    ],
    ("train", "fr"): [
        ("Le retard du coursier sur cette tournée a été : {dist}.",
         "Le magasin ferme {slack} minutes après l'arrivée prévue.",
         "Le coursier arrivera-t-il avant la fermeture du magasin ?", "à l'heure"),
        ("Les dépassements du traitement de nuit (minutes après la fin prévue) ont été : {dist}.",
         "Les rapports doivent sortir dans les {slack} minutes qui suivent la fin prévue.",
         "Les rapports de cette nuit sortiront-ils à temps ?", "aucun dépassement"),
        ("Retards relevés pour le train de 7 h 42 : {dist}.",
         "Le bus de correspondance attend {slack} minutes après l'arrivée prévue du train.",
         "Un voyageur du train de 7 h 42 aura-t-il le bus ?", "à l'heure"),
    ],
    ("heldout", "en"): [
        ("The ferry's late departures over the season: {dist}.",
         ("The coach transfer still connects if the ferry leaves at most {slack} minutes "
         "after its scheduled time."),
         "Will today's ferry passengers make the coach transfer?", "on schedule"),
    ],
    ("heldout", "fr"): [
        ("Retards du ferry au départ sur la saison : {dist}.",
         ("La navette en car assure encore la correspondance si le ferry part au plus "
         "{slack} minutes après l'heure prévue."),
         "Les passagers du ferry d'aujourd'hui auront-ils la navette ?", "à l'heure"),
    ],
}  # fmt: skip


def delay(rng, lang, split, srng) -> Group:
    tid, (dist_t, slack_t, q, zero) = _fam(srng, DELAY, split, lang, "pw_delay")
    k = int(rng.integers(3, 6))
    values = [0, *sorted(int(x) for x in rng.choice(range(5, 61, 5), size=k - 1, replace=False))]
    raw = [int(rng.integers(1, 10)) for _ in range(k)]
    shares = [round(100 * r / sum(raw)) for r in raw]
    shares[-1] = 100 - sum(shares[:-1])
    if min(shares) <= 0:  # rounding left the last value empty: even shares
        shares = [100 // k] * k
        shares[-1] = 100 - sum(shares[:-1])
    slack = int(rng.choice(range(5, values[-1], 5)))  # under the largest delay: p < 1
    late = {"en": "{v} minutes late in {s}% of cases", "fr": "{v} minutes dans {s} % des cas"}
    first = {"en": "{z} in {s}% of cases", "fr": "{z} dans {s} % des cas"}
    dist = ", ".join(first[lang].format(z=zero, s=s) if v == 0 else late[lang].format(v=v, s=s)
                     for v, s in zip(values, shares, strict=True))  # fmt: skip
    p = sum(s for v, s in zip(values, shares, strict=True) if v <= slack) / 100
    lines = [dist_t.format(dist=dist), slack_t.format(slack=slack)]
    return _group(
        "pw_delay",
        tid,
        lang,
        lines,
        _noul(q, lang),
        {"yes": p, "no": 1 - p},
        values=values,
        shares=shares,
        slack=slack,
    )


# ---------------------------------------------------------------------------------------
# A rule with an unrecorded condition of known frequency (noul)

POLICY = {
    ("train", "en"): [
        {"rule": "A return is refunded if it is within {w} days and the item is {cond}.",
         "known": "This return comes {n} days after purchase.",
         "unknown": "Whether the item is {cond} was not recorded; {p} of returns of this "
                    "product are {cond}.",
         "q": "Will this return be refunded?", "cond": "unopened"},
        {"rule": "A visitor gets free parking if they stay under {w} hours and hold a {cond}.",
         "known": "The visitor stayed {n} hours.",
         "unknown": "Nobody checked for a {cond}; {p} of visitors hold one.",
         "q": "Does this visitor get free parking?", "cond": "validated ticket"},
        {"rule": "A claim is fast-tracked if it is below {w} hundred dollars and the customer "
                 "is {cond}.",
         "known": "The claim is for {n} hundred dollars.",
         "unknown": "The customer's status was not looked up; {p} of claimants are {cond}.",
         "q": "Will the claim be fast-tracked?", "cond": "a premium member"},
    ],
    ("train", "fr"): [
        {"rule": "Un retour est remboursé s'il a lieu dans les {w} jours et si l'article est {cond}.",
         "known": "Ce retour a lieu {n} jours après l'achat.",
         "unknown": "On n'a pas noté si l'article est {cond} ; {p} des retours de ce produit "
                    "sont {conds}.",
         "q": "Ce retour sera-t-il remboursé ?", "cond": "non ouvert", "conds": "non ouverts"},
        {"rule": "Un visiteur stationne gratuitement s'il reste moins de {w} heures et a un {cond}.",
         "known": "Le visiteur est resté {n} heures.",
         "unknown": "Personne n'a vérifié le {cond} ; {p} des visiteurs en ont un.",
         "q": "Ce visiteur stationne-t-il gratuitement ?", "cond": "ticket validé"},
        {"rule": "Un dossier est traité en priorité s'il est inférieur à {w} centaines d'euros "
                 "et si le client est {cond}.",
         "known": "Le dossier porte sur {n} centaines d'euros.",
         "unknown": "Le statut du client n'a pas été consulté ; {p} des demandeurs sont {conds}.",
         "q": "Le dossier sera-t-il traité en priorité ?", "cond": "membre premium",
         "conds": "membres premium"},
    ],
    ("heldout", "en"): [
        {"rule": "A late assignment is accepted without penalty if it is under {w} days late "
                 "and the student has a {cond}.",
         "known": "The assignment is {n} days late.",
         "unknown": "The registry has not confirmed a {cond}; {p} of late students have one.",
         "q": "Is the assignment accepted without penalty?", "cond": "medical note"},
    ],
    ("heldout", "fr"): [
        {"rule": "Un devoir rendu en retard est accepté sans pénalité si le retard est inférieur "
                 "à {w} jours et si l'étudiant a un {cond}.",
         "known": "Le devoir a {n} jours de retard.",
         "unknown": "La scolarité n'a pas confirmé de {cond} ; {p} des étudiants en retard en "
                    "ont un.",
         "q": "Le devoir est-il accepté sans pénalité ?", "cond": "certificat médical"},
    ],
}  # fmt: skip


def policy(rng, lang, split, srng) -> Group:
    tid, f = _fam(srng, POLICY, split, lang, "pw_policy")
    w = int(rng.integers(3, 31))
    p_cond = int(rng.integers(10, 91)) / 100
    inside = rng.random() < 0.75
    # n < w or n > w: "within w days" (≤) and "under w" (<) read the same
    n = int(rng.integers(1, w)) if inside else int(rng.integers(w + 1, w + 15))
    p = p_cond if inside else 0.0
    lines = [f["rule"].format(w=w, cond=f["cond"]), f["known"].format(n=n),
             f["unknown"].format(cond=f["cond"], conds=f.get("conds", f["cond"]),
                                 p=_pct(p_cond, 0, lang))]  # fmt: skip
    return _group(
        "pw_policy",
        tid,
        lang,
        lines,
        _noul(f["q"], lang),
        {"yes": p, "no": 1 - p},
        w=w,
        n=n,
        p_cond=p_cond,
    )


BASES = {"pw_signals": signals, "pw_sampling": sampling, "pw_table": table, "pw_delay": delay,
         "pw_policy": policy}  # fmt: skip
