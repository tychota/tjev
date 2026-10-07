"""Public datasets reframed as JevBench-style rubric decisions (the mix's "replay" block).

Each adapter maps one HF row to zero or more item dicts (see ``item.py``). Labels are
*categories defined by criteria* (not answer strings), matching JevBench. Intent sets
become menus: gold plus sampled distractors, sometimes with an ``other`` option, so the
model learns routing over variable menus and abstention.

License flags are copied from dataset cards and must be re-checked before commercial
use; ``commercial_ok=False`` sources are excluded by ``--commercial-only``.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from tjev.data.sources.base import Adapter, Row, Source


def _human(label: str) -> str:
    return re.sub(r"[_/]+", " ", label).strip()


# French descriptions of the MASSIVE intents (the dataset's names are English identifiers).
MASSIVE_FR = {
    "alarm_query": "consulter les alarmes",
    "alarm_remove": "supprimer une alarme",
    "alarm_set": "régler une alarme",
    "audio_volume_down": "baisser le volume",
    "audio_volume_mute": "couper le son",
    "audio_volume_other": "un autre réglage du volume",
    "audio_volume_up": "monter le volume",
    "calendar_query": "consulter l'agenda",
    "calendar_remove": "supprimer un événement de l'agenda",
    "calendar_set": "ajouter un événement à l'agenda",
    "cooking_query": "une question de cuisine",
    "cooking_recipe": "une recette",
    "datetime_convert": "convertir une heure ou un fuseau horaire",
    "datetime_query": "la date ou l'heure",
    "email_addcontact": "ajouter un contact e-mail",
    "email_query": "consulter ses e-mails",
    "email_querycontact": "les coordonnées d'un contact",
    "email_sendemail": "envoyer un e-mail",
    "general_greet": "une salutation",
    "general_joke": "une blague",
    "general_quirky": "une remarque ou question sans demande précise",
    "iot_cleaning": "lancer l'aspirateur ou le ménage",
    "iot_coffee": "faire du café",
    "iot_hue_lightchange": "changer la couleur des lumières",
    "iot_hue_lightdim": "baisser la lumière",
    "iot_hue_lightoff": "éteindre la lumière",
    "iot_hue_lighton": "allumer la lumière",
    "iot_hue_lightup": "augmenter la lumière",
    "iot_wemo_off": "éteindre une prise connectée",
    "iot_wemo_on": "allumer une prise connectée",
    "lists_createoradd": "créer une liste ou y ajouter un élément",
    "lists_query": "consulter une liste",
    "lists_remove": "retirer un élément d'une liste",
    "music_dislikeness": "dire qu'on n'aime pas un morceau",
    "music_likeness": "dire qu'on aime un morceau",
    "music_query": "une question sur la musique en cours",
    "music_settings": "les réglages de lecture (aléatoire, répétition)",
    "news_query": "les actualités",
    "play_audiobook": "écouter un livre audio",
    "play_game": "jouer à un jeu",
    "play_music": "écouter de la musique",
    "play_podcasts": "écouter un podcast",
    "play_radio": "écouter la radio",
    "qa_currency": "un taux de change",
    "qa_definition": "la définition d'un mot",
    "qa_factoid": "une question de culture générale",
    "qa_maths": "un calcul",
    "qa_stock": "le cours d'une action",
    "recommendation_events": "des suggestions de sorties ou d'événements",
    "recommendation_locations": "des suggestions de lieux",
    "recommendation_movies": "des suggestions de films",
    "social_post": "publier sur un réseau social",
    "social_query": "consulter un réseau social",
    "takeaway_order": "commander un plat à emporter",
    "takeaway_query": "une question sur une commande à emporter",
    "transport_query": "une question de transport",
    "transport_taxi": "réserver un taxi",
    "transport_ticket": "réserver un billet",
    "transport_traffic": "l'état de la circulation",
    "weather_query": "la météo",
}
# French descriptions of the MTOP intents (the dataset's names are English identifiers).
MTOP_FR = {
    "ADD_TIME_TIMER": "ajouter du temps à un minuteur",
    "ADD_TO_PLAYLIST_MUSIC": "ajouter un morceau à une playlist",
    "ANSWER_CALL": "décrocher un appel",
    "CANCEL_MESSAGE": "annuler un message",
    "CREATE_ALARM": "créer une alarme",
    "CREATE_CALL": "passer un appel",
    "CREATE_PLAYLIST_MUSIC": "créer une playlist",
    "CREATE_REMINDER": "créer un rappel",
    "CREATE_TIMER": "lancer un minuteur",
    "DELETE_ALARM": "supprimer une alarme",
    "DELETE_PLAYLIST_MUSIC": "supprimer une playlist",
    "DELETE_REMINDER": "supprimer un rappel",
    "DELETE_TIMER": "supprimer un minuteur",
    "DISLIKE_MUSIC": "dire qu'on n'aime pas un morceau",
    "END_CALL": "raccrocher",
    "FAST_FORWARD_MUSIC": "avancer dans le morceau",
    "FOLLOW_MUSIC": "suivre un artiste ou une playlist",
    "GET_AGE": "l'âge d'une personne",
    "GET_AIRQUALITY": "la qualité de l'air",
    "GET_ALARM": "consulter les alarmes",
    "GET_ATTENDEE_EVENT": "les participants à un événement",
    "GET_AVAILABILITY": "la disponibilité d'une personne",
    "GET_CALL": "les informations sur un appel",
    "GET_CALL_CONTACT": "la personne au bout du fil",
    "GET_CALL_TIME": "la durée ou l'heure d'un appel",
    "GET_CATEGORY_EVENT": "le type d'un événement",
    "GET_CONTACT": "les coordonnées d'un contact",
    "GET_CONTACT_METHOD": "le moyen de joindre un contact",
    "GET_DATE_TIME_EVENT": "la date ou l'heure d'un événement",
    "GET_DETAILS_NEWS": "les détails d'une actualité",
    "GET_EDUCATION_DEGREE": "le diplôme d'une personne",
    "GET_EDUCATION_TIME": "les dates d'études d'une personne",
    "GET_EMPLOYER": "l'employeur d'une personne",
    "GET_EMPLOYMENT_TIME": "depuis quand une personne occupe son poste",
    "GET_EVENT": "les événements prévus",
    "GET_GENDER": "le genre d'une personne",
    "GET_INFO_CONTACT": "des informations sur un contact",
    "GET_INFO_RECIPES": "des informations sur une recette",
    "GET_JOB": "le métier d'une personne",
    "GET_LANGUAGE": "les langues parlées par une personne",
    "GET_LIFE_EVENT": "un événement de la vie d'une personne",
    "GET_LIFE_EVENT_TIME": "la date d'un événement de la vie d'une personne",
    "GET_LOCATION": "le lieu où se trouve une personne",
    "GET_LYRICS_MUSIC": "les paroles d'une chanson",
    "GET_MAJOR": "la spécialité d'études d'une personne",
    "GET_MESSAGE": "lire ses messages",
    "GET_MESSAGE_CONTACT": "l'expéditeur d'un message",
    "GET_MUTUAL_FRIENDS": "les amis en commun",
    "GET_RECIPES": "trouver une recette",
    "GET_REMINDER": "consulter ses rappels",
    "GET_REMINDER_AMOUNT": "le nombre de rappels",
    "GET_REMINDER_DATE_TIME": "la date ou l'heure d'un rappel",
    "GET_REMINDER_LOCATION": "le lieu associé à un rappel",
    "GET_STORIES_NEWS": "les actualités",
    "GET_SUNRISE": "l'heure du lever du soleil",
    "GET_SUNSET": "l'heure du coucher du soleil",
    "GET_TIMER": "consulter un minuteur",
    "GET_TRACK_INFO_MUSIC": "des informations sur le morceau en cours",
    "GET_UNDERGRAD": "les études supérieures d'une personne",
    "GET_WEATHER": "la météo",
    "HELP_REMINDER": "de l'aide sur les rappels",
    "HOLD_CALL": "mettre un appel en attente",
    "IGNORE_CALL": "ignorer un appel",
    "IS_TRUE_RECIPES": "vérifier une information de cuisine",
    "LIKE_MUSIC": "dire qu'on aime un morceau",
    "LOOP_MUSIC": "écouter un morceau en boucle",
    "MERGE_CALL": "fusionner des appels",
    "PAUSE_MUSIC": "mettre la musique en pause",
    "PAUSE_TIMER": "mettre un minuteur en pause",
    "PLAY_MEDIA": "lire un média (vidéo, podcast, livre audio)",
    "PLAY_MUSIC": "écouter de la musique",
    "PREFER": "exprimer une préférence",
    "PREVIOUS_TRACK_MUSIC": "revenir au morceau précédent",
    "QUESTION_MUSIC": "une question sur la musique",
    "QUESTION_NEWS": "une question sur l'actualité",
    "REMOVE_FROM_PLAYLIST_MUSIC": "retirer un morceau d'une playlist",
    "REPEAT_ALL_MUSIC": "répéter toute la playlist",
    "REPEAT_ALL_OFF_MUSIC": "arrêter la répétition de la playlist",
    "REPLAY_MUSIC": "réécouter un morceau",
    "RESTART_TIMER": "relancer un minuteur",
    "RESUME_CALL": "reprendre un appel",
    "RESUME_MUSIC": "reprendre la musique",
    "RESUME_TIMER": "reprendre un minuteur",
    "REWIND_MUSIC": "revenir en arrière dans le morceau",
    "SEND_MESSAGE": "envoyer un message",
    "SET_AVAILABLE": "se déclarer disponible",
    "SET_DEFAULT_PROVIDER_CALLING": "choisir l'application d'appel par défaut",
    "SET_DEFAULT_PROVIDER_MUSIC": "choisir le service de musique par défaut",
    "SET_RSVP_INTERESTED": "se dire intéressé par un événement",
    "SET_RSVP_NO": "décliner une invitation",
    "SET_RSVP_YES": "accepter une invitation",
    "SET_UNAVAILABLE": "se déclarer indisponible",
    "SHARE_EVENT": "partager un événement",
    "SILENCE_ALARM": "couper une alarme qui sonne",
    "SKIP_TRACK_MUSIC": "passer au morceau suivant",
    "SNOOZE_ALARM": "répéter une alarme plus tard",
    "START_SHUFFLE_MUSIC": "lancer la lecture aléatoire",
    "STOP_MUSIC": "arrêter la musique",
    "STOP_SHUFFLE_MUSIC": "arrêter la lecture aléatoire",
    "SUBTRACT_TIME_TIMER": "retirer du temps à un minuteur",
    "SWITCH_CALL": "basculer entre deux appels",
    "UNLOOP_MUSIC": "arrêter la lecture en boucle",
    "UPDATE_ALARM": "modifier une alarme",
    "UPDATE_CALL": "modifier un appel en cours",
    "UPDATE_METHOD_CALL": "passer un appel en vidéo ou en audio",
    "UPDATE_REMINDER": "modifier un rappel",
    "UPDATE_REMINDER_DATE_TIME": "changer la date ou l'heure d'un rappel",
    "UPDATE_REMINDER_LOCATION": "changer le lieu d'un rappel",
    "UPDATE_REMINDER_TODO": "changer le contenu d'un rappel",
    "UPDATE_TIMER": "modifier un minuteur",
}
MENU_TEXT = {"en": "The request is about {}", "fr": "La demande concerne : {}"}


def _describe(name: str, lang: str) -> str:
    words = (MASSIVE_FR.get(name) or MTOP_FR.get(name)) if lang == "fr" else None
    if lang == "fr" and words is None:
        raise KeyError(f"no French description for intent {name!r}")
    return MENU_TEXT[lang].format(words or _human(name))


def _menu(
    gold: str,
    names: list[str],
    rng,
    *,
    lang: str = "en",
    lo=4,
    hi=12,
    other_prob=0.3,
    other_gold_prob=0.12,
    other_text="None of these",
) -> tuple[dict, str]:
    """Gold plus sampled distractors, shuffled, sometimes with an ``other`` option.

    With ``other_gold_prob`` the true intent is left off the menu and ``other`` is the
    answer, so "none of these" is sometimes right (docs/data.md, known issues).
    """
    if gold != "other" and rng.random() < other_gold_prob:
        names, gold = [n for n in names if n != gold], "other"
    k = int(rng.integers(lo, min(hi, len(names)) + 1))
    pool = [n for n in names if n != gold]
    picked = list(rng.choice(pool, size=min(k - 1, len(pool)), replace=False))
    labels = picked + ([gold] if gold != "other" else [])
    labels = [labels[i] for i in rng.permutation(len(labels))]  # gold position carries no signal
    criteria = {n: _describe(n, lang) for n in labels}
    if gold == "other" or rng.random() < other_prob:
        criteria["other"] = other_text
    return criteria, gold


def _intent(text_key: str, names_of: Callable[[Row, dict], tuple[str, list[str]]], lang: str):
    instructions = {
        "en": "Route the message: which request does it make?",
        "fr": "Orientez le message : quelle demande fait-il ?",
    }[lang]
    other = {"en": "None of the listed requests", "fr": "Aucune des demandes listées"}[lang]

    def adapter(row, rng, meta):
        gold, names = names_of(row, meta)
        criteria, gold = _menu(gold, names, rng, lang=lang, other_text=other)
        return [
            {
                "state": row[text_key],
                "question": {"type": "choice", "instructions": instructions, "criteria": criteria},
                "expected": gold,
            }
        ]

    return adapter


NLI_TEXT: dict[str, dict[str, Any]] = {
    "en": {
        "choice": "How does the hypothesis relate to the premise?",
        "noul": "Does the premise imply that the hypothesis is true?",
        "criteria": {
            "entailment": "The premise guarantees the hypothesis is true",
            "neutral": "The premise neither confirms nor rules out the hypothesis",
            "contradiction": "The premise shows the hypothesis is false",
        },
        "false": "The hypothesis does not follow from the premise",
        "true": "The hypothesis follows from the premise",
        "state": "Premise: {p}\nHypothesis: {h}",
    },
    "fr": {
        "choice": "Quel est le lien entre l'hypothèse et la prémisse ?",
        "noul": "La prémisse implique-t-elle que l'hypothèse est vraie ?",
        "criteria": {
            "entailment": "La prémisse garantit que l'hypothèse est vraie",
            "neutral": "La prémisse ne confirme ni n'exclut l'hypothèse",
            "contradiction": "La prémisse montre que l'hypothèse est fausse",
        },
        "false": "L'hypothèse ne découle pas de la prémisse",
        "true": "L'hypothèse découle de la prémisse",
        "state": "Prémisse : {p}\nHypothèse : {h}",
    },
}
NLI_LABELS = ("entailment", "neutral", "contradiction")


def fr_detok(text: str) -> str:
    """Undo Moses-style tokenisation of French text (XNLI's machine-translated train):
    no space before , . ) or after (, elisions joined ("l' homme" -> "l'homme"), and
    "-T-il" lower-cased. French spaces before ; : ! ? and around « » are kept."""
    text = re.sub(r" +([,.)\]])", r"\1", text)
    text = re.sub(r"([(\[]) +", r"\1", text)
    text = re.sub(
        r"\b([cdjlmnst]|qu|jusqu|lorsqu|puisqu|quoiqu|aujourd)' +",
        r"\1'",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"-T-(il|elle|on|ils|elles)\b", r"-t-\1", text)
    return re.sub(r" {2,}", " ", text).strip()


def _nli(lang: str, detok: bool = False) -> Adapter:
    t = NLI_TEXT[lang]
    clean = fr_detok if detok else (lambda x: x)

    def adapter(row, rng, meta):
        if row["label"] not in (0, 1, 2):
            return []
        state = t["state"].format(
            p=clean(html.unescape(row["premise"])), h=clean(html.unescape(row["hypothesis"]))
        )
        gold = NLI_LABELS[row["label"]]
        if rng.random() < 0.5:
            return [
                {
                    "state": state,
                    "expected": gold,
                    "question": {
                        "type": "choice",
                        "instructions": t["choice"],
                        "criteria": t["criteria"],
                    },
                }
            ]
        return [
            {
                "state": state,
                "expected": "yes" if gold == "entailment" else "no",
                "question": {
                    "type": "noul",
                    "instructions": t["noul"],
                    "criteria": {"false": t["false"], "true": t["true"]},
                },
            }
        ]

    return adapter


def _paraphrase(lang: str) -> Adapter:
    q = {
        "en": "Do the two sentences mean the same thing?",
        "fr": "Les deux phrases ont-elles le même sens ?",
    }[lang]
    crit = {
        "en": {"false": "They differ in meaning", "true": "They are paraphrases"},
        "fr": {"false": "Leur sens diffère", "true": "Ce sont des paraphrases"},
    }[lang]

    def adapter(row, rng, meta):
        if not row["sentence1"].strip() or not row["sentence2"].strip():
            return []  # PAWS-X has empty pairs with random labels (DATA_AUDIT B4)
        state = f"1: {row['sentence1']}\n2: {row['sentence2']}"
        return [
            {
                "state": state,
                "expected": "yes" if row["label"] == 1 else "no",
                "question": {"type": "noul", "instructions": q, "criteria": crit},
            }
        ]

    return adapter


def boolq(row, rng, meta):
    return [
        {
            "state": row["passage"],
            "expected": "yes" if row["answer"] else "no",
            "question": {
                "type": "noul",
                "instructions": f"Based only on the text: {row['question']}?",
                "criteria": {"false": "The text implies no", "true": "The text implies yes"},
            },
        }
    ]


def stsb(row, rng, meta):
    value = float(row["score"]) * 5.0
    lo = int(np.floor(value))
    hi = min(lo + 1, 5)
    frac = value - lo
    target = {str(i): 0.0 for i in range(6)}
    target[str(lo)] += 1.0 - frac if hi != lo else 1.0
    if hi != lo:
        target[str(hi)] += frac
    levels = [
        "completely different topics",
        "different but on the same topic",
        "not equivalent, share some details",
        "roughly equivalent, some details differ",
        "mostly equivalent, minor details differ",
        "completely equivalent",
    ]
    return [
        {
            "state": f"1: {row['sentence1']}\n2: {row['sentence2']}",
            "target": target,
            "question": {
                "type": "score",
                "instructions": "How similar in meaning are the two sentences?",
                "criteria": levels,
            },
        }
    ]


def sst5(row, rng, meta):
    levels = ["very negative", "negative", "neutral", "positive", "very positive"]
    return [
        {
            "state": row["text"],
            "expected": str(row["label"]),
            "question": {
                "type": "score",
                "instructions": "Rate the sentiment of the text.",
                "criteria": levels,
            },
        }
    ]


def helpsteer(attribute: str) -> Adapter:
    levels = {
        "helpfulness": [
            "not helpful at all",
            "slightly helpful",
            "partially helpful",
            "mostly helpful",
            "fully helpful",
        ],
        "correctness": [
            "mostly wrong",
            "several errors",
            "some errors",
            "minor issues",
            "fully correct and complete",
        ],
    }[attribute]

    def adapter(row, rng, meta):
        prompt, response = html.unescape(row["prompt"]), html.unescape(row["response"])
        state = f"Prompt:\n{prompt}\n\nResponse:\n{response}"
        return [
            {
                "state": state,
                "expected": str(row[attribute]),
                "question": {
                    "type": "score",
                    "instructions": f"Rate the response's {attribute}.",
                    "criteria": levels,
                },
            }
        ]

    return adapter


def squad_abstain(lang: str) -> Adapter:
    q = {
        "en": "Does the passage contain the answer to: {q}",
        "fr": "Le passage contient-il la réponse à : {q}",
    }[lang]
    crit = {
        "en": {"false": "The answer is not in the passage", "true": "The passage answers it"},
        "fr": {"false": "La réponse n'est pas dans le passage", "true": "Le passage y répond"},
    }[lang]

    def adapter(row, rng, meta):
        answers = row["answers"]
        if isinstance(answers, str):  # some mirrors serialise the dict
            import ast

            answers = ast.literal_eval(answers)
        return [
            {
                "state": row["context"],
                "expected": "yes" if answers["text"] else "no",
                "question": {
                    "type": "noul",
                    "instructions": q.format(q=row["question"]),
                    "criteria": crit,
                },
            }
        ]

    return adapter


def piaf_load(split: str):
    """PIAF (one train split) with unanswerable rows: every question appears once with its
    own paragraph (answerable) and, half of the time instead, with another paragraph of the
    same article that does not contain the answer (or of another article when its article has
    one paragraph). Rows are grouped by article title for the held-out split."""
    import ast

    from datasets import Dataset, load_dataset

    ds = load_dataset("etalab-ia/piaf", split="train")
    by_title: dict[str, list[str]] = {}
    for r in ds:
        contexts = by_title.setdefault(r["title"], [])
        if r["context"] not in contexts:
            contexts.append(r["context"])
    titles = sorted(by_title)
    rng = np.random.default_rng(0)
    rows = []
    for r in ds:
        answers = r["answers"]
        if isinstance(answers, str):
            answers = ast.literal_eval(answers)
        texts = [a for a in answers["text"] if a.strip()]
        if not texts:
            continue
        context = r["context"]
        if rng.random() < 0.5:
            same = [c for c in by_title[r["title"]] if c != context]
            pool = same or by_title[_pick_title(rng, titles, r["title"])]
            pool = [c for c in pool if not any(a.lower() in c.lower() for a in texts)]
            if pool:
                context = pool[int(rng.integers(len(pool)))]
                texts = []
        rows.append({"title": r["title"], "context": context, "question": r["question"],
                     "answers": {"text": texts}})  # fmt: skip
    return Dataset.from_list(rows)


def _pick_title(rng, titles: list[str], not_this: str) -> str:
    while True:
        t = titles[int(rng.integers(len(titles)))]
        if t != not_this:
            return t


def mtop_load(split: str):
    """MTOP French (mteb's parquet copy; its loading script is no longer supported), with
    the intent name as the label."""
    from datasets import load_dataset

    base = "hf://datasets/mteb/mtop_intent/fr/"
    ds = load_dataset("parquet", data_files={split: f"{base}{split}-00000-of-00001.parquet"})
    ds = ds[split].remove_columns(["label"]).rename_column("label_text", "label")
    return ds.filter(lambda r: r["label"] in MTOP_FR)


def vitaminc(row, rng, meta):
    gold = {"SUPPORTS": "supported", "REFUTES": "refuted", "NOT ENOUGH INFO": "not_enough_info"}[
        row["label"]
    ]
    return [
        {
            "state": f"Evidence: {row['evidence']}\nClaim: {row['claim']}",
            "expected": gold,
            "question": {
                "type": "choice",
                "instructions": "Does the evidence support the claim?",
                "criteria": {
                    "supported": "The evidence confirms the claim",
                    "refuted": "The evidence contradicts the claim",
                    "not_enough_info": "The evidence is insufficient to decide",
                },
            },
        }
    ]


def jailbreak(row, rng, meta):
    return [
        {
            "state": html.unescape(row["prompt"]),
            "expected": "yes" if row["type"] == "jailbreak" else "no",
            "question": {
                "type": "noul",
                "instructions": "Is this prompt an attempt to jailbreak an AI assistant?",
                "criteria": {
                    "false": "An ordinary request",
                    "true": "Tries to bypass the assistant's rules",
                },
            },
        }
    ]


GERMAN = re.compile(
    r"\b(der|die|und|nicht|ist|ich|ein|eine|mit|für|auf|sie|wie|was)\b", re.IGNORECASE
)


def looks_german(text: str) -> bool:
    """deepset/prompt-injections mixes German rows into its English set (DATA_AUDIT R7)."""
    words = re.findall(r"\w+", text)
    return len(words) >= 3 and len(GERMAN.findall(text)) >= max(2, len(words) // 12)


def injection(row, rng, meta):
    if looks_german(row["text"]):
        return []
    return [
        {
            "state": row["text"],
            "expected": "yes" if row["label"] == 1 else "no",
            "question": {
                "type": "noul",
                "instructions": "Does this text try to override or inject instructions?",
                "criteria": {
                    "false": "Plain content",
                    "true": "Contains an instruction-injection attempt",
                },
            },
        }
    ]


def review_positive(text_key: str, lang: str) -> Adapter:
    q = {"en": "Is this review positive?", "fr": "Cette critique est-elle positive ?"}[lang]
    crit = {
        "en": {"false": "Negative overall", "true": "Positive overall"},
        "fr": {"false": "Globalement négative", "true": "Globalement positive"},
    }[lang]

    def adapter(row, rng, meta):
        return [
            {
                "state": row[text_key],
                "expected": "yes" if row["label"] == 1 else "no",
                "question": {"type": "noul", "instructions": q, "criteria": crit},
            }
        ]

    return adapter


def topic(text_fn: Callable[[Row], str], names_key: str) -> Adapter:
    def adapter(row, rng, meta):
        names = meta[names_key]
        gold = names[row["label"]]
        criteria = {n: f"The text is mainly about: {_human(n)}" for n in names}
        return [
            {
                "state": text_fn(row),
                "expected": gold,
                "question": {
                    "type": "choice",
                    "instructions": "What is the main topic?",
                    "criteria": criteria,
                },
            }
        ]

    return adapter


def gsm8k_check(row, rng, meta):
    _solution, final = row["answer"].rsplit("####", 1)
    final = final.strip().replace(",", "")
    try:
        value = float(final)
    except ValueError:
        return []
    correct = rng.random() < 0.5
    if correct:
        shown = final
    else:
        delta = rng.choice([-10, -2, -1, 1, 2, 5, 10, 100])
        wrong = value + delta if rng.random() < 0.7 else value * rng.choice([2, 0.5, 10])
        shown = str(int(wrong)) if float(wrong).is_integer() else f"{wrong:.2f}"
        if shown == final:
            return []
    return [
        {
            "state": f"Problem: {row['question']}\nProposed final answer: {shown}",
            "expected": "yes" if correct else "no",
            "question": {
                "type": "noul",
                "instructions": "Is the proposed final answer correct?",
                "criteria": {"false": "The answer is wrong", "true": "The answer is right"},
            },
        }
    ]


def _names(feature_names):
    def names_of(row, meta):
        names = meta["label_names"]
        return names[row[feature_names]], names

    return names_of


def _massive(row, meta):
    return row["label"], meta["label_names"]


SOURCES: dict[str, Source] = {
    s.name: s
    for s in [
        Source(
            "boolq",
            "google/boolq",
            None,
            "train",
            "validation",
            boolq,
            "reading",
            "en",
            "CC BY-SA 3.0",
            True,
        ),
        Source(
            "mnli",
            "nyu-mll/glue",
            "mnli",
            "train",
            "validation_matched",
            _nli("en"),
            "entailment",
            "en",
            "GLUE mixed (see card)",
            True,
        ),
        Source(
            "snli",
            "stanfordnlp/snli",
            None,
            "train",
            "test",
            _nli("en"),
            "entailment",
            "en",
            "CC BY-SA 4.0",
            True,
            cap=10000,
        ),
        # mix-v5: the human-translated XNLI / PAWS-X dev splits train, their human test
        # splits stay held out (the same rows as mix-v4); the machine-translated train sets
        # are kept, detokenised, as separate train-only sources at half weight (WEIGHT_SCALE)
        Source(
            "xnli_fr",
            "facebook/xnli",
            "fr",
            "validation",
            "test",
            _nli("fr"),
            "entailment",
            "fr",
            "not stated on the card",
            False,
            cap=15000,
        ),
        Source(
            "xnli_fr_mt",
            "facebook/xnli",
            "fr",
            "train",
            None,
            _nli("fr", detok=True),
            "entailment",
            "fr",
            "not stated on the card",
            False,
            cap=10000,
            train_only=True,
        ),
        Source(
            "paws",
            "google-research-datasets/paws",
            "labeled_final",
            "train",
            "test",
            _paraphrase("en"),
            "paraphrase",
            "en",
            "free use (card)",
            True,
            cap=10000,
        ),
        Source(
            "pawsx_fr",
            "google-research-datasets/paws-x",
            "fr",
            "validation",
            "test",
            _paraphrase("fr"),
            "paraphrase",
            "fr",
            "free use (card)",
            True,
            cap=10000,
        ),
        Source(
            "pawsx_fr_mt",
            "google-research-datasets/paws-x",
            "fr",
            "train",
            None,
            _paraphrase("fr"),
            "paraphrase",
            "fr",
            "free use (card)",
            True,
            cap=7000,
            train_only=True,
        ),
        Source(
            "stsb",
            "sentence-transformers/stsb",
            None,
            "train",
            "test",
            stsb,
            "similarity",
            "en",
            "CC BY-SA (card)",
            True,
        ),
        Source(
            "sst5",
            "SetFit/sst5",
            None,
            "train",
            "test",
            sst5,
            "sentiment",
            "en",
            "check card",
            True,
        ),
        Source(
            "allocine_fr",
            "tblard/allocine",
            None,
            "train",
            "test",
            review_positive("review", "fr"),
            "sentiment",
            "fr",
            "MIT",
            True,
            cap=16000,  # mix-v5: 8000 -> 16000 (native French, 160k available)
        ),
        Source(
            "mtop_fr",
            "mteb/mtop_intent",
            "fr",
            "train",
            "test",
            _intent("text", _massive, "fr"),
            "routing",
            "fr",
            "unknown (mteb card)",
            False,
            load=mtop_load,
        ),
        Source(
            "piaf_fr",
            "etalab-ia/piaf",
            None,
            "train",
            None,
            squad_abstain("fr"),
            "abstain",
            "fr",
            "MIT (card tag)",
            True,
            load=piaf_load,
            group=lambda r: r["title"],
        ),
        Source(
            "helpsteer2_helpfulness",
            "nvidia/HelpSteer2",
            None,
            "train",
            "validation",
            helpsteer("helpfulness"),
            "judge",
            "en",
            "CC BY 4.0",
            True,
            cap=8000,  # noisy one-hot ratings: a smaller share of the score type (R1)
        ),
        Source(
            "helpsteer2_correctness",
            "nvidia/HelpSteer2",
            None,
            "train",
            "validation",
            helpsteer("correctness"),
            "judge",
            "en",
            "CC BY 4.0",
            True,
            cap=8000,  # noisy one-hot ratings: a smaller share of the score type (R1)
        ),
        Source(
            "banking77",
            "legacy-datasets/banking77",
            None,
            "train",
            "test",
            _intent("text", _names("label"), "en"),
            "routing",
            "en",
            "CC BY 4.0",
            True,
        ),
        Source(
            "clinc_oos",
            "clinc/clinc_oos",
            "plus",
            "train",
            "test",
            _intent(
                "text",
                lambda r, m: (
                    m["label_names"][r["intent"]]
                    if m["label_names"][r["intent"]] != "oos"
                    else "other",
                    [n for n in m["label_names"] if n != "oos"],
                ),
                "en",
            ),
            "routing",
            "en",
            "CC BY 3.0",
            True,
        ),
        Source(
            "massive_en",
            "mteb/amazon_massive_intent",
            "en",
            "train",
            "test",
            _intent("text", _massive, "en"),
            "routing",
            "en",
            "CC BY 4.0",
            True,
        ),
        Source(
            "massive_fr",
            "mteb/amazon_massive_intent",
            "fr",
            "train",
            "test",
            _intent("text", _massive, "fr"),
            "routing",
            "fr",
            "CC BY 4.0",
            True,
        ),
        Source(
            "squad2",
            "rajpurkar/squad_v2",
            None,
            "train",
            "validation",
            squad_abstain("en"),
            "abstain",
            "en",
            "CC BY-SA 4.0",
            True,
        ),
        Source(
            "vitaminc",
            "tals/vitaminc",
            None,
            "train",
            "test",
            vitaminc,
            "fact_check",
            "en",
            "CC BY-SA 3.0",
            True,
        ),
        Source(
            "jailbreak",
            "jackhhao/jailbreak-classification",
            None,
            "train",
            "test",
            jailbreak,
            "safety",
            "en",
            "Apache 2.0",
            True,
            near_dedup=True,  # DAN / "Developer Mode" variants leak into held-out (R3)
        ),
        Source(
            "prompt_injection",
            "deepset/prompt-injections",
            None,
            "train",
            "test",
            injection,
            "safety",
            "en",
            "Apache 2.0",
            True,
        ),
        Source(
            "dbpedia",
            "fancyzhx/dbpedia_14",
            None,
            "train",
            "test",
            topic(lambda r: f"{r['title']}: {r['content']}", "label_names"),
            "classification",
            "en",
            "CC BY-SA 3.0",
            True,
            cap=8000,
        ),
        Source(
            "gsm8k_check",
            "openai/gsm8k",
            "main",
            "train",
            "test",
            gsm8k_check,
            "math",
            "en",
            "MIT",
            True,
        ),
    ]
}
