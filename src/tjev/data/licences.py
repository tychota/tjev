"""Licence and redistribution status of every source of the mix, as stated on its card.

``REDISTRIBUTABLE`` values: ``yes`` (a permissive licence on the card), ``share-alike``
(redistributable under the same licence: CC BY-SA), ``NC`` (non-commercial only), ``no``,
``unknown`` (the card states no licence, says "other" / "unknown", or adds terms that
restrict redistribution), and ``claude-generated`` (rows written by Claude for this
project: kept as a separate source so they can be left out).

Verdicts come from the dataset cards (HF ``license`` tag or the card's licensing section,
read 2026-10-01), never from memory: a source whose card says nothing is ``unknown``. The
builder records them in ``manifest.json`` and ``tjev data build --open-only`` keeps only
``yes`` and ``share-alike`` sources (docs/fr-data.md).
"""

from __future__ import annotations

# name -> (licence as stated on the card, redistributable, note)
LICENCES: dict[str, tuple[str, str, str]] = {
    # jev: contenders' public decision data
    "certo": ("MIT", "yes", "altslate/certo-decisions-v2 card tag"),
    "decider_teacher": ("Apache-2.0", "yes", "github.com/Mapika/decider LICENSE"),
    "mghafiri_scenarios": ("MIT", "yes", "card tag; rows written by Claude upstream"),
    "openjev": (
        "CC0-1.0 (generated) + CC BY 4.0 (WANLI)",
        "yes",
        "card licensing section (source-dependent)",
    ),
    "plumb": ("Apache-2.0", "yes", "card tag; rows written by a Qwen model upstream"),
    "jev_fr_pilot": (
        "translations by Claude of certo / decider / mghafiri / openjev rows",
        "claude-generated",
        "keep separable; upstream licences also apply",
    ),
    # replay
    "allocine_fr": ("MIT", "yes", "card tag"),
    "banking77": ("CC BY 4.0", "yes", "card tag"),
    "boolq": ("CC BY-SA 3.0", "share-alike", "card tag"),
    "clinc_oos": ("CC BY 3.0", "yes", "card tag"),
    "dbpedia": ("CC BY-SA 3.0", "share-alike", "card tag"),
    "gsm8k_check": ("MIT", "yes", "card tag"),
    "helpsteer2_correctness": ("CC BY 4.0", "yes", "card tag"),
    "helpsteer2_helpfulness": ("CC BY 4.0", "yes", "card tag"),
    "jailbreak": ("Apache-2.0", "yes", "card tag"),
    "massive_en": ("Apache-2.0", "yes", "mteb copy tag; AmazonScience/massive: CC BY 4.0"),
    "massive_fr": ("Apache-2.0", "yes", "mteb copy tag; AmazonScience/massive: CC BY 4.0"),
    "mnli": ("other", "unknown", "GLUE card refers to each task's original licence"),
    "mtop_fr": ("unknown", "unknown", "mteb/mtop_intent card: license unknown"),
    "paws": ("free use", "yes", "card: 'may be freely used for any purpose'"),
    "pawsx_fr": ("free use", "yes", "card: 'may be freely used for any purpose'"),
    "pawsx_fr_mt": ("free use", "yes", "card: 'may be freely used for any purpose'"),
    "piaf_fr": ("MIT", "yes", "etalab-ia/piaf card tag (licensing section empty)"),
    "prompt_injection": ("Apache-2.0", "yes", "card tag"),
    "snli": ("CC BY-SA 4.0", "share-alike", "card tag"),
    "squad2": ("CC BY-SA 4.0", "share-alike", "card tag"),
    "sst5": ("not stated", "unknown", "SetFit/sst5 card has no licence"),
    "stsb": ("not stated", "unknown", "sentence-transformers/stsb card has no licence"),
    "vitaminc": ("CC BY-SA 3.0", "share-alike", "card tag"),
    "xnli_fr": ("not stated", "unknown", "facebook/xnli card: licensing 'more information needed'"),
    "xnli_fr_mt": ("not stated", "unknown", "facebook/xnli card: licensing 'more information needed'"),
    # analysis
    "go_emotions": ("Apache-2.0", "yes", "card tag"),
    "hc3_fr": ("CC BY-SA 4.0", "share-alike", "card tag"),
    "mage": ("Apache-2.0", "yes", "card tag"),
    "mlma_en": ("MIT", "yes", "card tag (tweets)"),
    "mlma_fr": ("MIT", "yes", "card tag (tweets)"),
    "pavlick_formality": ("CC BY 3.0", "yes", "card tag"),
    "raid": ("MIT", "yes", "card tag"),
    "textdetox_fr": ("openrail++", "unknown", "card tag: a use-restricted licence"),
    "tweet_sentiment_en": (
        "CC BY 3.0 + Twitter ToS",
        "unknown",
        "card licensing section: also requires the Twitter terms of service",
    ),
    "tweet_sentiment_fr": (
        "CC BY 3.0 + Twitter ToS",
        "unknown",
        "card licensing section: also requires the Twitter terms of service",
    ),
    # targeted (mix-v6; cards read 2026-10-04)
    "eikos": (
        "CC BY 4.0",
        "yes",
        (
            "caiovicentino1/eikos-decisions card tag; rows written by GLM-5.3-Flash / "
            "Qwen3.8-27B or by its programs (outputs redistributable per the card); eval/ "
            "files not used"
        ),
    ),
    "eikos_long": ("CC BY 4.0", "yes", "caiovicentino1/eikos-decisions long_context, as eikos"),
    "procedural": ("Apache-2.0", "yes", "tasksource/procedural-typed-decisions card tag"),
    "tempreason": ("CC BY-SA 3.0", "share-alike", "tonytan48/TempReason card tag (L1 only)"),
    "helpsteer3": ("CC BY 4.0", "yes", "nvidia/HelpSteer3 card tag (preference, English)"),
    "cuad": ("CC BY 4.0", "yes", "theatticusproject/cuad-qa card tag (data.zip on GitHub)"),
    "cold_law_fr": (
        "CC BY 4.0",
        "yes",
        "harvard-lil/cold-french-law card tag (LEGI: Licence Ouverte); French text only",
    ),
    "argimi_fr": (
        "CC BY-SA 4.0",
        "share-alike",
        (
            "artefactory/Argimi-Legal-French-Jurisprudence card tag (pseudonymised, no "
            "re-identification)"
        ),
    ),
    "twin_squad2": ("CC BY-SA 4.0", "share-alike", "negated rajpurkar/squad_v2 rows"),
    "twin_boolq": ("CC BY-SA 3.0", "share-alike", "negated google/boolq rows"),
    "twin_piaf_fr": ("MIT", "yes", "negated etalab-ia/piaf rows (card tag)"),
}  # fmt: skip
GENERATED = ("Apache-2.0 (generated by tjev code)", "yes", "code-labelled generators")
# mix-v7.5: the v7 sources that are not plain generators (gen_v7.realtext seeds a rule line
# with a verbatim excerpt; licence as on its V7Source), the derived distractor rows and the
# machine-translated contender rows (their base rows' licences)
LICENCES.update({
    "gen7_rt_tau_pairs": ("MIT", "yes", ("sierra-research/tau-bench LICENSE @59a200c (policy "
                          "excerpts); cases generated by tjev code")),
    "gen7_rt_sharc_pairs": ("CC BY-SA 3.0", "share-alike", ("UCLNLP/sharc card tag (train "
                            "snippets); cases generated by tjev code")),
    "gen7_rt_kali_fr_pairs": ("odc-by", "yes", ("Nicolas-BZRD/KALI_opendata card tag (DILA "
                              "KALI open data; attribution); cases generated by tjev code")),
    "gen7_rt_cold_fr_pairs": ("CC BY 4.0", "yes", ("harvard-lil/cold-french-law card tag (LEGI: "
                              "Licence Ouverte); cases generated by tjev code")),
    "jev_distractor": (("the base rows' licences (certo, decider_teacher, mghafiri_scenarios, "
                       "openjev, plumb, procedural, eikos, cold_law_fr)"), "yes",
                       "real rows plus one opinion line generated by tjev code"),
    "jev_fr_mt": (("plumb Apache-2.0 / decider_teacher Apache-2.0 / mghafiri_scenarios MIT, "
                  "translated by Helsinki-NLP/opus-mt-tc-big-en-fr (CC BY 4.0)"), "yes",
                  ("rows translated by machine; QE filter Unbabel/wmt22-cometkiwi-da "
                  "(CC BY-NC-SA 4.0) only scored them (no output derived from it); mghafiri "
                  "rows are Claude-written upstream, plumb rows Qwen-written")),
})  # fmt: skip
OPEN = frozenset({"yes", "share-alike"})
# mix-v7: the licence use of each row ("commercial" / "non-commercial" / "unspecified",
# tjev.data.item.LICENSE_USES). Earlier sources: commercial when the card licence allows
# redistribution (permissive or share-alike) and the source is commercial-ok, otherwise
# unspecified, except these (the upstream licence is non-commercial although the HF card
# states none).
LICENSE_USE_OVERRIDES = {
    "xnli_fr": "non-commercial",  # XNLI: CC BY-NC 4.0 (github.com/facebookresearch/XNLI)
    "xnli_fr_mt": "non-commercial",
}


def _v7() -> dict[str, tuple[str, str, str]]:
    from tjev.data.sources import V7_LICENCES

    return V7_LICENCES


def licence_of(name: str) -> tuple[str, str, str]:
    """(card licence, redistributable, note); generators are tjev's own output."""
    if name.startswith("gen_") or (name.startswith("gen7_") and name not in LICENCES):
        return GENERATED
    return LICENCES.get(name) or _v7().get(name) or ("not recorded", "unknown", "")


def license_use_of(name: str) -> str:
    """The licence use of a source's rows (rows may carry their own, e.g. tasksource-jev)."""
    from tjev.data.sources import ALL_SOURCES

    if name.startswith(("gen_", "gen7_")) or name in ("jev_fr_mt", "jev_distractor"):
        # generated (or seeded by permissive / share-alike excerpts); the derived rows record
        # their base rows' use per row
        return "commercial" if licence_of(name)[1] in OPEN else "unspecified"
    if name in LICENSE_USE_OVERRIDES:
        return LICENSE_USE_OVERRIDES[name]
    source = ALL_SOURCES.get(name)
    if source is not None and source.license_use:
        return source.license_use
    redistributable = licence_of(name)[1]
    if redistributable == "NC":
        return "non-commercial"
    if source is not None and source.commercial_ok and redistributable in OPEN:
        return "commercial"
    return "unspecified"


def is_open(name: str) -> bool:
    """Kept by ``--open-only``: redistributable as is (permissive or share-alike)."""
    return licence_of(name)[1] in OPEN
