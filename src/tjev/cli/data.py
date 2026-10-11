"""``tjev data``: build the decision mixture and import the evaluation sets."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tjev.cli.common import comma_list, echo_json
from tjev.data.heldout import REPORT_SEED, SELECT_SEED

app = typer.Typer(help="Build the mix and import datasets.", no_args_is_help=True)

RebalanceNoul = Annotated[
    float | None,
    typer.Option(
        help="mix-v4b: subsample the majority class of noul sources whose train yes-share is "
        "outside [1-R, R] down to R (e.g. 0.7)"
    ),
]
NoRebalanceNoul = Annotated[
    bool,
    typer.Option(
        "--no-rebalance-noul",
        help="mix-v7.5-np: no yes/no rebalance (natural label priors; overrides --rebalance-noul)",
    ),
]
MaxLoss = Annotated[
    float, typer.Option(help="largest share of a source's train rows the rebalance may drop")
]
PreserveBlocks = Annotated[
    bool,
    typer.Option(
        help="with --max-epochs: the freed mass stays in the capped source's block (only a "
        "block whose sources are all capped passes its shortfall to the other blocks)"
    ),
]
MaxEpochs = Annotated[
    float | None,
    typer.Option(
        help="mix-v4b: scale weights so that no source exceeds this many effective epochs "
        "(simulated run: --steps × --tokens-per-step)"
    ),
]


@app.command()
def build(
    out: Path,
    sources: Annotated[str, typer.Option(help="comma list of sources (default: all)")] = "",
    blocks: Annotated[str, typer.Option(help="comma list of blocks (default: all)")] = "",
    commercial_only: bool = False,
    scale: Annotated[float, typer.Option(help="multiplies every source's cap")] = 1.0,
    gen_scale: Annotated[
        float,
        typer.Option(
            help="multiplies the generators' rows (on top of --scale): fewer repeats of each "
            "generated row at the same block share"
        ),
    ] = 1.0,
    seed: int = 0,
    jevbench_public: Annotated[
        Path | None,
        typer.Option(help="public.jsonl of `tjev data jevbench` (contamination filter)"),
    ] = None,
    tokenizer: Annotated[
        Path | None,
        typer.Option(
            help="tokenizer folder: drops prompts over the largest bucket "
            "(default: ~/tjev-work/models/Qwen3.5-*, if present)"
        ),
    ] = None,
    rebalance_noul: RebalanceNoul = None,
    max_loss: MaxLoss = 0.6,
    max_epochs: MaxEpochs = None,
    preserve_blocks: PreserveBlocks = True,
    max_prompt_tokens: Annotated[
        int,
        typer.Option(
            help="drop train prompts longer than this (the run's largest bucket; 16384 for "
            "the long buckets, which also builds the long generators)"
        ),
    ] = 4096,
    open_only: Annotated[
        bool,
        typer.Option(
            help="only sources whose card licence allows redistribution (permissive or "
            "share-alike; no unknown, NC or Claude-generated source)"
        ),
    ] = False,
    passage_dedup: Annotated[
        bool,
        typer.Option(
            help="mix-v6: drop train rows whose state equals or nearly equals a held-out state "
            "(whatever the question)"
        ),
    ] = True,
    jevbench_ngram: Annotated[
        bool,
        typer.Option(
            help="mix-v6: drop train rows sharing a word 8-gram (state, instructions, options) "
            "with a public JevBench item (needs --jevbench-public)"
        ),
    ] = True,
) -> None:
    """Build the decision mixture into OUT (train/, validation, calibration, test/, mix.yaml)."""
    from tjev.data.hygiene import find_tokenizer
    from tjev.data.mix import build as build_mix

    manifest = build_mix(
        out,
        sources=comma_list(sources),
        blocks=comma_list(blocks),
        commercial_only=commercial_only,
        scale=scale,
        gen_scale=gen_scale,
        seed=seed,
        jevbench_public=jevbench_public,
        tokenizer=tokenizer or find_tokenizer(),
        rebalance_noul=rebalance_noul,
        max_loss=max_loss,
        max_epochs=max_epochs,
        preserve_blocks=preserve_blocks,
        max_prompt_tokens=max_prompt_tokens,
        open_only=open_only,
        passage_dedup=passage_dedup,
        jevbench_ngram=jevbench_ngram,
    )
    echo_json({k: manifest[k] for k in ("train_rows_by_lang", "fr_sampled_share", "mixture")})


@app.command()
def derive(
    src: Annotated[Path, typer.Argument(help="a built mix (never modified)")],
    out: Path,
    rebalance_noul: RebalanceNoul = None,
    max_loss: MaxLoss = 0.6,
    max_epochs: MaxEpochs = None,
    preserve_blocks: PreserveBlocks = True,
    steps: int = 3000,
    tokens_per_step: int = 65536,
    microbatch_tokens: int = 4096,
    seed: int = 0,
    tokenizer: Annotated[
        Path | None, typer.Option(help="tokenizer folder (default: ~/tjev-work/models/Qwen3.5-*)")
    ] = None,
    packing: Annotated[str, typer.Option(help="packer placement simulated: bucket | best_fit")] = (
        "bucket"
    ),
    also_steps: Annotated[
        str, typer.Option(help="comma list: also report the epochs at these step counts")
    ] = "",
    drop_license_use: Annotated[
        str,
        typer.Option(
            help="comma list of licence uses whose train rows are dropped (mix-v7c: "
            "non-commercial; add unspecified for a strict commercial mix)"
        ),
    ] = "",
    refresh_shares: Annotated[
        bool, typer.Option(help="mix-v7: re-read the block / source shares from the code")
    ] = False,
    v75_shares: Annotated[
        float | None,
        typer.Option(help="mix-v7.5: the v7.5 shares from the code, jev_fr_mt at this share"),
    ] = None,
    tiers: Annotated[
        bool, typer.Option(help="mix-v7.5 epoch tiers: core (tier A) ≤ --tier-a-cap, rest ≤ 1")
    ] = False,
    tier_a_cap: Annotated[
        float | None, typer.Option(help="tier A cap (default --max-epochs)")
    ] = None,
    min_epochs: Annotated[
        float | None,
        typer.Option(help="sources under 0.9 × this are resized to their expected draws"),
    ] = None,
    fresh_generators: Annotated[
        bool, typer.Option(help="generators drawn more than their rows get new rows (rep1)")
    ] = False,
    jevbench_public: Annotated[
        Path | None, typer.Option(help="public.jsonl (the fresh generator rows' 8-gram check)")
    ] = None,
) -> None:
    """A corrected copy of a built mix (mix-v4 → mix-v4b, mix-v7 → mix-v7c): held-out files
    copied as is, train files filtered by licence use / rebalanced, weights capped at
    --max-epochs (OUT may already exist: it is rewritten)."""
    from tjev.data.hygiene import find_tokenizer
    from tjev.data.mix import derive as derive_mix

    manifest = derive_mix(
        src,
        out,
        rebalance_noul=rebalance_noul,
        max_loss=max_loss,
        max_epochs=max_epochs,
        preserve_blocks=preserve_blocks,
        tokenizer=tokenizer or find_tokenizer(),
        seed=seed,
        steps=steps,
        tokens_per_step=tokens_per_step,
        microbatch_tokens=microbatch_tokens,
        packing=packing,
        also_steps=tuple(int(k) for k in comma_list(also_steps)),
        drop_license_use=comma_list(drop_license_use),
        refresh_shares=refresh_shares,
        v75_shares=v75_shares,
        tiers=tiers,
        tier_a_cap=tier_a_cap,
        min_epochs=min_epochs,
        fresh_generators=fresh_generators,
        jevbench_public=jevbench_public,
    )
    cap = manifest.get("epoch_cap", {})
    echo_json(
        {
            "train_rows": sum(s.get("train", 0) for s in manifest["sources"].values()),
            "fr_sampled_share": manifest["fr_sampled_share"],
            "rebalanced": manifest.get("rebalance_noul", {}).get("sources", {}),
            "capped": cap.get("capped", []),
            "block_shares": cap.get("block_shares", {}),
            "overflow": cap.get("overflow", {}),
            "epochs": cap.get("epochs", {}),
            "min_epoch_subsample": manifest.get("min_epoch_subsample", {}),
            "fresh_generators": manifest.get("fresh_generators", {}),
        }
    )


@app.command()
def extend(
    base: Annotated[Path, typer.Argument(help="a built mix (mix-v6; never modified)")],
    out: Path,
    sources: Annotated[str, typer.Option(help="comma list of new sources (default: all v7)")] = "",
    jevbench_public: Annotated[
        Path | None, typer.Option(help="public.jsonl of `tjev data jevbench`")
    ] = None,
    tokenizer: Annotated[
        Path | None, typer.Option(help="tokenizer folder (default: ~/tjev-work/models/Qwen3.5-*)")
    ] = None,
    scale: float = 1.0,
    seed: int = 0,
    rebalance_noul: RebalanceNoul = 0.7,
    no_rebalance_noul: NoRebalanceNoul = False,
    max_loss: MaxLoss = 0.6,
    max_epochs: MaxEpochs = 2.0,
    preserve_blocks: PreserveBlocks = True,
    steps: int = 5000,
    tokens_per_step: int = 65536,
    microbatch_tokens: int = 4096,
    packing: Annotated[str, typer.Option(help="packer placement simulated")] = "best_fit",
    also_steps: Annotated[str, typer.Option(help="comma list of other step counts")] = "3000",
    references: Annotated[
        str, typer.Option(help="comma list of decontamination sets (default: all; 'none')")
    ] = "",
    import_from: Annotated[
        str, typer.Option(help="comma list of built mixes whose sources are added (gen-v7)")
    ] = "",
    replace_blocks: Annotated[
        str, typer.Option(help="with --import-from: drop the base sources of these blocks")
    ] = "",
) -> None:
    """mix-v7: BASE (mix-v6) plus the v7 sources: base held-out files byte-identical, new
    held-out rows in validation-v7 / calibration-v7 / test/, the v7 post-pass
    (decontamination, cross-source dedup), the per-source shares and the epoch cap."""
    from tjev.data.hygiene import find_tokenizer
    from tjev.data.mix import V7_REFERENCES
    from tjev.data.mix import extend as extend_mix

    refs = comma_list(references)
    manifest = extend_mix(
        base,
        out,
        sources=comma_list(sources) or None,
        jevbench_public=jevbench_public,
        tokenizer=tokenizer or find_tokenizer(),
        seed=seed,
        scale=scale,
        rebalance_noul=None if no_rebalance_noul else rebalance_noul,
        max_loss=max_loss,
        max_epochs=max_epochs,
        preserve_blocks=preserve_blocks,
        references=() if refs == ["none"] else (refs or V7_REFERENCES),
        import_from=comma_list(import_from),
        replace_blocks=comma_list(replace_blocks),
        steps=steps,
        tokens_per_step=tokens_per_step,
        microbatch_tokens=microbatch_tokens,
        packing=packing,
        also_steps=tuple(int(k) for k in comma_list(also_steps)),
    )
    cap = manifest.get("epoch_cap", {})
    echo_json(
        {
            "train_rows": sum(s.get("train", 0) for s in manifest["sources"].values()),
            "fr_sampled_share": manifest["fr_sampled_share"],
            "block_sampled_share": manifest.get("block_sampled_share"),
            "license_use_sampled_share": manifest.get("license_use_sampled_share"),
            "capped": cap.get("capped", []),
            "simulation": cap.get("simulation", {}),
        }
    )


@app.command(name="extend-v75")
def extend_v75(
    base: Annotated[Path, typer.Argument(help="mix-v7 (never modified)")],
    out: Path,
    fr_mt: Annotated[
        Path | None, typer.Option(help="jev_fr_mt mix folder (scripts/fr_mt_mix.py)")
    ] = None,
    jev_fr_mt_share: Annotated[
        float, typer.Option(help="jev_fr_mt group share (% of the mix at v7's jev scale)")
    ] = 2.4,
    jevbench_public: Annotated[
        Path | None, typer.Option(help="public.jsonl of `tjev data jevbench`")
    ] = None,
    tokenizer: Annotated[
        Path | None, typer.Option(help="tokenizer folder (default: ~/tjev-work/models/Qwen3.5-*)")
    ] = None,
    seed: int = 0,
    rebalance_noul: RebalanceNoul = 0.7,
    no_rebalance_noul: NoRebalanceNoul = False,
    max_epochs: MaxEpochs = 2.0,
    steps: int = 5000,
    tokens_per_step: int = 65536,
    microbatch_tokens: int = 4096,
    packing: Annotated[str, typer.Option(help="packer placement simulated")] = "best_fit",
    also_steps: Annotated[str, typer.Option(help="comma list of other step counts")] = "3000",
    references: Annotated[
        str, typer.Option(help="comma list of decontamination sets (default: all; 'none')")
    ] = "",
    generators: Annotated[str, typer.Option(help="comma list (default: every v7 one)")] = "",
    sources: Annotated[
        str, typer.Option(help="comma list of rebuilt sources (default: the preference ones)")
    ] = "",
) -> None:
    """mix-v7.5: mix-v7 cleaned, the noisy preference sources rebuilt (aggregated, smoothed
    votes), the v7 generators, jev_distractor and jev_fr_mt added, the v7.5 shares and the
    epoch cap (docs/data.md, mix-v7.5)."""
    from tjev.data.hygiene import find_tokenizer
    from tjev.data.mix import GEN7_NAMES, PREF_REBUILD, V7_REFERENCES
    from tjev.data.mix import extend_v75 as extend_mix

    refs = comma_list(references)
    manifest = extend_mix(
        base,
        out,
        fr_mt=fr_mt,
        jev_fr_mt_share=jev_fr_mt_share,
        sources=comma_list(sources) or PREF_REBUILD,
        generators=comma_list(generators) or GEN7_NAMES,
        jevbench_public=jevbench_public,
        tokenizer=tokenizer or find_tokenizer(),
        seed=seed,
        rebalance_noul=None if no_rebalance_noul else rebalance_noul,
        max_epochs=max_epochs,
        references=() if refs == ["none"] else (refs or V7_REFERENCES),
        steps=steps,
        tokens_per_step=tokens_per_step,
        microbatch_tokens=microbatch_tokens,
        packing=packing,
        also_steps=tuple(int(k) for k in comma_list(also_steps)),
    )
    cap = manifest.get("epoch_cap", {})
    echo_json(
        {
            "train_rows": sum(s.get("train", 0) for s in manifest["sources"].values()),
            "fr_sampled_share": manifest["fr_sampled_share"],
            "block_sampled_share": manifest.get("block_sampled_share"),
            "license_use_sampled_share": manifest.get("license_use_sampled_share"),
            "capped": cap.get("capped", []),
            "overflow": cap.get("overflow", {}),
            "simulation": cap.get("simulation", {}),
        }
    )


@app.command()
def reweight(out: Path) -> None:
    """Recompute mix.yaml (the source weights) of a built mix."""
    from tjev.data.mix import reweight as reweight_mix

    echo_json(reweight_mix(out)["mixture"])


@app.command()
def jevbench(
    repo: Annotated[Path, typer.Argument(help="a checkout of github.com/fstandhartinger/jevbench")],
    out: Path,
) -> None:
    """Import the JevBench public items (final measurement only, never training)."""
    from tjev.data.jevbench import prepare

    echo_json(prepare(repo, out))


@app.command()
def heldout(
    out: Path,
    per: Annotated[int, typer.Option(help="items per generator")] = 20,
    seed: int = SELECT_SEED,
) -> None:
    """Held-out generator items (data.heldout: part of the checkpoint selection score)."""
    from tjev.data.heldout import write_heldout

    if seed == REPORT_SEED:
        raise typer.BadParameter(f"seed {REPORT_SEED} is the post-training reporting set")
    echo_json({"items": write_heldout(out, per, seed), "out": str(out)})
