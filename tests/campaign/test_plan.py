"""The campaign planner: every planned job is a valid run config; the fit recovers rates from
the paired online loss, adopts arms and Muon on clear paired wins, applies the rank rule,
measures the horizon exponent, uses measured sizes; costs."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from tjev.campaign import plan as planner
from tjev.campaign.queue import parse_lines
from tjev.config import load_config


def _campaign(tmp_path, **kw) -> planner.Campaign:
    mix = tmp_path / "mix"
    mix.mkdir(parents=True, exist_ok=True)
    (mix / "mix.yaml").write_text("data:\n  validation: v.jsonl\n")
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    return planner.Campaign(runs=runs, mix=str(mix), models="/models", **kw)


def _fake_run(runs: Path, name: str, *, size="2B", lr=4e-5, seed=0, score=0.5, steps=600,
              tokens=65536, nll=None):  # fmt: skip
    """A finished run; ``nll(step)`` writes a per-step training NLL curve."""
    run = runs / name
    run.mkdir(parents=True)
    cfg = {"model": {"path": f"/m/Qwen3.5-{size}"}, "optim": {"lr": lr},
           "train": {"tokens_per_step": tokens, "steps": steps, "seed": seed}}  # fmt: skip
    (run / "config.json").write_text(json.dumps({"config": cfg, "identity": {}}))
    (run / "selection.json").write_text(json.dumps({"step": steps, "score": score}))
    if nll is not None:
        rows = [{"step": s, "learning/nll": nll(s)} for s in range(1, steps + 1)]
        (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "queue.done").touch()


def _batches(step, scale=1):
    """The difficulty of the batch at ``step``: the same for runs that see the same segments
    (``scale`` steps of a half/quarter batch cover one base step)."""
    window = (step - 1) // (planner.WINDOW * scale)
    return 0.02 * ((window * 7919) % 13 - 6) / 6


def _sweep(runs: Path, optimum=3e-5, curvature=0.05, prefix="sweep-lr", shift=0.0):
    grid = planner.MUON_GRID if "muon" in prefix else planner.SWEEP_GRID
    for lr in grid:
        offset = curvature * math.log2(lr / optimum) ** 2 + shift
        _fake_run(runs, f"{prefix}{lr:.3g}-s0", lr=lr, score=0.55,  # eval score: flat, noisy
                  nll=lambda s, o=offset: 0.6 + o + _batches(s))  # fmt: skip


def _centre(optimum=3e-5, curvature=0.05):
    return 0.6 + curvature * math.log2(planner.CENTRE / optimum) ** 2


@pytest.mark.parametrize("hardware", ["v6e", "v5e"])
def test_every_planned_job_is_a_valid_run_config(tmp_path, hardware):
    c = _campaign(tmp_path, hardware=hardware)
    counts = {}
    for phase in ("sweep", "transfer", "final"):
        jobs = parse_lines(planner.plan(phase, c), c.runs)
        counts[phase] = len(jobs)
        assert len({j.name for j in jobs}) == len(jobs), phase
        for job in jobs:
            files = [a for a in job.args if "=" not in a]
            cfg = load_config(*files, overrides=[a for a in job.args if "=" in a])
            assert files[0] == f"tpu-{hardware}"
            assert cfg.compute.gdn_impl == "pallas_tpu"
            assert cfg.data.heldout.endswith("heldout-select.jsonl")
            assert job.chips in (1, 2, 4, 8)
            if hardware == "v5e" and "Qwen3.5-4B" in cfg.model.path:
                assert cfg.mesh.fsdp == 2  # 4B sharded to fit 16 GB
                assert job.chips == 2
    # sweep: 4 AdamW rates + the centre on a 2nd seed + 3 Muon rates + 13 arms (+ the 3 LoRA
    # inits); transfer: 2 rates × (2400 + 2 branches) + 2 sizes × 3
    # rates; final (the decided recipe): 3 sizes × (long run + its h/2 branch)
    arms = 14 + len(planner.INIT_ARMS)  # + lora-merged; + pissa, eva, lora-ga
    assert counts == {"sweep": 8 + arms, "transfer": 12, "final": 6}
    final = parse_lines(planner.plan("final", c), c.runs)
    assert sum(1 for j in final if j.after) == 3
    assert all("@" in j.after for j in final if j.after)
    # the generic recipe (--final-recipe generic): 3 sizes × (long run + h/4 and h/2 branches)
    generic = parse_lines(planner.plan("final", replace(c, final_recipe="generic")), c.runs)
    assert len(generic) == 9 and sum(1 for j in generic if j.after) == 6


def test_budget_knobs(tmp_path):
    c = _campaign(tmp_path, seeds=2, arm_seeds=2, arms=("zclip", "b32k"), muon=False,
                  sweep_steps=300)  # fmt: skip
    jobs = parse_lines(planner.plan("sweep", c), c.runs)
    assert len(jobs) == 4 * 2 + 1 + 2 * 2
    steps = {j.name: next(a for a in j.args if a.startswith("train.steps=")) for j in jobs}
    assert steps["sweep-b32k-s1"] == "train.steps=600"  # twice the steps at half the batch
    assert steps["sweep-lr3.54e-05-s2"] == "train.steps=300"


def test_no_kernels_and_wandb_overrides(tmp_path):
    c = _campaign(tmp_path, kernels=False, wandb={"project": "p", "campaign": "c1"})
    lines = planner.plan("sweep", c)
    assert all("compute.gdn_impl=chunked" in line for line in lines)
    assert all("log.wandb_group=c1/sweep" in line for line in lines)


def test_fit_before_any_run_uses_the_priors(tmp_path):
    fit = planner.fit(_campaign(tmp_path))
    assert fit["decisions"]["proxy_lr"] == planner.PRIOR_LR
    assert fit["decisions"]["optimizer"] == "adamw"
    recipe = fit["recipe"]
    assert recipe["4B"]["rank"] == 64
    assert recipe["2B"]["rank"] == 32
    assert recipe["4B"]["sweep_lr"] < recipe["2B"]["sweep_lr"] < recipe["0.8B"]["sweep_lr"]
    factor = (1200 / 600) ** -planner.GAMMA[0]  # the default 1200-step horizon at the prior γ
    assert recipe["2B"]["lr"] == pytest.approx(recipe["2B"]["sweep_lr"] * factor, rel=0.05)


def test_fit_recovers_the_rate_from_the_paired_online_loss(tmp_path):
    c = _campaign(tmp_path)
    _sweep(c.runs, optimum=3e-5)
    fit = planner.fit(c)
    assert fit["curves"]["adamw"]["bracketed"]
    assert fit["curves"]["adamw"]["source"] == "online"
    assert fit["decisions"]["proxy_lr"] == pytest.approx(3e-5, rel=0.03)


def test_an_arm_is_adopted_on_a_clear_paired_win(tmp_path):
    c = _campaign(tmp_path)
    _sweep(c.runs)
    base = _centre()
    _fake_run(c.runs, "sweep-zclip-s0", lr=planner.CENTRE, score=0.55,
              nll=lambda s: base - 0.03 + _batches(s))  # fmt: skip
    _fake_run(c.runs, "sweep-noclip-s0", lr=planner.CENTRE, score=0.55,
              nll=lambda s: base + 0.03 + _batches(s))  # fmt: skip
    _fake_run(c.runs, "sweep-fp32-storage-s0", lr=planner.CENTRE, score=0.50,
              nll=lambda s: base - 0.05 + _batches(s))  # fmt: skip
    fit = planner.fit(c)
    effects = fit["effects"]
    assert effects["zclip"]["better"]
    assert not effects["noclip"]["better"]
    assert effects["zclip"]["online"][0] == pytest.approx(-0.03)
    assert not effects["fp32-storage"]["better"]  # report only
    assert "optim.clip_mode=zscore" in fit["recipe"]["2B"]["overrides"]


def test_the_best_clear_init_win_is_adopted(tmp_path):
    c = _campaign(tmp_path)
    _sweep(c.runs)
    base = _centre()
    _fake_run(c.runs, "sweep-pissa-s0", lr=planner.CENTRE, score=0.55,
              nll=lambda s: base - 0.02 + _batches(s))  # fmt: skip
    _fake_run(c.runs, "sweep-eva-s0", lr=planner.CENTRE, score=0.55,
              nll=lambda s: base - 0.04 + _batches(s))  # fmt: skip
    _fake_run(c.runs, "sweep-lora-ga-s0", lr=planner.CENTRE, score=0.55,
              nll=lambda s: base + 0.03 + _batches(s))  # fmt: skip
    fit = planner.fit(c)
    assert fit["effects"]["pissa"]["better"] and fit["effects"]["eva"]["better"]
    assert not fit["effects"]["lora-ga"]["better"]
    assert fit["decisions"]["lora_init"] == "eva"
    assert "lora.init=eva" in fit["recipe"]["2B"]["overrides"]
    assert planner.fit(_campaign(tmp_path / "none"))["decisions"]["lora_init"] == "default"


def test_batch_arms_compare_at_matching_tokens(tmp_path):
    """b32k runs 1200 steps of 32k tokens: its window at step 2k covers the base's step k."""
    c = _campaign(tmp_path)
    _sweep(c.runs)
    base = _centre()
    _fake_run(c.runs, "sweep-b32k-s0", lr=planner.CENTRE * planner.BATCH_LR, score=0.55,
              steps=1200, tokens=32768, nll=lambda s: base - 0.02 + _batches(s, 2))  # fmt: skip
    fit = planner.fit(c)
    online = fit["effects"]["b32k"]["online"]
    assert online[0] == pytest.approx(-0.02, abs=5e-3)
    assert fit["effects"]["b32k"]["better"]
    assert fit["decisions"]["tokens_per_step"] == 32768
    assert fit["recipe"]["2B"]["horizon"] == 2400  # 1200 steps' tokens at half the batch


def test_rank_arms_set_the_rank_and_its_rate_rule(tmp_path):
    c = _campaign(tmp_path)
    _sweep(c.runs)
    base = _centre()
    _fake_run(c.runs, "sweep-r64-s0", lr=planner.CENTRE * planner.ROOT_HALF, score=0.55,
              nll=lambda s: base - 0.02 + _batches(s))  # fmt: skip
    _fake_run(c.runs, "sweep-r64-samelr-s0", lr=planner.CENTRE, score=0.55,
              nll=lambda s: base - 0.05 + _batches(s))  # fmt: skip
    fit = planner.fit(c)
    assert fit["decisions"]["rank_small_sizes"] == 64
    assert fit["decisions"]["rank_lr_factor_per_doubling"] == 1.0  # the same rate won
    assert fit["recipe"]["2B"]["rank"] == 64


def test_muon_needs_a_clear_paired_win(tmp_path):
    c = _campaign(tmp_path / "a")
    _sweep(c.runs, optimum=3e-5)
    _sweep(c.runs, optimum=3e-5, prefix="sweep-muon-lr", shift=-0.04)
    fit = planner.fit(c)
    assert fit["muon"]["use"]
    assert fit["recipe"]["2B"]["overrides"][0] == "optim.name=muon"
    c2 = _campaign(tmp_path / "b")
    _sweep(c2.runs, optimum=3e-5)
    _sweep(c2.runs, optimum=3e-5, prefix="sweep-muon-lr", shift=-0.003)  # within the floor
    assert not planner.fit(c2)["muon"]["use"]


def test_the_horizon_exponent_is_measured_from_the_two_rates(tmp_path):
    """A lower rate that loses at 600 steps and wins at 2400: the optimum moves down as the
    horizon grows, so γ > 0."""
    c = _campaign(tmp_path)
    _sweep(c.runs, optimum=3e-5)
    for steps, lo_minus_hi in ((600, 0.01), (1200, 0.0), (2400, -0.02)):
        for tag, shift in (("hi", 0.0), ("lo", lo_minus_hi)):
            _fake_run(c.runs, f"transfer-2b-{tag}-h{steps}", steps=steps,
                      score=0.5 + shift, nll=lambda s, d=shift: 0.6 + d + _batches(s))  # fmt: skip
    fit = planner.fit(c)
    assert fit["horizon"]["measured"] > 0
    assert 0 < fit["decisions"]["horizon_gamma"] <= 0.6


def test_measured_sizes_and_the_best_horizon(tmp_path):
    c = _campaign(tmp_path)
    for lr, score in ((1e-5, 0.60), (2e-5, 0.50), (4e-5, 0.60)):
        _fake_run(c.runs, f"transfer-4b-lr{lr:.3g}", size="4B", lr=lr, score=score)
    for steps, score in ((600, 0.52), (1200, 0.55), (2400, 0.50)):
        _fake_run(c.runs, f"transfer-2b-hi-h{steps}", steps=steps, score=score)
    fit = planner.fit(c)
    assert fit["recipe"]["4B"]["sweep_lr"] == pytest.approx(2e-5, rel=0.05)
    assert fit["decisions"]["best_horizon"] == 2400
    assert all(r["horizon"] == 2400 for r in fit["recipe"].values())


def test_select_and_cost(tmp_path):
    c = _campaign(tmp_path)
    for name, score in (("final-2b-h1200", 0.5), ("final-2b-h300", 0.45), ("final-2b-h600", 0.47)):
        _fake_run(c.runs, name, score=score)
    assert planner.select(c.runs)["2B"]["name"] == "final-2b-h300"
    table = planner.cost(c, vm_chips=8, rate=1.0)
    arms = 14 + len(planner.INIT_ARMS)  # + lora-merged; + pissa, eva, lora-ga
    assert table["total"]["jobs"] == 8 + arms + 12 + 6
    assert table["total"]["usd"] == pytest.approx(table["total"]["chip_hours"], rel=1e-9)
    bench = {
        "tokens_per_second_per_chip": dict.fromkeys(planner.SIZES, 1000000000.0)
    }  # instant training
    assert planner.cost(c, bench=bench)["total"]["chip_hours"] < table["total"]["chip_hours"]


def _tags(line: str) -> list[str]:
    raw = next(a for a in line.split() if a.startswith("log.wandb_tags="))
    return raw.split("=", 1)[1].strip("[]").split(",")


def test_wandb_project_groups_and_tags(tmp_path):
    c = _campaign(tmp_path, wandb={"campaign": "c1", "entity": "tychota"})
    lines = {line.split()[0]: line for line in planner.plan("sweep", c)}
    assert all(f"log.wandb_project={planner.WANDB_PROJECT}" in x for x in lines.values())
    assert all("log.wandb_entity=tychota" in x for x in lines.values())
    grid, muon, arm = (lines["sweep-lr5e-05-s0"], lines["sweep-muon-lr2.5e-05-s0"],
                       lines["sweep-r64-s0"])  # fmt: skip
    assert "log.wandb_group=c1/sweep/lr" in grid and "log.wandb_group=c1/sweep/lr" in muon
    assert "log.wandb_group=c1/sweep/arms" in arm
    assert {"arm:grid", "opt:adamw", "lr:5e-05", "rank:32", "init:default", "dtype:bf16",
            "seed:0", "sweep", "2B"} <= set(_tags(grid))  # fmt: skip
    assert {"arm:muon-grid", "opt:muon"} <= set(_tags(muon))
    assert {"arm:r64", "rank:64"} <= set(_tags(arm))
    assert "dtype:fp32" in _tags(lines["sweep-fp32-storage-s0"])
    for line in lines.values():  # every tag list parses back as the run config's
        cfg = load_config(*[a for a in line.split()[3:] if "=" not in a],
                          overrides=[a for a in line.split()[3:] if "=" in a])  # fmt: skip
        assert cfg.log.wandb_project == "tjev-sweep" and len(cfg.log.wandb_tags) == 10


def test_sweep_stages_and_centre(tmp_path):
    c = _campaign(tmp_path, stage="lr")
    names = [x.split()[0] for x in planner.plan("sweep", c)]
    assert all("-lr" in n for n in names) and len(names) == 8
    c = _campaign(tmp_path, stage="arms", centre=2.5e-5, arm_seeds=2, arms=("zclip",))
    lines = planner.plan("sweep", c)
    names = [x.split()[0] for x in lines]
    # the arm on two seeds; seed 1 needs its base run at the new centre
    assert names == ["sweep-zclip-s0", "sweep-lr2.5e-05-s1", "sweep-zclip-s1"]
    assert "optim.lr=2.5e-05" in lines[0]
    with pytest.raises(ValueError, match="grid"):
        _campaign(tmp_path, centre=3e-5)
    # stage "all" plans a base run once
    c = _campaign(tmp_path, arm_seeds=2, arms=("zclip",), muon=False)
    names = [x.split()[0] for x in planner.plan("sweep", c)]
    assert len(names) == len(set(names))


def test_arms_pair_with_the_chosen_centre(tmp_path):
    c = _campaign(tmp_path, centre=2.5e-5)
    _sweep(c.runs)
    base = 0.6 + 0.05 * math.log2(2.5e-5 / 3e-5) ** 2
    _fake_run(c.runs, "sweep-zclip-s0", lr=2.5e-5, score=0.55,
              nll=lambda s: base - 0.03 + _batches(s))  # fmt: skip
    effects = planner.fit(c)["effects"]
    assert effects["zclip"]["online"][0] == pytest.approx(-0.03, abs=1e-6)
    assert effects["zclip"]["better"]


def test_killed_runs_are_left_out_of_the_fit(tmp_path):
    c = _campaign(tmp_path)
    _sweep(c.runs, optimum=3e-5)
    (c.runs / "sweep-lr7.07e-05-s0" / "health.json").write_text(json.dumps({"status": "killed"}))
    fit = planner.fit(c)
    assert fit["killed"] == ["sweep-lr7.07e-05-s0"]
    assert 7.07e-5 not in fit["curves"]["adamw"]["means"]
    assert fit["runs"] == 3


def test_colab_v6e1_plans_one_chip_runs_at_the_measured_speed(tmp_path):
    c = _campaign(tmp_path, hardware="v6e1", proxy="0.8B")
    lines = planner.plan("sweep", c)
    assert all(" chips=1 " in x and "qwen35-0.8b" in x for x in lines)
    jobs = parse_lines(lines, c.runs)
    cfg = load_config(*[a for a in jobs[0].args if "=" not in a],
                      overrides=[a for a in jobs[0].args if "=" in a])  # fmt: skip
    assert cfg.compute.gdn_impl == "pallas_tpu_mega2" and cfg.compute.remat == "core_states"
    assert cfg.compute.gdn_chunk == 128 and cfg.compute.gdn_inverse == "blocked"
    assert cfg.compute.lora_impl == "merged"  # 0.8B: merged LoRA pays (tjev-bench4)
    # perf/integration's measured wins, every size: best-fit packing, the flat GDN norm
    assert cfg.train.packing == "best_fit" and cfg.compute.gdn_norm == "flat"
    assert cfg.compute.head_rows == "full" and cfg.compute.remat_by_bucket == {}
    assert planner.tokens_per_second(c, "9B") == 9220.0
    final = parse_lines(planner.plan("final", c), c.runs)
    assert all(j.chips == 1 for j in final)
    c9 = _campaign(tmp_path, hardware="v6e1", proxy="0.8B", final_sizes=("9B", "4B"))
    by_size = {}
    for j in parse_lines(planner.plan("final", c9), c9.runs):
        cfg = _config(j)
        by_size[cfg.model.path.rsplit("-", 1)[1]] = cfg
        assert cfg.train.packing == "best_fit" and cfg.compute.gdn_norm == "flat"
        assert cfg.compute.remat == "core_states"
    assert by_size["4B"].compute.lora_impl == "merged" and by_size["4B"].compute.head_rows == "full"
    assert by_size["4B"].compute.remat_by_bucket == {}
    # 9B: unmerged LoRA, the letters-only head, minimal_states on its 1024 / 2048 buckets
    nine = by_size["9B"].compute
    assert nine.lora_impl == "unmerged" and nine.head_rows == "letters"
    assert nine.remat_by_bucket == {"1024": "minimal_states", "2048": "minimal_states"}
    assert nine.remat_at(4096) == "core_states"


def test_lean_sweep_drops_quick_evals_and_post_and_kills_unhealthy_runs(tmp_path):
    c = _campaign(tmp_path, lean=True, stage="lr")
    jobs = parse_lines(planner.plan("sweep", c), c.runs)
    for j in jobs:
        cfg = load_config(*[a for a in j.args if "=" not in a],
                          overrides=[a for a in j.args if "=" in a])  # fmt: skip
        assert j.post == "none"
        assert cfg.train.quick_eval_every == 0 and cfg.train.checkpoint_every == 100
        assert cfg.train.health_kill
    lean = planner.cost(_campaign(tmp_path, lean=True), ("sweep",))["sweep"]["chip_hours"]
    full = planner.cost(_campaign(tmp_path), ("sweep",))["sweep"]["chip_hours"]
    assert lean < full


def test_horizon_override_sets_the_final_runs_and_their_rate(tmp_path):
    c = _campaign(tmp_path, horizon=600, sweep_steps=300, proxy="0.8B", final_recipe="generic")
    recipe = planner.fit(c)["recipe"]
    assert recipe["4B"]["horizon"] == 600
    assert recipe["4B"]["lr"] == pytest.approx(
        recipe["4B"]["sweep_lr"] * 2 ** -planner.GAMMA[0], rel=0.05
    )
    names = [x.split()[0] for x in planner.plan("final", c)]
    assert "final-4b-h600" in names and "final-4b-h150" in names


def test_the_rank_axis_picks_the_best_adopted_rank_and_keeps_big_sizes_at_64(tmp_path):
    c = _campaign(tmp_path)
    _sweep(c.runs)
    base = _centre()
    for name, d in (("r16", -0.04), ("r64", -0.02), ("r128", 0.01)):
        rank = int(name[1:])
        _fake_run(c.runs, f"sweep-{name}-s0", lr=planner.CENTRE * (32 / rank) ** 0.5, score=0.55,
                  nll=lambda s, d=d: base + d + _batches(s))  # fmt: skip
    fit = planner.fit(c)
    assert fit["decisions"]["rank_small_sizes"] == 16
    assert fit["decisions"]["rank_lr_factor_per_doubling"] == planner.ROOT_HALF
    assert fit["recipe"]["2B"]["rank"] == 16 and fit["recipe"]["4B"]["rank"] == 64
    # 16 vs the sweep's 32: one halving at the 1/√r rule, × √2
    expect = fit["recipe"]["0.8B"]["sweep_lr"] / fit["recipe"]["2B"]["sweep_lr"]
    assert expect == pytest.approx((1024 / 2048) ** -planner.P_WIDTH, rel=0.02)


def test_a_same_rate_win_at_rank_128_keeps_the_rate_across_ranks(tmp_path):
    c = _campaign(tmp_path)
    _sweep(c.runs)
    base = _centre()
    _fake_run(c.runs, "sweep-r128-s0", lr=planner.CENTRE / 2, score=0.55,
              nll=lambda s: base + 0.03 + _batches(s))  # fmt: skip
    _fake_run(c.runs, "sweep-r128-samelr-s0", lr=planner.CENTRE, score=0.55,
              nll=lambda s: base + _batches(s))  # fmt: skip
    fit = planner.fit(c)
    assert fit["decisions"]["rank_lr_factor_per_doubling"] == 1.0
    assert fit["decisions"]["rank_small_sizes"] == 32  # no rank beat the centre


def test_rank_prices_the_lora_share_of_the_step():
    assert planner.rank_slowdown("0.8B", 32) == 1.0
    assert planner.rank_slowdown("0.8B", 16) == pytest.approx(0.935)
    assert planner.rank_slowdown("9B", 128) == pytest.approx(1.48)


def test_long_buckets_per_size_fit_a_v6e_chip(tmp_path):
    """--max-bucket 16384 on v6e1: each size up to its largest bucket that fits one chip by
    the memory model (16384 for 0.8B / 2B, 8192 for 4B with remat=full, 9B unchanged), at a
    microbatch whose rows fill it; every job is a valid run config."""
    c = _campaign(tmp_path, hardware="v6e1", proxy="0.8B", max_bucket=16384)
    want = {"0.8B": (16384, 16384, "core_states"), "2B": (16384, 16384, "core_states"),
            "4B": (8192, 8192, "full")}  # fmt: skip
    seen = set()
    for phase in ("sweep", "final"):
        for job in parse_lines(planner.plan(phase, c), c.runs):
            files = [a for a in job.args if "=" not in a]
            cfg = load_config(*files, overrides=[a for a in job.args if "=" in a])
            size = cfg.model.path.rsplit("-", 1)[1]
            top, microbatch, remat = want[size]
            assert cfg.train.seq_buckets == tuple(b for b in planner.LONG_BUCKETS if b <= top)
            assert cfg.train.microbatch_tokens == microbatch
            assert cfg.compute.remat == remat
            seen.add(size)
            hbm = planner.step_hbm_gb(size, cfg.train.microbatch_tokens_at(top), remat)
            assert hbm < planner.HBM_SAFE * planner.HBM_GB["v6e1"], (size, hbm)
    assert seen == {"0.8B", "2B", "4B"}
    assert planner.long_settings(c, "9B")[0] == (1024, 2048, 4096)
    # the default plans are unchanged: no bucket override
    default = planner.plan("final", _campaign(tmp_path, hardware="v6e1"))
    assert not any("seq_buckets" in line for line in default)
    with pytest.raises(ValueError, match="max_bucket"):
        _campaign(tmp_path, max_bucket=12000)


def test_step_memory_model_matches_the_bench_and_flags_the_long_rows():
    """The measured points are reproduced; 4B cannot take 8192 tokens with core_states, nor
    16384 at all, on a 32 GB chip; 0.8B / 2B take 16384."""
    for size, m in planner.STEP_HBM.items():
        assert planner.step_hbm_gb(size, m["tokens"]) == pytest.approx(m["arguments"] + m["temp"])
    limit = planner.HBM_SAFE * 32
    assert planner.step_hbm_gb("4B", 8192) > limit
    assert planner.step_hbm_gb("4B", 16384, "full") > limit
    assert planner.step_hbm_gb("2B", 16384) < limit and planner.step_hbm_gb("0.8B", 16384) < limit


def _config(job):
    return load_config(*[a for a in job.args if "=" not in a],
                       overrides=[a for a in job.args if "=" in a])  # fmt: skip


def test_the_long_final_run_protects_its_branch_points_and_checkpoints_its_cooldown(tmp_path):
    """An h750 branch once failed: its branch point (step 600 of h3000) had been pruned by
    keep_checkpoints before the sequential queue reached the branch."""
    c = _campaign(tmp_path, horizon=3000, sweep_steps=300, final_sizes=("4B",),
                  final_recipe="generic")  # fmt: skip
    jobs = parse_lines(planner.plan("final", c), c.runs)
    long = next(j for j in jobs if j.name == "final-4b-h3000")
    cfg = _config(long)
    points = sorted(int(j.after.split("@")[1]) for j in jobs if j.after)
    assert points == [600, 1200] and list(cfg.train.keep_steps) == points
    assert cfg.train.checkpoint_tail_every == 100 and cfg.train.checkpoint_every == 300
    from tjev.train.loop import Cadence

    cadence = Cadence(cfg)
    assert all(not cadence.resume_only(p) for p in points)
    tail = [s for s in range(2401, 3001) if cadence.full_eval(s)]
    assert tail == list(range(2500, 3001, 100))


def test_reg_stage_at_a_final_horizon_branches_rank16_and_keeps_the_late_trajectory(tmp_path):
    c = _campaign(tmp_path, hardware="v6e1", stage="reg", final_sizes=("4B",), horizon=5000,
                  arms=("rank16", "rank16-rep1", "nocentre"), wandb={"campaign": "c1"})  # fmt: skip
    (tmp_path / "mix-rep1").mkdir()
    (tmp_path / "mix-rep1" / "mix.yaml").write_text("data:\n  validation: v.jsonl\n")
    jobs = parse_lines(planner.plan("sweep", c), c.runs)
    assert [j.name for j in jobs] == ["sweep-reg-4b-rank16-s0", "sweep-reg-4b-rank16-h2500-s0",
                                      "sweep-reg-4b-rank16-rep1-s0"]  # fmt: skip
    long, branch, rep = (_config(j) for j in jobs)
    assert long.train.steps == 5000 and 2000 in long.train.keep_steps
    assert long.train.keep_tail and long.train.quick_eval_every == 25
    assert (long.train.checkpoint_every, long.train.keep_every, long.train.keep_from) == (
        200,
        200,
        2500,
    )
    assert branch.train.steps == 2500 and branch.train.branch_from.endswith("rank16-s0@2000")
    assert jobs[1].after == "sweep-reg-4b-rank16-s0@2000"
    # the repetition arm reads the sibling mix's train files; the held-out stays the campaign's
    assert f"{c.mix}-rep1/mix.yaml" in jobs[2].args and rep.data.heldout.startswith(c.mix + "/")
    assert rep.optim.lr == long.optim.lr == branch.optim.lr
    # final-length reg runs are proper runs: calibrated and benchmarked (post=full)
    assert [j.post for j in jobs] == ["full"] * 3


@pytest.mark.parametrize("horizon", [2500, 5000])
def test_final_plans_the_decided_recipe_for_every_size(tmp_path, horizon):
    """plan final: per size the long run and its 1-epoch cooldown branch to horizon/2, rank 16
    at the rank-rule rate of fit's recipe at the horizon (as the reg rank arms), CogCalib,
    the EMA, post=full, the final-length cadence; the branch never health-killed."""
    sizes = ("4B", "2B", "0.8B", "9B")
    c = _campaign(tmp_path, hardware="v6e1", final_sizes=sizes, horizon=horizon,
                  wandb={"campaign": "c1"})  # fmt: skip
    jobs = parse_lines(planner.plan("final", c), c.runs)
    half, point = horizon // 2, int(0.8 * (horizon // 2))
    assert [j.name for j in jobs] == [
        n
        for s in sizes
        for n in (f"final-{s.lower()}-h{horizon}", f"final-{s.lower()}-h{horizon}-b{half}")
    ]
    report = planner.fit(c)
    factor = report["decisions"]["rank_lr_factor_per_doubling"]
    for long, branch in zip(jobs[::2], jobs[1::2], strict=True):
        lc, bc = _config(long), _config(branch)
        size = lc.model.path.rsplit("-", 1)[1]
        r = report["recipe"][size]
        assert r["horizon"] == horizon
        lr = r["lr"] * factor ** math.log2(16 / r["rank"])
        for cfg in (lc, bc):
            assert cfg.lora.rank == 16 and cfg.optim.lr == pytest.approx(lr, rel=5e-3)
            assert cfg.loss.cogcalib_weight == 0.1 and cfg.loss.cogcalib_quantile == 0.3
            assert cfg.optim.ema_decay == 0.999 and not cfg.train.select_ema
            assert not set(cfg.lora.targets) & {"in_proj_a", "in_proj_b"}  # the frozen gates
            assert cfg.train.packing == "best_fit" and cfg.compute.gdn_norm == "flat"
            assert (cfg.train.checkpoint_every, cfg.train.checkpoint_tail_every) == (200, 50)
            assert (cfg.train.keep_every, cfg.train.keep_from) == (200, half)
            assert cfg.train.quick_eval_every == 25 and cfg.train.keep_tail
            assert f"Qwen3.5-{size}" in cfg.model.path
        assert lc.train.steps == horizon and list(lc.train.keep_steps) == [point]
        assert lc.train.health_kill and not bc.train.health_kill
        assert bc.train.steps == half and bc.train.branch_from.endswith(f"{long.name}@{point}")
        assert branch.after == f"{long.name}@{point}"
        assert long.post == branch.post == "full" and long.chips == 1
        assert "log.wandb_group=c1/final" in " ".join(long.args)
    # the rank rule from fit's ranks: 4B / 9B from 64 (lr × 2), 2B / 0.8B from 32 (lr × √2)
    rates = {_config(j).model.path.rsplit("-", 1)[1]: _config(j).optim.lr for j in jobs[::2]}
    assert rates["4B"] == pytest.approx(2 * report["recipe"]["4B"]["lr"], rel=5e-3)
    assert rates["2B"] == pytest.approx(2**0.5 * report["recipe"]["2B"]["lr"], rel=5e-3)
    if horizon == 5000:  # the reg stage's final-length 4B run at the same horizon: same rate
        reg = _campaign(tmp_path, hardware="v6e1", stage="reg", final_sizes=("4B",),
                        horizon=5000, arms=("rank16-cogcalib", "nocentre"))  # fmt: skip
        assert _config(parse_lines(planner.plan("sweep", reg), reg.runs)[0]).optim.lr == rates["4B"]


def test_final_on_a_sibling_mix_and_the_generic_recipe(tmp_path):
    (tmp_path / "mix-rep1").mkdir()
    (tmp_path / "mix-rep1" / "mix.yaml").write_text("data:\n  validation: v.jsonl\n")
    c = _campaign(tmp_path, hardware="v6e1", final_sizes=("4B",), horizon=2500, final_mix="rep1")
    jobs = parse_lines(planner.plan("final", c), c.runs)
    assert [j.name for j in jobs] == ["final-4b-h2500-rep1", "final-4b-h2500-b1250-rep1"]
    assert jobs[1].after == "final-4b-h2500-rep1@1000"
    for j in jobs:  # the sibling's train files, the campaign's held-out
        assert f"{c.mix}-rep1/mix.yaml" in j.args
        assert _config(j).data.heldout.startswith(c.mix + "/")
    generic = replace(c, final_mix="", final_recipe="generic")
    names = [x.split()[0] for x in planner.plan("final", generic)]
    assert names == ["final-4b-h2500", "final-4b-h625", "final-4b-h1250"]
    assert _config(parse_lines(planner.plan("final", generic), c.runs)[0]).lora.rank == 64
    with pytest.raises(ValueError, match="final_recipe"):
        _campaign(tmp_path, final_recipe="best")


def test_final_length_reg_runs_track_the_ema(tmp_path):
    """Past REG_STEPS every reg run tracks the EMA (raw and EMA scored at each full eval); at
    REG_STEPS only the centre does."""
    c = _campaign(tmp_path, hardware="v6e1", stage="reg", final_sizes=("4B",), horizon=5000,
                  arms=("rank16-cogcalib", "wd0.1"))  # fmt: skip
    lines = planner.plan("sweep", c)
    assert all(line.split().count("optim.ema_decay=0.999") == 1 for line in lines)
    assert all(_config(j).optim.ema_decay == 0.999 for j in parse_lines(lines, c.runs))
    short = replace(c, horizon=0)
    cfgs = {j.name: _config(j) for j in parse_lines(planner.plan("sweep", short), c.runs)}
    assert cfgs["sweep-reg-4b-centre-s0"].optim.ema_decay == 0.999
    assert cfgs["sweep-reg-4b-wd0.1-s0"].optim.ema_decay == 0.0


def test_reg_stage_at_a_final_horizon_branches_rank16_cogcalib_too(tmp_path):
    c = _campaign(tmp_path, hardware="v6e1", stage="reg", final_sizes=("4B",), horizon=5000,
                  arms=("rank16-cogcalib", "rank16-cogcalib-rep1", "nocentre"),
                  wandb={"campaign": "c1"})  # fmt: skip
    (tmp_path / "mix-rep1").mkdir()
    (tmp_path / "mix-rep1" / "mix.yaml").write_text("data:\n  validation: v.jsonl\n")
    jobs = parse_lines(planner.plan("sweep", c), c.runs)
    assert [j.name for j in jobs] == ["sweep-reg-4b-rank16-cogcalib-s0",
                                      "sweep-reg-4b-rank16-cogcalib-h2500-s0",
                                      "sweep-reg-4b-rank16-cogcalib-rep1-s0"]  # fmt: skip
    assert jobs[1].after == "sweep-reg-4b-rank16-cogcalib-s0@2000"
    assert [j.post for j in jobs] == ["full"] * 3
    assert f"{c.mix}-rep1/mix.yaml" in jobs[2].args
    for cfg in (_config(j) for j in jobs):
        assert cfg.lora.rank == 16
        assert cfg.loss.cogcalib_weight == 0.1 and cfg.loss.cogcalib_quantile == 0.3
        assert cfg.train.packing == "best_fit" and cfg.compute.gdn_norm == "flat"


def test_reg_stage_pairs_every_arm_with_its_own_sizes_centre(tmp_path):
    c = _campaign(tmp_path, hardware="v6e1", stage="reg", wandb={"campaign": "c1"})
    jobs = parse_lines(planner.plan("sweep", c), c.runs)
    names = [j.name for j in jobs]
    want = []
    for size, arms in planner.REG_PLAN.items():
        want += [f"sweep-reg-{size.lower()}-centre-s0",
                 *(f"sweep-reg-{size.lower()}-{a}-s0" for a in arms)]  # fmt: skip
    assert names == want
    cfgs = {j.name: _config(j) for j in jobs}
    recipe = planner.reg_recipe(c)
    # the planner's own rates at a 1500-step horizon (priors: no sweep run yet)
    assert recipe["4B"]["lr"] == 2.4e-5 and recipe["2B"]["lr"] == 3.5e-5
    for name, cfg in cfgs.items():
        size = "4B" if "-4b-" in name else "2B"
        assert f"Qwen3.5-{size}" in cfg.model.path and cfg.train.steps == planner.REG_STEPS
        assert cfg.data.heldout.startswith(c.mix)  # the campaign's mix (mix-v5)
        assert cfg.train.seed == 0 and cfg.train.tokens_per_step == 65536
        assert not set(cfg.lora.targets) & {"in_proj_a", "in_proj_b"}  # the frozen gates
        assert cfg.train.checkpoint_every == 100 and cfg.train.quick_eval_every == 0
        assert cfg.train.microbatch_tokens == planner.HARDWARE["v6e1"]["microbatch"][size]
        assert cfg.train.packing == "best_fit" and cfg.compute.gdn_norm == "flat"
        assert cfg.compute.remat == "core_states" and cfg.compute.remat_by_bucket == {}
        if "rank16" not in name:
            assert cfg.lora.rank == planner.RANK[size]
        if not name.endswith(("rank16-s0",)):
            assert cfg.optim.lr == recipe[size]["lr"]
    assert all(j.post == "none" and j.chips == 1 for j in jobs)
    for size in ("4b", "2b"):
        assert cfgs[f"sweep-reg-{size}-centre-s0"].optim.ema_decay == 0.999
    assert cfgs["sweep-reg-2b-ls0.1-gen-s0"].loss.smooth_sources == ("block:generators",)
    assert cfgs["sweep-reg-2b-rank16-s0"].lora.rank == 16
    assert cfgs["sweep-reg-2b-rank16-s0"].optim.lr == pytest.approx(3.5e-5 * 2**0.5, rel=1e-2)
    assert cfgs["sweep-reg-4b-wd0.1-s0"].optim.weight_decay == 0.1
    assert cfgs["sweep-reg-4b-floor0.1-s0"].optim.final_lr_fraction == 0.1
    assert cfgs["sweep-reg-2b-cool0.1-s0"].optim.decay_fraction == 0.1
    assert cfgs["sweep-reg-2b-drop0.05-s0"].lora.dropout == 0.05
    assert cfgs["sweep-reg-2b-hard-s0"].loss.hard_targets
    assert cfgs["sweep-reg-2b-kl0.1-s0"].loss.kl_base_weight == 0.1
    lines = planner.plan("sweep", c)
    assert all("log.wandb_group=c1/sweep/reg" in x for x in lines)
    assert "arm:reg-4b-wd0.1" in _tags(lines[1])
    # --sizes and --arms
    only = _campaign(tmp_path, hardware="v6e1", stage="reg", final_sizes=("4B",))
    assert [x.split()[0] for x in planner.plan("sweep", only)] == names[:3]
    some = _campaign(tmp_path, hardware="v6e1", stage="reg", arms=("ls0.05",))
    assert [x.split()[0] for x in planner.plan("sweep", some)] == [
        "sweep-reg-4b-centre-s0", "sweep-reg-4b-ls0.05-s0",
        "sweep-reg-2b-centre-s0", "sweep-reg-2b-ls0.05-s0"]  # fmt: skip
    # the budget: 55-70 CU on a v6e-1 (60.9 at the bench's 4B 14,120 / 2B 36,250 tokens/s,
    # 59.4 at the final line's measured 14,700 / 36,800)
    table = planner.cost(c, ("sweep",), vm_chips=1)["sweep"]
    assert 55 <= table["usd"] <= 70
    four = planner.cost(only, ("sweep",), vm_chips=1)["sweep"]["usd"]
    assert 27 <= four <= 33


def test_reg_cost_prices_the_kl_forward_dropout_and_the_ema_evals(tmp_path):
    def hours(*arms, size="2B"):
        c = _campaign(tmp_path, hardware="v6e1", stage="reg", arms=arms, final_sizes=(size,))
        return planner.cost(c, ("sweep",), vm_chips=1)["sweep"]["chip_hours"]

    base = hours("ls0.1")  # the centre + one plain arm
    plain = hours("ls0.1", "ls0.05") - base  # one plain 1500-step 2B arm
    assert hours("ls0.1", "kl0.1") - base > 1.25 * plain  # + the base's forward
    assert hours("ls0.1", "drop0.05") - base > 1.1 * plain  # unmerged LoRA
    assert base - plain > plain  # the centre: + its EMA evals
    c = _campaign(tmp_path, hardware="v6e1", stage="reg", arms=("ls0.1",), final_sizes=("2B",))
    fields = dict(f.split("=", 1) for f in planner.plan("sweep", c)[1].split() if "=" in f)
    assert planner.full_evals(fields, 1500) == 15


def test_fit_measures_the_reg_arms_against_their_sizes_centre(tmp_path):
    c = _campaign(tmp_path, stage="reg")
    for size, nll in (("4B", 0.6), ("2B", 0.7)):
        _fake_run(c.runs, f"sweep-reg-{size.lower()}-centre-s0", size=size, steps=1500,
                  score=0.50, nll=lambda s, n=nll: n + _batches(s))  # fmt: skip
    _fake_run(c.runs, "sweep-reg-2b-ls0.1-s0", size="2B", steps=1500, score=0.48,
              nll=lambda s: 0.68 + _batches(s))  # fmt: skip
    effects = planner.fit(c)["reg_effects"]
    assert effects["2B/ls0.1"]["better"]
    assert effects["2B/ls0.1"]["online"][0] == pytest.approx(-0.02)
    assert set(effects) == {"2B/ls0.1"}


def test_reg_calibration_arms_run_at_the_rank16_rate(tmp_path):
    """rank16-cogcalib / rank16-teacher: rank 16 at its rank-rule rate (reg_rate parses the
    rank from the name), plus their loss options."""
    c = _campaign(tmp_path, hardware="v6e1", stage="reg",
                  arms=("rank16", "rank16-cogcalib", "rank16-teacher"))  # fmt: skip
    cfgs = {j.name: _config(j) for j in parse_lines(planner.plan("sweep", c), c.runs)}
    for size in ("4b", "2b"):
        r16 = cfgs[f"sweep-reg-{size}-rank16-s0"]
        calib = cfgs[f"sweep-reg-{size}-rank16-cogcalib-s0"]
        teacher = cfgs[f"sweep-reg-{size}-rank16-teacher-s0"]
        for cfg in (calib, teacher):
            assert cfg.lora.rank == 16 and cfg.optim.lr == r16.optim.lr
        assert calib.loss.cogcalib_weight == 0.1 and calib.loss.cogcalib_quantile == 0.3
        assert calib.loss.cogcalib_adaptive and calib.loss.teacher_mix == 0.0
        assert teacher.loss.teacher_mix == 0.5 and teacher.loss.teacher_gate == 0.6
        assert teacher.loss.cogcalib_weight == 0.0
    assert cfgs["sweep-reg-2b-rank16-cogcalib-s0"].optim.lr == pytest.approx(
        3.5e-5 * 2**0.5, rel=1e-2
    )
    assert cfgs["sweep-reg-4b-rank16-cogcalib-s0"].optim.lr == pytest.approx(2.4e-5 * 2, rel=1e-2)


def test_reg_optimizer_arms_on_the_cc0_1_q0_6_base(tmp_path):
    """β2 raised in the cooldown and LoRA+ (B at 8x): rank16-cc0.1-q0.6 otherwise, at its
    rate (A's), final-length (post=full, EMA)."""
    arms = ("rank16-cc0.1-q0.6", "rank16-cc0.1-q0.6-b2cool", "rank16-cc0.1-q0.6-loraplus8")
    c = _campaign(tmp_path, hardware="v6e1", stage="reg", final_sizes=("2B",), horizon=2500,
                  arms=(*arms, "nocentre"), wandb={"campaign": "c1"})  # fmt: skip
    jobs = parse_lines(planner.plan("sweep", c), c.runs)
    assert [j.name for j in jobs] == [f"sweep-reg-2b-{a}-s0" for a in arms]
    assert all(j.post == "full" and not j.after for j in jobs)
    base, b2cool, loraplus = (_config(j) for j in jobs)
    for cfg in (b2cool, loraplus):
        assert cfg.lora.rank == 16 and cfg.optim.lr == base.optim.lr
        assert cfg.loss.cogcalib_weight == 0.1 and cfg.loss.cogcalib_quantile == 0.6
        assert cfg.optim.ema_decay == 0.999 and cfg.train.steps == 2500
    assert (base.optim.cooldown_b2, base.optim.loraplus_ratio) == (0.0, 1.0)
    assert (b2cool.optim.cooldown_b2, b2cool.optim.loraplus_ratio) == (0.9999, 1.0)
    assert (loraplus.optim.cooldown_b2, loraplus.optim.loraplus_ratio) == (0.0, 8.0)


def test_reg_cogcalib_strength_arms_on_2b_at_a_final_length(tmp_path):
    """The CogCalib sweep: off, the decided 0.1 / q0.3, 0.3 / q0.6, 1.0 / q0.6, all r16 at
    the same rate (paired with each other, no centre), on 2B at a 2500-step horizon: a
    final-length run (post=full, long_cadence, the EMA tracked), no cooldown branch."""
    arms = ("rank16-cc0", "rank16-cc0.1-q0.3", "rank16-cc0.3-q0.6", "rank16-cc1.0-q0.6")
    c = _campaign(tmp_path, hardware="v6e1", stage="reg", final_sizes=("2B",), horizon=2500,
                  arms=(*arms, "nocentre"), wandb={"campaign": "c1"})  # fmt: skip
    jobs = parse_lines(planner.plan("sweep", c), c.runs)
    assert [j.name for j in jobs] == [f"sweep-reg-2b-{a}-s0" for a in arms]
    assert all(j.post == "full" and not j.after and j.chips == 1 for j in jobs)
    cfgs = [_config(j) for j in jobs]
    want = [(0.0, 0.0), (0.1, 0.3), (0.3, 0.6), (1.0, 0.6)]
    assert [(cfg.loss.cogcalib_weight, cfg.loss.cogcalib_quantile) for cfg in cfgs] == want
    for cfg in cfgs:
        assert cfg.lora.rank == 16 and cfg.optim.lr == cfgs[0].optim.lr
        assert cfg.optim.ema_decay == 0.999 and cfg.train.steps == 2500
        assert cfg.train.checkpoint_every == 200 and cfg.train.checkpoint_tail_every == 50
        assert cfg.train.quick_eval_every == 25 and cfg.train.keep_tail
        assert cfg.train.keep_every == 200 and cfg.train.keep_from == 1250
        assert "Qwen3.5-2B" in cfg.model.path and cfg.train.health_kill
