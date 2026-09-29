"""The campaign planner: every planned job is a valid run config; the fit recovers rates from
the paired online loss, adopts arms, uses measured sizes and horizons; costs."""

import json
import math
from pathlib import Path

import pytest

from tjev.campaign import plan as planner
from tjev.campaign.queue import parse_lines
from tjev.config import load_config


def _campaign(tmp_path, **kw) -> planner.Campaign:
    mix = tmp_path / "mix"
    mix.mkdir(exist_ok=True)
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


def _batches(step):  # the same batch difficulty for every run with the same seed
    return 0.02 * ((step * 7919) % 13 - 6) / 6


def _sweep(runs: Path, optimum=3e-5, curvature=0.05):
    for lr in planner.SWEEP_GRID:
        offset = curvature * math.log2(lr / optimum) ** 2
        _fake_run(runs, f"sweep-lr{lr:.3g}-s0", lr=lr, score=0.55,  # eval score: flat, noisy
                  nll=lambda s, o=offset: 0.6 + o + _batches(s))  # fmt: skip


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
    # sweep: 4 rates + the centre on a 2nd seed + 3 arms; transfer: 2B horizon (+2 branches)
    # + 2 sizes × 3 rates; final: 3 sizes × (long + 2 branches)
    assert counts == {"sweep": 8, "transfer": 9, "final": 9}
    final = parse_lines(planner.plan("final", c), c.runs)
    assert all("@" in j.after for j in final if j.after)
    assert sum(1 for j in final if j.after) == 6


def test_no_kernels_and_wandb_overrides(tmp_path):
    c = _campaign(tmp_path, kernels=False, wandb={"project": "p", "campaign": "c1"})
    lines = planner.plan("sweep", c)
    assert all("compute.gdn_impl=chunked" in line for line in lines)
    assert all("log.wandb_group=c1/sweep" in line for line in lines)


def test_fit_before_any_run_uses_the_priors(tmp_path):
    fit = planner.fit(_campaign(tmp_path))
    assert fit["decisions"]["proxy_lr"] == planner.PRIOR_LR
    recipe = fit["recipe"]
    assert recipe["4B"]["rank"] == 64
    assert recipe["4B"]["sweep_lr"] < recipe["2B"]["sweep_lr"] < recipe["0.8B"]["sweep_lr"]
    factor = (1200 / 600) ** -planner.GAMMA  # the default 1200-step horizon at the prior γ
    assert recipe["2B"]["lr"] == pytest.approx(recipe["2B"]["sweep_lr"] * factor, rel=0.05)


def test_fit_recovers_the_rate_from_the_paired_online_loss(tmp_path):
    c = _campaign(tmp_path)
    _sweep(c.runs, optimum=3e-5)
    fit = planner.fit(c)
    assert fit["curve"]["bracketed"]
    assert fit["decisions"]["proxy_lr"] == pytest.approx(3e-5, rel=0.03)


def test_an_arm_is_adopted_on_a_clear_paired_win(tmp_path):
    c = _campaign(tmp_path)
    _sweep(c.runs)
    base_offset = 0.05 * math.log2(planner.CENTRE / 3e-5) ** 2
    _fake_run(c.runs, "sweep-zclip-s0", lr=planner.CENTRE, score=0.55,
              nll=lambda s: 0.6 + base_offset - 0.03 + _batches(s))  # fmt: skip
    _fake_run(c.runs, "sweep-noclip-s0", lr=planner.CENTRE, score=0.55,
              nll=lambda s: 0.6 + base_offset + 0.03 + _batches(s))  # fmt: skip
    fit = planner.fit(c)
    assert fit["effects"]["zclip"]["better"]
    assert not fit["effects"]["noclip"]["better"]
    assert fit["effects"]["zclip"]["online"][0] == pytest.approx(-0.03)
    assert "optim.clip_mode=zscore" in fit["recipe"]["2B"]["overrides"]


def test_measured_sizes_and_horizon_override_the_rules(tmp_path):
    c = _campaign(tmp_path)
    for lr, score in ((1e-5, 0.60), (2e-5, 0.50), (4e-5, 0.60)):
        _fake_run(c.runs, f"transfer-4b-lr{lr:.3g}", size="4B", lr=lr, score=score)
    for steps, score in ((600, 0.52), (1200, 0.55), (2400, 0.50)):
        _fake_run(c.runs, f"transfer-2b-h{steps}", steps=steps, score=score)
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
    assert table["total"]["jobs"] == 26
    assert table["total"]["usd"] == pytest.approx(table["total"]["chip_hours"], rel=1e-9)
    bench = {
        "tokens_per_second_per_chip": dict.fromkeys(planner.SIZES, 1000000000.0)
    }  # instant training
    assert planner.cost(c, bench=bench)["total"]["chip_hours"] < table["total"]["chip_hours"]
