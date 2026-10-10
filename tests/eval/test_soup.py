"""ΔW soups: a soup of one run is the run, a soup of two adds the mean of their deltas, the
rank-Σr form gives the merged-weights logits, the greedy scorer, and `tjev post` on a soup."""

import json
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest
from typer.testing import CliRunner

from tjev.cli import app
from tjev.config import load_config
from tjev.data.item import parse_item, write_jsonl
from tjev.eval import soup
from tjev.eval.runs import LoadedRun
from tjev.model import build_model
from tjev.testing import make_tiny_snapshot, tiny_items
from tjev.train.loop import train
from tjev.train.step import jit_eval_step, make_eval_step, split_model

TINY = [
    "model.dtype=float32", "train.seq_buckets=[1024]", "train.microbatch_tokens=1024",
    "train.tokens_per_step=8192", "train.max_segments=4", "compute.gdn_chunk=16",
    "log.tensorboard=false", "train.steps=3", "train.checkpoint_every=3",
    "train.quick_eval_every=0", "optim.lr=3e-3", "optim.warmup_steps=1",
]  # fmt: skip


def test_member_spec_parses_step_and_weights():
    assert soup.MemberSpec.parse("runs/a") == soup.MemberSpec(Path("runs/a"))
    assert soup.MemberSpec.parse("runs/a@1200") == soup.MemberSpec(Path("runs/a"), 1200)
    assert soup.MemberSpec.parse("runs/a@1200:ema") == soup.MemberSpec(Path("runs/a"), 1200, "ema")
    assert soup.MemberSpec.parse("runs/a:ema") == soup.MemberSpec(Path("runs/a"), None, "ema")
    assert str(soup.MemberSpec.parse("runs/a@7:ema")) == "runs/a@7:ema"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("soup")
    make_tiny_snapshot(root / "model")
    mix = root / "mix"
    write_jsonl(mix / "train.jsonl", [parse_item(r) for r in tiny_items(200, seed=1)])
    write_jsonl(mix / "validation.jsonl", [parse_item(r) for r in tiny_items(30, seed=2)])
    write_jsonl(mix / "calibration.jsonl", [parse_item(r) for r in tiny_items(60, seed=3)])
    # two runs of different ranks and seeds; the second keeps EMA adapters
    for name, extra in (("a", ["lora.rank=4", "train.seed=0"]),
                        ("b", ["lora.rank=2", "train.seed=1", "optim.ema_decay=0.5"])):  # fmt: skip
        overrides = [f"name={name}", f"output={root / 'runs'}", f"model.path={root / 'model'}",
                     f"data.train=[{mix / 'train.jsonl'}]", *TINY, *extra]  # fmt: skip
        train(load_config(overrides=overrides))
    return root


def logits(run: LoadedRun, data: Path) -> np.ndarray:
    evalset = run.evalset(data)
    return evalset.arrays(run.eval_fn, run.lora, run.frozen, run.mesh, run.vocab)["logits"]


def deltas(run_dir: Path, **kw) -> dict[str, np.ndarray]:
    run = LoadedRun(run_dir, **kw)
    return {p: np.asarray(jax.device_get(layer.delta()))
            for p, layer in soup.linears(run.model()).items() if layer.rank}  # fmt: skip


@pytest.mark.slow
def test_a_soup_of_one_run_is_the_run(runs):
    data = runs / "mix" / "validation.jsonl"
    out = runs / "soup-a"
    info = soup.soup([str(runs / "runs" / "a")], out)
    assert info["rank"] == 4 and info["members"][0]["weight"] == 1.0
    member, made = LoadedRun(runs / "runs" / "a"), LoadedRun(out)
    assert made.step == soup.SOUP_STEP and made.cfg.lora.scale == pytest.approx(1.0)
    np.testing.assert_allclose(logits(made, data), logits(member, data), atol=1e-5)
    assert made.checkpoint_identity != member.checkpoint_identity
    with pytest.raises(FileExistsError):
        soup.soup([str(runs / "runs" / "a")], out)


@pytest.mark.slow
def test_a_soup_of_two_adds_the_mean_delta_and_equals_merged_weights(runs):
    a, b = runs / "runs" / "a", runs / "runs" / "b"
    out = runs / "soup-ab"
    info = soup.soup([str(a), f"{b}@3:ema"], out)
    assert info["rank"] == 6 and [m["weights"] for m in info["members"]] == ["raw", "ema"]
    da, db, ds = deltas(a), deltas(b, step=3, weights="ema"), deltas(out)
    assert not np.allclose(da[next(iter(da))], db[next(iter(db))])
    assert set(ds) == set(da)
    for path in ds:
        np.testing.assert_allclose(ds[path], (da[path] + db[path]) / 2, atol=1e-6)
    # the EMA adapters differ from the raw ones (the :ema member is not the raw step)
    raw = deltas(b, step=3, weights="raw")
    assert any(not np.allclose(raw[p], db[p]) for p in raw)
    # the rank-Σr adapter gives the logits of the merged kernels W + mean ΔW
    made = LoadedRun(out)
    cfg = made.cfg
    _, base = build_model(cfg.model.path, cfg.compute, None, dtype=cfg.model.dtype)
    for path, layer in soup.linears(base).items():
        if path in ds:
            layer.kernel[...] = layer.weight() + ds[path]
    graphdef, lora, frozen = split_model(base)
    data = runs / "mix" / "validation.jsonl"
    evalset = made.evalset(data)
    merged = evalset.arrays(jit_eval_step(make_eval_step(graphdef)), lora, frozen, made.mesh,
                            made.vocab)["logits"]  # fmt: skip
    np.testing.assert_allclose(logits(made, data), merged, atol=1e-4)


@pytest.mark.slow
def test_members_must_share_base_and_targets(runs):
    a = soup.load_member(soup.MemberSpec(runs / "runs" / "a"))
    b = soup.load_member(soup.MemberSpec(runs / "runs" / "b"))
    soup.check_compatible([a, b])
    b.cfg = replace(b.cfg, lora=replace(b.cfg.lora, targets=("q_proj",)))
    with pytest.raises(ValueError, match="LoRA targets"):
        soup.check_compatible([a, b])
    b.identity = b.identity | {"model": "another"}
    with pytest.raises(ValueError, match="base model"):
        soup.check_compatible([a, b])


@pytest.mark.slow
def test_greedy_soup_scores_members_in_one_model(runs):
    from tjev.eval.runs import evaluate

    cal = runs / "mix" / "calibration.jsonl"
    members = [soup.load_member(soup.MemberSpec(runs / "runs" / n)) for n in ("a", "b")]
    scorer = soup.Scorer(members, cal, "nll")
    # one-hot weights: the other member's B block is zero, the score is the member's own
    for i, name in enumerate(("a", "b")):
        own = evaluate(runs / "runs" / name, cal)["all"]["nll"]
        assert scorer([float(i == j) for j in range(2)]) == pytest.approx(own, abs=1e-4)
    weights, log = soup.greedy(scorer)
    assert sum(weights) == pytest.approx(1.0) and sum(r.get("kept", False) for r in log) >= 1


@pytest.mark.slow
def test_post_takes_a_greedy_soup_from_the_cli(runs):
    out = runs / "soup-greedy"
    cal = runs / "mix" / "calibration.jsonl"
    result = CliRunner().invoke(app, ["soup", str(runs / "runs" / "a"), str(runs / "runs" / "b"),
                                      "--out", str(out), "--greedy", "--select-on", str(cal)])  # fmt: skip
    assert result.exit_code == 0, result.output
    info = json.loads((out / soup.SOUP_FILE).read_text())
    assert info["method"] == "greedy" and info["metric"] == "nll-cal"
    jev = runs / "mix" / "validation.jsonl"
    result = CliRunner().invoke(app, ["post", str(out), "--mix", str(runs / "mix"),
                                      "--jevbench", str(jev), "--quick"])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (out / "post" / "jevbench-cal.json").exists()
