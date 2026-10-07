"""Export: PEFT adapters and the merged snapshot equal W + s·AB, reference logits; the MLX
side (calibration bound to checkpoint and quantization, parity, eval) with a fake scorer."""

import json

import numpy as np
import pytest
from safetensors.numpy import load_file

from tjev.config import load_config
from tjev.data.item import parse_item, write_jsonl
from tjev.data.render import TEMPLATE_VERSION
from tjev.eval import calibrate as cal
from tjev.export import mlx
from tjev.export.peft import DECISION_META, export
from tjev.testing import make_tiny_snapshot, tiny_items
from tjev.train.loop import train

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("export")
    make_tiny_snapshot(root / "model")
    write_jsonl(root / "train.jsonl", [parse_item(r) for r in tiny_items(200, seed=1)])
    overrides = [
        "name=run", f"output={root / 'runs'}", f"model.path={root / 'model'}",
        "model.dtype=float32", f"data.train=[{root / 'train.jsonl'}]", "train.steps=3",
        "train.seq_buckets=[1024]", "train.microbatch_tokens=1024", "train.tokens_per_step=8192",
        "train.max_segments=4", "train.checkpoint_every=3", "compute.gdn_chunk=16", "lora.rank=4",
        "optim.lr=3e-3", "optim.warmup_steps=1", "log.tensorboard=false",
    ]  # fmt: skip
    train(load_config(overrides=overrides))
    out = root / "export"
    result = export(root / "runs" / "run", out, merged=True)
    return root, out, result


def test_merged_weights_are_the_base_plus_the_adapter_product(exported):
    root, out, result = exported
    assert result["step"] == 3
    assert result["merged_tensors"] == result["adapter_tensors"] // 2
    base = load_file(root / "model" / "model.safetensors")
    merged = load_file(out / "model.safetensors")
    adapters = load_file(out / "adapter_model.safetensors")
    config = json.loads((out / "adapter_config.json").read_text())
    scale = config["lora_alpha"] / np.sqrt(config["r"])  # rsLoRA
    name = "model.layers.0.mlp.gate_proj.weight"
    stem = "base_model.model.model.language_model.layers.0.mlp.gate_proj"
    a, b = adapters[stem + ".lora_A.weight"], adapters[stem + ".lora_B.weight"]
    want = base[name].astype(np.float32) + scale * (b @ a)
    np.testing.assert_allclose(merged[name].astype(np.float32), want, atol=1e-2, rtol=1e-2)
    assert not np.allclose(merged[name].astype(np.float32), base[name])  # the adapter moved
    meta = json.loads((out / DECISION_META).read_text())
    assert meta["template_version"] == TEMPLATE_VERSION
    with pytest.raises(FileExistsError):
        export(root / "runs" / "run", out)


class ReferenceScorer:
    """Returns the export's reference logits (a perfect conversion) for the parity check."""

    def __init__(self, path, reference):
        self.path = str(path)
        self.by_text = {}
        from tjev.data.render import RenderSpec, render

        for raw, z in zip(reference["items"], reference["logits"], strict=True):
            text, _ = render(parse_item(raw), RenderSpec(train=False))
            self.by_text[text] = np.asarray(z)

    def logits(self, text, k):
        return self.by_text[text][:k], len(text.split())


def test_reference_logits_and_parity(exported):
    _, out, _ = exported
    reference = json.loads((out / "reference.json").read_text())
    assert len(reference["items"]) == len(reference["logits"]) >= 12
    report = mlx.parity(ReferenceScorer(out, reference), reference)
    assert report["top1_agreement"] == 1.0
    assert report["max_abs_logit_diff"] == 0.0


class OverconfidentScorer:
    """Right ~70% of the time with huge margins: the fit must pick T > 1."""

    def __init__(self, path):
        self.path = str(path)
        self.rng = np.random.default_rng(0)

    def logits(self, text, k):
        z = self.rng.normal(size=k)
        z[self.rng.integers(k)] += 8.0
        return z, len(text.split())


def _mlx_model(tmp_path, quant):
    model = tmp_path / f"model-{quant}"
    model.mkdir()
    (model / DECISION_META).write_text(json.dumps({"checkpoint": "ckpt-abc", "quant": quant}))
    return model


def test_mlx_calibration_is_bound_to_checkpoint_and_quantization(tmp_path):
    items = [parse_item(r) for r in tiny_items(120, seed=3)]
    model4 = _mlx_model(tmp_path, "4bit")
    art = mlx.fit_calibration(OverconfidentScorer(model4), items, tmp_path / "c.json")
    assert art["backend"] == "mlx"
    assert art["checkpoint"] == "ckpt-abc/mlx-4bit"
    assert all(t > 1.0 for t in art["temperatures"].values()), art["temperatures"]
    saved = json.loads((tmp_path / "c.json").read_text())
    identity = mlx.mlx_identity(model4)
    assert cal.validate(saved, checkpoint=identity, template=TEMPLATE_VERSION,
                        backend="mlx") == art["temperatures"]  # fmt: skip
    with pytest.raises(ValueError, match="checkpoint"):
        cal.validate(saved, checkpoint=mlx.mlx_identity(_mlx_model(tmp_path, "8bit")),
                     template=TEMPLATE_VERSION, backend="mlx")  # fmt: skip
    report = mlx.evaluate(OverconfidentScorer(model4), items, art["temperatures"])
    assert report["all"]["n"] == 120
    assert set(report["type"]) == {"noul", "choice", "score"}
    assert report["all"]["p50_s"] >= 0


@pytest.mark.parametrize("init", ["pissa", "lora-ga", "eva"])
def test_an_init_run_exports_the_base_at_init(tmp_path, capsys, init):
    """lr = 0: the adapters stay at their init, so the export must be the base model: the
    merged snapshot W + s·(A0B0 − A0B0) = W, the PEFT adapter (rank 2r for residual inits:
    [A | A0]·[B ; −B0]) a zero delta on the original W. A resume restores the saved init."""
    from tjev.model.lora_init import INIT_FILE

    make_tiny_snapshot(tmp_path / "model")
    write_jsonl(tmp_path / "train.jsonl", [parse_item(r) for r in tiny_items(60, seed=1)])
    overrides = [
        "name=run", f"output={tmp_path / 'runs'}", f"model.path={tmp_path / 'model'}",
        "model.dtype=float32", f"data.train=[{tmp_path / 'train.jsonl'}]", "train.steps=1",
        "train.seq_buckets=[1024]", "train.microbatch_tokens=1024", "train.tokens_per_step=2048",
        "train.max_segments=4", "train.checkpoint_every=1", "compute.gdn_chunk=16", "lora.rank=4",
        "optim.lr=0.0", "optim.warmup_steps=1", "log.tensorboard=false", f"lora.init={init}",
        "lora.init_batches=2", "lora.init_niter=2", "train.quick_eval_every=0",
    ]  # fmt: skip
    cfg = load_config(overrides=overrides)
    train(cfg)
    run = tmp_path / "runs" / "run"
    assert (run / INIT_FILE).exists()
    assert '"event": "lora_init"' in capsys.readouterr().out
    train(cfg)  # a resume: the init is restored, never recomputed
    assert '"event": "lora_init_restored"' in capsys.readouterr().out
    out = tmp_path / "export"
    export(run, out, merged=True, reference=False)
    base = load_file(tmp_path / "model" / "model.safetensors")
    merged = load_file(out / "model.safetensors")
    adapters = load_file(out / "adapter_model.safetensors")
    config = json.loads((out / "adapter_config.json").read_text())
    residual = init != "eva"
    assert config["r"] == (8 if residual else 4)
    scale = config["lora_alpha"] / np.sqrt(config["r"])
    assert scale == pytest.approx(cfg.lora.scale)
    assert json.loads((out / DECISION_META).read_text())["lora_init"] == init
    for proj in ("mlp.gate_proj", "mlp.down_proj", "linear_attn.in_proj_qkv"):
        name = f"model.layers.0.{proj}.weight"
        want = base[name].astype(np.float32)
        np.testing.assert_allclose(merged[name].astype(np.float32), want, atol=2e-3, rtol=1e-2)
        stem = f"base_model.model.model.language_model.layers.0.{proj}"
        a, b = adapters[stem + ".lora_A.weight"], adapters[stem + ".lora_B.weight"]
        assert np.abs(scale * (b @ a)).max() < 1e-5
        if residual:  # the adapter half that moved W's part into s·A0B0 is not zero
            assert np.abs(a).max() > 0 and np.abs(b).max() > 0


def test_a_gate_full_run_exports_its_trained_gate_kernels(tmp_path):
    """lora.gate_full: the merged snapshot holds the trained in_proj_a / in_proj_b kernels,
    and the PEFT adapter carries them exactly as a rank-d_out LoRA (B = I, A = ΔW) at
    scale 1 (rank_pattern / alpha_pattern): W0 + s·BA = the merged weight."""
    make_tiny_snapshot(tmp_path / "model")
    write_jsonl(tmp_path / "train.jsonl", [parse_item(r) for r in tiny_items(60, seed=1)])
    overrides = [
        "name=run", f"output={tmp_path / 'runs'}", f"model.path={tmp_path / 'model'}",
        "model.dtype=float32", f"data.train=[{tmp_path / 'train.jsonl'}]", "train.steps=2",
        "train.seq_buckets=[1024]", "train.microbatch_tokens=1024", "train.tokens_per_step=2048",
        "train.max_segments=4", "train.checkpoint_every=2", "compute.gdn_chunk=16", "lora.rank=4",
        "optim.lr=1e-2", "optim.warmup_steps=1", "log.tensorboard=false", "lora.gate_full=true",
        "train.quick_eval_every=0",
    ]  # fmt: skip
    cfg = load_config(overrides=overrides)
    train(cfg)
    out = tmp_path / "export"
    export(tmp_path / "runs" / "run", out, merged=True, reference=False)
    base = load_file(tmp_path / "model" / "model.safetensors")
    merged = load_file(out / "model.safetensors")
    adapters = load_file(out / "adapter_model.safetensors")
    config = json.loads((out / "adapter_config.json").read_text())
    for proj in ("in_proj_a", "in_proj_b"):
        name = f"model.layers.0.linear_attn.{proj}.weight"
        stem = f"base_model.model.model.language_model.layers.0.linear_attn.{proj}"
        a, b = adapters[stem + ".lora_A.weight"], adapters[stem + ".lora_B.weight"]
        r = config["rank_pattern"][proj]
        assert a.shape[0] == r == base[name].shape[0] and np.array_equal(b, np.eye(r))
        scale = config["alpha_pattern"][proj] / np.sqrt(r)
        assert scale == pytest.approx(1.0)
        rebuilt = base[name].astype(np.float32) + scale * (b @ a)
        assert np.abs(a).max() > 0  # the gate kernel moved
        np.testing.assert_allclose(merged[name].astype(np.float32), rebuilt, atol=2e-3, rtol=1e-2)
    # the other projections keep their LoRA at the run's rank
    assert config["r"] == 4
