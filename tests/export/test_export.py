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
