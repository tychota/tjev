"""LoRA initialisations (lora.init): default unchanged; PiSSA, EVA and LoRA-GA keep the
model's output at init, hold the directions their papers prescribe, train, and persist."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from tjev.config import ComputeSpec, LoRASpec
from tjev.data.pack import Batch
from tjev.model import Qwen35, load_hf, stack_layers
from tjev.model.lora_init import (
    adapted_modules,
    initialize,
    randomized_svd,
    restore_init,
    save_init,
)
from tjev.model.norms import RMSNorm
from tjev.testing import make_tiny_snapshot
from tjev.train.objective import loss_sums
from tjev.train.step import split_model

RANK = 4


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tiny-init") / "model"
    make_tiny_snapshot(folder)
    return load_hf(folder, "float32")


def _model(snapshot, init="default", *, dtype=jnp.float32, niter=4, seed=0, **compute):
    config, tensors = snapshot
    lora = LoRASpec(rank=RANK, alpha=8.0, init=init, init_niter=niter, init_batches=3)
    compute = ComputeSpec(gdn_chunk=16, **compute)
    model = Qwen35(stack_layers(dict(tensors), config), config, compute, lora, dtype=dtype,
                   rngs=nnx.Rngs(seed))  # fmt: skip
    return model, lora


def _batches(n=3, rows=2, length=48, seed=0):
    rng = np.random.default_rng(seed)
    half = length // 2
    out = []
    for _ in range(n):
        target = np.zeros((rows, 2, 4), np.float32)
        target[np.arange(rows)[:, None], np.arange(2)[None], rng.integers(0, 4, (rows, 2))] = 1
        out.append(Batch(
            tokens=jnp.asarray(rng.integers(0, 512, (rows, length)), jnp.int32),
            segment_ids=jnp.asarray(np.repeat([[1, 2]], half, 1).repeat(rows, 0), jnp.int32),
            positions=jnp.asarray(np.tile(np.arange(half), 2)[None].repeat(rows, 0), jnp.int32),
            slots=jnp.asarray([[half - 1, length - 1]] * rows, jnp.int32),
            label_ids=jnp.asarray(np.broadcast_to(np.arange(10, 14), (rows, 2, 4)), jnp.int32),
            label_mask=jnp.ones((rows, 2, 4), bool),
            target=jnp.asarray(target),
            weight=jnp.ones((rows, 2), jnp.float32),
            type_id=jnp.zeros((rows, 2), jnp.int32),
            source_id=jnp.zeros((rows, 2), jnp.int32),
            lang_id=jnp.zeros((rows, 2), jnp.int32),
            family_id=jnp.zeros((rows, 2), jnp.int32),
            index=jnp.full((rows, 2), -1, jnp.int32),
        ))  # fmt: skip
    return out


BATCHES = _batches()


def _loss(model, batch):
    hidden = model.hidden(batch.tokens, batch.segment_ids, batch.positions)
    logits = model.label_logits(hidden, batch.slots, batch.label_ids)
    return loss_sums(logits, batch, brier_weight=0.1)[0]


def _hidden(model, batch=BATCHES[0]):
    return np.asarray(model.hidden(batch.tokens, batch.segment_ids, batch.positions), np.float32)


def _init(model, lora):
    return initialize(model, lora, seed=0, batches=lambda: BATCHES, loss=_loss)


def _cosines(basis, want):
    """Cosines of the principal angles between two column spans (orthonormalised)."""
    q1, q2 = np.linalg.qr(basis)[0], np.linalg.qr(want)[0]
    return np.linalg.svd(q1.T @ q2, compute_uv=False)


def _grads(model):
    graphdef, lora, frozen = split_model(model)
    grads = jax.grad(lambda lo: _loss(nnx.merge(graphdef, lo, frozen), BATCHES[0]))(lora)
    return nnx.merge(graphdef, grads, frozen)  # the gradients in the adapters' places


def _params(model):
    return [np.asarray(x) for x in jax.tree.leaves(nnx.state(model))]


def test_default_is_unchanged(snapshot):
    """lora.init=default is the LoRASpec of before the option: same A, B, kernels, no op."""
    config, tensors = snapshot
    model, lora = _model(snapshot)
    legacy = Qwen35(stack_layers(dict(tensors), config), config, ComputeSpec(gdn_chunk=16),
                    LoRASpec(rank=RANK, alpha=8.0), dtype=jnp.float32, rngs=nnx.Rngs(0))  # fmt: skip
    assert not _init(model, lora)
    for a, b in zip(_params(model), _params(legacy), strict=True):
        np.testing.assert_array_equal(a, b)
    for layer in adapted_modules(model).values():
        assert not np.any(np.asarray(layer.lora_b[...]))


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("lora_impl", ["unmerged", "merged"])
def test_pissa_keeps_the_output_and_holds_the_principal_part(snapshot, dtype, lora_impl):
    base, _ = _model(snapshot, dtype=dtype, lora_impl=lora_impl)
    want = _hidden(base)
    weights = {p: np.asarray(m.kernel[...], np.float32) for p, m in adapted_modules(base).items()}
    # the random fixture's spectrum is flat (no decay, unlike trained weights): enough power
    # iterations for the randomized SVD to converge on it
    model, lora = _model(snapshot, "pissa", dtype=dtype, niter=32, lora_impl=lora_impl)
    assert _init(model, lora)
    # the offset form: the adapter s·(AB − A0B0) is exactly zero at init, in any dtype
    np.testing.assert_array_equal(_hidden(model), want)
    if dtype == jnp.bfloat16:
        return
    for path, layer in adapted_modules(model).items():
        w = weights[path]
        np.testing.assert_array_equal(np.asarray(layer.kernel[...]), w)  # the kernel is W
        assert not np.any(np.asarray(layer.delta())), path  # s·(AB − A0B0) = 0
        a, b = np.asarray(layer.lora_a[...]), np.asarray(layer.lora_b[...])
        delta = layer.scale * a @ b  # s·AB
        for i in range(w.shape[0]):  # each stacked layer: s·AB = the top-r SVD of W
            u, s, _ = np.linalg.svd(w[i], full_matrices=False)
            r = min(RANK, len(s))
            got = np.linalg.svd(delta[i], compute_uv=False)[:r]
            np.testing.assert_allclose(got, s[:r], rtol=2e-3)  # the top-r spectrum
            if (
                r == len(s) or s[r] < 0.9 * s[r - 1]
            ):  # a gap after r: the top-r subspace is identifiable
                assert _cosines(np.asarray(layer.lora_a[...][i]), u[:, :r]).min() > 0.99, path


def test_randomized_svd_on_a_decaying_spectrum():
    """The fast SVD at PiSSA's few iterations, on a spectrum that decays like a weight's."""
    rng = np.random.default_rng(0)
    u = np.linalg.qr(rng.normal(size=(96, 96)))[0]
    v = np.linalg.qr(rng.normal(size=(64, 64)))[0]
    s = 1.0 / (1.0 + np.arange(64)) ** 0.7
    w = (u[:, :64] * s) @ v.T
    got_u, got_s, got_vt = randomized_svd(jnp.asarray(w, jnp.float32), 8, 4, jax.random.key(0))
    np.testing.assert_allclose(np.asarray(got_s[:4]), s[:4], rtol=1e-3)
    assert _cosines(np.asarray(got_u[:, :4]), u[:, :4]).min() > 0.999
    assert _cosines(np.asarray(got_vt[:4]).T, v[:, :4]).min() > 0.999


def test_pissa_trains_both_factors(snapshot):
    model, lora = _model(snapshot, "pissa")
    _init(model, lora)
    for path, layer in adapted_modules(_grads(model)).items():
        assert np.any(np.asarray(layer.lora_a[...])), path
        assert np.any(np.asarray(layer.lora_b[...])), path


def _first_gdn_input(model, batch):
    """The input rows of super-block 0's first GDN projections (masked at padding)."""
    x = model.embed[...][batch.tokens]
    norm = RMSNorm(np.asarray(model.blocks.input_norms[0].weight[...][0]),
                   model.config.rms_norm_eps)  # fmt: skip
    h = norm(x)
    h = jnp.where((batch.segment_ids > 0)[..., None], h, 0)
    return np.asarray(h, np.float64).reshape(-1, h.shape[-1])


def test_eva_keeps_the_output_and_holds_the_top_activation_directions(snapshot):
    base, _ = _model(snapshot)
    want = _hidden(base)
    model, lora = _model(snapshot, "eva", niter=32)  # flat random-fixture spectrum
    assert _init(model, lora)
    assert model.blocks.remat == "full" and not model.probing  # the probe pass is undone
    assert "probe" not in str(nnx.graphdef(model))
    np.testing.assert_array_equal(_hidden(model), want)  # B = 0: exactly the base
    for path, layer in adapted_modules(model).items():
        a = np.asarray(layer.lora_a[...])
        assert not np.any(np.asarray(layer.lora_b[...])), path
        eye = np.eye(a.shape[-1])
        for i in range(a.shape[0]):  # orthonormal columns: rows of the HF lora_A
            np.testing.assert_allclose(a[i].T @ a[i], eye, atol=1e-4)
    x = np.concatenate([_first_gdn_input(model, b) for b in BATCHES])
    _, _, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)  # centred: PCA, as PEFT's
    a = np.asarray(model.blocks.gdn[0].in_proj_qkv.lora_a[...][0])
    assert _cosines(a, vt[:RANK].T).min() > 0.99


def test_eva_reports_its_gain(snapshot, capsys):
    """eva_gain: rms(x·A) against a random A's, √(mean_j a_jᵀXᵀXa_j / (tr XᵀX / in))."""
    import json

    model, lora = _model(snapshot, "eva", niter=32)
    _init(model, lora)
    event = json.loads(next(line for line in capsys.readouterr().out.splitlines()
                            if '"eva_gain"' in line))  # fmt: skip
    x = np.concatenate([_first_gdn_input(model, b) for b in BATCHES])
    a = np.asarray(model.blocks.gdn[0].in_proj_qkv.lora_a[...][0], np.float64)
    want = np.sqrt(np.mean(np.sum((x @ a) ** 2, 0)) / (np.sum(x**2) / x.shape[1]))
    assert want > 1.5  # the top directions hold more than the average energy
    assert event["max"] >= event["median"] > 1.0
    assert event["in_proj_qkv"] == pytest.approx(want, rel=0.1)  # median of the 2 layers


def test_eva_trains_b(snapshot):
    model, lora = _model(snapshot, "eva")
    _init(model, lora)
    for path, layer in adapted_modules(_grads(model)).items():
        assert np.any(np.asarray(layer.lora_b[...])), path


def test_lora_ga_keeps_the_output_and_holds_the_gradient_directions(snapshot):
    base, _ = _model(snapshot)
    want = _hidden(base)
    # the full-weight gradient of the init batches, the ground truth
    graphdef, rest = nnx.split(base)

    def total(rest):
        model = nnx.merge(graphdef, rest)
        return sum(_loss(model, b) for b in BATCHES)

    full = nnx.merge(graphdef, jax.grad(total)(rest))
    grads = {p: np.asarray(m.kernel[...]) for p, m in adapted_modules(full).items()}
    model, lora = _model(snapshot, "lora-ga", niter=8)
    assert _init(model, lora)
    np.testing.assert_array_equal(_hidden(model), want)  # the offset is exactly zero
    checked = 0
    for path, layer in adapted_modules(model).items():
        a, b = np.asarray(layer.lora_a[...]), np.asarray(layer.lora_b[...])
        g = grads[path]
        fan_out = g.shape[-1]
        c = fan_out**0.25 / 16**0.5
        for i in range(g.shape[0]):
            u, _, vt = np.linalg.svd(g[i], full_matrices=False)
            if min(g[i].shape) <= 2 * RANK:
                continue  # in_proj_a / b: 4 outputs, fewer than 2r directions (B padded)
            np.testing.assert_allclose(np.linalg.norm(a[i], axis=0), c, rtol=1e-4)
            assert _cosines(a[i], u[:, :RANK]).min() > 0.98, path
            assert _cosines(b[i].T, vt[RANK : 2 * RANK].T).min() > 0.98, path
            checked += 1
    assert checked >= 20
    inits = adapted_modules(model)
    for path, layer in adapted_modules(_grads(model)).items():
        assert np.any(np.asarray(layer.lora_b[...])), path
        if np.any(np.asarray(inits[path].lora_b[...])):  # dA = s·xᵀ(dy Bᵀ)
            assert np.any(np.asarray(layer.lora_a[...])), path


@pytest.mark.parametrize("init", ["pissa", "lora-ga", "eva"])
def test_a_saved_init_restores_bitwise(snapshot, tmp_path, init):
    model, lora = _model(snapshot, init, dtype=jnp.bfloat16)
    _init(model, lora)
    save_init(tmp_path / "init.safetensors", model, init)
    fresh, _ = _model(snapshot, dtype=jnp.bfloat16, seed=7)
    assert restore_init(fresh, tmp_path / "init.safetensors") == init
    for a, b in zip(_params(model), _params(fresh), strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_legacy_init_file_restores_its_residual_kernels(snapshot, tmp_path):
    """An init file of before the offset form (no ``form`` metadata) rebuilds the residual
    kernels W − s·A0B0 that run trained on, without the frozen offset."""
    from safetensors.numpy import save_file

    from tjev.model.lora_init import init_adapters, residual_kernel

    model, lora = _model(snapshot, "pissa", dtype=jnp.bfloat16)
    _init(model, lora)
    tensors = {}
    for name, (a, b) in init_adapters(model).items():
        tensors[name + "/lora_a"], tensors[name + "/lora_b"] = a, b
    save_file(tensors, str(tmp_path / "old.safetensors"),
              metadata={"init": "pissa", "residual": "True"})  # fmt: skip
    fresh, _ = _model(snapshot, dtype=jnp.bfloat16, seed=7)
    kernels = {p: m.kernel[...] for p, m in adapted_modules(fresh).items()}
    assert restore_init(fresh, tmp_path / "old.safetensors") == "pissa"
    for path, layer in adapted_modules(fresh).items():
        assert not layer.offset, path
        a, b = tensors[path + "/lora_a"], tensors[path + "/lora_b"]
        want = residual_kernel(kernels[path], a, b, float(layer.scale), jnp.bfloat16)
        np.testing.assert_array_equal(np.asarray(layer.kernel[...]), np.asarray(want))
        np.testing.assert_array_equal(np.asarray(layer.lora_a[...]), a)


# ------------------------------------------------------------------------------------------
# bf16: the offset form keeps every init at the base model's loss (a residual kernel
# W − s·A0B0 in bf16 did not: LoRA-GA's s·A0B0 is several times W)


def _wide_snapshot(tmp_path_factory):
    """Wider than the tiny fixture, Qwen-like projection scale (~0.02), bf16-exact base."""
    import ml_dtypes

    folder = tmp_path_factory.mktemp("wide-init") / "model"
    make_tiny_snapshot(folder, hidden_size=256, intermediate_size=768, head_dim=64,
                       linear_key_head_dim=64, linear_value_head_dim=64)  # fmt: skip
    config, tensors = load_hf(folder, "float32")
    for k, v in tensors.items():
        if v.ndim == 2 and "embed" not in k and "norm" not in k:
            v = v * 0.4
        tensors[k] = v.astype(ml_dtypes.bfloat16).astype(np.float32)
    return config, tensors


@pytest.fixture(scope="module")
def wide(tmp_path_factory):
    return _wide_snapshot(tmp_path_factory)


WIDE_BATCHES = _batches(n=2, rows=2, length=64)


def _wide_model(snapshot, init, dtype, lora_impl="unmerged"):
    """Rank 32, alpha 32, rsLoRA: the sweep's s = 5.66."""
    config, tensors = snapshot
    lora = LoRASpec(rank=32, alpha=32.0, init=init, init_niter=2, init_batches=2)
    model = Qwen35(stack_layers(dict(tensors), config), config,
                   ComputeSpec(gdn_chunk=16, lora_impl=lora_impl), lora, dtype=dtype,
                   rngs=nnx.Rngs(0))  # fmt: skip
    return model, lora


def _wide_loss(snapshot, init, dtype, lora_impl="unmerged"):
    """Summed loss at init."""
    model, lora = _wide_model(snapshot, init, dtype, lora_impl)
    initialize(model, lora, seed=0, batches=lambda: WIDE_BATCHES, loss=_loss)
    return float(sum(_loss(model, b) for b in WIDE_BATCHES))


@pytest.mark.parametrize("lora_impl", ["unmerged", "merged"])
def test_bf16_inits_keep_the_base_loss(wide, lora_impl):
    """At init, in bf16, pissa / eva / lora-ga give the default init's loss (bitwise: the
    adapter of an offset init is exactly zero), whose distance to the fp32 truth is the
    base model's own bf16 noise."""
    truth = _wide_loss(wide, "default", jnp.float32)
    base = _wide_loss(wide, "default", jnp.bfloat16, lora_impl)
    assert base != truth  # bf16 noise exists, and is what the inits must stay within
    for init in ("pissa", "eva", "lora-ga"):
        assert _wide_loss(wide, init, jnp.bfloat16, lora_impl) == base, init


@pytest.mark.parametrize("init", ["pissa", "lora-ga"])
def test_bf16_offset_inits_track_fp32_after_steps(wide, tmp_path, init):
    """Away from init (A, B moved by ±1e-4, a few optimizer steps), the bf16 model stays
    within ~the base's bf16 noise of the fp32 model with the same A0, B0, A, B: the adapter
    is computed from the small differences A − A0, B − B0, never as the cancellation of
    s·AB against a bf16 residual kernel (which put LoRA-GA ~10x the noise away)."""
    model, lora = _wide_model(wide, init, jnp.bfloat16)
    initialize(model, lora, seed=0, batches=lambda: WIDE_BATCHES, loss=_loss)
    save_init(tmp_path / "init.safetensors", model, init)
    ref, _ = _wide_model(wide, "default", jnp.float32)
    restore_init(ref, tmp_path / "init.safetensors")
    rng = np.random.default_rng(1)
    for layer, other in zip(adapted_modules(model).values(), adapted_modules(ref).values(),
                            strict=True):  # fmt: skip
        for p, q in ((layer.lora_a, other.lora_a), (layer.lora_b, other.lora_b)):
            p[...] = p[...] + 1e-4 * rng.choice([-1.0, 1.0], p.shape).astype(np.float32)
            q[...] = p[...]
    got = abs(float(sum(_loss(model, b) - _loss(ref, b) for b in WIDE_BATCHES)))
    noise = abs(
        _wide_loss(wide, "default", jnp.bfloat16) - _wide_loss(wide, "default", jnp.float32)
    )
    assert got < 3 * noise, (got, noise)
