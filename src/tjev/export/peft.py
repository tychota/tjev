"""Export a trained run: PEFT adapters, a merged HF snapshot (for mlx_lm), reference logits.

``OUT/`` after ``tjev export RUN OUT --merged``::

    adapter_model.safetensors, adapter_config.json   PEFT LoRA (HF names, rsLoRA scaling)
    *.safetensors, config.json, tokenizer files       the base snapshot with W + s·AB merged
    tjev_decision.json                                template version, checkpoint identity
    reference.json                                    fp32 JAX letter logits on fixed items

Residual inits (``lora.init`` pissa, lora-ga: the run trained W + s·(AB − A0B0),
tjev.model.lora_init): the merged weight is W + s·(AB − A0B0) (exactly W at init), and the
PEFT adapter, which PEFT adds to the original W, is the rank-2r [A | A0]·[B ; −B0] (PEFT's
PiSSA-to-LoRA conversion). A run of before the offset form trained on a residual kernel
W_res = W − s·A0B0: its merged weight is its own W_res + s·AB.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import jax
import numpy as np

from tjev.config.schema import GATES
from tjev.data.render import TEMPLATE_VERSION
from tjev.eval.runs import LoadedRun
from tjev.export.reference import write_reference
from tjev.model.lora_init import RESIDUAL
from tjev.model.weights import TEXT_PREFIXES, tensor_files
from tjev.utils import write_json

PEFT_PREFIX = "base_model.model.model.language_model."
DECISION_META = "tjev_decision.json"


def export(
    run_dir: str | Path,
    out: str | Path,
    *,
    step: int | None = None,
    merged: bool = False,
    reference: bool = True,
    jevbench: str | Path | None = None,
) -> dict:
    """PEFT adapters (+ a merged HF snapshot for ``mlx_lm.convert``, + reference logits)."""
    from safetensors import safe_open
    from safetensors.numpy import save_file

    run = LoadedRun(run_dir, step)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    model = run.model()
    linears = model.adapted_linears()
    lora = run.cfg.lora
    full = model.full_linears()  # lora.gate_full: kernels trained directly
    residual = run.init in RESIDUAL
    init = residual_inits(model, run.init_adapters) if residual else {}
    adapters = {}
    for name, (linear, s) in linears.items():
        stem = PEFT_PREFIX + name.removesuffix(".weight")
        a, b = np.asarray(linear.lora_a[...][s]).T, np.asarray(linear.lora_b[...][s]).T
        if residual:  # [A | A0]·[B ; −B0]: PEFT adds it to the original W
            a0, b0 = init[id(linear)]
            a, b = np.concatenate([a, a0[s].T]), np.concatenate([b, -b0[s].T], axis=1)
        adapters[stem + ".lora_A.weight"] = a.copy()
        adapters[stem + ".lora_B.weight"] = b.copy()
    # a directly trained kernel as an exact PEFT LoRA: ΔW = W − W0 has at most d_out
    # (16 / 32) rows in HF orientation, so rank d_out with B = I and A = ΔW, at scale 1
    # (lora_alpha = √r with rsLoRA, else r): PEFT adds s·BA = ΔW to the original W
    full_rank: dict[str, int] = {}
    if full:
        originals = _original_weights(Path(run.cfg.model.path), set(full))
        for name, (linear, s) in full.items():
            delta = np.asarray(linear.kernel[...][s], np.float32).T - originals[name]
            r = delta.shape[0]
            stem = PEFT_PREFIX + name.removesuffix(".weight")
            adapters[stem + ".lora_A.weight"] = delta.copy()
            adapters[stem + ".lora_B.weight"] = np.eye(r, dtype=np.float32)
            full_rank[name.rsplit(".", 2)[-2]] = r
    save_file(adapters, str(out / "adapter_model.safetensors"))
    rank = 2 * lora.rank if residual else lora.rank
    # the same scale s at the exported rank (alpha/√r with rsLoRA, else alpha/r)
    alpha = lora.alpha if not residual else lora.scale * (rank**0.5 if lora.rslora else rank)
    write_json(
        out / "adapter_config.json",
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": rank,
            "lora_alpha": alpha,
            "use_rslora": lora.rslora,
            "lora_dropout": 0.0,
            "target_modules": list(lora.targets),
            # the gate projections' own rank (lora.gate_rank; PEFT scales each module by
            # lora_alpha / √r of its own r with rsLoRA, as tjev does)
            **(
                {"rank_pattern": {g: lora.gate_rank for g in GATES if g in lora.targets}}
                if lora.gate_rank and not residual and not full
                else {}
            ),
            **(
                {
                    "rank_pattern": full_rank,
                    "alpha_pattern": {
                        g: (r**0.5 if lora.rslora else r) for g, r in full_rank.items()
                    },
                }
                if full
                else {}
            ),
            "bias": "none",
        },
    )
    write_json(
        out / DECISION_META,
        {
            "template_version": TEMPLATE_VERSION,
            "checkpoint": run.checkpoint_identity,
            "step": run.step,
            "run": str(run.dir),
            "base_model": run.cfg.model.path,
            "letters": run.tok.letter_ids,
            "lora_init": run.init,
        },
    )
    result: dict = {"out": str(out), "adapter_tensors": len(adapters), "step": run.step}
    deltas: dict[int, np.ndarray] = {}
    if merged:
        import ml_dtypes

        src = Path(run.cfg.model.path)
        replaced = 0
        for file in tensor_files(src):
            tensors = {}
            with safe_open(str(file), framework="numpy") as f:
                for key in f.keys():  # noqa: SIM118 (safe_open is not iterable)
                    value = f.get_tensor(key)
                    prefix = next((p for p in TEXT_PREFIXES if key.startswith(p)), "")
                    short = key[len(prefix) :]
                    if short in full and not key.startswith("mtp."):  # the trained kernel
                        linear, s = full[short]
                        value = np.asarray(jax.device_get(linear.kernel[...][s]), np.float32).T
                        tensors[key] = value.astype(ml_dtypes.bfloat16, order="C")
                        replaced += 1
                        continue
                    if short in linears and not key.startswith("mtp."):
                        linear, s = linears[short]
                        if id(linear) not in deltas:  # one product per stacked Linear
                            deltas[id(linear)] = np.asarray(jax.device_get(linear.delta()))
                        if residual and not linear.offset:  # legacy W_res = W − s·A0B0
                            value = np.asarray(jax.device_get(linear.kernel[...][s])).T
                        # W + s·AB (s·(AB − A0B0); legacy W_res + s·AB) in fp32, as bf16
                        value = (value.astype(np.float32) + deltas[id(linear)][s].T).astype(
                            ml_dtypes.bfloat16,
                            order="C",  # safetensors writes the raw buffer
                        )
                        replaced += 1
                    tensors[key] = value
            save_file(tensors, str(out / file.name), metadata={"format": "pt"})
        for path in src.iterdir():
            if path.is_file() and not path.name.endswith(".safetensors"):
                shutil.copy2(path, out / path.name)
        if replaced != len(linears) + len(full):
            raise RuntimeError(f"merged {replaced} of {len(linears) + len(full)} trained weights")
        result["merged_tensors"] = replaced
    if reference:  # after the bf16 model is released: the reference holds an fp32 copy
        run_dir, step = run.dir, run.step
        del run, model, linears, deltas, init
        write_reference(run_dir, step, out / "reference.json", jevbench=jevbench)
    return result


def _original_weights(src: Path, names: set[str]) -> dict[str, np.ndarray]:
    """The base snapshot's weights of ``names`` (text-model names), fp32, HF orientation."""
    from safetensors import safe_open

    out = {}
    for file in tensor_files(src):
        with safe_open(str(file), framework="numpy") as f:
            for key in f.keys():  # noqa: SIM118 (safe_open is not iterable)
                prefix = next((p for p in TEXT_PREFIXES if key.startswith(p)), "")
                short = key[len(prefix) :]
                if short in names and not key.startswith("mtp."):
                    out[short] = np.asarray(f.get_tensor(key), np.float32)
    missing = names - set(out)
    if missing:
        raise KeyError(f"base weights not found: {sorted(missing)[:3]}")
    return out


def residual_inits(model, saved: dict | None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """id(stacked Linear) → its saved (A0, B0) (a residual init's run must have them)."""
    from tjev.model.lora_init import adapted_modules

    if saved is None:
        raise FileNotFoundError("a residual lora.init run without its lora_init.safetensors")
    return {id(layer): saved[path] for path, layer in adapted_modules(model).items()}
