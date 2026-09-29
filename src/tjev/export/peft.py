"""Export a trained run: PEFT adapters, a merged HF snapshot (for mlx_lm), reference logits.

``OUT/`` after ``tjev export RUN OUT --merged``::

    adapter_model.safetensors, adapter_config.json   PEFT LoRA (HF names, rsLoRA scaling)
    *.safetensors, config.json, tokenizer files       the base snapshot with W + s·AB merged
    tjev_decision.json                                template version, checkpoint identity
    reference.json                                    fp32 JAX letter logits on fixed items
"""

from __future__ import annotations

import shutil
from pathlib import Path

import jax
import numpy as np

from tjev.data.render import TEMPLATE_VERSION
from tjev.eval.runs import LoadedRun
from tjev.export.reference import write_reference
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
    linears = run.model().adapted_linears()
    lora = run.cfg.lora
    adapters = {}
    for name, (linear, s) in linears.items():
        stem = PEFT_PREFIX + name.removesuffix(".weight")
        adapters[stem + ".lora_A.weight"] = np.asarray(linear.lora_a[...][s]).T.copy()
        adapters[stem + ".lora_B.weight"] = np.asarray(linear.lora_b[...][s]).T.copy()
    save_file(adapters, str(out / "adapter_model.safetensors"))
    write_json(
        out / "adapter_config.json",
        {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": lora.rank,
            "lora_alpha": lora.alpha,
            "use_rslora": lora.rslora,
            "lora_dropout": 0.0,
            "target_modules": list(lora.targets),
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
                    if short in linears and not key.startswith("mtp."):
                        linear, s = linears[short]
                        if id(linear) not in deltas:  # one product per stacked Linear
                            deltas[id(linear)] = np.asarray(jax.device_get(linear.delta()))
                        # W + s·AB on the original HF weight, in fp32, stored as bf16
                        value = (value.astype(np.float32) + deltas[id(linear)][s].T).astype(
                            ml_dtypes.bfloat16
                        )
                        replaced += 1
                    tensors[key] = value
            save_file(tensors, str(out / file.name), metadata={"format": "pt"})
        for path in src.iterdir():
            if path.is_file() and not path.name.endswith(".safetensors"):
                shutil.copy2(path, out / path.name)
        if replaced != len(linears):
            raise RuntimeError(f"merged {replaced} of {len(linears)} adapted weights")
        result["merged_tensors"] = replaced
    if reference:  # after the bf16 model is released: the reference holds an fp32 copy
        run_dir, step = run.dir, run.step
        del run, linears, deltas
        write_reference(run_dir, step, out / "reference.json", jevbench=jevbench)
    return result
