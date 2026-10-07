"""Program names: one jitted copy per shape, named after its role and shape (xprof, dumps)."""

from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tjev.train.step import NamedJit, eval_step_name, jit_eval_step, named, train_step_name


def _batch(*shape):
    return SimpleNamespace(tokens=np.zeros(shape, np.int32))


def test_names_from_the_batch_shape():
    assert train_step_name(_batch(32, 2, 1024)) == "train_step_T1024_mb32"
    assert eval_step_name(_batch(16, 4096)) == "eval_step_T4096_b16"


def test_named_sets_the_module_name_and_keeps_the_outputs():
    def step(x):
        return jnp.sin(x) * 2

    f = jax.jit(named(step, "train_step_T8_mb1"))
    assert "module @jit_train_step_T8_mb1" in f.lower(jnp.ones(3)).as_text()
    assert np.array_equal(f(jnp.ones(3)), jax.jit(step)(jnp.ones(3)))


class _Rows(NamedTuple):  # a pytree batch
    tokens: jax.Array


def test_named_jit_one_program_per_shape():
    def step(lora, frozen, batch):
        return jnp.sum(batch.tokens) + lora

    jitted = jit_eval_step(step)
    assert isinstance(jitted, NamedJit)
    for rows, length in ((2, 8), (1, 16), (2, 8)):
        b = _Rows(jnp.ones((rows, length), jnp.int32))
        assert int(jitted(jnp.int32(1), None, b)) == rows * length + 1
    assert sorted(jitted.jits) == ["eval_step_T16_b1", "eval_step_T8_b2"]
    b = _Rows(jnp.ones((2, 8), jnp.int32))
    assert "module @jit_eval_step_T8_b2" in jitted.lower(jnp.int32(1), None, b).as_text()
