"""No ``lax.cond`` in ``env.step`` may take a lookup table as an operand.

Under ``jax.vmap``, a cond whose predicate differs between the envs runs both branches,
and JAX first copies each operand of the cond once per env. A branch that reads a
lookup table makes the table an operand. When the ``round_end="auto"`` round advance
dealt the next round inside a cond, the 22 MiB yaku table became 5.6 GiB at every step
of 256 envs, and east and half games ran 4.6x slower than single games. Compute such
a branch before the cond, and let the cond pick its result.
"""

import jax
import jax.numpy as jnp
import pytest

import mahjax
from mahjax.common.shanten import Shanten
from mahjax.common.yaku import Yaku

# The hand table (0.2 MiB) is left out: XLA folds its copies into the gathers that read them.
TABLES = {
    "yaku": Yaku.CACHE.shape,
    "shanten": Shanten.CACHE.shape,
    "flat shanten": Shanten.CACHE_FLAT.shape,
}


def _cond_operand_shapes(jaxpr):
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "cond":
            for var in eqn.invars:
                yield var.aval.shape
        for param in eqn.params.values():
            for sub in param if isinstance(param, (list, tuple)) else (param,):
                sub = getattr(sub, "jaxpr", sub)
                if hasattr(sub, "eqns"):
                    yield from _cond_operand_shapes(sub)


@pytest.mark.parametrize(
    "env_id, round_end",
    [("red_4p", "auto"), ("red_4p", "confirm"), ("red_3p", "auto")],
)
def test_no_cond_in_step_takes_a_lookup_table(env_id: str, round_end: str) -> None:
    env = mahjax.make(env_id, round_mode="half", round_end=round_end)
    key = jax.random.PRNGKey(0)
    state = jax.eval_shape(env.init, key)
    jaxpr = jax.make_jaxpr(env.step)(state, jnp.int32(0), key)
    shapes = set(_cond_operand_shapes(jaxpr.jaxpr))
    taken = sorted(name for name, table in TABLES.items() if table in shapes)
    assert not taken, f"a cond in env.step takes the {', '.join(taken)} table, which vmap copies once per env"
