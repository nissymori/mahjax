import jax
import jax.numpy as jnp
import pytest

from mahjax.common.yaku import Yaku
from mahjax.mahjong_4p.env import Mahjong4p


def _count_calls(jaxpr, name: str) -> int:
    count = 0
    for eqn in jaxpr.eqns:
        if eqn.params.get("name") == name:
            count += 1
            continue
        for param in eqn.params.values():
            for sub in param if isinstance(param, (list, tuple)) else (param,):
                sub = getattr(sub, "jaxpr", sub)
                if hasattr(sub, "eqns"):
                    count += _count_calls(sub, name)
    return count


@pytest.mark.parametrize("round_end", ["auto", "confirm"])
def test_a_step_calls_yaku_judge_four_times(monkeypatch, round_end: str) -> None:
    """A step computes every branch before picking one, so each ``Yaku.judge`` in any
    branch is computed on every step. The discard and the kan each judge every player's
    ron on the tile and tsumo on the next draw (one call each, batched over players),
    and the next round's deal judges the dealer's first draw twice: after a special
    abortive draw and after any other round end (under ``auto`` the latter is the
    round advance in ``env.step``)."""
    env = Mahjong4p(round_end=round_end)
    state = env.init(jax.random.PRNGKey(0))
    judge = Yaku.judge

    def counted_judge(*args, **kwargs):
        return judge(*args, **kwargs)

    monkeypatch.setattr(Yaku, "judge", staticmethod(jax.jit(counted_judge)))
    # The yaku look-ahead is jitted on its own: drop traces made with the real judge.
    jax.clear_caches()
    try:
        step = jax.make_jaxpr(env.step)(state, jnp.int32(0), jax.random.PRNGKey(1))
    finally:
        jax.clear_caches()

    assert _count_calls(step.jaxpr, "counted_judge") == 4
