"""The wall for every round after the first comes from the key given to ``step``.

The state carries no rng of its own. Before this was wired up, ``RoundState``
held an ``rng_key`` that ``_init`` never seeded and ``step`` never consumed, so
every wall after the first was one global sequence shared by all seeds, envs
and runs -- the second hand's deck, deals and dora were constants of the
library (#71).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mahjax

ENV_NAMES = ["red_mahjong", "no_red_mahjong"]
STYLES = ["auto", "dummy_share"]


def _make(env_name, style):
    return mahjax.make(env_name, round_mode="half", next_round_style=style)


def _walls(env, init_seed, step_seed, n_walls=3, max_steps=4000):
    """Play random legal actions and collect each distinct wall.

    A round transition is exactly when the deck array changes, which works
    under both ``next_round_style`` values (``auto`` completes the transition
    inside one step, so ``terminated_round`` is never observable between steps).
    """
    state = env.init(jax.random.PRNGKey(init_seed))
    step = jax.jit(env.step)
    walls = [np.asarray(state.round_state.deck).copy()]
    rng = jax.random.PRNGKey(step_seed)
    for _ in range(max_steps):
        if len(walls) >= n_walls or bool(state.terminated | state.truncated):
            break
        rng, k_act, k_step = jax.random.split(rng, 3)
        logits = jnp.where(state.legal_action_mask, 0.0, -jnp.inf)
        action = jax.random.categorical(k_act, logits).astype(jnp.int32)
        state = step(state, action, k_step)
        deck = np.asarray(state.round_state.deck)
        if not np.array_equal(deck, walls[-1]):
            walls.append(deck.copy())
    return walls


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_state_carries_no_rng(env_name):
    state = _make(env_name, "auto").init(jax.random.PRNGKey(0))
    assert not hasattr(state.round_state, "rng_key")


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_step_without_a_key_is_rejected(env_name):
    env = _make(env_name, "auto")
    state = env.init(jax.random.PRNGKey(0))
    action = int(jnp.argmax(state.legal_action_mask))
    with pytest.raises(ValueError, match="requires a PRNG key"):
        env.step(state, jnp.int32(action))


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_later_walls_follow_the_step_keys(env_name, style):
    """Same init key, different step keys -> the first wall is shared (it is
    dealt by ``init``) and every later wall differs. This is the bug: it used
    to be the other way round, with later walls identical across everything."""
    env = _make(env_name, style)
    a = _walls(env, init_seed=0, step_seed=1)
    b = _walls(env, init_seed=0, step_seed=2)
    assert len(a) >= 3 and len(b) >= 3, "no round transition within the step budget"
    np.testing.assert_array_equal(a[0], b[0])
    for i in range(1, 3):
        assert not np.array_equal(a[i], b[i]), f"wall {i} does not follow the step key"


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_walls_are_reproducible_from_the_same_keys(env_name, style):
    env = _make(env_name, style)
    a = _walls(env, init_seed=0, step_seed=1)
    b = _walls(env, init_seed=0, step_seed=1)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
