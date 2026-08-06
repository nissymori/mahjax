"""Regression tests: the wall-reshuffle chain must be seeded by the init key.

Bug: ``_init`` split a ``subkey`` off the init rng but never stored it, so
``round_state.rng_key`` stayed at the State-default ``PRNGKey(0)``. Since
``step()`` ignores its ``key`` argument and every reshuffle derives from
``round_state.rng_key`` alone, every wall after the first was drawn from one
global chain shared by all seeds, envs, and runs: the second hand's deck,
deals, and dora indicator were literal constants of the library.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import mahjax


@pytest.mark.parametrize("env_name", ["red_mahjong", "no_red_mahjong"])
def test_init_seeds_rng_key(env_name):
    env = mahjax.make(env_name, round_mode="half",
                      next_round_style="dummy_share")
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    r1 = np.asarray(env.init(k1).round_state.rng_key)
    r2 = np.asarray(env.init(k2).round_state.rng_key)
    assert not np.array_equal(r1, np.asarray(jax.random.PRNGKey(0))), \
        "rng_key still holds the State default PRNGKey(0)"
    assert not np.array_equal(r1, r2), "rng_key does not vary with the init key"
    # Determinism is preserved: same init key, same rng_key.
    np.testing.assert_array_equal(
        r1, np.asarray(env.init(k1).round_state.rng_key))


def _second_wall(env, seed, max_steps=800):
    """Play random legal actions until the first round transition; return
    (first deck, second deck)."""
    state = env.init(jax.random.PRNGKey(seed))
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed + 10_000)
    first = np.asarray(state.round_state.deck).copy()
    prev_term = False
    for _ in range(max_steps):
        rng, k1, k2 = jax.random.split(rng, 3)
        logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
        act = jax.random.categorical(k1, logits).astype(jnp.int32)
        state = step(state, act, k2)
        term = bool(state.round_state.terminated_round)
        if prev_term and not term:
            return first, np.asarray(state.round_state.deck).copy()
        prev_term = term
    pytest.fail("no round transition within the step budget")


def test_second_wall_differs_across_seeds():
    """The bug itself: before the fix, the second wall was bitwise identical
    across different init seeds. A collision between two random permutations
    of 136 tiles has probability ~0, so strict inequality is safe."""
    env = mahjax.make("red_mahjong", round_mode="half",
                      next_round_style="dummy_share")
    f1, w1 = _second_wall(env, seed=0)
    f2, w2 = _second_wall(env, seed=424242)
    assert not np.array_equal(f1, f2)
    assert not np.array_equal(w1, w2), \
        "second wall is identical across seeds -- reshuffle chain unseeded"
