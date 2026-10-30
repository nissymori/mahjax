"""State builders and constants shared by the red tests."""

import functools

import jax
import jax.numpy as jnp

from mahjax.common.hand import Hand
from mahjax.mahjong_4p.config import RED
from mahjax.mahjong_4p.env import Mahjong4p
from mahjax.mahjong_4p.logic import _init, _replace_state


STEP_KEY = jax.random.PRNGKey(0)  # wall key for round transitions


class _CompiledEnv:
    """An env whose ``init`` and ``step`` are jitted; every other attribute is the env's."""

    def __init__(self, env):
        self._env = env
        self.init = jax.jit(env.init)
        self.step = jax.jit(env.step)

    def __getattr__(self, name):
        return getattr(self._env, name)


@functools.lru_cache(maxsize=None)
def _compiled_env(round_mode, round_end, order_points, game_config):
    return _CompiledEnv(
        Mahjong4p(
            round_mode=round_mode,
            round_end=round_end,
            order_points=list(order_points),
            game_config=game_config,
        )
    )


def compiled_env(round_mode="half", round_end="auto", order_points=(0, 0, 0, 0), game_config=None):
    """``Mahjong4p(...)`` with ``init`` and ``step`` jitted. Tests that build the same
    env share one instance, so a configuration compiles once per test process."""
    return _compiled_env(round_mode, round_end, tuple(order_points), RED if game_config is None else game_config)


_RINSHAN_TILE = 12                                   # 4p


# Concealed remainder after ankan-ing 1m out of 1111m 234m 567m 99m 56p.
# +4p makes 234m 567m 99m 456p: discard 9m and it is tenpai on 9m, so riichi
# is legal for wall reasons alone.
_POST_ANKAN_HAND = [1, 2, 3, 4, 5, 6, 8, 8, 13, 14]  # 234m 567m 99m 56p


def _post_ankan_state(wall_left_after_rinshan, *, kan_dora_pre_flipped=False):
    """A state entering ``_draw_rinshan``: the current player has declared a
    closed kan and is about to take the replacement tile.

    ``kan_dora_pre_flipped`` picks the branch where ``_kan`` already revealed
    the kan dora and already advanced ``last_deck_ix``, so ``_draw_rinshan``
    must not advance it a second time.
    """
    state = _init(jax.random.PRNGKey(5))
    c_p = int(state.current_player)
    counts37 = jnp.zeros(37, dtype=state.players.hand_with_red.dtype)
    for t in _POST_ANKAN_HAND:
        counts37 = counts37.at[t].add(1)
    n_kan = state.players.n_kan.sum()                # 0: rinshan is deck[10]
    deck = state.round_state.deck.at[10 + n_kan].set(jnp.int8(_RINSHAN_TILE))
    # In the pre-flipped branch ``_kan`` already advanced the boundary, so the
    # state entering ``_draw_rinshan`` carries it and the call leaves it be.
    last_in = int(state.round_state.last_deck_ix) + int(kan_dora_pre_flipped)
    last_out = last_in if kan_dora_pre_flipped else last_in + 1
    return _replace_state(
        state,
        hand_with_red=state.players.hand_with_red.at[c_p].set(counts37),
        hand=state.players.hand.at[c_p].set(Hand.to_34(counts37)),
        deck=deck,
        n_kan_doras=jnp.int8(1 if kan_dora_pre_flipped else 0),
        last_deck_ix=jnp.int8(last_in),
        # _draw_rinshan leaves next_deck_ix alone, so the live wall left
        # after the rinshan draw is ``next - last_out + 1``.
        next_deck_ix=jnp.int32(last_out + wall_left_after_rinshan - 1),
    ), c_p
