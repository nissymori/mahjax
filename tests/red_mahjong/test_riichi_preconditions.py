"""Riichi preconditions in the after-draw mask: the 1000-point stick and the
4-live-draws rule (Tenhou rules; every mjai-compatible engine enforces both).

Why these are pinned: agents trained on an env that allows a sub-1000-point
or a 3-tiles-left riichi actually learn to declare them, and outside this env
that declaration is an illegal move (chombo). Both bugs were found by playing
mahjax-trained agents against libriichi-driven referees (2026-08-02): the
wall rule was counted with the between-turns formula ``next - last + 1`` at a
call site where ``next_deck_ix`` is not yet decremented, over-counting the
already-drawn tile by one.
"""
import jax
import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.env import (
    _init,
    _make_legal_action_mask_after_draw,
    _replace_state,
)
from mahjax.red_mahjong.hand import Hand


def _riichi_ready_state(**round_overrides):
    """A state whose current player holds a 14-tile closed hand that can
    declare riichi (123456789m 123p 55p), with round-state fields overridable
    per test. The hand is what makes ``Hand.can_riichi`` true; every test
    below then toggles exactly one precondition around it."""
    state = _init(jax.random.PRNGKey(5))
    c_p = int(state.current_player)
    counts37 = jnp.zeros(37, dtype=state.players.hand_with_red.dtype)
    for t in list(range(9)) + [9, 10, 11]:          # 1-9m, 123p
        counts37 = counts37.at[t].set(1)
    counts37 = counts37.at[13].set(2)               # 55p (plain)
    hand_with_red = state.players.hand_with_red.at[c_p].set(counts37)
    state = _replace_state(
        state,
        hand_with_red=hand_with_red,
        hand=state.players.hand.at[c_p].set(Hand.to_34(counts37)),
        **round_overrides,
    )
    return state, c_p, hand_with_red


def _mask(state, c_p, hand, **kw):
    return _make_legal_action_mask_after_draw(
        state, hand, jnp.int32(c_p), jnp.int8(13), **kw)


def test_the_crafted_hand_can_riichi_at_all():
    state, c_p, hand = _riichi_ready_state()
    assert bool(_mask(state, c_p, hand)[Action.RIICHI])


def test_riichi_needs_1000_points():
    state, c_p, hand = _riichi_ready_state()
    poor = _replace_state(state, score=state.round_state.score.at[c_p].set(9))
    assert not bool(_mask(poor, c_p, hand)[Action.RIICHI])
    exact = _replace_state(state, score=state.round_state.score.at[c_p].set(10))
    assert bool(_mask(exact, c_p, hand)[Action.RIICHI])


def test_riichi_needs_four_live_draws_after_the_draw():
    """The mask is built with the pre-decrement next_deck_ix, so the live
    draws left after the current draw is ``next - last`` exactly (verified
    against libriichi's tiles-left on a real game)."""
    state, c_p, hand = _riichi_ready_state()
    last = int(state.round_state.last_deck_ix)
    three_left = _replace_state(state, next_deck_ix=jnp.int32(last + 3))
    assert not bool(_mask(three_left, c_p, hand)[Action.RIICHI])
    four_left = _replace_state(state, next_deck_ix=jnp.int32(last + 4))
    assert bool(_mask(four_left, c_p, hand)[Action.RIICHI])


def test_explicit_live_draws_left_overrides_the_derivation():
    """The rinshan call site passes its own count (dead-wall draw: next is
    untouched, last already advanced -> between-turns formula)."""
    state, c_p, hand = _riichi_ready_state()
    assert not bool(_mask(state, c_p, hand,
                          live_draws_left=jnp.int32(3))[Action.RIICHI])
    assert bool(_mask(state, c_p, hand,
                      live_draws_left=jnp.int32(4))[Action.RIICHI])
