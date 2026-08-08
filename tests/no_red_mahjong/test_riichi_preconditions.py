"""Riichi preconditions in the after-draw mask: the 1000-point stick and the
4-live-draws rule (Tenhou rules; every mjai-compatible engine enforces both).

Mirror of ``tests/red_mahjong/test_riichi_preconditions.py`` -- the two envs
are hand-maintained copies of each other and this rule is exactly where they
drifted apart (red gated on ``next < last + 3``, no_red on ``+ 4``), so the
two files are kept diffable on purpose. The only structural difference is
that ``no_red_mahjong._draw_after_kan`` always reveals the kan dora itself,
so it has no ``kan_dora_pre_flipped`` branch to cover.
"""
import jax
import jax.numpy as jnp

from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.env import (
    _draw_after_kan,
    _init,
    _make_legal_action_mask_after_draw,
    _replace_state,
)


def _riichi_ready_state(**round_overrides):
    """A state whose current player holds a 14-tile closed hand that can
    declare riichi (123456789m 123p 55p), with round-state fields overridable
    per test. The hand is what makes ``Hand.can_riichi`` true; every test
    below then toggles exactly one precondition around it."""
    state = _init(jax.random.PRNGKey(5))
    c_p = int(state.current_player)
    counts34 = jnp.zeros(34, dtype=state.players.hand.dtype)
    for t in list(range(9)) + [9, 10, 11]:           # 1-9m, 123p
        counts34 = counts34.at[t].set(1)
    counts34 = counts34.at[13].set(2)                # 55p
    hand = state.players.hand.at[c_p].set(counts34)
    state = _replace_state(state, hand=hand, **round_overrides)
    return state, c_p, hand


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
    draws left after the current draw is ``next - last`` exactly."""
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


# --- the rinshan call site, end to end ------------------------------------
#
# The tests above pin the mask function in isolation, so they stay green even
# if ``_draw_after_kan`` hands it the wrong count. These pin the wiring: the
# replacement tile comes off the dead wall, so ``next_deck_ix`` is untouched
# while ``last_deck_ix`` advances by one (王牌繰り), and the live wall left
# after the draw is the between-turns count read off the RESULT state -- the
# same number ``visualization.remaining_tiles`` shows. Asserting the mask
# against the result state rather than against a formula keeps these honest if
# the index bookkeeping ever moves.

_RINSHAN_TILE = 12                                   # 4p
# Concealed remainder after ankan-ing 1m out of 1111m 234m 567m 99m 56p.
# +4p makes 234m 567m 99m 456p: discard 9m and it is tenpai on 9m, so riichi
# is legal for wall reasons alone.
_POST_ANKAN_HAND = [1, 2, 3, 4, 5, 6, 8, 8, 13, 14]  # 234m 567m 99m 56p


def _post_ankan_state(wall_left_after_rinshan):
    """A state entering ``_draw_after_kan``: the current player has declared a
    closed kan and is about to take the replacement tile."""
    state = _init(jax.random.PRNGKey(5))
    c_p = int(state.current_player)
    counts34 = jnp.zeros(34, dtype=state.players.hand.dtype)
    for t in _POST_ANKAN_HAND:
        counts34 = counts34.at[t].add(1)
    n_kan = state.players.n_kan.sum()                # 0: rinshan is deck[10]
    deck = state.round_state.deck.at[10 + n_kan].set(jnp.int8(_RINSHAN_TILE))
    last_in = int(state.round_state.last_deck_ix)
    last_out = last_in + 1
    return _replace_state(
        state,
        hand=state.players.hand.at[c_p].set(counts34),
        deck=deck,
        last_deck_ix=jnp.int8(last_in),
        # _draw_after_kan leaves next_deck_ix alone, so the live wall left
        # after the rinshan draw is ``next - last_out + 1``.
        next_deck_ix=jnp.int32(last_out + wall_left_after_rinshan - 1),
    ), c_p


def test_draw_after_kan_advances_only_the_dead_wall_boundary():
    """The premise the rinshan count rests on: 王牌繰り moves ``last_deck_ix``
    by exactly one and never touches ``next_deck_ix``."""
    state, _ = _post_ankan_state(6)
    out = _draw_after_kan(state)
    assert int(out.round_state.last_deck_ix) == int(state.round_state.last_deck_ix) + 1
    assert int(out.round_state.next_deck_ix) == int(state.round_state.next_deck_ix)


def test_rinshan_riichi_gate_matches_the_live_wall_after_the_kan():
    for wall_left in range(1, 8):
        state, c_p = _post_ankan_state(wall_left)
        out = _draw_after_kan(state)
        remaining = (
            int(out.round_state.next_deck_ix)
            - int(out.round_state.last_deck_ix)
            + 1
        )
        assert remaining == wall_left, (wall_left, remaining)
        riichi = bool(out.players.legal_action_mask[c_p, Action.RIICHI])
        assert riichi == (wall_left >= 4), (wall_left, riichi)


def test_rinshan_riichi_also_needs_1000_points():
    state, c_p = _post_ankan_state(6)
    poor = _replace_state(state, score=state.round_state.score.at[c_p].set(9))
    assert not bool(_draw_after_kan(poor).players.legal_action_mask[c_p, Action.RIICHI])
    exact = _replace_state(state, score=state.round_state.score.at[c_p].set(10))
    assert bool(_draw_after_kan(exact).players.legal_action_mask[c_p, Action.RIICHI])
