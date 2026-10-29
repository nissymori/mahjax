from __future__ import annotations

import jax
import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.env import _draw_after_kan, _init, _make_legal_action_mask_after_draw, _make_legal_action_mask_after_draw_w_riichi, _replace_state
from mahjax.red_mahjong.hand import Hand
from mahjax.red_mahjong.state import default_state
from mahjax.red_mahjong.tile import Tile

from ..helpers import _post_ankan_state


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


def test_rinshan_riichi_gate_matches_the_live_wall_after_the_kan():
    for pre_flipped in (False, True):
        for wall_left in range(1, 8):
            state, c_p = _post_ankan_state(wall_left, kan_dora_pre_flipped=pre_flipped)
            out = _draw_after_kan(state)
            remaining = (
                int(out.round_state.next_deck_ix)
                - int(out.round_state.last_deck_ix)
                + 1
            )
            assert remaining == wall_left, (pre_flipped, wall_left, remaining)
            riichi = bool(out.players.legal_action_mask[c_p, Action.RIICHI])
            assert riichi == (wall_left >= 4), (pre_flipped, wall_left, riichi)


def test_rinshan_riichi_also_needs_1000_points():
    state, c_p = _post_ankan_state(6)
    poor = _replace_state(state, score=state.round_state.score.at[c_p].set(9))
    assert not bool(_draw_after_kan(poor).players.legal_action_mask[c_p, Action.RIICHI])
    exact = _replace_state(state, score=state.round_state.score.at[c_p].set(10))
    assert bool(_draw_after_kan(exact).players.legal_action_mask[c_p, Action.RIICHI])


def _riichi_draw_state(concealed: list[int], drawn: int):
    hand_with_red = jnp.zeros((37,), dtype=jnp.int8)
    for tile in concealed:
        hand_with_red = hand_with_red.at[tile].add(1)
    can_win = jax.vmap(Hand.can_ron, in_axes=(None, 0))(
        Hand.to_34(hand_with_red), jnp.arange(34)
    )
    hand_with_red = hand_with_red.at[drawn].add(1)
    state = default_state()
    state = state.replace(
        current_player=jnp.int8(0),
        players=state.players.replace(
            hand_with_red=state.players.hand_with_red.at[0].set(hand_with_red),
            hand=state.players.hand.at[0].set(Hand.to_34(hand_with_red)),
            riichi=state.players.riichi.at[0].set(True),
            can_win=state.players.can_win.at[0].set(can_win),
        ),
    )
    return state, state.players.hand_with_red


def test_riichi_closed_kan_is_offered_only_for_the_drawn_tile() -> None:
    # 1111m 23m 456p 789p 9s waiting on 9s, then draws 4m: the 1m quad was
    # already complete at the riichi declaration, so it cannot be kanned.
    state, hand = _riichi_draw_state([0, 0, 0, 0, 1, 2, 12, 13, 14, 15, 16, 17, 26], 3)

    mask = _make_legal_action_mask_after_draw_w_riichi(state, hand, jnp.int8(0), jnp.int8(3))

    assert not bool(mask[Tile.NUM_TILE_TYPE_WITH_RED : Action.TSUMOGIRI].any())


def test_riichi_closed_kan_is_offered_for_the_drawn_tile() -> None:
    # 111m 234m 456p 789p 9s waiting on 9s, then draws the 4th 1m.
    state, hand = _riichi_draw_state([0, 0, 0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 26], 0)

    mask = _make_legal_action_mask_after_draw_w_riichi(state, hand, jnp.int8(0), jnp.int8(0))

    kan_mask = mask[Tile.NUM_TILE_TYPE_WITH_RED : Action.TSUMOGIRI]
    assert bool(kan_mask[0])
    assert int(kan_mask.sum()) == 1


def test_riichi_closed_kan_is_not_offered_once_four_kans_are_on_the_table() -> None:
    # The same hand and draw as above; riichi does not lift the four-kan limit.
    state, hand = _riichi_draw_state([0, 0, 0, 1, 2, 3, 12, 13, 14, 15, 16, 17, 26], 0)

    for kans, offered in (([0, 3, 0, 0], True), ([0, 4, 0, 0], False)):
        at_kans = state.replace(players=state.players.replace(n_kan=jnp.array(kans, dtype=jnp.int8)))

        mask = _make_legal_action_mask_after_draw_w_riichi(at_kans, hand, jnp.int8(0), jnp.int8(0))

        assert bool(mask[Tile.NUM_TILE_TYPE_WITH_RED + 0]) == offered
