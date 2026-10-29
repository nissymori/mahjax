from __future__ import annotations

import jax
import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.config import RED
from mahjax.red_mahjong.constants import FIRST_DRAW_IDX
from mahjax.red_mahjong.env import RedMahjong, _draw, _init, _make_legal_action_mask_after_draw, _step
from mahjax.red_mahjong.state import default_state
from mahjax.red_mahjong.tile import Tile

from ..helpers import STEP_KEY


def test_init_shape_and_first_draw_position() -> None:
    state = _init(jax.random.PRNGKey(1))
    assert bool(jnp.all(state.rewards == 0))
    assert not bool(state.terminated)
    assert not bool(state.truncated)
    assert int(state.round_state.next_deck_ix) == FIRST_DRAW_IDX - 1
    assert state.round_state.deck.shape == (136,)
    assert state.players.hand.shape == (4, Tile.NUM_TILE_TYPE)


def test_draw_decrements_deck_and_sets_last_draw() -> None:
    state = _init(jax.random.PRNGKey(2))
    before_ix = int(state.round_state.next_deck_ix)
    state = _draw(state)
    assert int(state.round_state.next_deck_ix) == before_ix - 1
    assert int(state.round_state.last_draw) >= 0


def test_mask_after_draw_has_discard_or_tsumogiri() -> None:
    state = _init(jax.random.PRNGKey(3))
    c_p = int(state.current_player)
    hand = state.players.hand_with_red
    new_tile = int(state.round_state.last_draw)
    mask = _make_legal_action_mask_after_draw(state, hand, c_p, new_tile)
    assert bool(mask[Action.TSUMOGIRI] | jnp.any(mask[: Tile.NUM_TILE_TYPE_WITH_RED]))


def test_tsumogiri_action_history_records_actual_tile_and_flag() -> None:
    state = _init(jax.random.PRNGKey(4))
    last_draw = int(state.round_state.last_draw)
    next_state = _step(state, jnp.int8(Action.TSUMOGIRI), STEP_KEY)

    assert int(next_state.round_state.action_history[0, 0]) == int(state.current_player)
    assert int(next_state.round_state.action_history[1, 0]) == last_draw
    assert int(next_state.round_state.action_history[2, 0]) == 1


def test_red_default_is_auto() -> None:
    e = RedMahjong()
    assert e.next_round_style == "auto"


def test_red_invalid_style_raises() -> None:
    import pytest as _pytest
    with _pytest.raises(ValueError):
        RedMahjong(next_round_style="bogus")  # type: ignore[arg-type]


def test_public_state_layout_and_defaults_for_red_env() -> None:
    state = default_state()
    config = RED

    assert state.players.hand.shape == (4, 34)
    assert state.players.hand_with_red.shape == (4, 37)
    assert state.players.legal_action_mask.shape == (4, Action.NUM_ACTION)
    assert state.legal_action_mask.shape == (Action.NUM_ACTION,)
    assert state.players.melds.shape == (4, 4)
    assert state.players.river.shape == (4, 24)
    assert state.round_state.action_history.shape == (3, 200)
    assert state.rewards.shape == (4,)
    assert bool(config.allow_double_ron)
    assert bool(config.enable_special_abortive_draw)
    assert bool(config.enable_pao)


def test_default_state_round_metadata_types() -> None:
    state = default_state()
    assert state.round_state.score.dtype == jnp.int32
    assert state.round_state.honba.dtype == jnp.int32  # no renchan limit, so no int8 wrap at 127
    assert state.round_state.kyotaku.dtype == jnp.int8
