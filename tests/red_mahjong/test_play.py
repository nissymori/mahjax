import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.state import default_state


def test_default_state_action_mask_shape() -> None:
    state = default_state()
    assert state.legal_action_mask.shape == (Action.NUM_ACTION,)
    assert state.players.legal_action_mask.shape == (4, Action.NUM_ACTION)


def test_default_state_player_arrays_shape() -> None:
    state = default_state()
    assert state.players.hand.shape == (4, 34)
    assert state.players.hand_with_red.shape == (4, 37)
    assert state.players.melds.shape == (4, 4)
    assert state.players.river.shape == (4, 24)
    assert state.rewards.shape == (4,)
