import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.env import _next_meld_player
from mahjax.red_mahjong.state import GameConfig, default_state


def test_double_ron_priority_is_left_to_discarder() -> None:
    legal = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    legal = legal.at[0, Action.RON].set(True)
    legal = legal.at[2, Action.RON].set(True)
    nxt, can_any = _next_meld_player(legal, jnp.int8(1))
    assert bool(can_any)
    assert int(nxt) == 2


def test_game_config_flags_default_to_expected_values() -> None:
    cfg = GameConfig()
    assert bool(cfg.allow_double_ron)
    assert bool(cfg.enable_special_abortive_draw)
    assert bool(cfg.enable_pao)


def test_default_state_round_metadata_types() -> None:
    state = default_state()
    assert state.round_state.score.dtype == jnp.int32
    assert state.round_state.honba.dtype == jnp.int8
    assert state.round_state.kyotaku.dtype == jnp.int8
