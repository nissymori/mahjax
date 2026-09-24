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
    assert state.round_state.honba.dtype == jnp.int32  # no renchan limit, so no int8 wrap at 127
    assert state.round_state.kyotaku.dtype == jnp.int8


def test_dealer_can_tsumo_a_complete_first_draw_as_a_single_blessing_of_heaven() -> None:
    """123m 33m 444p 666p 77s drawing 7s: TSUMO is legal, and three concealed pons are not four."""
    from unittest import mock

    import jax

    from mahjax.red_mahjong import env as m

    dealt = [0, 4, 8, 9, 10, 48, 49, 50, 56, 57, 58, 96, 97]  # tile ids; the tile is id // 4
    first_draw = 98  # 7s
    ids = [None] * 136
    ids[83] = first_draw  # the dealer's first draw
    ids[-52:-39] = dealt  # player 0's starting hand
    rest = iter(i for i in range(136) if i not in dealt and i != first_draw)
    ids = jnp.array([i if i is not None else next(rest) for i in ids])

    with mock.patch.object(jax.random, "permutation", lambda key, x: ids):
        state = m._init_for_next_round(jax.random.PRNGKey(0), default_state())

    assert int(state.current_player) == 0
    assert bool(state.legal_action_mask[Action.TSUMO])
    assert jnp.allclose(m._tsumo(state).rewards, jnp.array([480.0, -160.0, -160.0, -160.0]))
