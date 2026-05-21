import jax
import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.constants import FIRST_DRAW_IDX
from mahjax.red_mahjong.env import (
    RedMahjong,
    _abortive_draw_normal,
    _draw,
    _init,
    _make_legal_action_mask_after_draw,
    _next_meld_player,
    _next_ron_player,
    _replace_state,
    _step,
)
from mahjax.red_mahjong.state import default_state
from mahjax.red_mahjong.tile import Tile


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
    next_state = _step(state, jnp.int8(Action.TSUMOGIRI))

    assert int(next_state.round_state.action_history[0, 0]) == int(state.current_player)
    assert int(next_state.round_state.action_history[1, 0]) == last_draw
    assert int(next_state.round_state.action_history[2, 0]) == 1


def test_next_meld_player_prioritizes_ron_then_distance() -> None:
    legal = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    legal = legal.at[0, Action.PON].set(True)
    legal = legal.at[1, Action.RON].set(True)
    legal = legal.at[3, Action.RON].set(True)
    nxt, can_any = _next_meld_player(legal, jnp.int8(0))
    assert bool(can_any)
    assert int(nxt) == 1


def test_next_ron_player_returns_closest() -> None:
    legal = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    legal = legal.at[1, Action.RON].set(True)
    legal = legal.at[2, Action.RON].set(True)
    nxt, can_any = _next_ron_player(legal, jnp.int8(0))
    assert bool(can_any)
    assert int(nxt) == 1


def test_abortive_draw_payments_shape() -> None:
    state = default_state()
    state = state.replace(
        players=state.players.replace(
            can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[0].set(True)
        )
    )
    next_state = _abortive_draw_normal(state)
    assert bool(next_state.round_state.terminated_round)
    assert next_state.rewards.shape == (4,)


# ----------------- next_round_style tests -----------------


def _ron_legal_mask(ron_player: int = 0):
    return (
        jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        .at[ron_player, Action.RON].set(True)
    )


def test_red_default_is_auto() -> None:
    e = RedMahjong()
    assert e.next_round_style == "auto"


def test_red_invalid_style_raises() -> None:
    import pytest as _pytest
    with _pytest.raises(ValueError):
        RedMahjong(next_round_style="bogus")  # type: ignore[arg-type]


def test_red_auto_ron_advances_to_next_round_in_one_step() -> None:
    env_auto = RedMahjong(round_mode="half", next_round_style="auto")
    state = env_auto.init(jax.random.PRNGKey(7))
    state = _replace_state(
        state,
        legal_action_mask=_ron_legal_mask(0),
        current_player=jnp.int8(0),
    )
    next_state = env_auto.step(state, jnp.int32(Action.RON))
    assert not bool(next_state.terminated)
    assert not bool(next_state.round_state.terminated_round)
    assert int(next_state.round_state.dummy_count) == 0
    assert int(next_state.current_player) == int(next_state.round_state.dealer)
    # Legal action mask is NOT DUMMY-only.
    only_dummy = (
        bool(next_state.legal_action_mask[Action.DUMMY])
        and int(next_state.legal_action_mask.sum()) == 1
    )
    assert not only_dummy


def test_red_dummy_share_ron_keeps_dummy_phase() -> None:
    env_share = RedMahjong(round_mode="half", next_round_style="dummy_share")
    state = env_share.init(jax.random.PRNGKey(7))
    state = _replace_state(
        state,
        legal_action_mask=_ron_legal_mask(0),
        current_player=jnp.int8(0),
    )
    next_state = env_share.step(state, jnp.int32(Action.RON))
    assert not bool(next_state.terminated)
    assert bool(next_state.round_state.terminated_round)
    assert int(next_state.round_state.dummy_count) == 0
    # Only DUMMY is legal for every seat.
    assert bool(next_state.players.legal_action_mask[:, Action.DUMMY].all())


def test_red_auto_single_mode_terminates_like_legacy() -> None:
    env_auto = RedMahjong(round_mode="single", next_round_style="auto")
    state = env_auto.init(jax.random.PRNGKey(11))
    state = _replace_state(
        state,
        legal_action_mask=_ron_legal_mask(0),
        current_player=jnp.int8(0),
    )
    next_state = env_auto.step(state, jnp.int32(Action.RON))
    assert bool(next_state.terminated)


def test_red_auto_game_end_sets_terminated_with_final_score() -> None:
    env_auto = RedMahjong(round_mode="half", next_round_style="auto")
    state = env_auto.init(jax.random.PRNGKey(3))
    state = _replace_state(
        state,
        legal_action_mask=_ron_legal_mask(0),
        current_player=jnp.int8(0),
        dealer=jnp.int8(0),
        score=jnp.array([310, 310, 190, 190], dtype=jnp.int32),
        init_wind=jnp.array([0, 1, 2, 3], dtype=jnp.int8),
        round=jnp.int8(7),
        round_limit=jnp.int8(7),
        kyotaku=jnp.int8(3),
        has_won=jnp.array([False, False, False, False], dtype=jnp.bool_),
    )
    next_state = env_auto.step(state, jnp.int32(Action.RON))
    assert bool(next_state.terminated)
    expected = jnp.array([370, 320, 180, 160], dtype=jnp.int32)
    assert bool(jnp.all(next_state.round_state.score == expected)), (
        f"got {next_state.round_state.score}, expected {expected}"
    )


# ---------------- parity: auto vs dummy_share ----------------
#
# These tests assert that ``auto`` mode collapses the dummy_share rotation
# phase into a single env.step while producing the same end state.
# ``mahjax_tenhou_test`` validates the dummy_share trajectory against real
# tenhou mjlogs; this parity test bridges that validation across to ``auto``.


def _force_ron_state(env: RedMahjong, key, ron_player: int = 0):
    state = env.init(key)
    return _replace_state(
        state,
        legal_action_mask=_ron_legal_mask(ron_player),
        current_player=jnp.int8(ron_player),
    )


def test_red_auto_matches_dummy_share_at_mid_game_round_transition() -> None:
    """auto's 1-step round transition == dummy_share's 5-step (RON + 4 DUMMY)
    transition, modulo:
    - ``step_count`` (auto +1, share +5)
    - ``rewards`` (auto preserves the round-end vector from the RON step;
      dummy_share resets it to zero in the ``_make_state``-based init).
    """
    env_auto = RedMahjong(round_mode="half", next_round_style="auto")
    env_share = RedMahjong(round_mode="half", next_round_style="dummy_share")
    key = jax.random.PRNGKey(2026)

    state_auto = env_auto.step(_force_ron_state(env_auto, key), jnp.int32(Action.RON))
    state_share = env_share.step(_force_ron_state(env_share, key), jnp.int32(Action.RON))
    rewards_at_ron = state_share.rewards  # round-end reward delivered at RON step in dummy_share
    for _ in range(4):
        state_share = env_share.step(state_share, jnp.int32(Action.DUMMY))

    # Both must land in the next-round init: mid-game ⇒ not terminated, not terminated_round.
    assert not bool(state_auto.terminated)
    assert not bool(state_share.terminated)
    assert not bool(state_auto.round_state.terminated_round)
    assert not bool(state_share.round_state.terminated_round)
    assert int(state_auto.round_state.dummy_count) == 0
    assert int(state_share.round_state.dummy_count) == 0

    rs_a, rs_s = state_auto.round_state, state_share.round_state
    assert int(state_auto.current_player) == int(state_share.current_player)
    assert int(rs_a.dealer) == int(rs_s.dealer)
    assert int(rs_a.round) == int(rs_s.round)
    assert int(rs_a.honba) == int(rs_s.honba)
    assert int(rs_a.kyotaku) == int(rs_s.kyotaku)
    assert bool(jnp.all(rs_a.score == rs_s.score))
    assert bool(jnp.all(rs_a.deck == rs_s.deck))
    assert bool(jnp.all(rs_a.dora_indicators == rs_s.dora_indicators))
    assert int(rs_a.next_deck_ix) == int(rs_s.next_deck_ix)
    assert int(rs_a.last_draw) == int(rs_s.last_draw)

    ps_a, ps_s = state_auto.players, state_share.players
    assert bool(jnp.all(ps_a.hand == ps_s.hand))
    assert bool(jnp.all(ps_a.has_won == ps_s.has_won))
    assert bool(jnp.all(ps_a.legal_action_mask == ps_s.legal_action_mask))
    assert bool(jnp.all(state_auto.legal_action_mask == state_share.legal_action_mask))

    # auto preserves the round-end rewards; dummy_share's were delivered at the
    # RON step (captured in ``rewards_at_ron``) and zeroed afterwards.
    assert bool(jnp.all(state_auto.rewards == rewards_at_ron))


def test_red_auto_matches_dummy_share_at_game_end() -> None:
    """When RON ends the game, auto terminates after the RON step; dummy_share
    terminates one step later (DUMMY 1 detects ``_is_game_end`` at ``dc==0``).
    Compare the two terminal states: same ``terminated``, same final ``score``,
    same ``rewards``.
    """
    env_auto = RedMahjong(round_mode="half", next_round_style="auto")
    env_share = RedMahjong(round_mode="half", next_round_style="dummy_share")
    key = jax.random.PRNGKey(2026)
    forced = dict(
        dealer=jnp.int8(0),
        score=jnp.array([310, 310, 190, 190], dtype=jnp.int32),
        init_wind=jnp.array([0, 1, 2, 3], dtype=jnp.int8),
        round=jnp.int8(7),
        round_limit=jnp.int8(7),
        kyotaku=jnp.int8(3),
        has_won=jnp.array([False, False, False, False], dtype=jnp.bool_),
    )

    state_auto = _replace_state(_force_ron_state(env_auto, key), **forced)
    state_share = _replace_state(_force_ron_state(env_share, key), **forced)

    state_auto = env_auto.step(state_auto, jnp.int32(Action.RON))
    state_share = env_share.step(state_share, jnp.int32(Action.RON))
    state_share = env_share.step(state_share, jnp.int32(Action.DUMMY))

    assert bool(state_auto.terminated)
    assert bool(state_share.terminated)
    assert bool(jnp.all(state_auto.round_state.score == state_share.round_state.score))
    assert bool(jnp.all(state_auto.rewards == state_share.rewards))
