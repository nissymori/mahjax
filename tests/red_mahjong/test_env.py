import jax
import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.constants import FIRST_DRAW_IDX
from mahjax.red_mahjong.env import (
    RedMahjong,
    _abortive_draw_normal,
    _draw,
    _init,
    _kan,
    _make_legal_action_mask_after_draw,
    _mask_for_chi,
    _next_meld_player,
    _next_ron_player,
    _replace_state,
    _step,
)
from mahjax.red_mahjong.hand import Hand
from mahjax.red_mahjong.meld import Meld
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


def test_mask_for_chi_forbids_forced_kuikae() -> None:
    # Regression test for issue #61: a CHI that leaves no legal discard after
    # kuikae (swap-calling) restrictions must not be offered as a legal action,
    # otherwise executing it produces an active state with an all-false mask.
    # Hand 1123444m (with two open melds held elsewhere). Calling 1m or 4m
    # consumes 2m3m and leaves only 1m/4m, all forbidden as discards by kuikae.
    hand = jnp.zeros(Tile.NUM_TILE_TYPE_WITH_RED, dtype=jnp.int8)
    hand = hand.at[0].set(2).at[1].set(1).at[2].set(1).at[3].set(3)  # 1123444m
    chi_slice = slice(Action.CHI_L, Action.CHI_R_RED + 1)
    # Chi 1m: only CHI_L (1m+23m) is possible; every remaining discard is kuikae.
    mask_1m = _mask_for_chi(hand, jnp.int8(0))
    assert not bool(jnp.any(mask_1m[chi_slice]))
    # Chi 4m: only CHI_R (234m) is possible; every remaining discard is kuikae.
    mask_4m = _mask_for_chi(hand, jnp.int8(3))
    assert not bool(jnp.any(mask_4m[chi_slice]))
    # Sanity: with a spare 9m in hand, chi 1m leaves a legal discard and is allowed.
    hand_ok = hand.at[3].set(2).at[8].set(1)  # 1123449m
    mask_ok = _mask_for_chi(hand_ok, jnp.int8(0))
    assert bool(mask_ok[Action.CHI_L])


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


def test_robbing_kan_mask_drops_stale_self_turn_actions() -> None:
    tile_ids = {
        **{f"{n}m": n - 1 for n in range(1, 10)},
        **{f"{n}p": 9 + n - 1 for n in range(1, 10)},
        **{f"{n}s": 18 + n - 1 for n in range(1, 10)},
    }

    def counts37(tiles: list[str]):
        counts = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8)
        for tile in tiles:
            counts = counts.at[tile_ids[tile]].add(1)
        return counts

    actor1_tiles = "3m 4m 5p 5p 6p 7p 8p 6s 6s 6s 7s 8s 9s".split()
    hand_with_red = jnp.zeros((4, Tile.NUM_TILE_TYPE_WITH_RED), dtype=jnp.int8)
    hand_with_red = hand_with_red.at[1].set(counts37(actor1_tiles))
    hand_with_red = hand_with_red.at[2].set(counts37(["2m"]))
    hand = jnp.stack([Hand.to_34(hand_with_red[player]) for player in range(4)])

    assert bool(Hand.can_ron(hand[1], tile_ids["2m"]))

    stale_actor1_actions = [
        tile_ids["3m"],
        tile_ids["4m"],
        tile_ids["5p"],
        tile_ids["6p"],
        tile_ids["7p"],
        tile_ids["8p"],
        tile_ids["6s"],
        tile_ids["7s"],
        tile_ids["8s"],
        tile_ids["9s"],
        Tile.RED_FIVE["p"],
        Action.TSUMOGIRI,
        Action.RIICHI,
    ]
    legal_action_mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    for action_id in stale_actor1_actions:
        legal_action_mask = legal_action_mask.at[1, action_id].set(True)

    action = Tile.NUM_TILE_TYPE_WITH_RED + tile_ids["2m"]
    legal_action_mask = legal_action_mask.at[2, action].set(True)
    pon = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int32).at[2, tile_ids["2m"]].set(1)
    melds = default_state().players.melds.at[2, 1].set(Meld.init(Action.PON, tile_ids["2m"], 1))
    can_win = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[1, tile_ids["2m"]].set(True)

    state = _replace_state(
        default_state(),
        current_player=jnp.int8(2),
        legal_action_mask=legal_action_mask,
        hand=hand,
        hand_with_red=hand_with_red,
        pon=pon,
        melds=melds,
        can_win=can_win,
        last_draw=jnp.int8(tile_ids["2m"]),
        last_player=jnp.int8(2),
        deck=jnp.zeros((136,), dtype=jnp.int8).at[10].set(30),
        dora_indicators=jnp.array([tile_ids["5s"], -1, -1, -1, -1], dtype=jnp.int8),
    )

    kan_state = _kan(state, jnp.int32(action))

    expected = (
        jnp.zeros((Action.NUM_ACTION,), dtype=jnp.bool_)
        .at[Action.RON].set(True)
        .at[Action.PASS].set(True)
    )
    assert int(kan_state.current_player) == 1
    assert bool(kan_state.round_state.kan_declared)
    assert bool(jnp.all(kan_state.players.legal_action_mask[1] == expected))
    assert bool(jnp.all(kan_state.legal_action_mask == expected))


def test_robbing_kan_ron_finalizes_to_dummy_share_mask() -> None:
    tile_ids = {
        **{f"{n}m": n - 1 for n in range(1, 10)},
        **{f"{n}p": 9 + n - 1 for n in range(1, 10)},
        **{f"{n}s": 18 + n - 1 for n in range(1, 10)},
    }

    def counts37(tiles: list[str]):
        counts = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8)
        for tile in tiles:
            counts = counts.at[tile_ids[tile]].add(1)
        return counts

    actor1_tiles = "3m 4m 5p 5p 6p 7p 8p 6s 6s 6s 7s 8s 9s".split()
    hand_with_red = jnp.zeros((4, Tile.NUM_TILE_TYPE_WITH_RED), dtype=jnp.int8)
    hand_with_red = hand_with_red.at[1].set(counts37(actor1_tiles))
    hand_with_red = hand_with_red.at[2].set(counts37(["2m"]))
    hand = jnp.stack([Hand.to_34(hand_with_red[player]) for player in range(4)])
    action = Tile.NUM_TILE_TYPE_WITH_RED + tile_ids["2m"]

    state = _replace_state(
        default_state(),
        current_player=jnp.int8(2),
        legal_action_mask=jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_).at[2, action].set(True),
        hand=hand,
        hand_with_red=hand_with_red,
        pon=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int32).at[2, tile_ids["2m"]].set(1),
        melds=default_state().players.melds.at[2, 1].set(Meld.init(Action.PON, tile_ids["2m"], 1)),
        can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[1, tile_ids["2m"]].set(True),
        last_draw=jnp.int8(tile_ids["2m"]),
        last_player=jnp.int8(2),
        deck=jnp.zeros((136,), dtype=jnp.int8).at[10].set(30),
        dora_indicators=jnp.array([tile_ids["5s"], -1, -1, -1, -1], dtype=jnp.int8),
    )

    env = RedMahjong(round_mode="half", next_round_style="dummy_share")
    kan_state = _kan(state, jnp.int32(action))
    ron_state = env.step(kan_state, jnp.int32(Action.RON))

    expected = jnp.zeros((Action.NUM_ACTION,), dtype=jnp.bool_).at[Action.DUMMY].set(True)
    assert bool(ron_state.round_state.terminated_round)
    assert bool(jnp.all(ron_state.legal_action_mask == expected))
    assert bool(jnp.all(ron_state.players.legal_action_mask[:, Action.DUMMY]))
    assert not bool(ron_state.round_state.kan_declared)
    assert int(ron_state.players.n_kan[1]) == 0
    assert int(ron_state.players.hand_with_red[1].sum()) == 13
    assert int(ron_state.round_state.n_kan_doras) == 0


def test_robbing_kan_second_candidate_pass_after_ron_does_not_draw_after_kan() -> None:
    tile_ids = {
        **{f"{n}m": n - 1 for n in range(1, 10)},
        **{f"{n}p": 9 + n - 1 for n in range(1, 10)},
        **{f"{n}s": 18 + n - 1 for n in range(1, 10)},
    }

    def counts37(tiles: list[str]):
        counts = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8)
        for tile in tiles:
            counts = counts.at[tile_ids[tile]].add(1)
        return counts

    winner_tiles = "3m 4m 5p 5p 6p 7p 8p 6s 6s 6s 7s 8s 9s".split()
    hand_with_red = jnp.zeros((4, Tile.NUM_TILE_TYPE_WITH_RED), dtype=jnp.int8)
    hand_with_red = hand_with_red.at[1].set(counts37(winner_tiles))
    hand_with_red = hand_with_red.at[2].set(counts37(["2m"]))
    hand_with_red = hand_with_red.at[3].set(counts37(winner_tiles))
    hand = jnp.stack([Hand.to_34(hand_with_red[player]) for player in range(4)])
    action = Tile.NUM_TILE_TYPE_WITH_RED + tile_ids["2m"]

    state = _replace_state(
        default_state(),
        current_player=jnp.int8(2),
        legal_action_mask=jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_).at[2, action].set(True),
        hand=hand,
        hand_with_red=hand_with_red,
        pon=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int32).at[2, tile_ids["2m"]].set(1),
        melds=default_state().players.melds.at[2, 1].set(Meld.init(Action.PON, tile_ids["2m"], 1)),
        can_win=(
            jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_)
            .at[1, tile_ids["2m"]].set(True)
            .at[3, tile_ids["2m"]].set(True)
        ),
        last_draw=jnp.int8(tile_ids["2m"]),
        last_player=jnp.int8(2),
        deck=jnp.zeros((136,), dtype=jnp.int8).at[10].set(30),
        dora_indicators=jnp.array([tile_ids["5s"], -1, -1, -1, -1], dtype=jnp.int8),
    )

    env = RedMahjong(round_mode="half", next_round_style="dummy_share")
    kan_state = _kan(state, jnp.int32(action))
    first_ron_state = env.step(kan_state, jnp.int32(Action.RON))
    pass_state = env.step(first_ron_state, jnp.int32(Action.PASS))

    expected = jnp.zeros((Action.NUM_ACTION,), dtype=jnp.bool_).at[Action.DUMMY].set(True)
    assert bool(pass_state.round_state.terminated_round)
    assert bool(jnp.all(pass_state.legal_action_mask == expected))
    assert bool(jnp.all(pass_state.players.legal_action_mask[:, Action.DUMMY]))
    assert not bool(pass_state.round_state.kan_declared)
    assert int(pass_state.players.n_kan[2]) == 0
    assert int(pass_state.players.hand_with_red[3].sum()) == 13
    assert int(pass_state.round_state.n_kan_doras) == 0


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
