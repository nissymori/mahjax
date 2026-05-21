from __future__ import annotations

import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.constants import FIRST_DRAW_IDX
from mahjax.red_mahjong.env import (
    _abortive_draw_normal,
    _draw,
    _make_legal_action_mask_after_draw,
    _mangan_tsumo,
    _pao,
    _pass,
    _ron,
    _special_next_round,
)
from mahjax.red_mahjong.hand import Hand
from mahjax.red_mahjong.meld import Meld
from mahjax.red_mahjong.state import GameConfig, default_game_config, default_state
from mahjax.red_mahjong.tile import River, Tile
from mahjax.red_mahjong.yaku import Yaku


def test_public_state_layout_and_defaults_for_red_env() -> None:
    state = default_state()
    config = default_game_config()

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


def test_red_call_logic_matches_expected_reference_cases() -> None:
    hand = jnp.zeros((37,), dtype=jnp.int8)
    hand = hand.at[Tile.BLACK_FIVE['p']].set(2)
    hand = hand.at[Tile.RED_FIVE['p']].set(1)
    hand = hand.at[12].set(1)  # 4p

    assert bool(Hand.can_red_pon(hand, Tile.BLACK_FIVE['p']))
    assert bool(Hand.can_open_kan(hand, Tile.BLACK_FIVE['p']))
    assert bool(Hand.can_open_kan(hand, Tile.RED_FIVE['p']))
    assert bool(Hand.can_chi(hand, 14, Action.CHI_R_RED))

    pon_hand = Hand.pon(hand, Tile.BLACK_FIVE['p'], Action.PON_RED)
    assert int(pon_hand[Tile.BLACK_FIVE['p']]) == 1
    assert int(pon_hand[Tile.RED_FIVE['p']]) == 0

    chi_hand = Hand.chi(hand, 14, Action.CHI_R_RED)
    assert int(chi_hand[12]) == 0
    assert int(chi_hand[Tile.RED_FIVE['p']]) == 0
    assert int(chi_hand[14]) == 0

    open_kan_black_target = Hand.open_kan(hand, Tile.BLACK_FIVE['p'])
    assert int(open_kan_black_target[Tile.BLACK_FIVE['p']]) == 0
    assert int(open_kan_black_target[Tile.RED_FIVE['p']]) == 0

    hand_red_target = jnp.zeros((37,), dtype=jnp.int8).at[Tile.BLACK_FIVE['m']].set(3)
    open_kan_red_target = Hand.open_kan(hand_red_target, Tile.RED_FIVE['m'])
    assert int(open_kan_red_target[Tile.BLACK_FIVE['m']]) == 0

    closed_kan_hand = jnp.zeros((37,), dtype=jnp.int8)
    closed_kan_hand = closed_kan_hand.at[Tile.BLACK_FIVE['s']].set(3)
    closed_kan_hand = closed_kan_hand.at[Tile.RED_FIVE['s']].set(1)
    after_closed_kan = Hand.closed_kan(closed_kan_hand, Tile.BLACK_FIVE['s'])
    assert int(after_closed_kan[Tile.BLACK_FIVE['s']]) == 0
    assert int(after_closed_kan[Tile.RED_FIVE['s']]) == 0


def test_meld_red_flags_match_action_and_target() -> None:
    pon_red = Meld.init(Action.PON_RED, Tile.BLACK_FIVE['m'], 1)
    chi_red = Meld.init(Action.CHI_M_RED, 13, 3)
    open_kan_red_target = Meld.init(Action.OPEN_KAN, Tile.RED_FIVE['s'], 2)
    normal_chi = Meld.init(Action.CHI_M, 13, 3)

    assert bool(Meld.contains_red(pon_red))
    assert bool(Meld.contains_red(chi_red))
    assert bool(Meld.contains_red(open_kan_red_target))
    assert not bool(Meld.contains_red(normal_chi))
    assert int(Meld.target(open_kan_red_target)) == Tile.BLACK_FIVE['s']
    assert int(Meld.target_tile(open_kan_red_target)) == Tile.RED_FIVE['s']


def test_yaku_red_dora_adds_one_fan_without_changing_fu() -> None:
    base_hand = jnp.zeros((34,), dtype=jnp.int8)
    for tile in (1, 1, 2, 2, 12, 12, 14, 14, 19, 19, 20, 20):
        base_hand = base_hand.at[tile].add(1)
    base_hand = base_hand.at[13].add(1)  # single 5p before the winning tile

    red_hand = jnp.zeros((37,), dtype=jnp.int8)
    red_hand = red_hand.at[:34].set(base_hand)

    base_state = default_state()
    black_state = base_state.replace(
        round_state=base_state.round_state.replace(last_draw=jnp.int8(13))
    )
    red_state = base_state.replace(
        round_state=base_state.round_state.replace(last_draw=jnp.int8(Tile.RED_FIVE['p']))
    )

    yaku_black, fan_black, fu_black = Yaku.judge(
        base_hand,
        jnp.bool_(False),
        jnp.int8(0),
        black_state,
    )
    yaku_red, fan_red, fu_red = Yaku.judge(
        red_hand,
        jnp.bool_(False),
        jnp.int8(0),
        red_state,
    )

    assert yaku_black.shape == (52,)
    assert Yaku.FourKans == 51
    assert bool(yaku_black[Yaku.SevenPairs])
    assert jnp.array_equal(yaku_black, yaku_red)
    assert int(fan_red) == int(fan_black) + 1
    assert int(fu_red) == int(fu_black)


def test_pinfu_is_rejected_on_honor_tanki_wait() -> None:
    hand = jnp.zeros((34,), dtype=jnp.int8)
    for tile in (4, 5, 6, 9, 10, 11, 13, 14, 15, 18, 19, 20, 27):
        hand = hand.at[tile].add(1)

    base = default_state()
    state = base.replace(
        players=base.players.replace(
            riichi=base.players.riichi.at[0].set(jnp.bool_(True)),
        ),
        round_state=base.round_state.replace(
            round=jnp.int8(1),  # East 2, prevalent wind = East
            seat_wind=base.round_state.seat_wind.at[0].set(jnp.int8(1)),  # South seat
            target=jnp.int8(27),  # ron on East
        ),
    )

    yaku, fan, fu = Yaku.judge(
        hand,
        jnp.bool_(True),
        jnp.int8(0),
        state,
    )

    assert bool(yaku[Yaku.Riichi])
    assert not bool(yaku[Yaku.Pinfu])
    assert int(fan) == 1
    assert int(fu) == 40


def test_pinfu_is_rejected_when_closed_kan_exists() -> None:
    hand = jnp.zeros((34,), dtype=jnp.int8)
    for tile in (3, 4, 5, 11, 12, 13, 22, 23, 24, 15):
        hand = hand.at[tile].add(1)
    hand = hand.at[14].add(2)

    base = default_state()
    melds = base.players.melds.at[0, 0].set(Meld.init(40, 27, 0))
    state = base.replace(
        players=base.players.replace(
            melds=melds,
            meld_counts=base.players.meld_counts.at[0].set(1),
            riichi=base.players.riichi.at[0].set(jnp.bool_(True)),
        ),
        round_state=base.round_state.replace(
            round=jnp.int8(1),
            seat_wind=base.round_state.seat_wind.at[0].set(jnp.int8(1)),
            last_draw=jnp.int8(15),
        ),
    )

    yaku, fan, fu = Yaku.judge(
        hand,
        jnp.bool_(False),
        jnp.int8(0),
        state,
    )

    assert bool(yaku[Yaku.Riichi])
    assert bool(yaku[Yaku.FullyConcealedHand])
    assert not bool(yaku[Yaku.Pinfu])
    assert int(fan) >= 2
    assert int(fu) > 20


def test_kyuushu_action_is_enabled_on_first_turn() -> None:
    state = default_state()
    hand_with_red = state.players.hand_with_red.at[0].set(
        jnp.zeros((37,), dtype=jnp.int8)
        .at[jnp.array([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 0])]
        .add(1)
    )
    state = state.replace(players=state.players.replace(hand_with_red=hand_with_red))

    mask = _make_legal_action_mask_after_draw(state, hand_with_red, jnp.int8(0), jnp.int8(0))

    assert bool(mask[Action.KYUUSHU])


def test_kyuushu_action_is_disabled_by_game_config() -> None:
    state = default_state()
    hand_with_red = state.players.hand_with_red.at[0].set(
        jnp.zeros((37,), dtype=jnp.int8)
        .at[jnp.array([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 0])]
        .add(1)
    )
    state = state.replace(players=state.players.replace(hand_with_red=hand_with_red))
    config = GameConfig(enable_special_abortive_draw=jnp.bool_(False))

    mask = _make_legal_action_mask_after_draw(
        state,
        hand_with_red,
        jnp.int8(0),
        jnp.int8(0),
        config,
    )

    assert not bool(mask[Action.KYUUSHU])


def test_four_winds_abortive_draw_sets_kyuushu_mask() -> None:
    state = default_state()
    river = state.players.river
    for player in range(4):
        river = River.add_discard(river, jnp.int8(27), jnp.int8(player), jnp.int8(0), False, False)
    state = state.replace(
        players=state.players.replace(
            river=river,
            discard_counts=jnp.ones((4,), dtype=jnp.int8),
        ),
        round_state=state.round_state.replace(next_deck_ix=jnp.int32(FIRST_DRAW_IDX)),
    )

    next_state = _draw(state)

    assert bool(next_state.legal_action_mask[Action.KYUUSHU])
    assert bool(next_state.players.legal_action_mask[:, Action.KYUUSHU].all())


def test_special_next_round_keeps_dealer_and_increments_honba() -> None:
    state = default_state().replace(
        current_player=jnp.int8(2),
        round_state=default_state().round_state.replace(
            dealer=jnp.int8(2),
            round=jnp.int8(3),
            honba=jnp.int8(2),
            kyotaku=jnp.int8(1),
            score=jnp.array([300, 250, 200, 250], dtype=jnp.int32),
        ),
    )

    next_state = _special_next_round(state)

    assert int(next_state.current_player) == 2
    assert int(next_state.round_state.dealer) == 2
    assert int(next_state.round_state.round) == 3
    assert int(next_state.round_state.honba) == 3
    assert int(next_state.round_state.kyotaku) == 1
    assert jnp.array_equal(next_state.round_state.score, jnp.array([300, 250, 200, 250], dtype=jnp.int32))


def test_pao_and_nagashi_mangan_scoring() -> None:
    base = default_state()
    melds = base.players.melds.at[0, 0].set(Meld.init(Action.PON, 31, 1))
    melds = melds.at[0, 1].set(Meld.init(Action.PON, 32, 3))
    melds = melds.at[0, 2].set(Meld.init(Action.OPEN_KAN, 33, 2))
    ron_state = base.replace(
        current_player=jnp.int8(0),
        players=base.players.replace(
            melds=melds,
            meld_counts=base.players.meld_counts.at[0].set(3),
            fan=base.players.fan.at[0, 0].set(jnp.int32(5)),
            fu=base.players.fu.at[0, 0].set(jnp.int32(30)),
        ),
        round_state=base.round_state.replace(
            dealer=jnp.int8(1),
            last_player=jnp.int8(1),
            honba=jnp.int8(1),
            score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        ),
    )

    is_pao, pao_player = _pao(ron_state, jnp.int8(0))
    ron_after = _ron(ron_state)
    basic = Yaku.score(jnp.int32(5), jnp.int32(30))
    ron_payment = jnp.ceil(basic * 4 / 100)
    expected_ron = jnp.array(
        [ron_payment + 3, -ron_payment / 2 - 3, -ron_payment / 2, 0],
        dtype=jnp.float32,
    )

    assert bool(is_pao)
    assert int(pao_player) == 2
    assert jnp.array_equal(ron_after.rewards, expected_ron)

    nagashi_state = base.replace(
        players=base.players.replace(
            has_nagashi_mangan=jnp.array([True, False, False, False], dtype=jnp.bool_)
        ),
        round_state=base.round_state.replace(
            dealer=jnp.int8(0),
            honba=jnp.int8(1),
            score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        ),
    )
    nagashi_after = _abortive_draw_normal(nagashi_state)
    expected_nagashi = _mangan_tsumo(jnp.int8(0), jnp.int8(0), jnp.int8(0))

    assert jnp.array_equal(nagashi_after.rewards, expected_nagashi.astype(jnp.float32))


def test_double_ron_config_chains_ron_and_pass_resolution() -> None:
    base = default_state()
    legal_action_mask = base.players.legal_action_mask
    legal_action_mask = legal_action_mask.at[0, Action.RON].set(True)
    legal_action_mask = legal_action_mask.at[1, Action.RON].set(True)
    ron_state = base.replace(
        current_player=jnp.int8(0),
        legal_action_mask=legal_action_mask[0],
        players=base.players.replace(
            legal_action_mask=legal_action_mask,
            fan=base.players.fan.at[0, 0].set(jnp.int32(5)),
            fu=base.players.fu.at[0, 0].set(jnp.int32(30)),
        ),
        round_state=base.round_state.replace(
            dealer=jnp.int8(2),
            last_player=jnp.int8(3),
            honba=jnp.int8(1),
            kyotaku=jnp.int8(2),
            score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        ),
    )
    config = GameConfig(allow_double_ron=jnp.bool_(True))

    after_first_ron = _ron(ron_state, config)

    assert not bool(after_first_ron.round_state.terminated_round)
    assert int(after_first_ron.current_player) == 1
    assert bool(after_first_ron.players.legal_action_mask[1, Action.RON])
    assert bool(after_first_ron.players.legal_action_mask[1, Action.PASS])
    assert int(after_first_ron.round_state.kyotaku) == 0

    after_pass = _pass(after_first_ron, config)

    assert bool(after_pass.round_state.terminated_round)
    assert bool(after_pass.players.legal_action_mask[:, Action.DUMMY].all())


def test_single_ron_when_double_ron_is_disabled() -> None:
    base = default_state()
    legal_action_mask = base.players.legal_action_mask
    legal_action_mask = legal_action_mask.at[0, Action.RON].set(True)
    legal_action_mask = legal_action_mask.at[1, Action.RON].set(True)
    ron_state = base.replace(
        current_player=jnp.int8(0),
        legal_action_mask=legal_action_mask[0],
        players=base.players.replace(
            legal_action_mask=legal_action_mask,
            fan=base.players.fan.at[0, 0].set(jnp.int32(5)),
            fu=base.players.fu.at[0, 0].set(jnp.int32(30)),
        ),
        round_state=base.round_state.replace(
            dealer=jnp.int8(2),
            last_player=jnp.int8(3),
            honba=jnp.int8(1),
            score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        ),
    )
    config = GameConfig(allow_double_ron=jnp.bool_(False))

    after_ron = _ron(ron_state, config)

    assert bool(after_ron.round_state.terminated_round)
    assert bool(after_ron.players.legal_action_mask[:, Action.DUMMY].all())


def test_pao_can_be_disabled_by_game_config() -> None:
    base = default_state()
    melds = base.players.melds.at[0, 0].set(Meld.init(Action.PON, 31, 1))
    melds = melds.at[0, 1].set(Meld.init(Action.PON, 32, 3))
    melds = melds.at[0, 2].set(Meld.init(Action.OPEN_KAN, 33, 2))
    ron_state = base.replace(
        current_player=jnp.int8(0),
        players=base.players.replace(
            melds=melds,
            meld_counts=base.players.meld_counts.at[0].set(3),
            fan=base.players.fan.at[0, 0].set(jnp.int32(5)),
            fu=base.players.fu.at[0, 0].set(jnp.int32(30)),
        ),
        round_state=base.round_state.replace(
            dealer=jnp.int8(1),
            last_player=jnp.int8(1),
            honba=jnp.int8(1),
            score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        ),
    )
    config = GameConfig(enable_pao=jnp.bool_(False))
    basic = Yaku.score(jnp.int32(5), jnp.int32(30))
    ron_payment = jnp.ceil(basic * 4 / 100)

    after_ron = _ron(ron_state, config)

    assert jnp.array_equal(
        after_ron.rewards,
        jnp.array([ron_payment + 3, -ron_payment - 3, 0, 0], dtype=jnp.float32),
    )
