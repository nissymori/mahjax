from __future__ import annotations

import jax
import jax.numpy as jnp

from mahjax.mahjong_4p.action import Action

from mahjax.mahjong_4p.meld import Meld
from mahjax.mahjong_4p.state import default_state
from mahjax.mahjong_4p.tile import Tile
from mahjax.mahjong_4p.yaku import Yaku

from ..helpers import compiled_env


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


def test_big_four_winds_is_a_single_yakuman() -> None:
    """Every yakuman counts once; only different yakuman in one hand add up."""
    base = default_state()
    state = base.replace(round_state=base.round_state.replace(target=jnp.int8(30)))  # ron on North
    for pair, expected_fan in ((0, 1), (31, 2)):  # 11m pair; White pair also makes All Honors
        hand = jnp.zeros((34,), dtype=jnp.int8)
        hand = hand.at[27].set(3).at[28].set(3).at[29].set(3).at[30].set(2).at[pair].set(2)

        yaku, fan, fu = Yaku.judge(hand, jnp.bool_(True), jnp.int8(1), state)

        assert bool(yaku[Yaku.BigFourWinds])
        assert int(fu) == 0
        assert int(fan) == expected_fan


def test_thirteen_orphans_waiting_on_all_thirteen_is_completed_thirteen_orphans() -> None:
    """One of each terminal and honor waits on all thirteen; winning it is
    CompletedThirteenOrphans instead of ThirteenOrphans, and still one yakuman."""
    orphans = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)
    base = default_state()
    state = base.replace(  # the winning tile is Red, by ron or by tsumo
        round_state=base.round_state.replace(target=jnp.int8(33), last_draw=jnp.int8(33))
    )
    for tiles, thirteen_sided in (
        (orphans, True),  # one of each: waits on all thirteen
        (orphans[:-1] + (0,), False),  # 1m pair and no Red: waits on Red only
    ):
        hand = jnp.zeros((34,), dtype=jnp.int8)
        for tile in tiles:
            hand = hand.at[tile].add(1)
        for judge in (Yaku.judge, Yaku.judge_yakuman):
            for is_ron in (True, False):
                yaku, fan, fu = judge(hand, jnp.bool_(is_ron), jnp.int8(1), state)

                assert bool(yaku[Yaku.CompletedThirteenOrphans]) == thirteen_sided
                assert bool(yaku[Yaku.ThirteenOrphans]) != thirteen_sided
                assert int(fan) == 1
                assert int(fu) == 0


def test_nine_gates_waiting_on_all_nine_is_pure_nine_gates() -> None:
    """1112345678999 of one suit waits on all nine; winning it is PureNineGates
    instead of NineGates, and still one yakuman."""
    shape = (0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8)  # 1112345678999m
    base = default_state()
    for tiles, winning_tile, nine_sided in (
        (shape, 4, True),  # wins on 5m
        (shape, 0, True),  # wins on 1m, the fourth one
        ((0, 0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 8, 8), 8, False),  # 1112345567899m wins on 9m
    ):
        hand = jnp.zeros((34,), dtype=jnp.int8)
        for tile in tiles:
            hand = hand.at[tile].add(1)
        state = base.replace(  # by ron or by tsumo
            round_state=base.round_state.replace(target=jnp.int8(winning_tile), last_draw=jnp.int8(winning_tile))
        )
        for judge in (Yaku.judge, Yaku.judge_yakuman):
            for is_ron in (True, False):
                yaku, fan, fu = judge(hand, jnp.bool_(is_ron), jnp.int8(1), state)

                assert bool(yaku[Yaku.PureNineGates]) == nine_sided
                assert bool(yaku[Yaku.NineGates]) != nine_sided
                assert int(fan) == 1
                assert int(fu) == 0


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


def test_all_green_counts_the_four_tiles_of_a_kan() -> None:
    """With a kan the hand holds 15 tiles, so All Green means no other tile, not 14 green ones."""
    base = default_state()
    closed_kan_8s = Meld.init(Tile.NUM_TILE_TYPE_WITH_RED + 25, 25, 0)
    state = base.replace(
        players=base.players.replace(
            melds=base.players.melds.at[0, 0].set(closed_kan_8s),
            meld_counts=base.players.meld_counts.at[0].set(1),
        ),
        round_state=base.round_state.replace(last_draw=jnp.int8(21)),  # tsumo 4s
    )
    for tiles, is_all_green in (
        ((19, 19, 20, 20, 21, 23, 23, 23, 32, 32), True),  # 8888s + 234s 234s 666s GrGr
        ((18, 19, 19, 20, 20, 21, 21, 23, 23, 23), False),  # 8888s + 123s 234s 666s 44s: 1s is not green
    ):
        hand = jnp.zeros((34,), dtype=jnp.int8)
        for tile in tiles:
            hand = hand.at[tile].add(1)

        yaku, _, _ = Yaku.judge(hand, jnp.bool_(False), jnp.int8(0), state)

        assert bool(yaku[Yaku.AllGreen]) == is_all_green


def test_score_is_a_limit_hand_from_five_fan_whatever_the_fu() -> None:
    """13 fan or more is a counted yakuman; a large fan used to overflow the int32 shift."""
    score = jax.jit(jax.vmap(jax.vmap(Yaku.score, in_axes=(None, 0)), in_axes=(0, None)))
    fus = jnp.array([20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130], dtype=jnp.int32)
    limits = {5: 2000, 6: 3000, 7: 3000, 8: 4000, 9: 4000, 10: 4000, 11: 6000, 12: 6000}
    fans = jnp.arange(5, 41, dtype=jnp.int32)
    expected = jnp.array([limits.get(fan, 8000) for fan in range(5, 41)], dtype=jnp.int32)
    assert jnp.array_equal(score(fans, fus), jnp.broadcast_to(expected[:, None], (fans.size, fus.size)))

    # Up to 4 fan the basic points are fu << (fan + 2), capped at a mangan.
    fans = jnp.arange(1, 5, dtype=jnp.int32)
    assert jnp.array_equal(score(fans, fus), jnp.minimum(fus[None, :] << (fans[:, None] + 2), 2000))
    assert int(Yaku.score(jnp.int32(2), jnp.int32(0))) == 16000  # a double yakuman


def test_red_kan_of_fives_scores_no_red_dora_without_red_fives() -> None:
    from mahjax.mahjong_4p.meld import Meld as RedMeld
    from mahjax.mahjong_4p.config import GameConfig as RedGameConfig
    from mahjax.mahjong_4p.yaku import Yaku as RedYaku

    def fan_for(use_red_fives: bool) -> int:
        env = compiled_env(
            round_mode="half", game_config=RedGameConfig(use_red_fives=use_red_fives)
        )
        state = env.init(jax.random.PRNGKey(0))
        hand = jnp.zeros((37,), dtype=jnp.int8)
        for tile in (0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27):
            hand = hand.at[tile].add(1)
        state = state.replace(
            players=state.players.replace(
                melds=state.players.melds.at[0, 0].set(RedMeld.init(Action.OPEN_KAN, 4, 1)),
                meld_counts=state.players.meld_counts.at[0].set(1),
            ),
            round_state=state.round_state.replace(
                target=jnp.int8(27), dora_indicators=jnp.full((5,), -1, dtype=jnp.int8)
            ),
        )
        _, fan, _ = RedYaku.judge(hand, jnp.bool_(True), jnp.int8(0), state)
        return int(fan)

    assert fan_for(False) == fan_for(True) - 1


def test_dealer_can_tsumo_a_complete_first_draw_as_a_single_blessing_of_heaven() -> None:
    """123m 33m 444p 666p 77s drawing 7s: TSUMO is legal, and three concealed pons are not four."""
    from unittest import mock

    import jax

    from mahjax.mahjong_4p import logic as m

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
