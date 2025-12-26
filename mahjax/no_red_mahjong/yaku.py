# Copyright 2025 The Mahjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from mahjax._src.types import Array
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.hand import Hand
from mahjax.no_red_mahjong.meld import EMPTY_MELD, Meld

DIR = os.path.join(os.path.dirname(__file__), "cache")


def load_yaku_cache():
    with np.load(os.path.join(DIR, "yaku_cache.npz"), allow_pickle=False) as data:
        return jnp.asarray(data["data"], dtype=jnp.uint32)


WIND_TILE = jnp.array([27, 28, 29, 30], dtype=jnp.int8)
OUTSIDE_TILE = jnp.array([0, 8, 9, 17, 18, 26], dtype=jnp.int8)
TANYAO_TILE = jnp.array(
    [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25],
    dtype=jnp.int8,
)
KOKUSHI_TILE = jnp.array(
    [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33], dtype=jnp.int8
)
ALL_GREEN_TILE = jnp.array([19, 20, 21, 23, 25, 32], dtype=jnp.int8)
SCORES = jnp.array(
    [2000, 2000, 3000, 3000, 4000, 4000, 4000, 6000, 6000, 8000, 8000, 8000],
    dtype=jnp.int32,
)  # 5-15 fan TODO: Add the scores for yakuman over 15 fan


def one_hot_at(x: Array, idx: Array) -> Array:
    one_hot = jax.nn.one_hot(idx, x.shape[0])
    return x @ one_hot


powers_of_5_full = jnp.concatenate(
    [
        5 ** jnp.arange(8, -1, -1),  # for suit 0
        5 ** jnp.arange(8, -1, -1),  # for suit 1
        5 ** jnp.arange(8, -1, -1),  # for suit 2
    ]
)  # shape = (27,)


class Yaku:
    CACHE = load_yaku_cache()
    MAX_PATTERNS = 3
    # The terminology basically follows:
    # http://mahjong-europe.org/portal/images/docs/riichi_scoresheet_EN.pdf
    Pinfu = 0  # 平和
    PureDoubleChis = 1  # 一盃口
    TwicePureDoubleChis = 2  # 二盃口
    OutsideHand = 3  # 混全帯么九
    TerminalsInAllSets = 4  # 純全帯么九
    PureStraight = 5  # 一気通貫
    MixedTripleChis = 6  # 三色同順
    TriplePons = 7  # 三色同刻
    AllPons = 8  # 対々和
    ThreeConcealedPons = 9  # 三暗刻
    ThreeKans = 10  # 三槓子
    SevenPairs = 11  # 七対子
    AllSimples = 12  # 断么九
    HalfFlush = 13  # 混一色
    FullFlush = 14  # 清一色
    AllTerminalsAndHonors = 15  # 混老頭
    LittleThreeDragons = 16  # 小三元
    WhiteDragon = 17  # 白
    GreenDragon = 18  # 發
    RedDragon = 19  # 中
    PrevelantWind = 20  # 場風
    SeatWind = 21  # 自風
    FullyConcealedHand = 22  # 門前清自摸和
    Riichi = 23  # 立直
    # yakuman
    BigThreeDragons = 24  # 大三元
    LittleFourWinds = 25  # 小四喜
    BigFourWinds = 26  # 大四喜
    NineGates = 27  # 九蓮宝燈
    ThirteenOrphans = 28  # 国士無双
    AllTerminals = 29  # 清老頭
    AllHonors = 30  # 字一色
    AllGreen = 31  # 緑一色
    FourConcealedPons = 32  # 四暗刻 TODO: Maybe need to distinguish 四暗刻単騎
    FourKans = 33  # 四槓子

    # fmt: off
    FAN = jnp.array([
        [0,0,0,1,2,1,1,2,2,2,2,0,1,2,5,2,2,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],  # noqa
        [1,1,3,2,3,2,2,2,2,2,2,2,1,3,6,2,2,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],  # noqa
    ])
    YAKUMAN = jnp.array([
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,1,1,1,1,1,1,1  # noqa
    ])
    # fmt: on

    YAKU_UPDATE_INDICES = jnp.array(
        [
            Pinfu,  # 平和
            PureDoubleChis,  # 一盃口
            TwicePureDoubleChis,  # 二盃口
            OutsideHand,  # 混全帯么九
            TerminalsInAllSets,  # 純全帯么九
            PureStraight,  # 一気通貫
            MixedTripleChis,  # 三色同順
            TriplePons,  # 三色同刻
            AllPons,  # 対々和
            ThreeConcealedPons,  # 三暗刻
            ThreeKans,  # 三槓子
        ],
        dtype=jnp.int32,
    )

    YAKU_BEST_UPDATE_INDICES = jnp.array(
        [
            AllSimples,  # 断么九
            HalfFlush,  # 混一色
            FullFlush,  # 清一色
            AllTerminalsAndHonors,  # 混老頭
            WhiteDragon,  # 白
            GreenDragon,  # 發
            RedDragon,  # 中
            LittleThreeDragons,  # 小三元
            PrevelantWind,  # 場風
            SeatWind,  # 自風
            FullyConcealedHand,  # 門前清自摸和
            Riichi,  # 立直
        ],
        dtype=jnp.int32,
    )

    YAKUMAN_UPDATE_INDICES = jnp.array(
        [
            BigThreeDragons,  # 大三元
            LittleFourWinds,  # 小四喜
            BigFourWinds,  # 大四喜
            NineGates,  # 九蓮宝燈
            ThirteenOrphans,  # 国士無双
            AllTerminals,  # 清老頭
            AllHonors,  # 字一色
            AllGreen,  # 緑一色
            FourConcealedPons,  # 四暗刻
            FourKans,  # 四槓子
        ],
        dtype=jnp.int32,
    )

    @staticmethod
    def score(
        fan: Array,
        fu: Array,
    ) -> int:
        """
        Calculate the score from fan and fu
        - For yakuman, the score is 8000 * fan
        - For other yaku, the score is fu << (fan + 2)
        """
        score = fu << (fan + 2)
        return jax.lax.cond(
            fu == 0,
            lambda: jnp.int32(
                8000 * fan
            ),  # In the case of yakuman, the fan contains the number of yakuman.
            lambda: (score < 2000) * score + (score >= 2000) * SCORES[fan - 4],
        )

    @staticmethod
    def head(code: Array) -> Array:
        return Yaku.CACHE[code] & 0b1111

    @staticmethod
    def chow(code: Array) -> Array:
        return Yaku.CACHE[code] >> 4 & 0b1111111

    @staticmethod
    def pung(code: Array) -> Array:
        return Yaku.CACHE[code] >> 11 & 0b111111111

    @staticmethod
    def n_pung(code: Array) -> Array:
        return Yaku.CACHE[code] >> 20 & 0b111

    @staticmethod
    def n_double_chow(code: Array) -> Array:
        return Yaku.CACHE[code] >> 23 & 0b11

    @staticmethod
    def outside(code: Array) -> Array:
        return Yaku.CACHE[code] >> 25 & 1

    @staticmethod
    def nine_gates(code: Array) -> Array:
        return Yaku.CACHE[code] >> 26

    @staticmethod
    def is_pure_straight(chow: Array) -> Array:
        return (
            ((chow & 0b1001001) == 0b1001001)
            | ((chow >> 9 & 0b1001001) == 0b1001001)
            | ((chow >> 18 & 0b1001001) == 0b1001001)
        ) == 1

    @staticmethod
    def is_triple_chow(chow: Array) -> Array:
        return (
            ((chow & 0b1000000001000000001) == 0b1000000001000000001)
            | ((chow >> 1 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((chow >> 2 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((chow >> 3 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((chow >> 4 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((chow >> 5 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((chow >> 6 & 0b1000000001000000001) == 0b1000000001000000001)
        ) == 1

    @staticmethod
    def is_triple_pung(pung: Array) -> Array:
        return (
            ((pung & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 1 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 2 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 3 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 4 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 5 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 6 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 7 & 0b1000000001000000001) == 0b1000000001000000001)
            | ((pung >> 8 & 0b1000000001000000001) == 0b1000000001000000001)
        ) == 1

    @staticmethod
    def update(
        is_pinfu: Array,
        has_outside: Array,
        n_double_chow: Array,
        all_chow: Array,
        all_pung: Array,
        n_concealed_pung: Array,
        nine_gates: Array,
        fu: Array,
        code: Array,
        suit: Array,
        last_tile_type: Array,
        is_ron: Array,
    ) -> Tuple:
        """
        Update the statistics to calculate yaku.
        - They are precalculated for meld.
        - We update the statistics based on the hand information.
        - Each statistics is encoded into cache suitwise.
        """
        chow = Yaku.chow(code)
        pung = Yaku.pung(code)

        open_end = (chow ^ (chow & 1)) << 2 | (chow ^ (chow & 0b1000000))
        # An open ended wait (両面待ち) can be made at this position
        in_range = suit == last_tile_type // 9
        pos = last_tile_type % 9

        is_pinfu &= (in_range == 0) | (open_end >> pos & 1) == 1
        is_pinfu &= pung == 0
        has_outside &= Yaku.outside(code) == 1

        n_double_chow += Yaku.n_double_chow(code)
        all_chow |= chow << 9 * suit
        all_pung |= pung << 9 * suit

        n_pung = Yaku.n_pung(code)
        # The number of pungs (刻子)
        chow_range = chow | chow << 1 | chow << 2

        loss = is_ron & in_range & ((chow_range >> pos & 1) == 0) & (pung >> pos & 1)
        # If the player ron and the chow becomes a meld, the loss is counted
        n_concealed_pung += n_pung - loss

        nine_gates |= Yaku.nine_gates(code) == 1
        outside_pung = pung & 0b100000001
        strong = (
            in_range
            & (
                (1 << Yaku.head(code))
                | ((chow & 1) << 2)
                | (chow & 0b1000000)
                | (chow << 1)
            )
            >> pos
            & 1
        )
        # A strong wait (カンチャン, ペンチャン, 単騎) can be made at this position
        loss <<= outside_pung >> pos & 1
        fu += 4 * (n_pung + (outside_pung > 0)) - 2 * loss + 2 * strong
        return (
            is_pinfu,
            has_outside,
            n_double_chow,
            all_chow,
            all_pung,
            n_concealed_pung,
            nine_gates,
            fu,
        )

    @staticmethod
    def judge(
        hand: Array,
        melds: Array,
        n_meld: Array,
        last_tile: Array,
        riichi: Array,
        is_ron: Array,
        prevalent_wind: Array,
        seat_wind: Array,
        dora: Array,
    ) -> Tuple:
        """
        Judge the yaku of the hand.
        - hand: Hand vector (34,) has 13 tiles
        - melds: Melds vector (4,)
        - n_meld: Number of melds (1,)
        - last_tile: The tile that is added to the hand
        - riichi: Whether the player has riichi (4,)
        - is_ron: Whether the winning type is ron (4,)
        - prevalent_wind: Prevalent wind (0-3)
        - seat_wind: Seat wind (0-3)
        - dora: Dora vector (2, 34)
        """
        # Add the last tile to the hand and determine the yaku
        hand = Hand.add(hand, last_tile)
        last_tile_type = last_tile

        # Depending on the presence of riichi, the dora is obtained. dora is of shape (2, 34). dora[0] is the front dora only, dora[1] includes the ura dora.
        dora = jnp.where(riichi, dora.sum(axis=0), dora[0])
        # Set the winds
        seat_wind_tile_type = WIND_TILE[seat_wind]
        prevalent_wind_tile_type = WIND_TILE[prevalent_wind]

        # Check the necessary conditions for the hand FullyConcealedHand and Pinfu
        is_hand_concealed = jnp.all(
            (Action.is_selfkan(Meld.action(melds)) & (Meld.src(melds) == 0))
            | (melds == EMPTY_MELD)
        )  # The parts without melds are melds==0
        is_pinfu = jnp.full(
            Yaku.MAX_PATTERNS,
            is_hand_concealed
            & jnp.all(hand[27:31] < 3)
            & (hand[seat_wind_tile_type] == 0)
            & (hand[prevalent_wind_tile_type] == 0)
            & jnp.all(hand[31:34] == 0),
        )

        # Calculate the variables necessary for the yaku of the OutsideHand(混全帯么九), pung, and shuntsu (chow)
        has_outside = jnp.full(
            Yaku.MAX_PATTERNS, jnp.all(Meld.has_outside(melds) | (melds == EMPTY_MELD))
        )
        all_chow = jnp.full(
            Yaku.MAX_PATTERNS, jnp.any(Meld.chow(melds) & (melds != EMPTY_MELD))
        ).astype(jnp.int32)
        all_pung = jnp.full(
            Yaku.MAX_PATTERNS, jnp.any(Meld.suited_pung(melds) & (melds != EMPTY_MELD))
        ).astype(jnp.int32)
        # The number of kans (槓子)
        n_kan = jnp.sum(Meld.is_kan(melds) & (melds != EMPTY_MELD))
        n_closed_kan = jnp.sum(Meld.is_closed_kan(melds) & (melds != EMPTY_MELD))
        # Initialize the variables necessary for the yaku of TwicePureDoubleChis(二盃口), Yaku related to Pons, and NineGates(九蓮宝燈)
        n_double_chow = jnp.full(Yaku.MAX_PATTERNS, 0)
        n_concealed_pung = jnp.full(Yaku.MAX_PATTERNS, 0)
        nine_gates = jnp.full(Yaku.MAX_PATTERNS, False)
        # Closed kans are also counted as pungs
        n_concealed_pung += (
            jnp.sum(hand[27:] >= 3)
            - (is_ron & (last_tile_type >= 27) & (hand[last_tile_type] >= 3))
            + n_closed_kan
        )
        # Calculate the fu of the melds and honors in the hand
        fu = jnp.full(
            Yaku.MAX_PATTERNS,
            2 * (is_ron == 0)
            + jnp.sum(Meld.fu(melds))
            + (hand[seat_wind_tile_type] == 2) ** 2  # Head fu for seat wind
            + (hand[prevalent_wind_tile_type] == 2) ** 2  # Head fu for prevalent wind
            + jnp.any(hand[31:] == 2) * 2  # Head fu for dragons
            + jnp.sum(
                (hand[27:34] == 3)
                * 4
                * (2 - (is_ron & (jnp.arange(27, 34) == last_tile_type)))
            )  # Fu for concealed tiles of honors
            + ((27 <= last_tile_type) & (hand[last_tile_type] == 2)),
            dtype=jnp.int32,
        )

        codes = (
            (hand[:27].astype(int) * powers_of_5_full).reshape(3, 9).sum(axis=1)
        )  # (3,)

        # Update the variables based on the information of the tiles (since there are chows, it is complicated)
        def _update_yaku(suit: Array, tpl: Tuple) -> Tuple:
            code = codes[suit]
            return Yaku.update(
                tpl[0],
                tpl[1],
                tpl[2],
                tpl[3],
                tpl[4],
                tpl[5],
                tpl[6],
                tpl[7],
                code,
                suit,
                last_tile_type,
                is_ron,
            )

        (
            is_pinfu,
            has_outside,
            n_double_chow,
            all_chow,
            all_pung,
            n_concealed_pung,
            nine_gates,
            fu,
        ) = jax.lax.fori_loop(
            0,
            3,
            _update_yaku,
            (
                is_pinfu,
                has_outside,
                n_double_chow,
                all_chow,
                all_pung,
                n_concealed_pung,
                nine_gates,
                fu,
            ),
        )

        fu *= is_pinfu == 0
        fu += 20 + 10 * (is_hand_concealed & is_ron)
        fu += 10 * ((is_hand_concealed == 0) & (fu == 20))

        # Combine the melds and the hand
        flatten = Yaku.flatten(hand, melds, n_meld)  # (34,)

        four_winds = jnp.sum(flatten[27:31] >= 3)
        three_dragons = jnp.sum(flatten[31:34] >= 3)

        has_tanyao = jnp.any(
            jax.vmap(one_hot_at, in_axes=(None, 0))(flatten, TANYAO_TILE)
        )
        has_honor = jnp.any(flatten[27:] > 0)
        is_flush = (
            jnp.any(flatten[0:9] > 0).astype(int)
            + jnp.any(flatten[9:18] > 0).astype(int)
            + jnp.any(flatten[18:27] > 0).astype(int)
        ) == 1

        yaku_update_values = jnp.stack(
            [
                is_pinfu,  # Pinfu(平和)
                is_hand_concealed & (n_double_chow == 1),  # PureDoubleChis(一盃口)
                n_double_chow == 2,  # TwicePureDoubleChis(二盃口)
                has_outside & has_honor & has_tanyao,  # OutsideHand(混全帯么九)
                has_outside & (has_honor == 0),  # TerminalsInAllSets(純全帯么九)
                Yaku.is_pure_straight(all_chow),  # PureStraight(一気通貫)
                Yaku.is_triple_chow(all_chow),  # MixedTripleChis(三色同順)
                Yaku.is_triple_pung(all_pung),  # TriplePons(三色同刻)
                all_chow == 0,  # AllPons(対々和)
                n_concealed_pung == 3,  # ThreeConcealedPons(三暗刻)
                jnp.repeat(n_kan == 3, Yaku.MAX_PATTERNS),  # ThreeKans(三槓子)
            ]
        )  # (13,)

        yaku = jnp.full((Yaku.FAN.shape[1], Yaku.MAX_PATTERNS), False)
        yaku = yaku.at[Yaku.YAKU_UPDATE_INDICES].set(yaku_update_values)

        fan = Yaku.FAN[jnp.where(is_hand_concealed, 1, 0)]
        best_pattern = jnp.argmax(jnp.dot(fan, yaku) * 200 + fu)

        yaku_best = yaku.T[best_pattern]
        fu_best = fu[best_pattern]
        fu_best += -fu_best % 10

        # Seven pairs judgment
        is_mentsu_hand = yaku_best[Yaku.TwicePureDoubleChis] | (jnp.sum(hand == 2) < 7)
        yaku_best = yaku_best * is_mentsu_hand + (1 - is_mentsu_hand) * jnp.full(
            Yaku.FAN.shape[1], False
        ).at[Yaku.SevenPairs].set(jnp.bool_(True))
        fu_best = fu_best * is_mentsu_hand + (1 - is_mentsu_hand) * 25
        has_outside_in_flatten = jnp.any(flatten[OUTSIDE_TILE] > 0)

        yaku_best_update_values = jnp.array(
            [
                jnp.logical_not(
                    (has_honor | has_outside_in_flatten)
                ),  # AllSimples(断么九)
                is_flush & has_honor,  # HalfFlush(混一色)
                is_flush & (has_honor == 0),  # FullFlush(清一色)
                has_tanyao == 0,  # AllTerminalsAndHonors(混老頭)
                flatten[31] >= 3,  # WhiteDragon(白)
                flatten[32] >= 3,  # GreenDragon(發)
                flatten[33] >= 3,  # RedDragon(中)
                jnp.all(flatten[31:34] >= 2)
                & (three_dragons >= 2),  # LittleThreeDragons(小三元)
                flatten[prevalent_wind_tile_type] >= 3,  # PrevelantWind(場風)
                flatten[seat_wind_tile_type] >= 3,  # SeatWind(自風)
                is_hand_concealed & (is_ron == 0),  # FullyConcealedHand(門前清自摸和)
                riichi,  # Riichi(立直)
            ],
            dtype=jnp.bool_,
        )

        yaku_best = yaku_best.at[Yaku.YAKU_BEST_UPDATE_INDICES].set(
            yaku_best_update_values
        )
        yakuman_update_values = jnp.array(
            [
                three_dragons == 3,  # BigThreeDragons(大三元)
                jnp.all(flatten[27:31] >= 2)
                & (four_winds == 3),  # LittleFourWinds(小四喜)
                four_winds == 4,  # BigFourWinds(大四喜)
                jnp.any(nine_gates),  # NineGates(九蓮宝燈)
                jnp.all(hand[KOKUSHI_TILE] > 0)
                & (has_tanyao == 0),  # ThirteenOrphans(国士無双)
                (has_tanyao == 0) & (has_honor == 0),  # AllTerminals(清老頭)
                jnp.all(flatten[0:27] == 0),  # AllHonors(字一色)
                jnp.sum(flatten[ALL_GREEN_TILE]) == 14,  # AllGreen(緑一色)
                jnp.any(n_concealed_pung == 4),  # FourConcealedPons(四暗刻)
                n_kan == 4,  # FourKans(四槓子)
            ],
            dtype=jnp.bool_,
        )
        yakuman = jnp.full(Yaku.FAN.shape[1], False)
        yakuman = yakuman.at[Yaku.YAKUMAN_UPDATE_INDICES].set(yakuman_update_values)
        yakuman_num = jnp.dot(yakuman, Yaku.YAKUMAN)
        return jax.lax.cond(
            jnp.any(yakuman),
            lambda: (
                yakuman.astype(jnp.bool_),
                yakuman_num.astype(jnp.int32),
                jnp.int32(0),
            ),
            lambda: (
                yaku_best.astype(jnp.bool_),
                (jnp.dot(fan, yaku_best) + jnp.dot(flatten, dora)).astype(jnp.int32),
                fu_best.astype(jnp.int32),
            ),
        )

    @staticmethod
    def flatten(hand: Array, melds: Array, n_meld: Array) -> Array:
        """
        Return the hand with the melds added
        """
        addition = jax.vmap(Yaku._calc_addition, in_axes=(0))(melds).sum(axis=0)
        return hand + addition

    @staticmethod
    def _calc_addition(meld: Array) -> Array:
        target, action = Meld.target(meld), Meld.action(meld)
        idx = action - Action.PON + 1
        addition = jnp.zeros(34, dtype=jnp.int8)
        addition_pon_kan = addition.at[target].set(
            (3 + (idx == 0) + (idx == 2)) * Meld.is_pon(meld) + 4 * Meld.is_kan(meld)
        )
        start = target - (
            idx - 3
        )  # idx = 3 => CHI_L, idx = 4 => CHI_M, idx = 5 => CHI_R
        addition_chi = addition.at[jnp.array([start, start + 1, start + 2])].set(
            1
        ) * Meld.is_chi(meld)
        return (addition_pon_kan + addition_chi) * (
            meld != EMPTY_MELD
        )  # When meld is empty, it does not exist

    @staticmethod
    def judge_yakuman(
        hand: Array,
        melds: Array,
        n_meld: Array,
        last_tile: Array,
        riichi: Array,
        is_ron: Array,
        prevalent_wind: Array,
        seat_wind: Array,
        dora: Array,
    ) -> Tuple:
        """
        Judge only yakuman
        """
        # Judge the hand with the last tile added
        hand = Hand.add(hand, last_tile)
        last_tile_type = last_tile
        # Get the dora based on the presence or absence of riichi dora is (2, 34) shape dora[0] is the visible dora, dora[1] includes the hidden dora
        dora = jnp.where(riichi, dora[1], dora[0])

        # Judgment of the necessary conditions for Pinfu and the hand concealed
        is_hand_concealed = jnp.all(
            (Action.is_selfkan(Meld.action(melds)) & (Meld.src(melds) == 0))
            | (melds == EMPTY_MELD)
        )  # When meld is empty, melds==EMPTY_MELD
        n_kan = jnp.sum(Meld.is_kan(melds) & (melds != EMPTY_MELD))
        n_closed_kan = jnp.sum(Meld.is_closed_kan(melds) & (melds != EMPTY_MELD))

        n_concealed_pung = jnp.full(Yaku.MAX_PATTERNS, 0)
        nine_gates = jnp.full(Yaku.MAX_PATTERNS, False)
        # Concealed kans are also counted as pungs
        n_concealed_pung += (
            jnp.sum(hand >= 3) - (is_ron & (hand[last_tile_type] >= 3)) + n_closed_kan
        )

        codes = (
            (hand[:27].astype(int) * powers_of_5_full).reshape(3, 9).sum(axis=1)
        )  # (3,)

        nine_gates = Yaku.nine_gates(codes)

        # Combine the melds and the hand
        flatten = Yaku.flatten(hand, melds, n_meld)  # (34,)

        four_winds = jnp.sum(flatten[27:31] >= 3)
        three_dragons = jnp.sum(flatten[31:34] >= 3)

        has_tanyao = jnp.any(
            jax.vmap(one_hot_at, in_axes=(None, 0))(flatten, TANYAO_TILE)
        )
        has_honor = jnp.any(flatten[27:] > 0)

        yakuman_update_values = jnp.array(
            [
                three_dragons == 3,  # BigThreeDragons(大三元)
                jnp.all(flatten[27:31] >= 2)
                & (four_winds == 3),  # LittleFourWinds(小四喜)
                four_winds == 4,  # BigFourWinds(大四喜)
                jnp.any(nine_gates),  # NineGates(九蓮宝燈)
                jnp.all(hand[KOKUSHI_TILE] > 0)
                & (has_tanyao == 0),  # ThirteenOrphans(国士無双)
                (has_tanyao == 0) & (has_honor == 0),  # AllTerminals(清老頭)
                jnp.all(flatten[0:27] == 0),  # AllHonors(字一色)
                jnp.sum(flatten[ALL_GREEN_TILE]) == 14,  # AllGreen(緑一色)
                jnp.any(n_concealed_pung == 4)
                & is_hand_concealed,  # FourConcealedPons(四暗刻)
                n_kan == 4,  # FourKans(四槓子)
            ],
            dtype=jnp.bool_,
        )
        yakuman = jnp.full(Yaku.FAN.shape[1], False)
        yakuman = yakuman.at[Yaku.YAKUMAN_UPDATE_INDICES].set(yakuman_update_values)
        yakuman_num = jnp.dot(yakuman, Yaku.YAKUMAN)
        return yakuman, jnp.int32(yakuman_num), jnp.int32(0)
