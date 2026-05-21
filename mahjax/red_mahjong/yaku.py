from __future__ import annotations

from pathlib import Path
from typing import Tuple
import importlib.resources as resources

import jax
import jax.numpy as jnp
import numpy as np

from .constants import DORA_ARRAY
from .hand import Hand
from .meld import EMPTY_MELD, Meld
from .tile import Tile
from .types import Array


def load_yaku_cache() -> jnp.ndarray:
    with resources.as_file(resources.files("mahjax._src.cache").joinpath("yaku_cache.npz")) as path:
        with np.load(path, allow_pickle=False) as data:
            return jnp.asarray(data["data"], dtype=jnp.uint32)


WIND_TILE = jnp.array([27, 28, 29, 30], dtype=jnp.int8)
OUTSIDE_TILE = jnp.array([0, 8, 9, 17, 18, 26], dtype=jnp.int8)
TANYAO_TILE = jnp.array(
    [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25],
    dtype=jnp.int8,
)
KOKUSHI_TILE = jnp.array([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33], dtype=jnp.int8)
ALL_GREEN_TILE = jnp.array([19, 20, 21, 23, 25, 32], dtype=jnp.int8)
SCORES = jnp.array(
    [2000, 2000, 3000, 3000, 4000, 4000, 4000, 6000, 6000, 8000, 8000, 8000],
    dtype=jnp.int32,
)
NUM_TENHOU_YAKU = 52


powers_of_5_full = jnp.concatenate([5 ** jnp.arange(8, -1, -1)] * 3)


def _one_hot_at(x: Array, idx: Array) -> Array:
    return x @ jax.nn.one_hot(idx, x.shape[0])


def _dora_array_from_state(state: object) -> jnp.ndarray:
    def update_dora_counts(dora_counts: Array, dora_indicator: Array) -> Array:
        is_valid = dora_indicator != -1
        dora_tile_type = Tile.to_tile_type(dora_indicator)
        return dora_counts.at[DORA_ARRAY[dora_tile_type]].add(is_valid)

    dora_counts = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.int8)
    dora_counts = jax.vmap(update_dora_counts, in_axes=(None, 0))(
        dora_counts, state.round_state.dora_indicators
    ).sum(axis=0)
    ura_dora_counts = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.int8)
    ura_dora_counts = jax.vmap(update_dora_counts, in_axes=(None, 0))(
        ura_dora_counts, state.round_state.ura_dora_indicators
    ).sum(axis=0)
    return jnp.stack([dora_counts, ura_dora_counts], axis=0)


class _Internal:
    FullyConcealedHand = 0
    Riichi = 1
    Ippatsu = 2
    RobbingKan = 3
    DrawAfterKan = 4
    BottomOfTheSea = 5
    BottomOfTheRiver = 6
    Pinfu = 7
    AllSimples = 8
    PureDoubleChis = 9
    SeatWindEast = 10
    SeatWindSouth = 11
    SeatWindWest = 12
    SeatWindNorth = 13
    PrevalentWindEast = 14
    PrevalentWindSouth = 15
    PrevalentWindWest = 16
    PrevalentWindNorth = 17
    WhiteDragon = 18
    GreenDragon = 19
    RedDragon = 20
    DoubleRiichi = 21
    SevenPairs = 22
    OutsideHand = 23
    PureStraight = 24
    MixedTripleChis = 25
    TriplePons = 26
    ThreeKans = 27
    AllPons = 28
    ThreeConcealedPons = 29
    LittleThreeDragons = 30
    AllTerminalsAndHonors = 31
    TwicePureDoubleChis = 32
    TerminalsInAllSets = 33
    HalfFlush = 34
    FullFlush = 35
    Renhou = 36
    BlessingOfHeaven = 37
    BlessingOfEarth = 38
    BigThreeDragons = 39
    FourConcealedPons = 40
    CompletedFourConcealedPons = 41
    AllHonors = 42
    AllGreen = 43
    AllTerminals = 44
    NineGates = 45
    PureNineGates = 46
    ThirteenOrphans = 47
    CompletedThirteenOrphans = 48
    BigFourWinds = 49
    LittleFourWinds = 50
    FourKans = 51
    MAX_PATTERNS = 3

    _FAN_OPEN = jnp.zeros((NUM_TENHOU_YAKU,), dtype=jnp.int32)
    _FAN_OPEN = _FAN_OPEN.at[OutsideHand].set(1)
    _FAN_OPEN = _FAN_OPEN.at[TerminalsInAllSets].set(2)
    _FAN_OPEN = _FAN_OPEN.at[PureStraight].set(1)
    _FAN_OPEN = _FAN_OPEN.at[MixedTripleChis].set(1)
    _FAN_OPEN = _FAN_OPEN.at[TriplePons].set(2)
    _FAN_OPEN = _FAN_OPEN.at[AllPons].set(2)
    _FAN_OPEN = _FAN_OPEN.at[ThreeConcealedPons].set(2)
    _FAN_OPEN = _FAN_OPEN.at[ThreeKans].set(2)
    _FAN_OPEN = _FAN_OPEN.at[SevenPairs].set(2)
    _FAN_OPEN = _FAN_OPEN.at[AllSimples].set(1)
    _FAN_OPEN = _FAN_OPEN.at[HalfFlush].set(2)
    _FAN_OPEN = _FAN_OPEN.at[FullFlush].set(5)
    _FAN_OPEN = _FAN_OPEN.at[AllTerminalsAndHonors].set(2)
    _FAN_OPEN = _FAN_OPEN.at[LittleThreeDragons].set(2)
    _FAN_OPEN = _FAN_OPEN.at[WhiteDragon].set(1)
    _FAN_OPEN = _FAN_OPEN.at[GreenDragon].set(1)
    _FAN_OPEN = _FAN_OPEN.at[RedDragon].set(1)
    _FAN_OPEN = _FAN_OPEN.at[SeatWindEast].set(1).at[SeatWindSouth].set(1).at[SeatWindWest].set(1).at[SeatWindNorth].set(1)
    _FAN_OPEN = _FAN_OPEN.at[PrevalentWindEast].set(1).at[PrevalentWindSouth].set(1).at[PrevalentWindWest].set(1).at[PrevalentWindNorth].set(1)
    _FAN_CLOSED = _FAN_OPEN
    _FAN_CLOSED = _FAN_CLOSED.at[FullyConcealedHand].set(1)
    _FAN_CLOSED = _FAN_CLOSED.at[Riichi].set(1)
    _FAN_CLOSED = _FAN_CLOSED.at[Pinfu].set(1)
    _FAN_CLOSED = _FAN_CLOSED.at[PureDoubleChis].set(1)
    _FAN_CLOSED = _FAN_CLOSED.at[TwicePureDoubleChis].set(3)
    _FAN_CLOSED = _FAN_CLOSED.at[OutsideHand].set(2)
    _FAN_CLOSED = _FAN_CLOSED.at[TerminalsInAllSets].set(3)
    _FAN_CLOSED = _FAN_CLOSED.at[PureStraight].set(2)
    _FAN_CLOSED = _FAN_CLOSED.at[MixedTripleChis].set(2)
    _FAN_CLOSED = _FAN_CLOSED.at[HalfFlush].set(3)
    _FAN_CLOSED = _FAN_CLOSED.at[FullFlush].set(6)
    FAN = jnp.stack([_FAN_OPEN, _FAN_CLOSED], axis=0)

    YAKUMAN = jnp.zeros((NUM_TENHOU_YAKU,), dtype=jnp.int32)
    YAKUMAN = YAKUMAN.at[BigThreeDragons].set(1)
    YAKUMAN = YAKUMAN.at[FourConcealedPons].set(1)
    YAKUMAN = YAKUMAN.at[CompletedFourConcealedPons].set(1)
    YAKUMAN = YAKUMAN.at[AllHonors].set(1)
    YAKUMAN = YAKUMAN.at[AllGreen].set(1)
    YAKUMAN = YAKUMAN.at[AllTerminals].set(1)
    YAKUMAN = YAKUMAN.at[NineGates].set(1)
    YAKUMAN = YAKUMAN.at[PureNineGates].set(1)
    YAKUMAN = YAKUMAN.at[ThirteenOrphans].set(1)
    YAKUMAN = YAKUMAN.at[CompletedThirteenOrphans].set(1)
    YAKUMAN = YAKUMAN.at[BigFourWinds].set(2)
    YAKUMAN = YAKUMAN.at[LittleFourWinds].set(1)
    YAKUMAN = YAKUMAN.at[FourKans].set(1)

    YAKU_UPDATE_INDICES = jnp.array(
        [
            Pinfu,
            PureDoubleChis,
            TwicePureDoubleChis,
            OutsideHand,
            TerminalsInAllSets,
            PureStraight,
            MixedTripleChis,
            TriplePons,
            AllPons,
            ThreeConcealedPons,
            ThreeKans,
        ],
        dtype=jnp.int32,
    )
    YAKU_BEST_UPDATE_INDICES = jnp.array(
        [
            AllSimples,
            HalfFlush,
            FullFlush,
            AllTerminalsAndHonors,
            WhiteDragon,
            GreenDragon,
            RedDragon,
            LittleThreeDragons,
            FullyConcealedHand,
            Riichi,
        ],
        dtype=jnp.int32,
    )
    YAKUMAN_UPDATE_INDICES = jnp.array(
        [
            BigThreeDragons,
            BigFourWinds,
            LittleFourWinds,
            NineGates,
            ThirteenOrphans,
            AllTerminals,
            AllHonors,
            AllGreen,
            FourConcealedPons,
            FourKans,
        ],
        dtype=jnp.int32,
    )


class Yaku:
    CACHE = load_yaku_cache()
    MAX_PATTERNS = _Internal.MAX_PATTERNS

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
        pat = 0b1000000001000000001
        out = (chow & pat) == pat
        for s in range(1, 8):
            out = out | ((chow >> s & pat) == pat)
        return out

    @staticmethod
    def is_triple_pung(pung: Array) -> Array:
        pat = 0b1000000001000000001
        out = (pung & pat) == pat
        for s in range(1, 9):
            out = out | ((pung >> s & pat) == pat)
        return out

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
        chow = Yaku.chow(code)
        pung = Yaku.pung(code)
        open_end = (chow ^ (chow & 1)) << 2 | (chow ^ (chow & 0b1000000))
        in_range = suit == last_tile_type // 9
        pos = last_tile_type % 9
        is_pinfu = is_pinfu & (((in_range == 0) | (((open_end >> pos) & 1) == 1)) & (pung == 0))
        has_outside = has_outside & (Yaku.outside(code) == 1)
        n_double_chow = n_double_chow + Yaku.n_double_chow(code)
        all_chow = all_chow | (chow << (9 * suit))
        all_pung = all_pung | (pung << (9 * suit))
        n_pung = Yaku.n_pung(code)
        chow_range = chow | (chow << 1) | (chow << 2)
        loss = is_ron & in_range & (((chow_range >> pos) & 1) == 0) & (((pung >> pos) & 1) == 1)
        n_concealed_pung = n_concealed_pung + n_pung - loss
        nine_gates = nine_gates | (Yaku.nine_gates(code) == 1)
        outside_pung = pung & 0b100000001
        n_outside_pung = (outside_pung & 1) + ((outside_pung >> 8) & 1)
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
        outside_loss = loss & ((outside_pung >> pos) & 1)
        fu = fu + 4 * n_pung + 4 * n_outside_pung - 2 * loss - 2 * outside_loss + 2 * strong
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
    def _chi_index(action: Array) -> Array:
        return jnp.where(
            (action == 78) | (action == 79),
            jnp.int32(0),
            jnp.where(
                (action == 80) | (action == 81),
                jnp.int32(1),
                jnp.where((action == 82) | (action == 83), jnp.int32(2), jnp.int32(-1)),
            ),
        )

    @staticmethod
    def _calc_addition(meld: Array) -> Array:
        target = Meld.target(meld)
        action = Meld.action(meld)
        addition = jnp.zeros(34, dtype=jnp.int8)
        addition = addition.at[target].set(3 * Meld.is_pon(meld).astype(jnp.int8) + 4 * Meld.is_kan(meld).astype(jnp.int8))
        chi_idx = Yaku._chi_index(action)
        start = jnp.clip(target - chi_idx, 0, 31)
        is_chi = Meld.is_chi(meld).astype(jnp.int8)
        addition = addition.at[start].set(addition[start] + is_chi)
        addition = addition.at[start + 1].set(addition[start + 1] + is_chi)
        addition = addition.at[start + 2].set(addition[start + 2] + is_chi)
        return addition * (meld != EMPTY_MELD).astype(jnp.int8)

    @staticmethod
    def flatten(hand: Array, melds: Array, n_meld: Array) -> Array:
        del n_meld
        addition = jax.vmap(Yaku._calc_addition)(melds).sum(axis=0)
        return Hand.to_34(hand) + addition

    @staticmethod
    def score(fan: Array, fu: Array) -> Array:
        raw = fu * jnp.left_shift(1, fan + 2)
        return jax.lax.cond(
            fu == 0,
            lambda: 8000 * fan,
            lambda: jnp.where(raw < 2000, raw, SCORES[jnp.clip(fan - 4, 0, 11)]),
        )

    @staticmethod
    def judge_hand_related(
        hand: Array,
        melds: Array,
        n_meld: Array,
        last_tile: Array,
        riichi: Array,
        is_ron: Array,
        prevalent_wind: Array,
        seat_wind: Array,
        dora: Array,
    ) -> Tuple[Array, Array, Array]:
        hand = Hand.add(hand, last_tile)
        red_fan = jnp.int32(0)
        if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED:
            red_fan = jnp.sum(hand[Tile.NUM_TILE_TYPE:]).astype(jnp.int32) + jnp.sum(Meld.contains_red(melds)).astype(jnp.int32)
            hand = Hand.to_34(hand)
            last_tile_type = Tile.to_tile_type(last_tile)
        else:
            last_tile_type = last_tile
        dora = jnp.where(riichi, dora.sum(axis=0), dora[0])
        seat_wind_tile_type = WIND_TILE[seat_wind]
        prevalent_wind_tile_type = WIND_TILE[prevalent_wind]

        is_hand_concealed = jnp.all(Meld.is_closed_kan(melds) | (melds == EMPTY_MELD))
        is_pinfu = jnp.full(
            Yaku.MAX_PATTERNS,
            is_hand_concealed
            & (n_meld == 0)
            & (last_tile_type < 27)
            & jnp.all(hand[27:31] < 3)
            & (hand[seat_wind_tile_type] == 0)
            & (hand[prevalent_wind_tile_type] == 0)
            & jnp.all(hand[31:34] == 0),
        )
        has_outside = jnp.full(
            Yaku.MAX_PATTERNS,
            jnp.all(Meld.has_outside(melds) | (melds == EMPTY_MELD)),
        )
        meld_chow_bits = jax.lax.reduce(
            jax.vmap(Meld.chow)(melds).astype(jnp.int32),
            jnp.int32(0), jax.lax.bitwise_or, (0,),
        )
        meld_pung_bits = jax.lax.reduce(
            jax.vmap(Meld.suited_pung)(melds).astype(jnp.int32),
            jnp.int32(0), jax.lax.bitwise_or, (0,),
        )
        all_chow = jnp.full(Yaku.MAX_PATTERNS, meld_chow_bits, dtype=jnp.int32)
        all_pung = jnp.full(Yaku.MAX_PATTERNS, meld_pung_bits, dtype=jnp.int32)
        n_kan = jnp.sum(Meld.is_kan(melds) & (melds != EMPTY_MELD))
        n_closed_kan = jnp.sum(Meld.is_closed_kan(melds) & (melds != EMPTY_MELD))
        n_concealed_pung = (
            jnp.sum(hand[27:] >= 3)
            - (is_ron & (last_tile_type >= 27) & (hand[last_tile_type] >= 3))
            + n_closed_kan
        )

        honor_tile_types = jnp.arange(27, 34, dtype=jnp.int32)
        ron_penalty = (is_ron & (honor_tile_types == last_tile_type)).astype(jnp.int32)
        # Yakuhai pair fu: 役牌対子 2 符; 連風（場風＝自風の同じ牌の対子）は 4 符（2+2 を重ねない）
        seat_tt = seat_wind_tile_type
        prev_tt = prevalent_wind_tile_type
        seat_pair = hand[seat_tt] == 2
        prev_pair = hand[prev_tt] == 2
        renfu_pair = (seat_tt == prev_tt) & seat_pair
        wind_pair_fu = jnp.where(renfu_pair, jnp.int32(4), jnp.int32(0))
        wind_pair_fu = wind_pair_fu + jnp.where((~renfu_pair) & seat_pair, jnp.int32(2), jnp.int32(0))
        wind_pair_fu = wind_pair_fu + jnp.where((~renfu_pair) & prev_pair, jnp.int32(2), jnp.int32(0))
        fu = jnp.full(
            Yaku.MAX_PATTERNS,
            2 * (is_ron == 0)
            + jnp.sum(Meld.fu(melds))
            + wind_pair_fu
            + jnp.any(hand[31:] == 2).astype(jnp.int32) * 2
            + jnp.sum((hand[27:34] == 3).astype(jnp.int32) * 4 * (2 - ron_penalty))
            + ((27 <= last_tile_type) & (hand[last_tile_type] == 2)).astype(jnp.int32),
            dtype=jnp.int32,
        )
        codes = (hand[:27].astype(jnp.int32) * powers_of_5_full).reshape(3, 9).sum(axis=1)

        def _update_yaku(suit: int, tpl: Tuple) -> Tuple:
            code = codes[suit]
            return Yaku.update(
                tpl[0], tpl[1], tpl[2], tpl[3], tpl[4], tpl[5], tpl[6], tpl[7],
                code, jnp.int32(suit), last_tile_type, is_ron,
            )

        init = (
            is_pinfu,
            has_outside,
            jnp.zeros((Yaku.MAX_PATTERNS,), dtype=jnp.int32),
            all_chow,
            all_pung,
            jnp.full(Yaku.MAX_PATTERNS, n_concealed_pung),
            jnp.full(Yaku.MAX_PATTERNS, False),
            fu,
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
        ) = jax.lax.fori_loop(0, 3, _update_yaku, init)

        fu = fu * (is_pinfu == 0)
        fu = fu + 20 + 10 * (is_hand_concealed & is_ron)
        fu = fu + 10 * ((~is_hand_concealed) & (fu == 20))

        flatten = Yaku.flatten(hand, melds, n_meld)
        four_winds = jnp.sum(flatten[27:31] >= 3)
        three_dragons = jnp.sum(flatten[31:34] >= 3)
        has_tanyao = jnp.any(jax.vmap(lambda i: _one_hot_at(flatten, TANYAO_TILE[i]))(jnp.arange(21)))
        has_honor = jnp.any(flatten[27:] > 0)
        is_flush = (
            jnp.any(flatten[0:9] > 0).astype(jnp.int32)
            + jnp.any(flatten[9:18] > 0).astype(jnp.int32)
            + jnp.any(flatten[18:27] > 0).astype(jnp.int32)
        ) == 1

        yaku_update_values = jnp.stack(
            [
                is_pinfu,
                is_hand_concealed & (n_double_chow == 1),
                n_double_chow == 2,
                has_outside & has_honor & has_tanyao,
                has_outside & (has_honor == 0),
                Yaku.is_pure_straight(all_chow),
                Yaku.is_triple_chow(all_chow),
                Yaku.is_triple_pung(all_pung),
                all_chow == 0,
                n_concealed_pung == 3,
                jnp.repeat(n_kan == 3, Yaku.MAX_PATTERNS),
            ]
        )
        yaku = jnp.zeros((NUM_TENHOU_YAKU, Yaku.MAX_PATTERNS), dtype=jnp.bool_)
        yaku = yaku.at[_Internal.YAKU_UPDATE_INDICES, :].set(yaku_update_values)

        fan_row = _Internal.FAN[jnp.where(is_hand_concealed, 1, 0)]
        best_pattern = jnp.argmax(jnp.dot(fan_row, yaku) * 200 + fu)
        yaku_best = yaku[:, best_pattern]
        fu_best = fu[best_pattern]
        fu_best = fu_best + (-fu_best % 10)

        is_mentsu_hand = yaku_best[_Internal.TwicePureDoubleChis] | (jnp.sum(hand == 2) < 7)
        yaku_best = jnp.where(
            is_mentsu_hand,
            yaku_best,
            jnp.zeros(NUM_TENHOU_YAKU, dtype=jnp.bool_).at[_Internal.SevenPairs].set(True),
        )
        fu_best = jnp.where(is_mentsu_hand, fu_best, 25)
        has_outside_in_flatten = jnp.any(flatten[OUTSIDE_TILE] > 0)

        yaku_best_update = jnp.array(
            [
                ~(has_honor | has_outside_in_flatten),
                is_flush & has_honor,
                is_flush & (has_honor == 0),
                has_tanyao == 0,
                flatten[31] >= 3,
                flatten[32] >= 3,
                flatten[33] >= 3,
                jnp.all(flatten[31:34] >= 2) & (three_dragons >= 2),
                is_hand_concealed & (is_ron == 0),
                riichi,
            ],
            dtype=jnp.bool_,
        )
        yaku_best = yaku_best.at[_Internal.YAKU_BEST_UPDATE_INDICES].set(yaku_best_update)
        yaku_best = yaku_best.at[_Internal.PrevalentWindEast + prevalent_wind].set(flatten[prevalent_wind_tile_type] >= 3)
        yaku_best = yaku_best.at[_Internal.SeatWindEast + seat_wind].set(flatten[seat_wind_tile_type] >= 3)

        win_tile_count = hand[last_tile_type]
        four_concealed_tsumo = jnp.any((n_concealed_pung == 4) & (win_tile_count >= 3) & (is_ron == 0))
        four_concealed_single = jnp.any((n_concealed_pung == 4) & (win_tile_count == 2))
        yakuman_update_values = jnp.array(
            [
                three_dragons == 3,
                four_winds == 4,
                jnp.all(flatten[27:31] >= 2) & (four_winds == 3),
                jnp.any(nine_gates),
                jnp.all(hand[KOKUSHI_TILE] > 0) & (has_tanyao == 0),
                (has_tanyao == 0) & (has_honor == 0),
                jnp.all(flatten[0:27] == 0),
                jnp.sum(flatten[ALL_GREEN_TILE]) == 14,
                four_concealed_tsumo,
                n_kan == 4,
            ],
            dtype=jnp.bool_,
        )
        yakuman = jnp.zeros(NUM_TENHOU_YAKU, dtype=jnp.bool_)
        yakuman = yakuman.at[_Internal.YAKUMAN_UPDATE_INDICES].set(yakuman_update_values)
        yakuman = yakuman.at[_Internal.CompletedFourConcealedPons].set(four_concealed_single)
        yakuman_num = jnp.dot(yakuman.astype(jnp.int32), _Internal.YAKUMAN)

        def _ret_yakuman() -> Tuple[Array, Array, Array]:
            return yakuman, yakuman_num.astype(jnp.int32), jnp.int32(0)

        def _ret_normal() -> Tuple[Array, Array, Array]:
            fan_val = jnp.dot(_Internal.FAN[jnp.where(is_hand_concealed, 1, 0)], yaku_best.astype(jnp.int32)) + jnp.dot(flatten, dora) + red_fan
            return yaku_best, fan_val.astype(jnp.int32), fu_best.astype(jnp.int32)

        return jax.lax.cond(jnp.any(yakuman), _ret_yakuman, _ret_normal)

    @staticmethod
    def judge(
        hand: Array,
        is_ron: Array,
        player: Array,
        rs: object,
    ) -> Tuple[Array, Array, Array]:
        p = jnp.int32(player)
        melds = rs.players.melds[p]
        n_meld = rs.players.meld_counts[p]
        last_tile = jnp.where(is_ron, rs.round_state.target, rs.round_state.last_draw)
        riichi = rs.players.riichi[p]
        prevalent_wind = jnp.int32(rs.round_state.round // 4)
        seat_wind = jnp.int32(rs.round_state.seat_wind[p])
        dora = _dora_array_from_state(rs)
        return Yaku.judge_hand_related(
            hand=hand,
            melds=melds,
            n_meld=n_meld,
            last_tile=last_tile,
            riichi=riichi,
            is_ron=is_ron,
            prevalent_wind=prevalent_wind,
            seat_wind=seat_wind,
            dora=dora,
        )

    @staticmethod
    def judge_other(
        *,
        is_ron: Array,
        is_riichi: Array,
        is_ippatsu: Array = False,
        is_robbing_kan: Array = False,
        is_after_kan: Array = False,
        is_bottom_of_the_sea: Array = False,
        is_bottom_of_the_river: Array = False,
        is_double_riichi: Array = False,
        is_blessing_of_heaven: Array = False,
        is_blessing_of_earth: Array = False,
    ) -> Tuple[Array, Array, Array, Array]:
        is_ron = jnp.bool_(is_ron)
        is_riichi = jnp.bool_(is_riichi)
        is_ippatsu = jnp.bool_(is_ippatsu) & is_riichi
        is_robbing_kan = jnp.bool_(is_robbing_kan) & is_ron
        is_after_kan = jnp.bool_(is_after_kan) & (~is_ron)
        is_bottom_of_the_sea = jnp.bool_(is_bottom_of_the_sea) & (~is_ron)
        is_bottom_of_the_river = jnp.bool_(is_bottom_of_the_river) & is_ron
        is_double_riichi = jnp.bool_(is_double_riichi) & is_riichi
        is_blessing_of_heaven = jnp.bool_(is_blessing_of_heaven) & (~is_ron)
        is_blessing_of_earth = jnp.bool_(is_blessing_of_earth) & (~is_ron)

        normal = jnp.zeros(NUM_TENHOU_YAKU, dtype=jnp.bool_)
        normal = normal.at[_Internal.Ippatsu].set(is_ippatsu)
        normal = normal.at[_Internal.RobbingKan].set(is_robbing_kan)
        normal = normal.at[_Internal.DrawAfterKan].set(is_after_kan)
        normal = normal.at[_Internal.BottomOfTheSea].set(is_bottom_of_the_sea)
        normal = normal.at[_Internal.BottomOfTheRiver].set(is_bottom_of_the_river)
        normal = normal.at[_Internal.DoubleRiichi].set(is_double_riichi)

        yakuman = jnp.zeros(NUM_TENHOU_YAKU, dtype=jnp.bool_)
        yakuman = yakuman.at[_Internal.BlessingOfHeaven].set(is_blessing_of_heaven)
        yakuman = yakuman.at[_Internal.BlessingOfEarth].set(is_blessing_of_earth)

        normal_fan = jnp.sum(normal.astype(jnp.int32))
        yakuman_num = jnp.sum(yakuman.astype(jnp.int32))
        return normal, yakuman, normal_fan.astype(jnp.int32), yakuman_num.astype(jnp.int32)

    @staticmethod
    def judge_yakuman(
        hand: Array,
        is_ron: Array,
        player: Array,
        rs: object,
    ) -> Tuple[Array, Array, Array]:
        p = jnp.int32(player)
        melds = rs.players.melds[p]
        last_tile = jnp.where(is_ron, rs.round_state.target, rs.round_state.last_draw)
        hand = Hand.add(hand, last_tile)
        if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED:
            hand = Hand.to_34(hand)
            last_tile_type = Tile.to_tile_type(last_tile)
        else:
            last_tile_type = last_tile
        n_kan = jnp.sum(Meld.is_kan(melds) & (melds != EMPTY_MELD))
        n_closed_kan = jnp.sum(Meld.is_closed_kan(melds) & (melds != EMPTY_MELD))
        n_concealed_pung = jnp.sum(hand >= 3) - (is_ron & (hand[last_tile_type] >= 3)) + n_closed_kan
        codes = (hand[:27].astype(jnp.int32) * powers_of_5_full).reshape(3, 9).sum(axis=1)
        nine_gates = jnp.any(jax.vmap(Yaku.nine_gates)(codes))
        flatten = Yaku.flatten(hand, melds, jnp.int8(0))
        four_winds = jnp.sum(flatten[27:31] >= 3)
        three_dragons = jnp.sum(flatten[31:34] >= 3)
        has_tanyao = jnp.any(jax.vmap(lambda i: _one_hot_at(flatten, TANYAO_TILE[i]))(jnp.arange(21)))
        has_honor = jnp.any(flatten[27:] > 0)
        win_tile_count = hand[last_tile_type]
        four_concealed_tsumo = (n_concealed_pung == 4) & (win_tile_count >= 3) & (is_ron == 0)
        four_concealed_single = (n_concealed_pung == 4) & (win_tile_count == 2)
        yakuman_update_values = jnp.array(
            [
                three_dragons == 3,
                four_winds == 4,
                jnp.all(flatten[27:31] >= 2) & (four_winds == 3),
                nine_gates,
                jnp.all(hand[KOKUSHI_TILE] > 0) & (has_tanyao == 0),
                (has_tanyao == 0) & (has_honor == 0),
                jnp.all(flatten[0:27] == 0),
                jnp.sum(flatten[ALL_GREEN_TILE]) == 14,
                four_concealed_tsumo,
                n_kan == 4,
            ],
            dtype=jnp.bool_,
        )
        yakuman = jnp.zeros(NUM_TENHOU_YAKU, dtype=jnp.bool_)
        yakuman = yakuman.at[_Internal.YAKUMAN_UPDATE_INDICES].set(yakuman_update_values)
        yakuman = yakuman.at[_Internal.CompletedFourConcealedPons].set(four_concealed_single)
        yakuman_num = jnp.dot(yakuman.astype(jnp.int32), _Internal.YAKUMAN)
        return yakuman, yakuman_num.astype(jnp.int32), jnp.int32(0)


for _name in dir(_Internal):
    if _name.startswith('_'):
        continue
    _value = getattr(_Internal, _name)
    if isinstance(_value, (int, jnp.integer)):
        setattr(Yaku, _name, int(_value))


__all__ = ['NUM_TENHOU_YAKU', 'Yaku']
