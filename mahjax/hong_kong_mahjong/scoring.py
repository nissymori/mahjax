"""Hong Kong Old Style faan calculation and settlement."""

from dataclasses import dataclass
from typing import Dict, Tuple

import jax.numpy as jnp

from mahjax._src.types import Array
from mahjax.hong_kong_mahjong.meld import EMPTY_MELD, Meld
from mahjax.hong_kong_mahjong.rules import HKOS_V1, Rules
from mahjax.no_red_mahjong.yaku import Yaku

# Values intentionally live in one table so variants can be reviewed and changed
# without touching the environment transition code.
HKOS_PATTERN_FAAN: Dict[int, int] = {
    Yaku.Pinfu: 1,
    Yaku.AllPons: 3,
    Yaku.ThreeConcealedPons: 3,
    Yaku.ThreeKans: 3,
    Yaku.AllSimples: 1,
    Yaku.HalfFlush: 3,
    Yaku.FullFlush: 7,
    Yaku.AllTerminalsAndHonors: 10,
    Yaku.LittleThreeDragons: 5,
    Yaku.WhiteDragon: 1,
    Yaku.GreenDragon: 1,
    Yaku.RedDragon: 1,
    Yaku.PrevelantWind: 1,
    Yaku.SeatWind: 1,
}

LIMIT_INDICES = jnp.array(
    [
        Yaku.BigThreeDragons,
        Yaku.LittleFourWinds,
        Yaku.BigFourWinds,
        Yaku.NineGates,
        Yaku.ThirteenOrphans,
        Yaku.AllTerminals,
        Yaku.AllHonors,
        Yaku.AllGreen,
        Yaku.FourConcealedPons,
        Yaku.FourKans,
    ],
    dtype=jnp.int32,
)

PATTERN_VALUES = jnp.zeros(Yaku.FAN.shape[1], dtype=jnp.int32)
for _pattern, _value in HKOS_PATTERN_FAAN.items():
    PATTERN_VALUES = PATTERN_VALUES.at[_pattern].set(_value)


@dataclass(frozen=True)
class FaanResult:
    faan: Array
    patterns: Array
    flower_faan: Array
    is_limit: Array

    @property
    def can_win(self) -> Array:
        return self.faan >= HKOS_V1.minimum_faan


class HongKongScoring:
    """Score complete hands using the HKOS_V1 pattern and payment tables."""

    @staticmethod
    def flower_faan(flowers: Array, seat: Array) -> Array:
        """Score flowers by the traditional seat-matching Hong Kong rule.

        A flower matching the player's seat scores one faan. No flowers scores
        one faan. A complete flower or season set scores two faan instead of its
        individual seat match; all eight flowers therefore score four faan.
        """
        count = flowers.sum()
        flower_counts = flowers.astype(jnp.int32)
        own = flower_counts[seat] + flower_counts[seat + 4]
        first_set = flowers[:4].all()
        second_set = flowers[4:].all()
        sets_score = 2 * (first_set.astype(jnp.int32) + second_set.astype(jnp.int32))
        own_not_in_set = (
            own
            - flower_counts[seat] * first_set.astype(jnp.int32)
            - flower_counts[seat + 4] * second_set.astype(jnp.int32)
        )
        return jnp.where(count == 0, 1, sets_score + own_not_in_set)

    @staticmethod
    def judge(
        hand: Array,
        melds: Array,
        n_meld: Array,
        winning_tile: Array,
        flowers: Array,
        prevalent_wind: Array,
        seat_wind: Array,
        *,
        is_self_draw: Array,
        is_robbing_kong: Array = jnp.bool_(False),
        is_kong_replacement: Array = jnp.bool_(False),
        is_last_tile: Array = jnp.bool_(False),
        is_heavenly_hand: Array = jnp.bool_(False),
        is_earthly_hand: Array = jnp.bool_(False),
        rules: Rules = HKOS_V1,
    ) -> FaanResult:
        empty_dora = jnp.zeros((2, 34), dtype=jnp.int8)
        patterns, _, _ = Yaku.judge(
            hand,
            melds,
            n_meld,
            winning_tile,
            jnp.bool_(False),
            ~is_self_draw,
            prevalent_wind,
            seat_wind,
            empty_dora,
        )
        is_limit = patterns[LIMIT_INDICES].any() | is_heavenly_hand | is_earthly_hand
        pattern_faan = jnp.dot(patterns.astype(jnp.int32), PATTERN_VALUES)
        flower_faan = HongKongScoring.flower_faan(flowers, seat_wind)
        is_concealed = jnp.all((melds == EMPTY_MELD) | Meld.is_closed_kan(melds))
        concealed_faan = is_concealed.astype(jnp.int32)
        bonus = (
            is_self_draw.astype(jnp.int32)
            + is_robbing_kong.astype(jnp.int32)
            + is_kong_replacement.astype(jnp.int32)
            + is_last_tile.astype(jnp.int32)
        )
        raw = pattern_faan + flower_faan + concealed_faan + bonus
        faan = jnp.where(is_limit & rules.special_limit_hands, rules.maximum_faan, raw)
        faan = jnp.minimum(faan, rules.maximum_faan).astype(jnp.int32)
        return FaanResult(faan=faan, patterns=patterns, flower_faan=flower_faan, is_limit=is_limit)

    @staticmethod
    def payout(faan: Array, rules: Rules = HKOS_V1) -> Array:
        """Return the capped half-spicy HKOS v1 base payment."""
        capped = jnp.clip(faan, rules.minimum_faan, rules.maximum_faan)
        table = jnp.array([8, 16, 24, 32, 48, 64, 96, 128], dtype=jnp.int32)
        return table[capped - rules.minimum_faan]

    @staticmethod
    def settle(
        winner: Array,
        discarder: Array,
        faan: Array,
        is_self_draw: Array,
        dealer: Array = jnp.int8(-1),
        rules: Rules = HKOS_V1,
    ) -> Tuple[Array, Array]:
        """Return zero-sum rewards and the per-player payment amount."""
        amount = HongKongScoring.payout(faan, rules)
        players = jnp.arange(4)
        winner_is_dealer = winner == dealer
        payer_multiplier = jnp.where(winner_is_dealer | (players == dealer), 2, 1)
        self_draw_payments = jnp.where(players == winner, 0, amount * payer_multiplier)
        discard_multiplier = jnp.where(winner_is_dealer | (discarder == dealer), 2, 1)
        discard_payments = jnp.where(players == discarder, amount * discard_multiplier, 0)
        payments = jnp.where(is_self_draw, self_draw_payments, discard_payments)
        rewards = -payments
        rewards = rewards.at[winner].set(payments.sum())
        return rewards.astype(jnp.int32), payments.astype(jnp.int32)
