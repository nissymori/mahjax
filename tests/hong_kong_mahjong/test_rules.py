import jax.numpy as jnp
import pytest

from mahjax.hong_kong_mahjong.hand import Hand
from mahjax.hong_kong_mahjong.meld import EMPTY_MELD
from mahjax.hong_kong_mahjong.rules import HKOS_V1, Rules
from mahjax.hong_kong_mahjong.scoring import HongKongScoring
from mahjax.hong_kong_mahjong.tile import Tile


def test_hkos_v1_matches_requested_rules():
    assert HKOS_V1.tile_count == 144
    assert HKOS_V1.hand_size == 13
    assert HKOS_V1.include_flowers
    assert HKOS_V1.minimum_faan == 3
    assert HKOS_V1.maximum_faan == 10
    assert not HKOS_V1.allow_seven_pairs
    assert HKOS_V1.allow_thirteen_orphans
    assert not HKOS_V1.multiple_winners
    assert HKOS_V1.dealer_repeats_on_win
    assert HKOS_V1.dealer_repeats_on_draw
    assert HKOS_V1.robbing_a_kong
    assert HKOS_V1.replacement_tile_for_kong
    assert HKOS_V1.replacement_tile_for_flower
    assert HKOS_V1.discard_priority == "nearest_winner"
    assert HKOS_V1.self_draw_payment == "all_players_pay"
    assert HKOS_V1.discard_payment == "discarder_only"
    assert HKOS_V1.special_limit_hands
    assert HKOS_V1.flowers_score == "hong_kong"
    assert HKOS_V1.payout_table == "hk_old_style_v1"


def test_invalid_rule_variants_fail_fast():
    values = vars(HKOS_V1) | {"tile_count": 136}
    with pytest.raises(ValueError, match="144-tile"):
        Rules(**values)


def test_physical_tile_set_has_136_standard_and_eight_unique_flowers():
    tiles = Tile.FROM_TILE_ID_TO_TILE
    assert tiles.shape == (144,)
    assert jnp.array_equal(jnp.bincount(tiles[:136], length=34), jnp.full(34, 4))
    assert jnp.array_equal(tiles[136:], jnp.arange(34, 42))


def test_thirteen_orphans_is_legal_but_seven_pairs_is_not():
    orphan_indices = jnp.array([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])
    orphans = jnp.zeros(34, dtype=jnp.int8).at[orphan_indices].set(1).at[0].add(1)
    seven_pairs = jnp.zeros(34, dtype=jnp.int8).at[jnp.array([0, 2, 4, 6, 8, 9, 11])].set(2)
    also_standard = jnp.zeros(34, dtype=jnp.int8).at[jnp.arange(7)].set(2)
    assert bool(Hand.can_tsumo(orphans))
    assert not bool(Hand.can_tsumo(seven_pairs))
    assert bool(Hand.can_tsumo(also_standard))


def test_hong_kong_flower_faan():
    no_flowers = jnp.zeros(8, dtype=jnp.bool_)
    own_two = no_flowers.at[1].set(True).at[5].set(True)
    complete_seasons = no_flowers.at[:4].set(True)
    all_flowers = jnp.ones(8, dtype=jnp.bool_)
    assert HongKongScoring.flower_faan(no_flowers, jnp.int8(1)) == 1
    assert HongKongScoring.flower_faan(own_two, jnp.int8(1)) == 2
    assert HongKongScoring.flower_faan(complete_seasons, jnp.int8(1)) == 2
    assert HongKongScoring.flower_faan(all_flowers, jnp.int8(1)) == 4


def test_old_style_payout_table_caps_at_ten_faan():
    assert HongKongScoring.payout(jnp.int32(3)) == 8
    assert HongKongScoring.payout(jnp.int32(5)) == 24
    assert HongKongScoring.payout(jnp.int32(6)) == 32
    assert HongKongScoring.payout(jnp.int32(10)) == 128
    assert HongKongScoring.payout(jnp.int32(15)) == 128


def test_discarder_only_and_all_player_payments_are_zero_sum():
    ron_rewards, ron_payments = HongKongScoring.settle(jnp.int8(2), jnp.int8(0), jnp.int32(3), jnp.bool_(False))
    assert jnp.array_equal(ron_payments, jnp.array([8, 0, 0, 0]))
    assert jnp.array_equal(ron_rewards, jnp.array([-8, 0, 8, 0]))
    tsumo_rewards, tsumo_payments = HongKongScoring.settle(jnp.int8(2), jnp.int8(0), jnp.int32(3), jnp.bool_(True))
    assert jnp.array_equal(tsumo_payments, jnp.array([8, 8, 0, 8]))
    assert jnp.array_equal(tsumo_rewards, jnp.array([-8, -8, 24, -8]))
    assert ron_rewards.sum() == tsumo_rewards.sum() == 0


def test_dealer_pays_and_collects_double():
    dealer_wins, _ = HongKongScoring.settle(
        jnp.int8(2), jnp.int8(0), jnp.int32(3), jnp.bool_(False), dealer=jnp.int8(2)
    )
    dealer_discards, _ = HongKongScoring.settle(
        jnp.int8(2), jnp.int8(0), jnp.int32(3), jnp.bool_(False), dealer=jnp.int8(0)
    )
    assert jnp.array_equal(dealer_wins, jnp.array([-16, 0, 16, 0]))
    assert jnp.array_equal(dealer_discards, jnp.array([-16, 0, 16, 0]))


def test_three_faan_boundary_for_same_hand_by_ron_and_self_draw():
    # 123456789m 234p 5s, winning on 5s. The complete hand is concealed
    # and flowerless: 2 faan by discard, plus self-draw makes exactly 3.
    hand = jnp.zeros(34, dtype=jnp.int8).at[jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12])].set(1).at[22].set(1)
    melds = jnp.full(4, EMPTY_MELD, dtype=jnp.uint16)
    flowers = jnp.zeros(8, dtype=jnp.bool_)
    ron = HongKongScoring.judge(
        hand,
        melds,
        jnp.int8(0),
        jnp.int8(22),
        flowers,
        jnp.int8(0),
        jnp.int8(0),
        is_self_draw=jnp.bool_(False),
    )
    self_draw = HongKongScoring.judge(
        hand,
        melds,
        jnp.int8(0),
        jnp.int8(22),
        flowers,
        jnp.int8(0),
        jnp.int8(0),
        is_self_draw=jnp.bool_(True),
    )
    assert ron.faan == 2
    assert self_draw.faan == 3
    assert not ron.can_win
    assert self_draw.can_win


def test_heavenly_and_earthly_hands_are_special_limits():
    common = dict(
        hand=jnp.zeros(34, dtype=jnp.int8),
        melds=jnp.full(4, EMPTY_MELD, dtype=jnp.uint16),
        n_meld=jnp.int8(0),
        winning_tile=jnp.int8(0),
        flowers=jnp.zeros(8, dtype=jnp.bool_),
        prevalent_wind=jnp.int8(0),
        seat_wind=jnp.int8(0),
    )
    heavenly = HongKongScoring.judge(**common, is_self_draw=jnp.bool_(True), is_heavenly_hand=jnp.bool_(True))
    earthly = HongKongScoring.judge(**common, is_self_draw=jnp.bool_(False), is_earthly_hand=jnp.bool_(True))
    assert heavenly.is_limit and heavenly.faan == 10
    assert earthly.is_limit and earthly.faan == 10
