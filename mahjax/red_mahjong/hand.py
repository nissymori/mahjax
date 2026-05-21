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


from pathlib import Path
import importlib.resources as resources

import jax
import jax.numpy as jnp
import numpy as np

from .types import Array
from .action import Action
from .tile import Tile


def load_hand_cache():
    with resources.as_file(resources.files("mahjax._src.cache").joinpath("hand_cache.npz")) as path:
        with np.load(path, allow_pickle=False) as data:
            return jnp.asarray(data["data"], dtype=jnp.uint32)


THIRTEEN_ORPHAN_IDX = jnp.array([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])
POWERS_OF_5_FULL = jnp.concatenate(
    [
        5 ** jnp.arange(8, -1, -1),
        5 ** jnp.arange(8, -1, -1),
        5 ** jnp.arange(8, -1, -1),
    ]
)


class Hand:
    CACHE = load_hand_cache()

    @staticmethod
    def _is_red_chi_action(action: Array) -> Array:
        return (action == Action.CHI_L_RED) | (action == Action.CHI_M_RED) | (action == Action.CHI_R_RED)

    @staticmethod
    def _chi_index(action: Array) -> Array:
        return jnp.where(
            (action == Action.CHI_L) | (action == Action.CHI_L_RED),
            jnp.int32(0),
            jnp.where(
                (action == Action.CHI_M) | (action == Action.CHI_M_RED),
                jnp.int32(1),
                jnp.where(
                    (action == Action.CHI_R) | (action == Action.CHI_R_RED),
                    jnp.int32(2),
                    jnp.int32(-1),
                ),
            ),
        )

    @staticmethod
    def _base_chi_action(chi_idx: Array) -> Array:
        return jnp.where(
            chi_idx == 0,
            jnp.int32(Action.CHI_L),
            jnp.where(chi_idx == 1, jnp.int32(Action.CHI_M), jnp.int32(Action.CHI_R)),
        )

    KYUUSHU_MASK = jnp.array(
        [
            1, 0, 0, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 0, 0, 0, 1,
            1, 1, 1, 1, 1, 1, 1,
        ],
        dtype=jnp.uint8,
    )

    @staticmethod
    def make_init_hand(deck: Array) -> Array:
        hand = jnp.zeros((4, Tile.NUM_TILE_TYPE_WITH_RED), dtype=jnp.int8)
        hand_ids = deck[-(13 * 4) :].reshape(4, 13)

        def add_tiles(h, tiles):
            return h.at[tiles].add(1)

        return jax.vmap(add_tiles)(hand, hand_ids)

    @staticmethod
    def to_34(hand: Array) -> Array:
        if hand.shape[0] == Tile.NUM_TILE_TYPE:
            return hand
        hand_34 = hand[: Tile.NUM_TILE_TYPE]
        hand_34 = hand_34.at[Tile.BLACK_FIVE["m"]].add(hand[Tile.RED_FIVE["m"]])
        hand_34 = hand_34.at[Tile.BLACK_FIVE["p"]].add(hand[Tile.RED_FIVE["p"]])
        hand_34 = hand_34.at[Tile.BLACK_FIVE["s"]].add(hand[Tile.RED_FIVE["s"]])
        return hand_34

    @staticmethod
    def cache(code):
        return (Hand.CACHE[code >> 5] >> (code & 0b11111)) & 1

    @staticmethod
    def has_red_of(hand: Array, tile_type: Array) -> Array:
        return jnp.where(
            Tile.is_tile_type_five(tile_type),
            hand[Tile.to_red(tile_type)] > 0,
            jnp.bool_(False),
        )

    @staticmethod
    def can_chi(hand: Array, tile: Array, action: Array) -> bool:
        is_red_chi = Hand._is_red_chi_action(action)
        return jax.lax.select(
            is_red_chi,
            Hand.can_red_chi(hand, tile, action),
            Hand.can_no_red_chi(hand, tile, action),
        )

    @staticmethod
    def can_no_red_chi(hand: Array, tile: Array, action: Array) -> bool:
        hand_nr = hand if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED else Hand.to_34(hand)
        tile_type = Tile.to_tile_type(tile)
        chi_idx = Hand._chi_index(action)
        can_black_chi = jax.lax.switch(
            chi_idx,
            [
                lambda: (tile_type % 9 < 7) & (hand_nr[tile_type + 1] > 0) & (hand_nr[tile_type + 2] > 0),
                lambda: (
                    (tile_type % 9 < 8)
                    & (tile_type % 9 > 0)
                    & (hand_nr[tile_type - 1] > 0)
                    & (hand_nr[tile_type + 1] > 0)
                ),
                lambda: (tile_type % 9 > 1) & (hand_nr[tile_type - 2] > 0) & (hand_nr[tile_type - 1] > 0),
            ],
        )
        return can_black_chi & (tile_type < 27)

    @staticmethod
    def can_red_chi(hand: Array, tile: Array, action: Array) -> bool:
        hand_34 = Hand.to_34(hand)
        tile_type = Tile.to_tile_type(tile)
        chi_idx = Hand._chi_index(action)
        base_action = Hand._base_chi_action(chi_idx)
        can_black_chi = Hand.can_no_red_chi(hand_34, tile_type, base_action)
        has_red_neighbor = jax.lax.switch(
            chi_idx,
            [
                lambda: Hand.has_red_of(hand, tile_type + 1) | Hand.has_red_of(hand, tile_type + 2),
                lambda: Hand.has_red_of(hand, tile_type - 1) | Hand.has_red_of(hand, tile_type + 1),
                lambda: Hand.has_red_of(hand, tile_type - 2) | Hand.has_red_of(hand, tile_type - 1),
            ],
        )
        return can_black_chi & has_red_neighbor & (tile_type < 27)

    @staticmethod
    def can_pon(hand: Array, tile: Array) -> bool:
        return Hand.can_no_red_pon(hand, tile) | Hand.can_red_pon(hand, tile)

    @staticmethod
    def can_no_red_pon(hand: Array, tile: Array) -> bool:
        tile_type = Tile.to_tile_type(tile)
        hand_nr = hand if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED else Hand.to_34(hand)
        return hand_nr[tile_type] >= 2

    @staticmethod
    def can_red_pon(hand: Array, tile: Array) -> bool:
        if hand.shape[0] != Tile.NUM_TILE_TYPE_WITH_RED:
            return jnp.bool_(False)
        tile_type = Tile.to_tile_type(tile)
        return Tile.is_tile_type_five(tile_type) & (hand[tile_type] > 0) & (hand[Tile.to_red(tile_type)] > 0)

    @staticmethod
    def can_open_kan(hand: Array, tile: Array) -> bool:
        tile_type = Tile.to_tile_type(tile)
        hand_34 = Hand.to_34(hand)
        if hand.shape[0] != Tile.NUM_TILE_TYPE_WITH_RED:
            return hand_34[tile_type] == 3
        return jax.lax.select(
            Tile.is_tile_type_five(tile_type),
            (hand[tile_type] == 3) | ((hand[Tile.to_red(tile_type)] == 1) & (hand[tile_type] == 2)),
            hand[tile_type] == 3,
        )

    @staticmethod
    def can_added_kan(hand: Array, tile: Array) -> bool:
        tile_type = Tile.to_tile_type(tile)
        hand_34 = Hand.to_34(hand)
        return hand_34[tile_type] == 1

    @staticmethod
    def can_closed_kan(hand: Array, tile: Array) -> bool:
        tile_type = Tile.to_tile_type(tile)
        hand_34 = Hand.to_34(hand)
        if hand.shape[0] != Tile.NUM_TILE_TYPE_WITH_RED:
            return hand_34[tile_type] == 4
        return jax.lax.select(
            Tile.is_tile_type_five(tile_type),
            (hand[tile_type] == 3) & (hand[Tile.to_red(tile_type)] == 1),
            hand[tile_type] == 4,
        )

    @staticmethod
    def can_closed_kan_after_riichi(hand: Array, tile: Array, original_can_win: Array) -> bool:
        tile_type = Tile.to_tile_type(tile)

        def _check_identity():
            new_hand = Hand.closed_kan(hand, tile_type)
            new_can_win = jax.vmap(Hand.can_ron, in_axes=(None, 0))(new_hand, jnp.arange(Tile.NUM_TILE_TYPE))
            return jnp.all(original_can_win == new_can_win)

        can_closed_kan = Hand.can_closed_kan(hand, tile_type)
        return jax.lax.cond(can_closed_kan, _check_identity, lambda: jnp.bool_(False))

    @staticmethod
    def can_tsumo(hand: Array):
        hand_34 = Hand.to_34(hand)
        thirteen_orphan = (hand_34[THIRTEEN_ORPHAN_IDX] > 0).all() & (hand_34[THIRTEEN_ORPHAN_IDX].sum() == 14)
        seven_pairs = jnp.sum(hand_34 == 2) == 7
        codes = (hand_34[:27].astype(int) * POWERS_OF_5_FULL).reshape(3, 9).sum(axis=1)

        def _is_valid(suit):
            return Hand.cache(codes[suit])

        valid = jax.vmap(_is_valid)(jnp.arange(3)).all()
        suit_sums = jnp.sum(hand_34[:27].reshape(3, 9), axis=1)
        heads = jnp.sum((suit_sums % 3 == 2).astype(jnp.int32))
        heads_honors = jnp.sum(hand_34[27:34] == 2)
        heads += heads_honors
        valid = valid & jnp.all((hand_34[27:34] != 1) & (hand_34[27:34] != 4))
        return ((valid & (heads == 1)) | thirteen_orphan | seven_pairs) == 1

    @staticmethod
    def can_ron(hand: Array, tile: Array):
        tile_for_hand = tile if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED else Tile.to_tile_type(tile)
        return Hand.can_tsumo(Hand.add(hand, tile_for_hand))

    @staticmethod
    def is_tenpai(hand: Array):
        hand_34 = Hand.to_34(hand)
        return jax.vmap(lambda tile_type: (hand_34[tile_type] != 4) & Hand.can_ron(hand_34, tile_type))(jnp.arange(Tile.NUM_TILE_TYPE)).any()

    @staticmethod
    def can_riichi(hand: Array):
        if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED:
            return jax.vmap(lambda i: (hand[i] != 0) & Hand.is_tenpai(Hand.sub(hand, i)))(jnp.arange(Tile.NUM_TILE_TYPE_WITH_RED)).any()
        return jax.vmap(lambda i: (hand[i] != 0) & Hand.is_tenpai(Hand.sub(hand, i)))(jnp.arange(Tile.NUM_TILE_TYPE)).any()

    @staticmethod
    def can_kyuushu(hand: Array) -> bool:
        hand_34 = Hand.to_34(hand)
        return jnp.sum((hand_34 * Hand.KYUUSHU_MASK) > 0) >= 9

    @staticmethod
    def add(hand: Array, tile: Array, x: int = 1) -> Array:
        target = tile if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED else Tile.to_tile_type(tile)
        return hand.at[target].set(hand[target] + x)

    @staticmethod
    def sub(hand: Array, tile: Array, x: int = 1) -> Array:
        return Hand.add(hand, tile, -x)

    @staticmethod
    def _remove_one_of_tile_type(hand: Array, tile_type: Array) -> Array:
        if hand.shape[0] != Tile.NUM_TILE_TYPE_WITH_RED:
            return Hand.sub(hand, tile_type)
        red = Tile.to_red(tile_type)
        return jax.lax.cond(
            Tile.is_tile_type_five(tile_type),
            lambda: jax.lax.cond(
                hand[tile_type] > 0,
                lambda: Hand.sub(hand, tile_type),
                lambda: Hand.sub(hand, red),
            ),
            lambda: Hand.sub(hand, tile_type),
        )

    @staticmethod
    def chi(hand: Array, tile: Array, action: Array) -> Array:
        is_red_chi = Hand._is_red_chi_action(action)
        return jax.lax.select(is_red_chi, Hand.chi_red(hand, tile, action), Hand.chi_no_red(hand, tile, action))

    @staticmethod
    def chi_no_red(hand: Array, tile: Array, action: Array) -> Array:
        tile_type = Tile.to_tile_type(tile)
        chi_idx = Hand._chi_index(action)
        start = tile_type - chi_idx
        def _remove(i: Array, h: Array) -> Array:
            return jax.lax.cond(
                i == tile_type,
                lambda: h,
                lambda: Hand.sub(h, i),
            )

        return jax.lax.fori_loop(start, start + 3, _remove, hand)

    @staticmethod
    def chi_red(hand: Array, tile: Array, action: Array) -> Array:
        tile_type = Tile.to_tile_type(tile)
        chi_idx = Hand._chi_index(action)
        start = tile_type - chi_idx
        def _remove(i: Array, h: Array) -> Array:
            remove_tile = jax.lax.select(Hand.has_red_of(h, i), Tile.to_red(i), i)
            return jax.lax.cond(
                i == tile_type,
                lambda: h,
                lambda: Hand.sub(h, remove_tile),
            )

        return jax.lax.fori_loop(start, start + 3, _remove, hand)

    @staticmethod
    def pon(hand: Array, tile: Array, action: Array) -> Array:
        is_red_pon = action == Action.PON_RED
        return jax.lax.select(is_red_pon, Hand.pon_red(hand, tile), Hand.pon_no_red(hand, tile))

    @staticmethod
    def pon_no_red(hand: Array, tile: Array) -> Array:
        tile_type = Tile.to_tile_type(tile)
        return Hand.sub(hand, tile_type, 2)

    @staticmethod
    def pon_red(hand: Array, tile: Array) -> Array:
        tile_type = Tile.to_tile_type(tile)
        hand = Hand.sub(hand, tile_type)
        return Hand.sub(hand, Tile.to_red(tile_type))

    @staticmethod
    def open_kan(hand: Array, tile: Array) -> Array:
        tile_type = Tile.to_tile_type(tile)
        if hand.shape[0] != Tile.NUM_TILE_TYPE_WITH_RED:
            return Hand.sub(hand, tile_type, 3)

        def _five_case() -> Array:
            return jax.lax.cond(
                Tile.is_tile_red(tile),
                lambda: Hand.sub(hand, tile_type, 3),
                lambda: jax.lax.cond(
                    hand[Tile.to_red(tile_type)] > 0,
                    lambda: Hand.sub(Hand.sub(hand, tile_type, 2), Tile.to_red(tile_type)),
                    lambda: Hand.sub(hand, tile_type, 3),
                ),
            )

        return jax.lax.cond(
            Tile.is_tile_type_five(tile_type),
            _five_case,
            lambda: Hand.sub(hand, tile_type, 3),
        )

    @staticmethod
    def added_kan(hand: Array, tile: Array) -> Array:
        tile_type = Tile.to_tile_type(tile)
        if hand.shape[0] == Tile.NUM_TILE_TYPE_WITH_RED:
            return Hand._remove_one_of_tile_type(hand, tile_type)
        return Hand.sub(hand, tile_type)

    @staticmethod
    def closed_kan(hand: Array, tile: Array) -> Array:
        tile_type = Tile.to_tile_type(tile)
        if hand.shape[0] != Tile.NUM_TILE_TYPE_WITH_RED:
            return Hand.sub(hand, tile_type, 4)
        return jax.lax.cond(
            Tile.is_tile_type_five(tile_type),
            lambda: Hand.sub(Hand.sub(hand, Tile.to_red(tile_type)), tile_type, 3),
            lambda: Hand.sub(hand, tile_type, 4),
        )
