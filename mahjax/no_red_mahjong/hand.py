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

import jax
import jax.numpy as jnp
import numpy as np

from mahjax._src.types import Array
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.tile import Tile

DIR = os.path.join(os.path.dirname(__file__), "../no_red_mahjong/cache")


def load_hand_cache():
    with np.load(os.path.join(DIR, "hand_cache.npz"), allow_pickle=False) as data:
        return jnp.asarray(data["data"], dtype=jnp.uint32)


THIRTEEN_ORPHAN_IDX = jnp.array([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])
powers_of_5_full = jnp.concatenate(
    [
        5 ** jnp.arange(8, -1, -1),  # for suit 0
        5 ** jnp.arange(8, -1, -1),  # for suit 1
        5 ** jnp.arange(8, -1, -1),  # for suit 2
    ]
)  # shape = (27,)


class Hand:
    """
    Hand class

    Hand is assumed to be a count vector of 34 tiles.
    e.g. if hand is 1112225678999m, then
    hand = [3, 3, 0, 0, 1, 1, 1, 1, 3, 0, ...]
    """
    # Load the cache
    CACHE = load_hand_cache()
    # Mask for kyuushu
    KYUUSHU_MASK = jnp.array(
        [
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        dtype=jnp.uint8,
    )

    @staticmethod
    def make_init_hand(deck: Array) -> Array:
        """
        Generate the initial hand
        Each player is assigned the last 13 tiles from the deck,
        and converted into a 34-dimensional tile representation.
        """
        hand = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int8)
        # Assign the last 4*13 tiles from the deck to each player
        hand_ids = deck[-(13 * 4) :].reshape(4, 13)

        # For each player's hand, add 1 to each tile ID
        def add_tiles(h, tiles):
            return h.at[tiles].add(1)  # TODO: set cannot be parallelized?

        return jax.vmap(add_tiles)(hand, hand_ids)

    @staticmethod
    def cache(code):
        return (Hand.CACHE[code >> 5] >> (code & 0b11111)) & 1

    @staticmethod
    def can_chi(hand: Array, tile: Array, action: Array) -> bool:
        """
        Judge whether it is possible to play chi
        - We need to have tiles to form a set with the target tile
        """
        can_black_chi = jax.lax.switch(
            action - Action.CHI_L,
            [
                lambda: (tile % 9 < 7)
                & (hand[tile + 1] > 0)
                & (hand[tile + 2] > 0),  # CHI_L
                lambda: (
                    (tile % 9 < 8)
                    & (tile % 9 > 0)
                    & (hand[tile - 1] > 0)
                    & (hand[tile + 1] > 0)
                ),  # CHI_M
                lambda: (tile % 9 > 1)
                & (hand[tile - 2] > 0)
                & (hand[tile - 1] > 0),  # CHI_R
            ],
        )
        return can_black_chi & (tile < 27)

    @staticmethod
    def can_pon(hand: Array, tile) -> bool:
        """
        Judge whether it is possible to play pon
        - The tile count should be 2
        """
        return hand[tile] >= 2  # type: ignore

    @staticmethod
    def can_open_kan(hand: Array, tile) -> bool:
        """
        Judge whether it is possible to play open kan
        """
        return hand[tile] == 3

    @staticmethod
    def can_added_kan(hand: Array, tile) -> bool:
        """
        Judge whether it is possible to play added kan
        - The tile count should be 1
        """
        return hand[tile] == 1  # type: ignore

    @staticmethod
    def can_closed_kan(hand: Array, tile) -> bool:
        """
        Judge whether it is possible to play closed kan
        - The tile count should be 4
        """
        return hand[tile] == 4

    @staticmethod
    def can_closed_kan_after_riichi(hand: Array, tile, original_can_win: Array) -> bool:
        """
        - Jundge whether it is possible to play closed kan after riichi
        - The waiting tile should not change by playing closed kan
        - TODO: In the future, we should consider the rule of how to get the waiting tile.
        - original_can_win: The waiting tile before playing closed kan (34,)
        Example:
        1111456m ... => can closed kan
        12222234m ... => cannot closed kan (3m will be removed from waiting tile)
        """

        def _check_identity():
            new_hand = Hand.sub(hand, tile, 4)
            new_can_win = jax.vmap(Hand.can_ron, in_axes=(None, 0))(
                new_hand, jnp.arange(Tile.NUM_TILE_TYPE)
            )
            return jnp.all(original_can_win == new_can_win)

        can_closed_kan = Hand.can_closed_kan(hand, tile)
        return can_closed_kan & _check_identity()

    @staticmethod
    def can_tsumo(hand: Array):
        """
        Judge whether it is possible to win by drawing a tile (hand is 14 tiles)
        The possible winning hands are:
        - thirteen orphan e.g. 19m19p19sESWNWGR
        - seven pairs e.g. 11223344556677s
        - 1head + 4 sets e.g. 1112345678m123p11s
        """
        thirteen_orphan = (hand[THIRTEEN_ORPHAN_IDX] > 0).all() & (
            hand[THIRTEEN_ORPHAN_IDX].sum() == 14
        )
        seven_pairs = jnp.sum(hand == 2) == 7
        codes = (hand[:27].astype(int) * powers_of_5_full).reshape(3, 9).sum(axis=1)

        def _is_valid(suit):
            return Hand.cache(codes[suit])

        valid = jax.vmap(_is_valid)(jnp.arange(3)).all()
        # For each suit (0~26), calculate the sum of 9 tiles
        suit_sums = jnp.sum(hand[:27].reshape(3, 9), axis=1)
        heads = jnp.sum((suit_sums % 3 == 2).astype(jnp.int32))
        # For each honor (27~33), calculate the sum of 2 tiles
        heads_honors = jnp.sum(hand[27:34] == 2)
        heads += heads_honors
        valid = valid & jnp.all((hand[27:34] != 1) & (hand[27:34] != 4))
        return ((valid & (heads == 1)) | thirteen_orphan | seven_pairs) == 1

    @staticmethod
    def can_ron(hand: Array, tile):
        """
        Judge whether the hand can be won by the tile (hand has 14 tiles)
        It is done by checking if the hand added by the tile can be won by tsumo.
        """
        return Hand.can_tsumo(Hand.add(hand, tile))

    @staticmethod
    def is_tenpai(hand: Array):
        """
        Judge whether the hand is tenpai, ready to win (hand has 14 tiles)
        It is done by checking if any tile can be discarded to make the hand ron.
        e.g.
        - 1112345678999m ... => True
        - 111234567899mE ... => False
        """
        return jax.vmap(
            lambda tile_type: (hand[tile_type] != 4) & Hand.can_ron(hand, tile_type)
        )(jnp.arange(Tile.NUM_TILE_TYPE)).any()

    @staticmethod
    def can_riichi(hand: Array):
        """
        Judge whether it is possible to play riichi (hand has 14 tiles)
        It is done by checking if any tile can be discarded to make the hand tenpai.
        """
        return jax.vmap(lambda i: (hand[i] != 0) & Hand.is_tenpai(Hand.sub(hand, i)))(
            jnp.arange(Tile.NUM_TILE_TYPE)
        ).any()

    @staticmethod
    def add(hand: Array, tile, x=1) -> Array:
        """
        Add x tiles to the hand
        """
        return hand.at[tile].set(hand[tile] + x)

    @staticmethod
    def sub(hand: Array, tile, x=1) -> Array:
        """
        Subtract x tiles from the hand
        """
        return Hand.add(hand, tile, -x)

    @staticmethod
    def chi(hand: Array, tile, action) -> Array:
        """
        Apply the results of chi to the hand
        """
        chi_idx = action - Action.CHI_L
        start = tile - chi_idx
        hand = hand.at[jnp.array([start, start + 1, start + 2])].add(-1)
        hand = hand.at[tile].add(1)
        return hand

    @staticmethod
    def pon(hand: Array, tile) -> Array:
        """
        Apply the results of pon to the hand
        """
        return Hand.sub(hand, tile, 2)

    @staticmethod
    def open_kan(hand: Array, tile) -> Array:
        """
        Apply the results of open kan to the hand
        """
        return Hand.sub(hand, tile, 3)

    @staticmethod
    def added_kan(hand: Array, tile) -> Array:
        """
        Apply the results of added kan to the hand
        """
        return Hand.sub(hand, tile)

    @staticmethod
    def closed_kan(hand: Array, tile) -> Array:
        """
        Apply the results of closed kan to the hand
        """
        return Hand.sub(hand, tile, 4)
