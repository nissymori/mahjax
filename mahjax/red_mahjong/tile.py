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


import jax
import jax.numpy as jnp

from .types import Array
from .action import Action
from .constants import RED_FIVE_TILE_IDS, RED_FIVE_TILE_TYPES


class Tile:
    """
    tile_id: when all 136 tiles are distinguished
    tile_type: tile type (0-33) refer to mjx
    tile: local tile index (0-36) where 34..36 are red 5m/5p/5s
    """

    NUM_TILE_ID = 136
    NUM_TILE_TYPE = 34
    NUM_TILE_TYPE_WITH_RED = 37
    BLACK_FIVE = {"m": 4, "p": 13, "s": 22}
    RED_FIVE = {"m": 34, "p": 35, "s": 36}

    _from_tile_id = (jnp.arange(136, dtype=jnp.int32) // 4).astype(jnp.int8)
    _from_tile_id = _from_tile_id.at[jnp.array(RED_FIVE_TILE_IDS, dtype=jnp.int32)].set(
        jnp.array([RED_FIVE["m"], RED_FIVE["p"], RED_FIVE["s"]], dtype=jnp.int8)
    )
    FROM_TILE_ID_TO_TILE = _from_tile_id

    @staticmethod
    def from_tile_id_to_tile(tile_id: Array) -> Array:
        return Tile.FROM_TILE_ID_TO_TILE[tile_id]

    @staticmethod
    def is_tile_red(tile: Array) -> Array:
        return tile >= Tile.NUM_TILE_TYPE

    @staticmethod
    def to_tile_type(tile: Array) -> Array:
        tile = jnp.asarray(tile, dtype=jnp.int32)
        return jnp.where(
            tile == Tile.RED_FIVE["m"],
            Tile.BLACK_FIVE["m"],
            jnp.where(
                tile == Tile.RED_FIVE["p"],
                Tile.BLACK_FIVE["p"],
                jnp.where(tile == Tile.RED_FIVE["s"], Tile.BLACK_FIVE["s"], tile),
            ),
        ).astype(jnp.int32)

    @staticmethod
    def to_red(tile_type: Array) -> Array:
        tile_type = jnp.asarray(tile_type, dtype=jnp.int32)
        return jnp.where(
            tile_type == Tile.BLACK_FIVE["m"],
            Tile.RED_FIVE["m"],
            jnp.where(
                tile_type == Tile.BLACK_FIVE["p"],
                Tile.RED_FIVE["p"],
                jnp.where(tile_type == Tile.BLACK_FIVE["s"], Tile.RED_FIVE["s"], tile_type),
            ),
        ).astype(jnp.int32)

    @staticmethod
    def is_tile_type_five(tile_type: Array) -> Array:
        tile_type = jnp.asarray(tile_type, dtype=jnp.int32)
        return (tile_type == RED_FIVE_TILE_TYPES[0]) | (tile_type == RED_FIVE_TILE_TYPES[1]) | (tile_type == RED_FIVE_TILE_TYPES[2])

    @staticmethod
    def is_tile_type_seven(tile_type: Array) -> bool:
        tile_type = Tile.to_tile_type(tile_type)
        return (tile_type % 9 == 6) & (tile_type < 27)

    @staticmethod
    def is_tile_type_three(tile_type: Array) -> bool:
        tile_type = Tile.to_tile_type(tile_type)
        return (tile_type % 9 == 2) & (tile_type < 27)

    @staticmethod
    def is_tile_four_wind(tile: Array) -> bool:
        tile_type = Tile.to_tile_type(tile)
        return (27 <= tile_type) & (tile_type < 31)

    @staticmethod
    def is_yaochu(tile: Array) -> Array:
        tile_type = Tile.to_tile_type(tile)
        num = tile_type % 9
        return (tile_type >= 27) | (num == 0) | (num == 8)


TILE_MASK = jnp.uint16(0b0000000000111111)
BIT_RIICHI = jnp.uint16(1 << 6)
BIT_GRAY = jnp.uint16(1 << 7)
BIT_TSUMOGIRI = jnp.uint16(1 << 8)
SRC_SHIFT = 9
MT_SHIFT = 11
SRC_MASK = jnp.uint16(0b11 << SRC_SHIFT)
MT_MASK = jnp.uint16(0b111 << MT_SHIFT)
EMPTY_RIVER = jnp.uint16(0xFFFF)


class River:
    @staticmethod
    def add_discard(
        river: Array,
        tile: Array,
        player: Array,
        idx: Array,
        is_tsumogiri: bool,
        is_riichi: bool,
    ) -> Array:
        tile_u16 = jnp.uint16(tile) & TILE_MASK
        tile_u16 = tile_u16 | BIT_TSUMOGIRI * jnp.uint16(is_tsumogiri)
        tile_u16 = tile_u16 | BIT_RIICHI * jnp.uint16(is_riichi)
        tile_u16 = tile_u16 | BIT_GRAY * jnp.uint16(False)
        tile_u16 = tile_u16 | ((jnp.uint16(0) & jnp.uint16(0b11)) << SRC_SHIFT)
        tile_u16 = tile_u16 | ((jnp.uint16(0) & jnp.uint16(0b111)) << MT_SHIFT)
        return river.at[player, idx].set(tile_u16)

    @staticmethod
    def add_meld(
        river: Array, action: Array, player: Array, idx: Array, src: Array
    ) -> Array:
        tile_u16 = river[player, idx]
        meld_type = jnp.where(
            (action == Action.PON) | (action == Action.PON_RED),
            jnp.uint16(1),
            jnp.where(
                action == Action.OPEN_KAN,
                jnp.uint16(2),
                jnp.where(
                    (action == Action.CHI_L) | (action == Action.CHI_L_RED),
                    jnp.uint16(3),
                    jnp.where(
                        (action == Action.CHI_M) | (action == Action.CHI_M_RED),
                        jnp.uint16(4),
                        jnp.where(
                            (action == Action.CHI_R) | (action == Action.CHI_R_RED),
                            jnp.uint16(5),
                            jnp.uint16(0),
                        ),
                    ),
                ),
            ),
        )
        tile_u16 = tile_u16 & ~BIT_GRAY
        tile_u16 = tile_u16 & ~SRC_MASK
        tile_u16 = tile_u16 & ~MT_MASK
        tile_u16 = tile_u16 | BIT_GRAY
        tile_u16 = tile_u16 | ((jnp.uint16(src) & jnp.uint16(0b11)) << SRC_SHIFT)
        tile_u16 = tile_u16 | ((meld_type & jnp.uint16(0b111)) << MT_SHIFT)
        return river.at[player, idx].set(tile_u16)

    @staticmethod
    def decode_river(river: Array) -> Array:
        empty = river == EMPTY_RIVER
        tile = (river & TILE_MASK).astype(jnp.int32)
        riichi = (river & BIT_RIICHI) != 0
        gray = (river & BIT_GRAY) != 0
        tsumogiri = (river & BIT_TSUMOGIRI) != 0
        src = ((river & SRC_MASK) >> SRC_SHIFT).astype(jnp.int32)
        meld_type = ((river & MT_MASK) >> MT_SHIFT).astype(jnp.int32)

        tile = jnp.where(empty, -1, tile)
        riichi_i = jnp.where(empty, 0, riichi.astype(jnp.int32))
        gray_i = jnp.where(empty, 0, gray.astype(jnp.int32))
        tsumog_i = jnp.where(empty, 0, tsumogiri.astype(jnp.int32))
        src_i = jnp.where(empty, 0, src)
        mt_i = jnp.where(empty, 0, meld_type)
        return jnp.stack([tile, riichi_i, gray_i, tsumog_i, src_i, mt_i], axis=0)

    @staticmethod
    def decode_tile(river: Array) -> Array:
        empty = river == EMPTY_RIVER
        tile = (river & TILE_MASK).astype(jnp.int32)
        return jnp.where(empty, -1, tile)
