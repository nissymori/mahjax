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

from mahjax._src.types import Array
from mahjax.no_red_mahjong.action import Action


class Tile:
    """
    tile_id: when all 136 tiles are distinguished
    tile_type: tile type (0-33) refer to mjx
    """

    NUM_TILE_ID = 136
    NUM_TILE_TYPE = 34
    # convert tile_id (0-135) to tile (0-34)
    # basically, divide tile_id by 4.
    FROM_TILE_ID_TO_TILE = (jnp.arange(136) // 4).astype(jnp.uint8)

    @staticmethod
    def from_tile_id_to_tile(tile_id: Array) -> Array:
        """
        Convert tile_id (0-135) to tile (0-34).
        """
        return Tile.FROM_TILE_ID_TO_TILE[tile_id]

    @staticmethod
    def is_tile_type_seven(tile_type: Array) -> bool:
        """
        Check if the given tile type is 7.
        Used for swap-calling judgment.
        """
        return (tile_type % 9 == 6) & (tile_type < 27)

    @staticmethod
    def is_tile_type_three(tile_type: Array) -> bool:
        """
        Check if the given tile type is 3.
        Used for swap-calling judgment.
        """
        return (tile_type % 9 == 2) & (tile_type < 27)

    def is_tile_four_wind(tile: Array) -> bool:
        """
        Check if the given tile is four winds.
        """
        return (27 <= tile) & (tile < 31)


# ---- 16-bit layout ----
# [15..14] unused
# [13..11] meld_type (0:none,1:pon,2:open_kan,3:chi_l,4:chi_m,5:chi_r)
# [10..9]  src (0..3; 0=not set(there is no self-kan in the river))
# [8]      TSUMOGIRI (1=tsumogiri)
# [7]      GRAY  (1=gray)
# [6]      RIICHI(1=riichi)
# [5..0]   TILE  (0..33)

TILE_MASK = jnp.uint16(0b0000000000111111)  # bits 0..5
BIT_RIICHI = jnp.uint16(1 << 6)
BIT_GRAY = jnp.uint16(1 << 7)
BIT_TSUMOGIRI = jnp.uint16(1 << 8)
SRC_SHIFT = 9
MT_SHIFT = 11
SRC_MASK = jnp.uint16(0b11 << SRC_SHIFT)  # bits 10..9
MT_MASK = jnp.uint16(0b111 << MT_SHIFT)  # bits 13..11
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
        """
        Record discard at (player, idx). Tsumogiri is automatically determined by action==Action.TSUMOGIRI.
        src=0 (not set), meld_type=0 (none).
        """
        tile_u16 = jnp.uint16(tile) & TILE_MASK
        tile_u16 = tile_u16 | BIT_TSUMOGIRI * jnp.uint16(is_tsumogiri)
        tile_u16 = tile_u16 | BIT_RIICHI * jnp.uint16(is_riichi)
        tile_u16 = tile_u16 | BIT_GRAY * jnp.uint16(False)
        tile_u16 = tile_u16 | (
            (jnp.uint16(0) & jnp.uint16(0b11)) << SRC_SHIFT
        )  # src=0 (not set)
        tile_u16 = tile_u16 | (
            (jnp.uint16(0) & jnp.uint16(0b111)) << MT_SHIFT
        )  # meld_type=0 (none)
        return river.at[player, idx].set(tile_u16)

    @staticmethod
    def add_meld(
        river: Array, action: Array, player: Array, idx: Array, src: Array
    ) -> Array:
        """
        W   hen meld is established: update the tile at (player, idx) to "gray=1, src=src, meld_type=meld_type".
        """
        tile_u16 = river[player, idx]
        meld_type = (
            action - Action.PON + 1
        )  # 0:none, 1:pon, 2:open_kan, 3:chi_l, 4:chi_m, 5:chi_r
        # --- reset related bits ---
        tile_u16 = tile_u16 & ~BIT_GRAY  # reset gray
        tile_u16 = tile_u16 & ~SRC_MASK  # reset src
        tile_u16 = tile_u16 & ~MT_MASK  # reset meld_type
        # --- set after resetting ---
        tile_u16 = tile_u16 | BIT_GRAY
        tile_u16 = tile_u16 | (
            (jnp.uint16(src) & jnp.uint16(0b11)) << SRC_SHIFT
        )  # set src
        tile_u16 = tile_u16 | (
            (jnp.uint16(meld_type) & jnp.uint16(0b111)) << MT_SHIFT
        )  # set meld_type
        return river.at[player, idx].set(tile_u16)

    @staticmethod
    def decode_river(river: Array) -> Array:
        """
        (4,18) uint16 → (6,4,18) int32 tensor (jittable single array).
        Channel order: [tile, riichi, gray, tsumogiri, src, meld_type]
        - empty(0xFFFF): tile=-1, riichi/gray/tsumo/src/meld_type=0
        """
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

    def decode_tile(river: Array) -> Array:
        """
        (4,18) uint16 → (6,4,18) int32 tensor (jittable single array).
        Channel order: [tile, riichi, gray, tsumogiri, src, meld_type]
        - empty(0xFFFF): tile=-1, riichi/gray/tsumo/src/meld_type=0
        """
        empty = river == EMPTY_RIVER
        tile = (river & TILE_MASK).astype(jnp.int32)
        tile = jnp.where(empty, -1, tile)
        return tile
