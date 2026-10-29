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
from .tile import Tile

EMPTY_MELD = jnp.uint16(0xFFFF)


class Meld:
    @staticmethod
    def init(action: Array, target: Array, src: Array) -> Array:
        target_is_red = Tile.is_tile_red(target)
        target_tile_type = Tile.to_tile_type(target)
        return (
            (jnp.uint16(target_is_red) << 15)
            | (jnp.uint16(src) << 13)
            | (jnp.uint16(target_tile_type) << 7)
            | jnp.uint16(action)
        )

    @staticmethod
    def empty() -> Array:
        return EMPTY_MELD

    @staticmethod
    def is_empty(meld: Array) -> bool:
        return meld == EMPTY_MELD

    @staticmethod
    def is_target_red(meld: Array) -> Array:
        return ((meld >> 15) & jnp.uint16(0b1)) & (~Meld.is_empty(meld))

    @staticmethod
    def src(meld: Array) -> Array:
        is_emp = Meld.is_empty(meld)
        return jnp.where(is_emp, jnp.int32(-1), (meld >> 13) & jnp.uint16(0b11))

    @staticmethod
    def target(meld: Array) -> Array:
        is_emp = Meld.is_empty(meld)
        return jnp.where(is_emp, jnp.int32(-1), (meld >> 7) & jnp.uint16(0b111111))

    @staticmethod
    def target_tile(meld: Array) -> Array:
        target = Meld.target(meld)
        return jnp.where(Meld.is_target_red(meld), Tile.to_red(target), target)

    @staticmethod
    def action(meld: Array) -> Array:
        is_emp = Meld.is_empty(meld)
        return jnp.where(is_emp, jnp.int32(-1), meld & jnp.uint16(0b1111111))

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
    def suited_pung(meld: Array) -> Array:
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        target = Meld.target(meld)
        is_pung = (
            (action == Action.PON)
            | (action == Action.PON_RED)
            | (action == Action.OPEN_KAN)
            | Action.is_selfkan(action)
        )
        is_suited_pon = is_pung & (target < 27) & (~is_emp)
        return is_suited_pon.astype(jnp.int32) << target

    @staticmethod
    def chow(meld: Array) -> Array:
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        is_chi = Meld.is_chi(meld)
        pos = Meld.target(meld) - Meld._chi_index(action)
        pos = pos * is_chi.astype(jnp.int32)
        return is_chi.astype(jnp.int32) << pos

    @staticmethod
    def is_kan(meld: Array) -> bool:
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        return ((action == Action.OPEN_KAN) | Action.is_selfkan(action)) & (~is_emp)

    @staticmethod
    def is_closed_kan(meld: Array) -> bool:
        is_emp = Meld.is_empty(meld)
        src = Meld.src(meld)
        action = Meld.action(meld)
        return (src == 0) & Action.is_selfkan(action) & (~is_emp)

    @staticmethod
    def is_added_kan(meld: Array) -> bool:
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        return Action.is_selfkan(action) & (Meld.src(meld) != 0) & (~is_emp)

    @staticmethod
    def is_chi(meld: Array) -> bool:
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        return (((Action.CHI_L <= action) & (action <= Action.CHI_R)) | ((Action.CHI_L_RED <= action) & (action <= Action.CHI_R_RED))) & (~is_emp)

    @staticmethod
    def is_pon(meld: Array) -> bool:
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        return ((action == Action.PON) | (action == Action.PON_RED)) & (~is_emp)

    @staticmethod
    def is_outside(meld: Array) -> Array:
        is_emp = Meld.is_empty(meld)
        target = Meld.target(meld)
        action = Meld.action(meld)
        is_pon_or_kan = Meld.is_pon(meld) | (action == Action.OPEN_KAN) | Action.is_selfkan(action)
        num = target % 9
        return ((target >= 27) | (num == 0) | (num == 8)) & (~is_emp) & is_pon_or_kan

    @staticmethod
    def has_outside(meld: Array) -> Array:
        is_emp = Meld.is_empty(meld)
        target = Meld.target(meld)
        action = Meld.action(meld)
        num = target % 9
        is_outside = Meld.is_outside(meld)
        chi_index = Meld._chi_index(action)
        for_chi_l = Meld.is_chi(meld) & (chi_index == 0) & ((num == 0) | (num == 6))
        for_chi_m = Meld.is_chi(meld) & (chi_index == 1) & ((num == 1) | (num == 7))
        for_chi_r = Meld.is_chi(meld) & (chi_index == 2) & ((num == 2) | (num == 8))
        return is_outside | for_chi_l | for_chi_m | for_chi_r | is_emp

    @staticmethod
    def fu(meld: Array) -> Array:
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        base = (
            (Meld.is_pon(meld)) * 2
            + (action == Action.OPEN_KAN) * 8
            + (Action.is_selfkan(action) * 8 * (1 + (Meld.src(meld) == 0)))
        )
        base = base * (~is_emp)
        return base * (1 + Meld.is_outside(meld))

    @staticmethod
    def contains_red(meld: Array) -> Array:
        action = Meld.action(meld)
        target = Meld.target(meld)
        red_kan = (action == Action.OPEN_KAN) | Action.is_selfkan(action)
        red_kan = red_kan & Tile.is_tile_type_five(target)
        red_chi = (action == Action.CHI_L_RED) | (action == Action.CHI_M_RED) | (action == Action.CHI_R_RED)
        return Meld.is_target_red(meld) | (action == Action.PON_RED) | red_chi | red_kan

    @staticmethod
    def exist_prohibitive_tile_type_after_chi(action: Array, target: Array) -> Array:
        chi_index = Meld._chi_index(action)
        target_tile_type = Tile.to_tile_type(target)
        is_chi = (chi_index >= 0)
        for_chi_l = (chi_index == 0) & is_chi & ~Tile.is_tile_type_seven(target_tile_type)
        for_chi_r = (chi_index == 2) & is_chi & ~Tile.is_tile_type_three(target_tile_type)
        return for_chi_l | for_chi_r

    @staticmethod
    def prohibitive_tile_type_after_chi(action: Array, target: Array) -> Array:
        chi_index = Meld._chi_index(action)
        target_tile_type = Tile.to_tile_type(target)
        for_chi_l = (chi_index == 0) & ~Tile.is_tile_type_seven(target_tile_type)
        for_chi_r = (chi_index == 2) & ~Tile.is_tile_type_three(target_tile_type)
        return jax.lax.cond(
            Meld.exist_prohibitive_tile_type_after_chi(action, target),
            lambda: jnp.int8(target_tile_type + 3) * for_chi_l + jnp.int8(target_tile_type - 3) * for_chi_r,
            lambda: jnp.int8(-1),
        )
