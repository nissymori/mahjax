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
from mahjax.no_red_mahjong.tile import Tile

EMPTY_MELD = jnp.uint16(0xFFFF)


class Meld:
    @staticmethod
    def init(action: Array, target: Array, src: Array) -> Array:
        """
        - action: action type (0-78)
        - target: tile index (0-34)
        - src: relative position (0-3), 0: self (only for closed kan), 1: right, 2: opposite, 3: left
        16-bit layout: [15..14] unused, [13..11] src, [10..7] target, [6..0] action
        """
        val = (jnp.uint16(src) << 13) | (jnp.uint16(target) << 7) | jnp.uint16(action)
        return val

    @staticmethod
    def empty() -> Array:
        return EMPTY_MELD

    @staticmethod
    def is_empty(meld: Array) -> bool:
        return meld == EMPTY_MELD

    @staticmethod
    def src(meld: Array) -> Array:
        """
        - Encoded in relative position (0-3), 0: self (only for closed kan), 1: right, 2: opposite, 3: left
        - return -1 if empty
        """
        is_emp = Meld.is_empty(meld)
        return jnp.where(is_emp, jnp.int32(-1), (meld >> 13) & jnp.uint16(0b11))

    @staticmethod
    def target(meld: Array) -> Array:
        """
        Target tile index (0-34)
        Return -1 if empty
        """
        is_emp = Meld.is_empty(meld)
        return jnp.where(is_emp, jnp.int32(-1), (meld >> 7) & jnp.uint16(0b111111))

    @staticmethod
    def action(meld: Array) -> Array:
        """
        Action type (0-78)
        return -1 if empty
        """
        is_emp = Meld.is_empty(meld)
        return jnp.where(is_emp, jnp.int32(-1), meld & jnp.uint16(0b1111111))

    @staticmethod
    def suited_pung(meld: Array) -> Array:
        """
        Is suited pung << target tile
        """
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        target = Meld.target(meld)
        is_pung = (
            (action == Action.PON)
            | (action == Action.OPEN_KAN)
            | Action.is_selfkan(action)
        )
        is_suited_pon = is_pung & (target < 27) & (~is_emp)
        return is_suited_pon.astype(jnp.int32) << target

    @staticmethod
    def chow(meld: Array) -> Array:
        """
        Is chow << target tile
        """
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        is_chi = ((Action.CHI_L <= action) & (action <= Action.CHI_R)) & (~is_emp)
        pos = Meld.target(meld) - (action - Action.CHI_L)
        pos = pos * is_chi.astype(jnp.int32)
        return is_chi.astype(jnp.int32) << pos

    @staticmethod
    def is_kan(meld: Array) -> bool:
        """
        Whether the meld is a kan (open kan or self kan)
        """
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        return (
            (action == Action.OPEN_KAN)
            | (
                (Tile.NUM_TILE_TYPE <= action) & (action < Action.TSUMOGIRI)
            )  # selfkan (34~67)
        ) & (~is_emp)

    @staticmethod
    def is_closed_kan(meld: Array) -> bool:
        """
        Whether the meld is a closed kan (暗槓)
        """
        is_emp = Meld.is_empty(meld)
        src = Meld.src(meld)
        action = Meld.action(meld)
        return (
            (src == 0)
            & (Tile.NUM_TILE_TYPE <= action)
            & (action < Action.TSUMOGIRI)
            & (~is_emp)
        )

    @staticmethod
    def is_added_kan(meld: Array) -> bool:
        """
        Whether the meld is a added kan (加槓)
        """
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        return (
            (Tile.NUM_TILE_TYPE <= action)
            & (action < Action.TSUMOGIRI)
            & (Meld.src(meld) != 0)
        ) & (~is_emp)

    @staticmethod
    def is_chi(meld: Array) -> bool:
        """
        Whether the meld is a chi (吃)
        """
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        return ((Action.CHI_L <= action) & (action <= Action.CHI_R)) & (~is_emp)

    @staticmethod
    def is_pon(meld: Array) -> bool:
        """
        Whether the meld is a pon (碰)
        """
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        return (action == Action.PON) & (~is_emp)

    @staticmethod
    def is_outside(meld: Array) -> Array:
        """
        Whether the meld is constructed by (1, 9 or hornors) tiles (幺九牌 or 字牌)
        Example:
        - 111m -> True
        - EEE  -> True
        - 123m -> False
        """
        is_emp = Meld.is_empty(meld)
        target = Meld.target(meld)
        action = Meld.action(meld)
        is_pon_or_kan = (
            (action == Action.PON)
            | (action == Action.OPEN_KAN)
            | Action.is_selfkan(action)
        )
        num = target % 9
        return ((target >= 27) | (num == 0) | (num == 8)) & (~is_emp) & (is_pon_or_kan)

    @staticmethod
    def has_outside(meld: Array) -> Array:
        """
        Whether the meld has outside tiles (幺九牌 or 字牌) (1, 9 or hornors)
        Example:
        - 111m -> True
        - EEE  -> True
        - 123m -> True
        - 234m -> False
        """
        is_emp = Meld.is_empty(meld)
        target = Meld.target(meld)
        action = Meld.action(meld)
        num = target % 9
        is_outside = Meld.is_outside(meld)
        # For chi
        for_chi_l = (action == Action.CHI_L) & (
            (num == 0) | (num == 6)
        )  # [1]23 or [7]89
        for_chi_m = (action == Action.CHI_M) & (
            (num == 1) | (num == 7)
        )  # 1[2]3 or 7[8]9
        for_chi_r = (action == Action.CHI_R) & (
            (num == 2) | (num == 8)
        )  # 12[3] or 78[9]
        return is_outside | for_chi_l | for_chi_m | for_chi_r

    @staticmethod
    def fu(meld: Array) -> Array:
        """
        Calculate the fu of the meld
        - Pon is 2 Fu
        - Open Kan is 8 Fu
        - Added Kan is 8 Fu, Closed Kan is 16 Fu
        - Double Fu for Outside and Honor tiles
        """
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        base = (
            (action == Action.PON) * 2  # Pon is 2 Fu
            + (action == Action.OPEN_KAN) * 8  # Open Kan is 8 Fu
            + (
                Action.is_selfkan(action) * 8 * (1 + (Meld.src(meld) == 0))
            )  # Added Kan is 8 Fu, Closed Kan is 16 Fu
        )
        base = base * (~is_emp)
        return base * (
            1 + (Meld.is_outside(meld))
        )  # Double Fu for Outside and Honor tiles

    @staticmethod
    def exist_prohibitive_tile_type_after_chi(action: Array, target: Array) -> Array:
        """
        To detect swap-calling.
        """
        for_chi_l = (action == Action.CHI_L) & ~Tile.is_tile_type_seven(target)
        for_chi_r = (action == Action.CHI_R) & ~Tile.is_tile_type_three(target)
        return for_chi_l | for_chi_r

    @staticmethod
    def prohibitive_tile_type_after_chi(action: Array, target: Array) -> Array:
        """
        Limit swap-calling for chi_l and chi_r.
        Example:
        - [1]23 -> 4m
        - 67[8] -> 5m
        are prohibited to discard.
        """
        for_chi_l = ((action == Action.CHI_L) & ~Tile.is_tile_type_seven(target)) * (
            target + 3
        )
        for_chi_r = ((action == Action.CHI_R) & ~Tile.is_tile_type_three(target)) * (
            target - 3
        )
        return jax.lax.cond(
            Meld.exist_prohibitive_tile_type_after_chi(action, target),
            lambda: jnp.int8(for_chi_l + for_chi_r),
            lambda: jnp.int8(-1),
        )

    @staticmethod
    def to_str(meld: Array) -> str:
        is_emp = Meld.is_empty(meld)
        action = Meld.action(meld)
        target = Meld.target(meld)
        src = Meld.src(meld)

        def fmt():
            suit, num = target // 9, target % 9 + 1
            if action == Action.PON:
                if src == 1:
                    return "{}{}[{}]{}".format(
                        num, num, num, ["m", "p", "s", "z"][suit]
                    )
                elif src == 2:
                    return "{}[{}]{}{}".format(
                        num, num, num, ["m", "p", "s", "z"][suit]
                    )
                elif src == 3:
                    return "[{}]{}{}{}".format(
                        num, num, num, ["m", "p", "s", "z"][suit]
                    )
            elif Action.is_selfkan(action):
                if src == 0:
                    return "{}{}{}{}{}".format(
                        num, num, num, num, ["m", "p", "s", "z"][suit]
                    )
                if src == 1:
                    return "{}{}[{}{}]{}".format(
                        num, num, num, num, ["m", "p", "s", "z"][suit]
                    )
                elif src == 2:
                    return "{}[{}{}]{}{}".format(
                        num, num, num, num, ["m", "p", "s", "z"][suit]
                    )
                elif src == 3:
                    return "[{}{}]{}{}{}".format(
                        num, num, num, num, ["m", "p", "s", "z"][suit]
                    )
            elif action == Action.OPEN_KAN:
                if src == 1:
                    return "{}{}{}[{}]{}".format(
                        num, num, num, num, ["m", "p", "s", "z"][suit]
                    )
                elif src == 2:
                    return "{}[{}]{}{}{}".format(
                        num, num, num, num, ["m", "p", "s", "z"][suit]
                    )
                elif src == 3:
                    return "[{}]{}{}{}{}".format(
                        num, num, num, num, ["m", "p", "s", "z"][suit]
                    )
            elif Action.CHI_L <= action <= Action.CHI_R:
                assert src == 3
                pos = action - Action.CHI_L
                t = [num - pos + i for i in range(3)]
                t.insert(0, t.pop(pos))
                return "[{}]{}{}{}".format(*t, ["m", "p", "s", "z"][suit])
            return ""  # Invalid or empty

        return jax.lax.cond(is_emp, lambda: "", fmt)
