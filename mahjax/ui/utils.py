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


from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import jax.numpy as jnp

HONOR_NAMES_JP = ["東", "南", "西", "北", "白", "發", "中"]
SUIT_NAMES = ["m", "p", "s"]

_NUMERAL_LABELS = [
    f"{num}{SUIT_NAMES[suit]}" for suit in range(3) for num in range(1, 10)
]
TILE_LABELS = _NUMERAL_LABELS + HONOR_NAMES_JP


def tile_label(tile: int) -> str:
    if tile == 34:
        return "5mr"
    if tile == 35:
        return "5pr"
    if tile == 36:
        return "5sr"
    if 0 <= tile < len(TILE_LABELS):
        return TILE_LABELS[tile]
    raise ValueError(f"Invalid tile index: {tile}")


def tile_labels(tiles: Sequence[int]) -> List[str]:
    return [tile_label(int(t)) for t in tiles]


@dataclass
class MeldDisplay:
    action: int
    tiles: List[int]

    @property
    def labels(self) -> List[str]:
        return tile_labels(self.tiles)


def flatten_bool_mask(mask: jnp.ndarray) -> List[int]:
    """Return indices where the mask is truthy as Python ints."""
    indices: Iterable[int] = jnp.where(mask)[0].tolist()
    return [int(i) for i in indices]
