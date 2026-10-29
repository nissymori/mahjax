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
from typing import Tuple
import importlib.resources as resources

import jax
import jax.numpy as jnp
import numpy as np

from .types import Array
from .hand import THIRTEEN_ORPHAN_IDX


def load_shanten_cache():
    with resources.as_file(resources.files("mahjax._src.cache").joinpath("shanten_cache.npz")) as path:
        with np.load(path, allow_pickle=False) as data:
            return jnp.asarray(data["data"], dtype=jnp.uint32)


class Shanten:
    # See the link below for the algorithm details.
    # https://github.com/sotetsuk/pgx/pull/123
    CACHE = load_shanten_cache()
    # Flatten once at load time to avoid per-instance materialization in XLA.
    CACHE_FLAT = CACHE.reshape(-1)

    @staticmethod
    def discard(hand: Array) -> Array:
        # Cond-free formulation: wrapping Shanten.number in lax.cond under the
        # 34-way vmap makes XLA materialize a copy of the ~70MiB CACHE per lane
        # (74GiB temp at batch 32); computing all candidates and masking with
        # where compiles to plain gathers with ~0 temp memory.
        eye = jnp.eye(34, dtype=hand.dtype)
        cand = jnp.maximum(hand[None, :] - eye, 0)  # (34, 34)
        res = jax.vmap(Shanten.number)(cand)  # (34,)
        return jnp.where(hand > 0, res, jnp.int32(6))

    @staticmethod
    def detailed_discard(hand: Array) -> Array:
        # See discard() for why this avoids lax.cond.
        eye = jnp.eye(34, dtype=hand.dtype)
        cand = jnp.maximum(hand[None, :] - eye, 0)  # (34, 34)
        res = jax.vmap(Shanten.detailed_number)(cand)  # (34, 3)
        return jnp.where((hand > 0)[:, None], res, jnp.int32(6))

    # Sentinel meaning "the hand does not hold this tile" in
    # detailed_discard_shanten. It must sit above every reachable normalized
    # value; the widest column is thirteen orphans, which tops out at 13.
    NOT_IN_HAND: int = 14

    @staticmethod
    def detailed_discard_shanten(hand: Array, fill: int = 14) -> Array:
        """Standard-notation shanten after discarding each tile type, split by hand shape.

        Returns ``(34, 3)`` with columns ``[normal, seven pairs, thirteen orphans]``,
        each already in standard shanten notation (the ``- 1`` that :meth:`number`
        applies), so ``out.min(axis=1) == Shanten.discard(hand)`` on held tiles.

        Reachable ranges are ``[0, 8]`` / ``[0, 13]`` / ``[0, 13]``. Note the normal
        column is *not* bounded by 6: the combined shanten is, but only because the
        seven-pairs and thirteen-orphans decompositions undercut the normal one on
        exactly the hands where it is worst. ``min(axis=1)`` is in ``[0, 6]``.

        13 is therefore attainable and :attr:`NOT_IN_HAND` == 14 clears it by exactly
        one. Do not lower the sentinel.

        This is the observation-facing variant of :meth:`detailed_discard`, which
        returns *raw* costs and fills absent tiles with ``6`` -- a value that collides
        with real entries in all three columns. ``fill`` defaults to :attr:`NOT_IN_HAND`
        (14), outside every column's range, so "not in hand" stays distinguishable from
        a genuinely awful discard. :meth:`detailed_discard` keeps its old behaviour
        because ``players.py`` depends on it.

        Cost is the same as :meth:`discard`: ``number`` and ``detailed_number`` issue
        the identical normal / seven-pairs / thirteen-orphans trio, so the two extra
        columns are the same cache lookups with a wider output. See :meth:`discard` for
        why this must stay free of ``lax.cond``.
        """
        eye = jnp.eye(34, dtype=hand.dtype)
        cand = jnp.maximum(hand[None, :] - eye, 0)  # (34, 34)
        res = jax.vmap(Shanten.detailed_number)(cand) - jnp.int32(1)  # (34, 3)
        return jnp.where((hand > 0)[:, None], res, jnp.int32(fill))

    @staticmethod
    def number(hand: Array) -> Array:
        return (
            jnp.min(
                jnp.array(
                    [
                        Shanten.normal(hand),
                        Shanten.seven_pairs(hand),
                        Shanten.thirteen_orphan(hand),
                    ]
                )
            )
            - 1
        )  # Standard shanten number notation

    @staticmethod
    def detailed_number(hand: Array) -> Array:
        return jnp.array(
            [
                Shanten.normal(hand),
                Shanten.seven_pairs(hand),
                Shanten.thirteen_orphan(hand),
            ]
        )

    @staticmethod
    def seven_pairs(hand: Array) -> Array:
        n_pair = jnp.sum(hand >= 2)
        n_kind = jnp.sum(hand > 0)
        return 7 - n_pair + jax.lax.max(7 - n_kind, 0)

    @staticmethod
    def thirteen_orphan(hand: Array) -> Array:
        n_pair = jnp.sum(hand[THIRTEEN_ORPHAN_IDX] >= 2)
        n_kind = jnp.sum(hand[THIRTEEN_ORPHAN_IDX] > 0)
        return 14 - n_kind - (n_pair > 0)

    @staticmethod
    def normal(hand: Array) -> Array:
        # Marginal meld costs can decrease, so greedy allocation is not exact.
        # Enumerate all allocations of up to four melds and choose the pair suit.
        powers = jnp.asarray([5**i for i in range(8, -1, -1)], dtype=jnp.int32)
        suited = hand[:27].astype(jnp.int32).reshape(3, 9) @ powers
        honors = hand[27:34].astype(jnp.int32) @ powers[2:] + 5**9
        codes = jnp.concatenate([suited, honors[None]])
        rows = jnp.take(
            Shanten.CACHE_FLAT, codes[:, None] * 9 + jnp.arange(9, dtype=jnp.int32)
        ).astype(jnp.int32)

        # Cache columns: four meld increments, pair cost, four increments with a pair.
        zero = jnp.zeros((4, 1), dtype=jnp.int32)
        without_pair = jnp.concatenate([zero, jnp.cumsum(rows[:, :4], axis=1)], axis=1)
        with_pair = rows[:, 4:5] + jnp.concatenate([zero, jnp.cumsum(rows[:, 5:], axis=1)], axis=1)
        allocations = jnp.asarray(
            [
                (a, b, c, d)
                for a in range(5)
                for b in range(5 - a)
                for c in range(5 - a - b)
                for d in range(5 - a - b - c)
            ],
            dtype=jnp.int32,
        )
        suits = jnp.arange(4)[None, :]
        costs = without_pair[suits, allocations]
        pair_extra = with_pair[suits, allocations] - costs
        totals = costs.sum(axis=1) + pair_extra.min(axis=1)
        n_set = jnp.minimum(jnp.sum(hand, dtype=jnp.int32) // 3, 4)
        return jnp.min(jnp.where(allocations.sum(axis=1) == n_set, totals, jnp.iinfo(jnp.int32).max)).astype(jnp.int32)
