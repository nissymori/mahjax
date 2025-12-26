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
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from mahjax._src.types import Array
from mahjax.no_red_mahjong.hand import THIRTEEN_ORPHAN_IDX

DIR = os.path.join(os.path.dirname(__file__), "cache")


def load_shanten_cache():
    with np.load(os.path.join(DIR, "shanten_cache.npz"), allow_pickle=False) as data:
        return jnp.asarray(data["data"], dtype=jnp.uint32)


class Shanten:
    # See the link below for the algorithm details.
    # https://github.com/sotetsuk/pgx/pull/123
    CACHE = load_shanten_cache()

    @staticmethod
    def discard(hand: Array) -> Array:
        # i to int32, align the dtype of both branches of cond
        def f(i):
            i = jnp.int32(i)
            return jax.lax.cond(
                hand[i] == 0,
                lambda: jnp.int32(6),
                lambda: Shanten.number(hand.at[i].set(hand[i] - 1)),
            )

        return jax.vmap(f)(jnp.arange(34, dtype=jnp.int32))

    def detailed_discard(hand: Array) -> Array:
        def f(i):
            i = jnp.int32(i)
            return jax.lax.cond(
                hand[i] == 0,
                lambda: jnp.array([6, 6, 6]),
                lambda: Shanten.detailed_number(hand.at[i].set(hand[i] - 1)),
            )

        return jax.vmap(f)(jnp.arange(34, dtype=jnp.int32))  # (34, 3)

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
        # --- code generation (int32 fixed) ---
        def encode_suit(suit):
            def loop_rng(start, stop):
                def body(i, code):
                    return code * jnp.int32(5) + hand[i].astype(jnp.int32)

                return jax.lax.fori_loop(start, stop, body, jnp.int32(0))

            return jax.lax.cond(
                suit == 3,
                lambda: loop_rng(27, 34) + jnp.int32(1953125),  # 5**9
                lambda: loop_rng(9 * suit, 9 * (suit + 1)),
            )

        code = jax.vmap(encode_suit)(jnp.arange(4, dtype=jnp.int32))  # (4,)
        # n_set is OK for tracer. Guarded by a fixed number of loops and cond later.
        n_set = jnp.minimum(
            jnp.sum(hand, dtype=jnp.int32) // jnp.int32(3), jnp.int32(4)
        )
        # --- 1 element only gather ---
        CACHE = Shanten.CACHE  # (N_code, 9)
        J = jnp.int32(CACHE.shape[1])  # == 9

        def gather_elem(c, idx):
            lin = c * J + idx
            return jnp.take(CACHE.reshape(-1), lin)

        # 4 variants simultaneously calculate
        base_costs = gather_elem(code, jnp.full((4,), 4, dtype=jnp.int32))  # (4,)

        idx = jnp.zeros((4, 4), dtype=jnp.int32)  # (variant, suit)
        idx = idx.at[jnp.arange(4), jnp.arange(4)].set(jnp.int32(5))
        codes_rect = jnp.broadcast_to(code, (4, 4))  # (4,4)

        def one_step(t, carry):
            cost, idx = carry
            # Get the candidate costs (4,4) and select the minimum suit
            cand = gather_elem(codes_rect, idx)  # (4,4)
            pick = jnp.argmin(cand, axis=1)  # (4,)
            delta = cand[jnp.arange(4), pick]  # (4,)
            inc = (jnp.arange(4)[None, :] == pick[:, None]).astype(jnp.int32)

            new_cost = cost + delta
            new_idx = idx + inc

            # Only update when t < n_set (scalar pred guards all variants)
            return jax.lax.cond(
                t < n_set,
                lambda: (new_cost, new_idx),
                lambda: (cost, idx),
            )

        # Fixed 4 steps (no variable length, so avoid Concretization)
        def fori_body(t, carry):
            return one_step(jnp.int32(t), carry)

        costs, _ = jax.lax.fori_loop(0, 4, fori_body, (base_costs, idx))
        return costs.min().astype(jnp.int32)
