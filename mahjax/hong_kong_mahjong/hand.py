"""Winning-hand and meld operations for Hong Kong mahjong."""

import jax
import jax.numpy as jnp

from mahjax._src.types import Array
from mahjax.no_red_mahjong.hand import Hand as RiichiHand

THIRTEEN_ORPHAN_IDX = jnp.array([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])


class Hand(RiichiHand):
    """A 34-count concealed hand; flowers are tracked outside the hand."""

    @staticmethod
    def can_tsumo(hand: Array) -> Array:
        thirteen_orphans = (hand[THIRTEEN_ORPHAN_IDX] > 0).all() & (hand[THIRTEEN_ORPHAN_IDX].sum() == 14)
        codes = (hand[:27].astype(int) * _POWERS_OF_5).reshape(3, 9).sum(axis=1)
        valid_suits = jax.vmap(RiichiHand.cache)(codes).all()
        suit_sums = jnp.sum(hand[:27].reshape(3, 9), axis=1)
        heads = jnp.sum((suit_sums % 3) == 2) + jnp.sum(hand[27:34] == 2)
        valid_honors = jnp.all((hand[27:34] != 1) & (hand[27:34] != 4))
        standard = valid_suits & valid_honors & (heads == 1)
        return (standard | thirteen_orphans) & (hand.sum() == 14)

    @staticmethod
    def can_ron(hand: Array, tile: Array) -> Array:
        return Hand.can_tsumo(Hand.add(hand, tile))

    @staticmethod
    def is_tenpai(hand: Array) -> Array:
        return jax.vmap(lambda tile: (hand[tile] != 4) & Hand.can_ron(hand, tile))(jnp.arange(34)).any()


_POWERS_OF_5 = jnp.concatenate([5 ** jnp.arange(8, -1, -1)] * 3)
