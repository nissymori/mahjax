import jax
import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.hand import Hand
from mahjax.red_mahjong.tile import Tile


def test_make_init_hand_returns_red_aware_shape() -> None:
    rng = jax.random.PRNGKey(1)
    deck = Tile.from_tile_id_to_tile(jax.random.permutation(rng, jnp.arange(136)))
    hand = Hand.make_init_hand(deck)
    assert hand.shape == (4, Tile.NUM_TILE_TYPE_WITH_RED)
    assert bool(jnp.all(hand.sum(axis=1) == 13))


def test_red_pon_consumes_red_five_first() -> None:
    hand = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8)
    hand = hand.at[Tile.BLACK_FIVE["m"]].set(2)
    hand = hand.at[Tile.RED_FIVE["m"]].set(1)
    hand = hand.at[3].set(1)  # 4m
    hand = hand.at[5].set(1)  # 6m

    assert bool(Hand.can_red_pon(hand, Tile.BLACK_FIVE["m"]))
    after_pon = Hand.pon(hand, Tile.BLACK_FIVE["m"], Action.PON_RED)
    assert int(after_pon[Tile.BLACK_FIVE["m"]]) == 1
    assert int(after_pon[Tile.RED_FIVE["m"]]) == 0

def test_open_closed_added_kan_for_red_five_family() -> None:
    hand_open = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8)
    hand_open = hand_open.at[Tile.BLACK_FIVE["p"]].set(3)
    hand_open = hand_open.at[Tile.RED_FIVE["p"]].set(1)
    assert bool(Hand.can_open_kan(hand_open, Tile.BLACK_FIVE["p"]))
    open_after = Hand.open_kan(hand_open, Tile.BLACK_FIVE["p"])
    assert int(open_after.sum()) == int(hand_open.sum()) - 3

    hand_closed = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8)
    hand_closed = hand_closed.at[Tile.BLACK_FIVE["s"]].set(3)
    hand_closed = hand_closed.at[Tile.RED_FIVE["s"]].set(1)
    assert bool(Hand.can_closed_kan(hand_closed, Tile.BLACK_FIVE["s"]))
    closed_after = Hand.closed_kan(hand_closed, Tile.BLACK_FIVE["s"])
    assert int(closed_after.sum()) == int(hand_closed.sum()) - 4

    hand_added = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8)
    hand_added = hand_added.at[Tile.BLACK_FIVE["m"]].set(1)
    added_after = Hand.added_kan(hand_added, Tile.BLACK_FIVE["m"])
    assert int(added_after[Tile.BLACK_FIVE["m"]]) == 0
