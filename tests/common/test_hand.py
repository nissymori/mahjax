import jax
import jax.numpy as jnp
import numpy as np

from mahjax.common.action import Action
from mahjax.common.hand import Hand
from mahjax.common.tile import Tile


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


def test_can_ron_is_false_on_a_fifth_copy() -> None:
    """66 7777 99m EEE CC holds every 7m; a fifth one would carry into 6m and read 666 99m EEE CC."""
    hand = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8)
    hand = hand.at[5].set(2).at[6].set(4).at[8].set(2).at[27].set(3).at[33].set(2)
    for h in (hand, Hand.to_34(hand)):
        assert not bool(jax.vmap(Hand.can_ron, in_axes=(None, 0))(h, jnp.arange(Tile.NUM_TILE_TYPE)).any())

    # A red five counts toward the four: 44 5550 77m EEE CC.
    hand = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8)
    hand = hand.at[3].set(2).at[4].set(3).at[Tile.RED_FIVE["m"]].set(1).at[6].set(2).at[27].set(3).at[33].set(2)
    assert not bool(Hand.can_ron(hand, Tile.BLACK_FIVE["m"]))
    assert not bool(Hand.can_ron(hand, Tile.RED_FIVE["m"]))
    assert not bool(Hand.can_ron(Hand.to_34(hand), Tile.BLACK_FIVE["m"]))

    # Melded copies are not in the hand: a 7m single wait next to a pon of 7m still counts, as on Tenhou.
    hand = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8).at[6].set(1).at[9:18].set(1)  # 7m 123456789p
    assert bool(Hand.can_ron(hand, 6))


def test_can_ron_never_waits_on_a_tile_held_four_times() -> None:
    """Every one-suit shape with a quad, padded to 13 tiles with honor pons and a pair."""
    shapes = np.stack(np.meshgrid(*[np.arange(5, dtype=np.int8)] * 9, indexing="ij"), axis=-1).reshape(-1, 9)
    n = shapes.sum(axis=1)
    shapes = shapes[(shapes == 4).any(axis=1) & (n % 3 != 0) & (n <= 13)]
    n = shapes.sum(axis=1)
    hands = np.zeros((len(shapes), Tile.NUM_TILE_TYPE), dtype=np.int8)
    hands[:, :9] = shapes
    hands[:, 27:31] = 3 * (np.arange(4) < ((13 - n) // 3)[:, None])  # EEE SSS WWW NNN
    hands[:, 33] = 2 * (n % 3 == 2)  # CC
    assert (hands.sum(axis=1) == 13).all()

    can_win = jax.jit(jax.vmap(jax.vmap(Hand.can_ron, in_axes=(None, 0)), in_axes=(0, None)))(
        jnp.asarray(hands), jnp.arange(9)
    )

    assert not bool((can_win & (hands[:, :9] == 4)).any())


def test_red_call_logic_matches_expected_reference_cases() -> None:
    hand = jnp.zeros((37,), dtype=jnp.int8)
    hand = hand.at[Tile.BLACK_FIVE['p']].set(2)
    hand = hand.at[Tile.RED_FIVE['p']].set(1)
    hand = hand.at[12].set(1)  # 4p

    assert bool(Hand.can_red_pon(hand, Tile.BLACK_FIVE['p']))
    assert bool(Hand.can_open_kan(hand, Tile.BLACK_FIVE['p']))
    assert bool(Hand.can_open_kan(hand, Tile.RED_FIVE['p']))
    assert bool(Hand.can_chi(hand, 14, Action.CHI_R_RED))

    pon_hand = Hand.pon(hand, Tile.BLACK_FIVE['p'], Action.PON_RED)
    assert int(pon_hand[Tile.BLACK_FIVE['p']]) == 1
    assert int(pon_hand[Tile.RED_FIVE['p']]) == 0

    chi_hand = Hand.chi(hand, 14, Action.CHI_R_RED)
    assert int(chi_hand[12]) == 0
    assert int(chi_hand[Tile.RED_FIVE['p']]) == 0
    assert int(chi_hand[14]) == 0

    open_kan_black_target = Hand.open_kan(hand, Tile.BLACK_FIVE['p'])
    assert int(open_kan_black_target[Tile.BLACK_FIVE['p']]) == 0
    assert int(open_kan_black_target[Tile.RED_FIVE['p']]) == 0

    hand_red_target = jnp.zeros((37,), dtype=jnp.int8).at[Tile.BLACK_FIVE['m']].set(3)
    open_kan_red_target = Hand.open_kan(hand_red_target, Tile.RED_FIVE['m'])
    assert int(open_kan_red_target[Tile.BLACK_FIVE['m']]) == 0

    closed_kan_hand = jnp.zeros((37,), dtype=jnp.int8)
    closed_kan_hand = closed_kan_hand.at[Tile.BLACK_FIVE['s']].set(3)
    closed_kan_hand = closed_kan_hand.at[Tile.RED_FIVE['s']].set(1)
    after_closed_kan = Hand.closed_kan(closed_kan_hand, Tile.BLACK_FIVE['s'])
    assert int(after_closed_kan[Tile.BLACK_FIVE['s']]) == 0
    assert int(after_closed_kan[Tile.RED_FIVE['s']]) == 0
