import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from mahjax.no_red_mahjong.tile import Tile
from mahjax.no_red_mahjong.hand import Hand
from mahjax.no_red_mahjong.action import Action

jitted_sub = jax.jit(Hand.sub)
jitted_add = jax.jit(Hand.add)
jitted_can_pon = jax.jit(Hand.can_pon)
jitted_can_chi = jax.jit(Hand.can_chi)
jitted_make_init_hand = jax.jit(Hand.make_init_hand)
jitted_chi = jax.jit(Hand.chi)
jitted_can_closed_kan_after_riichi = jax.jit(Hand.can_closed_kan_after_riichi)
jitted_can_open_kan = jax.jit(Hand.can_open_kan)
jitted_open_kan = jax.jit(Hand.open_kan)
jitted_added_kan = jax.jit(Hand.added_kan)
jitted_closed_kan = jax.jit(Hand.closed_kan)
jitted_can_tsumo = jax.jit(Hand.can_tsumo)
jitted_can_win = jax.jit(Hand.can_ron)
jitted_is_tenpai = jax.jit(Hand.is_tenpai)
jitted_can_closed_kan = jax.jit(Hand.can_closed_kan)
jitted_pon = jax.jit(Hand.pon)


class TestHand(unittest.TestCase):
    def test_add_sub(self):
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 0)  # 1m
        self.assertEqual(hand[0], 1)
        hand = jitted_sub(hand, 0)
        self.assertEqual(hand[0], 0)


    def test_can_pon(self):
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 0)  # 1m
        hand = jitted_add(hand, 0)  # 1m
        self.assertTrue(jitted_can_pon(hand, 0))  # 1m can be pon
        self.assertFalse(jitted_can_pon(hand, 1))  # 2m cannot be pon
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 4)  # 5m
        hand = jitted_add(hand, 4)  # 5m
        self.assertTrue(jitted_can_pon(hand, 4))  # 5m can be pon


    def test_can_chi(self):
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 1)  # 2m
        hand = jitted_add(hand, 2)  # 3m
        self.assertTrue(jitted_can_chi(hand, 0, Action.CHI_L))  # 1m can be chi_l
        self.assertFalse(jitted_can_chi(hand, 0, Action.CHI_M))  # 1m cannot be chi_m
        self.assertFalse(jitted_can_chi(hand, 0, Action.CHI_R))  # 1m cannot be chi_r


    def test_make_init_hand(self):
        rng_key = jax.random.PRNGKey(1)
        # Generate a deck of 136 tiles
        deck = jnp.arange(136, dtype=jnp.int8)
        # Shuffle the deck
        deck = jax.random.permutation(rng_key, deck) // 4  # Divide by 4 to get indices 0-33
        # Generate the initial hand
        hand = jitted_make_init_hand(deck)
        # Check the shape of the hand
        self.assertEqual(hand.shape, (4, Tile.NUM_TILE_TYPE))
        # Check if each player's initial hand has 13 tiles
        for player in range(4):
            self.assertEqual(jnp.sum(hand[player]), 13)

    def test_can_open_kan(self):
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 0, 3)  # 1m 3 tiles
        self.assertTrue(jitted_can_open_kan(hand, 0))  # 1m can be open kan


    def test_can_closed_kan(self):
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 0, 4)  # 1m 4 tiles
        self.assertTrue(jitted_can_closed_kan(hand, 0))  # 1m can be closed kan


    def test_can_tsumo(self):
        # Test seven pairs
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = hand.at[:7].set(2)
        self.assertTrue(jitted_can_tsumo(hand))
        # Test four melds and one head
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        # 111m + 234m + 567p + 888s + 99s
        hand = Hand.add(hand, 0, 3)  # 1m x 3
        hand = Hand.add(hand, 1)  # 2m
        hand = Hand.add(hand, 2)  # 3m
        hand = Hand.add(hand, 3)  # 4m
        hand = Hand.add(hand, 13)  # 5p
        hand = Hand.add(hand, 14)  # 6p
        hand = Hand.add(hand, 15)  # 7p
        hand = Hand.add(hand, 25, 3)  # 8s x 3
        hand = Hand.add(hand, 26, 2)  # 9s x 2
        self.assertTrue(jitted_can_tsumo(hand))
        self.assertFalse(jitted_can_tsumo(hand.at[25].set(2))) # 13 tiles, cannot tsumo

    def test_can_win(self):
        # Test seven pairs
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = hand.at[:7].set(2).at[6].set(1)
        self.assertTrue(jitted_can_win(hand, 6))
        # Test four melds and one head
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        # 111m + 34m + 567p + 888s + 99s
        hand = Hand.add(hand, 0, 3)  # 1m x 3
        hand = Hand.add(hand, 2)  # 3m
        hand = Hand.add(hand, 3)  # 4m
        hand = Hand.add(hand, 13)  # 5p
        hand = Hand.add(hand, 14)  # 6p
        hand = Hand.add(hand, 15)  # 7p
        hand = Hand.add(hand, 25, 3)  # 8s x 3
        hand = Hand.add(hand, 26, 2)  # 9s x 2
        self.assertTrue(jitted_can_win(hand, 4)) # 5m can be ron
        self.assertTrue(jitted_can_win(hand, 1)) # 2m can be ron
        self.assertFalse(jitted_can_win(hand, 25)) # 8s cannot be ron

    def test_can_win_is_false_on_a_fifth_copy(self):
        # 66 7777 99m EEE CC holds every 7m; a fifth one would carry into 6m and read 666 99m EEE CC.
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.int8)
        hand = hand.at[5].set(2).at[6].set(4).at[8].set(2).at[27].set(3).at[33].set(2)
        self.assertFalse(jax.vmap(jitted_can_win, in_axes=(None, 0))(hand, jnp.arange(Tile.NUM_TILE_TYPE)).any())
        # Melded copies are not in the hand: a 7m single wait next to a pon of 7m still counts, as on Tenhou.
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.int8).at[6].set(1).at[9:18].set(1)  # 7m 123456789p
        self.assertTrue(jitted_can_win(hand, 6))

    def test_can_win_never_waits_on_a_tile_held_four_times(self):
        # Every one-suit shape with a quad, padded to 13 tiles with honor pons and a pair.
        shapes = np.stack(np.meshgrid(*[np.arange(5, dtype=np.int8)] * 9, indexing="ij"), axis=-1).reshape(-1, 9)
        n = shapes.sum(axis=1)
        shapes = shapes[(shapes == 4).any(axis=1) & (n % 3 != 0) & (n <= 13)]
        n = shapes.sum(axis=1)
        hands = np.zeros((len(shapes), Tile.NUM_TILE_TYPE), dtype=np.int8)
        hands[:, :9] = shapes
        hands[:, 27:31] = 3 * (np.arange(4) < ((13 - n) // 3)[:, None])  # EEE SSS WWW NNN
        hands[:, 33] = 2 * (n % 3 == 2)  # CC
        self.assertTrue((hands.sum(axis=1) == 13).all())

        can_win = jax.jit(jax.vmap(jax.vmap(Hand.can_ron, in_axes=(None, 0)), in_axes=(0, None)))(
            jnp.asarray(hands), jnp.arange(9)
        )

        self.assertFalse((can_win & (hands[:, :9] == 4)).any())

    def test_is_tenpai(self):
        # Test seven pairs
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = hand.at[:7].set(2).at[6].set(1)
        self.assertTrue(jitted_is_tenpai(hand))
        # Test four melds and one head
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        # 111m + 34m + 567p + 888s + 99s
        hand = Hand.add(hand, 0, 3)  # 1m x 3
        hand = Hand.add(hand, 2)  # 3m
        hand = Hand.add(hand, 3)  # 4m
        hand = Hand.add(hand, 13)  # 5p
        hand = Hand.add(hand, 14)  # 6p
        hand = Hand.add(hand, 15)  # 7p
        hand = Hand.add(hand, 25, 3)  # 8s x 3
        hand = Hand.add(hand, 26, 2)  # 9s x 2
        self.assertTrue(jitted_is_tenpai(hand))
        self.assertFalse(jitted_is_tenpai(hand.at[25].set(2).at[24].set(1)))

    def test_chi(self):
        # チーのテスト
        # Test chi_l (1m, 2m, 3m)
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 1)  # 2m
        hand = jitted_add(hand, 2)  # 3m
        # Play chi_l (0=1m for chi_l): call 1m and consume 2m, 3m
        new_hand = jitted_chi(hand, 0, Action.CHI_L)
        # 2m and 3m are consumed
        self.assertEqual(new_hand[1], 0)  # 2m is consumed
        self.assertEqual(new_hand[2], 0)  # 3m is consumed
        # Test chi_m (2m, 3m, 4m)
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 1)  # 2m
        hand = jitted_add(hand, 3)  # 4m
        # Play chi_m (2=3m for chi_m): call 3m and consume 2m, 4m
        new_hand = jitted_chi(hand, 2, Action.CHI_M)
        # 2m and 4m are consumed
        self.assertEqual(new_hand[1], 0)  # 2m is consumed
        self.assertEqual(new_hand[3], 0)  # 4m is consumed
        # Test chi_r (3m, 4m, 5m)
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 2)  # 3m
        hand = jitted_add(hand, 3)  # 4m
        # Play chi_r (4=5m for chi_r): call 5m and consume 3m, 4m
        new_hand = jitted_chi(hand, 4, Action.CHI_R)
        # 3m and 4m are consumed
        self.assertEqual(new_hand[2], 0)  # 3m is consumed
        self.assertEqual(new_hand[3], 0)  # 4m is consumed

    def test_pon(self):
        # Test pon
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 0, 2)  # 1m x 2
        # Play pon
        new_hand = jitted_pon(hand, 0)
        # 1m is consumed
        self.assertEqual(new_hand[0], 0)
        # Test pon for 5m
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 4, 2)  # 5m x 2
        # Play pon
        new_hand = jitted_pon(hand, 4)
        # 5m is consumed
        self.assertEqual(new_hand[4], 0)

    def test_open_kan(self):
        # Test open kan
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 0, 3)  # 1m x 3
        self.assertTrue(jitted_can_open_kan(hand, 0))  # 1m can be open kan
        result_hand = jitted_open_kan(hand, 0)
        self.assertEqual(result_hand[0], 0)   # 1m is consumed

    def test_added_kan(self):
        # Test added kan
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 0)  # 1m
        result_hand = jitted_added_kan(hand, 0)
        self.assertEqual(result_hand[0], 0)  # 1m is consumed

    def test_closed_kan(self):
        # Test closed kan
        hand = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.uint8)
        hand = jitted_add(hand, 0, 4)  # 1m x 4
        result_hand = jitted_closed_kan(hand, 0)
        self.assertEqual(result_hand[0], 0)   # 1m is consumed


    def test_can_closed_kan_after_riichi(self):
        """
        Test closed kan after riichi
        Tenhou considers only the change in the waiting tile.
        Reference: https://mj-dragon.com/rule/prac/closed_kan.html
        """
        can_kan_hand = jnp.array([
            1, 1, 1, 1, 1, 1, 0, 0, 0,
            0, 4, 0, 1, 0, 0, 0, 0, 0,
            3, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0
        ], dtype=jnp.uint8) # 123456m 2223p EEEE (waiting tile: 4p)
        can_kan_original_can_win = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.bool_).at[12].set(True)
        cannot_kan_hand = jnp.array([
            1, 1, 1, 1, 1, 1, 0, 0, 0,
            0, 4, 1, 0, 0, 0, 0, 0, 0,
            3, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0
        ], dtype=jnp.uint8) # 123456m 2223p EEEE (waiting tile: 14p)
        cannot_kan_original_can_win = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.bool_).at[9].set(True).at[12].set(True)

        self.assertTrue(jitted_can_closed_kan_after_riichi(can_kan_hand, 10, can_kan_original_can_win))
        self.assertFalse(jitted_can_closed_kan_after_riichi(cannot_kan_hand, 10, cannot_kan_original_can_win))



if __name__ == "__main__":
    unittest.main()