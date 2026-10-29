import unittest

import jax
import jax.numpy as jnp
import json
from pathlib import Path

from mahjax.red_mahjong.shanten import Shanten
import numpy as np

jitted_number = jax.jit(Shanten.number)


class TestShanten(unittest.TestCase):
    def test_shanten(self):
        # Load test data
        test_dir = Path(__file__).resolve().parent
        test_file = test_dir / "assets" / "shanten.json"
        if not test_file.exists():
            test_file = test_dir.parent / "no_red_mahjong" / "assets" / "shanten.json"
        with open(test_file, "r") as f:
            data = json.load(f)

        for shanten_name, content in data.items():
            with self.subTest(shanten=shanten_name):
                hand = jnp.int32(content["hand"])
                num_tiles = jnp.int32(content["num_tiles"])
                expected_shanten = jnp.int32(content["shanten"])
                if num_tiles == 14:
                    shanten = jitted_number(hand)
                else:
                    shanten = jitted_number(hand)
                self.assertEqual(shanten, expected_shanten,
                                 f"Shanten mismatch for {shanten_name}: expected {expected_shanten}, got {shanten}")
                print(shanten_name, "passed")


jit_discard = jax.jit(Shanten.discard)


jit_detailed_discard = jax.jit(Shanten.detailed_discard)


jit_number = jax.jit(Shanten.number)


jit_detailed_number = jax.jit(Shanten.detailed_number)


class TestShantenDiscard(unittest.TestCase):
    """discard/detailed_discard must equal number/detailed_number applied to
    each one-tile-removed hand (and 6 for tiles not in hand)."""

    def test_discard_matches_number(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            tiles = rng.choice(136, size=14, replace=False)
            hand = np.bincount(tiles // 4, minlength=34).astype(np.int32)
            hand_j = jnp.asarray(hand)
            got = np.asarray(jit_discard(hand_j))
            got_detail = np.asarray(jit_detailed_discard(hand_j))
            for i in range(34):
                if hand[i] == 0:
                    self.assertEqual(got[i], 6)
                    self.assertTrue((got_detail[i] == 6).all())
                else:
                    reduced = hand_j.at[i].add(-1)
                    self.assertEqual(got[i], int(jit_number(reduced)))
                    np.testing.assert_array_equal(
                        got_detail[i], np.asarray(jit_detailed_number(reduced))
                    )
