import unittest

import jax
import jax.numpy as jnp
import json
from pathlib import Path

from mahjax.red_mahjong.shanten import Shanten

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
