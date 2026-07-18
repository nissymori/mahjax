import unittest

import jax
import jax.numpy as jnp
import numpy as np

from mahjax.red_mahjong.shanten import Shanten

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


if __name__ == "__main__":
    unittest.main()
