import unittest
import json
import os

import jax
import jax.numpy as jnp
from jax import jit
from mahjax.no_red_mahjong.meld import EMPTY_MELD, Meld
from mahjax.no_red_mahjong.yaku import Yaku

class TestYaku(unittest.TestCase):
    def test_yaku(self):
        # Load test data
        test_file = os.path.join(os.path.dirname(__file__), "assets/yaku_test.json")
        with open(test_file, "r") as f:
            data = json.load(f)

        # Execute each test case
        for yaku_name, content in data.items():
            melds = jnp.where(jnp.array(content["melds"], dtype=jnp.uint16) == 0, jnp.array([EMPTY_MELD]*4, dtype=jnp.uint16), jnp.array(content["melds"], dtype=jnp.uint16))
            with self.subTest(yaku=yaku_name):
                hand = jnp.int8(content["hand"])
                melds = jnp.uint16(melds)
                n_meld = jnp.int8(content["n_meld"])
                last_tile = jnp.int8(content["last"])
                riichi = jnp.bool_(content["riichi"])
                dora = jnp.zeros((2, 34), dtype=jnp.bool_).at[0,content["dora_indices"][0]].set(True)
                is_ron = jnp.bool_(content["is_ron"])
                prevalent_wind = jnp.int8(content["prevalent_wind"])
                seat_wind = jnp.int8(content["seat_wind"])
                expected_score = jnp.int32(content["score"])
                yaku_indices = jnp.int32(content["yaku_indices"])
                expected_fan = jnp.int32(content["fan"])
                expected_fu = jnp.int32(content["fu"])

                # Judge yaku and calculate score (use jit for optimization)
                yaku, fan, fu = jit(Yaku.judge)(
                    hand=hand,
                    melds=melds,
                    n_meld=n_meld,
                    last_tile=last_tile,
                    riichi=riichi,
                    is_ron=is_ron,
                    dora=dora,
                    prevalent_wind=prevalent_wind,
                    seat_wind=seat_wind,
                )

                score = jit(Yaku.score)(
                    fan, fu
                )

                # Test assertions
                # Check if the value of the index of the corresponding yaku is all 1
                self.assertTrue(jnp.all(yaku[yaku_indices] == 1),
                                f"Yaku check failed for {yaku_name}, {yaku}")

                self.assertEqual(score, expected_score,
                                 f"Score mismatch for {yaku_name}: expected {expected_score}, got {score}")

                self.assertEqual(fan, expected_fan,
                                    f"Fan mismatch for {yaku_name}: expected {expected_fan}, got {fan}")
                self.assertEqual(fu, expected_fu,
                                    f"Fu mismatch for {yaku_name}: expected {expected_fu}, got {fu}")

                print(yaku_name, "passed")

    def test_yakuman(self):
        # Load test data
        test_file = os.path.join(os.path.dirname(__file__), "assets/yaku_test.json")
        with open(test_file, "r") as f:
            data = json.load(f)

        # Execute each test case
        for yakuman_name, content in data.items():
            melds = jnp.where(jnp.array(content["melds"], dtype=jnp.uint16) == 0, jnp.array([EMPTY_MELD]*4, dtype=jnp.uint16), jnp.array(content["melds"], dtype=jnp.uint16))
            with self.subTest(yakuman=yakuman_name):
                hand = jnp.int8(content["hand"])
                melds = jnp.uint16(melds)
                n_meld = jnp.int8(content["n_meld"])
                last_tile = jnp.int8(content["last"])
                riichi = jnp.bool_(content["riichi"])
                dora = jnp.zeros((2, 34), dtype=jnp.bool_).at[0,content["dora_indices"][0]].set(True)
                is_ron = jnp.bool_(content["is_ron"])
                prevalent_wind = jnp.int8(content["prevalent_wind"])
                seat_wind = jnp.int8(content["seat_wind"])
                expected_score = jnp.int32(content["score"])
                yaku_indices = jnp.int32(content["yaku_indices"])
                expected_fan = jnp.int32(content["fan"])
                expected_fu = jnp.int32(content["fu"])
                if expected_fu == 0:
                    yakuman, fan, fu = jit(Yaku.judge_yakuman)(
                        hand=hand,
                        melds=melds,
                        n_meld=n_meld,
                        last_tile=last_tile,
                        riichi=riichi,
                        is_ron=is_ron,
                        dora=dora,
                        prevalent_wind=prevalent_wind,
                        seat_wind=seat_wind,
                    )

                    score = jit(Yaku.score)(
                        fan, fu
                    )

                    # Test assertions
                    # Check if the value of the index of the corresponding yaku is all 1
                    self.assertTrue(jnp.all(yakuman[yaku_indices] == 1),
                                    f"Yakuman check failed for {yakuman_name}, {yakuman}")

                    self.assertEqual(score, expected_score,
                                     f"Score mismatch for {yakuman_name}: expected {expected_score}, got {score}")

                    self.assertEqual(fan, expected_fan,
                                    f"Fan mismatch for {yakuman_name}: expected {expected_fan}, got {fan}")
                    self.assertEqual(fu, expected_fu,
                                    f"Fu mismatch for {yakuman_name}: expected {expected_fu}, got {fu}")

                    print(yakuman_name, "passed")

    def test_all_green_counts_the_four_tiles_of_a_kan(self):
        # With a kan the hand holds 15 tiles, so All Green means no other tile, not 14 green ones.
        closed_kan_8s = Meld.init(34 + 25, 25, 0)
        melds = jnp.array([EMPTY_MELD] * 4, dtype=jnp.uint16).at[0].set(closed_kan_8s)
        for tiles, is_all_green in (
            ((19, 19, 20, 20, 21, 23, 23, 23, 32, 32), True),  # 8888s + 234s 234s 666s GrGr
            ((18, 19, 19, 20, 20, 21, 21, 23, 23, 23), False),  # 8888s + 123s 234s 666s 44s: 1s is not green
        ):
            hand = jnp.zeros((34,), dtype=jnp.int8)
            for tile in tiles:
                hand = hand.at[tile].add(1)
            yaku, _, _ = Yaku.judge(
                hand=hand,
                melds=melds,
                n_meld=jnp.int8(1),
                last_tile=jnp.int8(21),  # tsumo 4s
                riichi=jnp.bool_(False),
                is_ron=jnp.bool_(False),
                dora=jnp.zeros((2, 34), dtype=jnp.bool_),
                prevalent_wind=jnp.int8(0),
                seat_wind=jnp.int8(0),
            )
            self.assertEqual(bool(yaku[Yaku.AllGreen]), is_all_green, tiles)

    def test_score_is_a_limit_hand_from_five_fan_whatever_the_fu(self):
        # 13 fan or more is a counted yakuman; a large fan used to overflow the int32 shift.
        score = jit(jax.vmap(jax.vmap(Yaku.score, in_axes=(None, 0)), in_axes=(0, None)))
        fus = jnp.array([20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130], dtype=jnp.int32)
        limits = {5: 2000, 6: 3000, 7: 3000, 8: 4000, 9: 4000, 10: 4000, 11: 6000, 12: 6000}
        fans = jnp.arange(5, 41, dtype=jnp.int32)
        expected = jnp.array([limits.get(fan, 8000) for fan in range(5, 41)], dtype=jnp.int32)
        self.assertTrue(jnp.array_equal(score(fans, fus), jnp.broadcast_to(expected[:, None], (fans.size, fus.size))))

        # Up to 4 fan the basic points are fu << (fan + 2), capped at a mangan. 3 fan 70+ fu and
        # 2 fan 130 fu used to index SCORES with fan - 4 < 0 and pay a yakuman.
        fans = jnp.arange(1, 5, dtype=jnp.int32)
        self.assertTrue(jnp.array_equal(score(fans, fus), jnp.minimum(fus[None, :] << (fans[:, None] + 2), 2000)))
        self.assertEqual(int(Yaku.score(jnp.int32(2), jnp.int32(0))), 16000)  # a double yakuman

if __name__ == '__main__':
    unittest.main()
