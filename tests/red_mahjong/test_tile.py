import unittest
import jax.numpy as jnp

from mahjax.red_mahjong.tile import (
    Tile,
    River,
    EMPTY_RIVER,
)
from mahjax.red_mahjong.action import Action


class TestTile(unittest.TestCase):
    def test_from_tile_id_to_tile_mapping(self):
        self.assertEqual(int(Tile.from_tile_id_to_tile(0)), 0)
        self.assertEqual(int(Tile.from_tile_id_to_tile(3)), 0)
        self.assertEqual(int(Tile.from_tile_id_to_tile(4)), 1)
        self.assertEqual(int(Tile.from_tile_id_to_tile(35)), 8)
        self.assertEqual(int(Tile.from_tile_id_to_tile(36)), 9)
        self.assertEqual(int(Tile.from_tile_id_to_tile(133)), 33)
        self.assertEqual(int(Tile.from_tile_id_to_tile(135)), 33)

    def test_is_tile_type_three(self):
        # Among suit (0..26), tiles with tile_type % 9 == 2 are True
        for t in range(34):
            expect = (t < 27) and (t % 9 == 2)
            got = bool(Tile.is_tile_type_three(t))
            self.assertEqual(
                got, expect, msg=f"is_tile_type_three mismatch at tile_type={t}"
            )

    def test_is_tile_type_seven(self):
        # Among suit (0..26), tiles with tile_type % 9 == 6 are True
        for t in range(34):
            expect = (t < 27) and (t % 9 == 6)
            got = bool(Tile.is_tile_type_seven(t))
            self.assertEqual(
                got, expect, msg=f"is_tile_type_seven mismatch at tile_type={t}"
            )

    def test_is_tile_four_wind(self):
        # 27..30 are four winds (東南西北)
        for t in range(34):
            expect = (27 <= t < 31)
            got = bool(Tile.is_tile_four_wind(t))
            self.assertEqual(
                got, expect, msg=f"is_tile_four_wind mismatch at tile={t}"
            )


class TestRiver(unittest.TestCase):
    def setUp(self):
        self.river = jnp.full((4, 24), EMPTY_RIVER, dtype=jnp.uint16)

    def test_decode_empty(self):
        decoded = River.decode_river(self.river)  # (6,4,24)
        # Empty space specification:
        # tile=-1, riichi=0, gray=0, tsumogiri=0, src=0, meld_type=0
        self.assertEqual(int(decoded.shape[0]), 6)
        self.assertTrue(jnp.all(decoded[0] == -1))
        self.assertTrue(jnp.all(decoded[1] == 0))
        self.assertTrue(jnp.all(decoded[2] == 0))
        self.assertTrue(jnp.all(decoded[3] == 0))
        self.assertTrue(jnp.all(decoded[4] == 0))
        self.assertTrue(jnp.all(decoded[5] == 0))

    def test_add_discard(self):
        # player=2 discard at idx=5
        player, idx = 2, 5
        for tile in range(34):
            for is_tsumogiri in [True, False]:
                for is_riichi in [True, False]:
                    test_river = River.add_discard(
                        self.river, tile=tile, player=player, idx=idx,
                        is_tsumogiri=is_tsumogiri, is_riichi=is_riichi
                    )  # add discard
                    decoded = River.decode_river(test_river)  # decode river
                    self.assertEqual(int(decoded[0, player, idx]), tile)          # tile
                    self.assertEqual(int(decoded[1, player, idx]), int(is_riichi))             # riichi
                    self.assertEqual(int(decoded[2, player, idx]), 0)             # gray
                    self.assertEqual(int(decoded[3, player, idx]), int(is_tsumogiri))             # tsumogiri
                    self.assertEqual(int(decoded[4, player, idx]), 0)             # src (not set)
                    self.assertEqual(int(decoded[5, player, idx]), 0)             # meld_type
                    # Other spaces are empty
                    self.assertTrue(jnp.all(decoded[0, player, :idx] == -1))
                    self.assertTrue(jnp.all(decoded[0, player, idx+1:] == -1))

    def test_add_meld(self):
        # first, place a discard
        player, idx = 0, 3
        test_river = River.add_discard(
            self.river, tile=5, player=player, idx=idx,
            is_tsumogiri=False, is_riichi=False
        )
        meld_types = {
            Action.PON: 1,
            Action.OPEN_KAN: 2,
            Action.CHI_L: 3,
            Action.CHI_M: 4,
            Action.CHI_R: 5,
        }

        for action in [Action.PON, Action.OPEN_KAN]:
            for src in [1, 2, 3]:
                test_river_for_pon_and_open_kan = River.add_meld(test_river, action=action, player=player, idx=idx, src=src)  # add meld
                decoded = River.decode_river(test_river_for_pon_and_open_kan)  # decode river
                self.assertEqual(int(decoded[0, player, idx]), 5)             # tile
                self.assertEqual(int(decoded[1, player, idx]), 0)             # riichi (not set)
                self.assertEqual(int(decoded[2, player, idx]), 1)             # gray
                self.assertEqual(int(decoded[3, player, idx]), 0)             # tsumogiri (not set)
                self.assertEqual(int(decoded[4, player, idx]), src)             # src
                self.assertEqual(int(decoded[5, player, idx]), meld_types[action])             # meld_type
                # Other spaces are empty
                self.assertTrue(jnp.all(decoded[0, player, :idx] == -1))
                self.assertTrue(jnp.all(decoded[0, player, idx+1:] == -1))

        for action in [Action.CHI_L, Action.CHI_M, Action.CHI_R]:
            test_river_for_chi = River.add_meld(test_river, action=action, player=player, idx=idx, src=3)
            decoded = River.decode_river(test_river_for_chi)
            self.assertEqual(int(decoded[0, player, idx]), 5)             # tile
            self.assertEqual(int(decoded[2, player, idx]), 1)             # gray
            self.assertEqual(int(decoded[4, player, idx]), 3)             # src
            self.assertEqual(int(decoded[5, player, idx]), meld_types[action])             # meld_type
            # Other spaces are empty
            self.assertTrue(jnp.all(decoded[0, player, :idx] == -1))
            self.assertTrue(jnp.all(decoded[0, player, idx+1:] == -1))


    def test_decode_tile(self):
        player, idx = 0, 3
        for tile in range(34):
            test_river = River.add_discard(
                self.river, tile=tile, player=player, idx=idx,
                is_tsumogiri=False, is_riichi=False
            )
            tile_decoded = River.decode_tile(test_river)
            self.assertEqual(int(tile_decoded[player, idx]), tile)             # tile
            # Other spaces are empty
            self.assertTrue(jnp.all(tile_decoded[player, :idx] == -1))
            self.assertTrue(jnp.all(tile_decoded[player, idx+1:] == -1))



if __name__ == "__main__":
    unittest.main()
