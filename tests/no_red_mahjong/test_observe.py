from unittest import TestCase
import jax
import jax.numpy as jnp
from mahjax.no_red_mahjong.observation import _observe_dict, _observe_2D
from mahjax.no_red_mahjong.state import State
from mahjax.no_red_mahjong.tile import EMPTY_RIVER, River
from mahjax.no_red_mahjong.meld import Meld, EMPTY_MELD
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.env import _replace_state
import numpy as np

jitted_observe_dict = jax.jit(_observe_dict)
jitted_observe_2D = jax.jit(_observe_2D)


class TestObserve2D(TestCase):
    def setUp(self):
        self.state = State()
        # Collect test items for player 0, change the perspective and test.
        # --- hand related ---
        # hand
        test_hand = jnp.zeros((4, 34), dtype=jnp.int32)
        test_hand = test_hand.at[0, 1].set(1) # 1m 1 tile
        test_hand = test_hand.at[0, 2].set(2) # 2m 2 tiles
        test_hand = test_hand.at[0, 3].set(3) # 3m 3 tiles
        test_hand = test_hand.at[0, 4].set(4) # 4m 4 tiles
        expected_0p_hand = jnp.zeros((4, 34)).at[0, jnp.array([1, 2, 3, 4], dtype=jnp.int32)].set(1)
        expected_0p_hand = expected_0p_hand.at[1, jnp.array([2, 3, 4], dtype=jnp.int32)].set(1)
        expected_0p_hand = expected_0p_hand.at[2, jnp.array([3, 4], dtype=jnp.int32)].set(1)
        expected_0p_hand = expected_0p_hand.at[3, jnp.array([4], dtype=jnp.int32)].set(1)
        self.expected_0p_hand = expected_0p_hand

        self.state = _replace_state(self.state, hand=test_hand, shanten_current_player=jnp.int8(1))
        # can ron
        test_can_win = jnp.zeros((4, 34), dtype=jnp.int32).at[0, :].set(1)  # player0 can ron
        self.state = _replace_state(self.state, can_win=test_can_win)
        # furiten
        test_furiten = jnp.zeros((4,), dtype=jnp.int32).at[0].set(1)  # player0 all tiles are furiten
        self.state = _replace_state(self.state, furiten_by_discard=test_furiten)
        # --- game related ---
        # score
        test_score = jnp.array([260, 240, 270, 230], dtype=jnp.int32)
        self.state = _replace_state(self.state, score=test_score)
        # kyotaku
        test_kyotaku = jnp.int8(1)
        self.state = _replace_state(self.state, kyotaku=test_kyotaku)
        # Dora indicators
        test_dora_indicators = jnp.array([0, 1, -1, -1, -1], dtype=jnp.int32)  # dora tiles are 1m, 2m
        self.state = _replace_state(self.state, dora_indicators=test_dora_indicators)
        # --- river related ---
        # river
        test_river = jnp.full((4, 24), EMPTY_RIVER, dtype=jnp.uint16)
        test_river = River.add_discard(test_river, 0, 0, 0, False, False) # 1m discard
        test_river = River.add_discard(test_river, 1, 0, 1, False, False) # 2m discard
        test_river = River.add_discard(test_river, 2, 0, 2, True, False) # 3m tsumogiri
        test_river = River.add_discard(test_river, 3, 0, 3, False, True) # 4m riichi
        self.state = _replace_state(self.state, river=test_river)
        river_tile_block = jnp.zeros((96, 34))
        river_tile_block = river_tile_block.at[0, 0].set(1)
        river_tile_block = river_tile_block.at[1, 1].set(1)
        river_tile_block = river_tile_block.at[2, 2].set(1)
        river_tile_block = river_tile_block.at[3, 3].set(1)
        self.expected_river_tile_block = river_tile_block
        river_riichi_block = jnp.zeros((96, 34))
        river_riichi_block = river_riichi_block.at[3].set(1)
        self.expected_river_riichi_block = river_riichi_block
        river_tsumogiri_block = jnp.zeros((96, 34))
        river_tsumogiri_block = river_tsumogiri_block.at[2].set(1)
        self.expected_river_tsumogiri_block = river_tsumogiri_block

        # --- melds related ---
        test_melds = jnp.full((4, 4), EMPTY_MELD, dtype=jnp.int32)
        test_melds = test_melds.at[0, 0].set(Meld.init(Action.PON, 1, 2)) # 1m pon (from across)
        test_melds = test_melds.at[0, 1].set(Meld.init(Action.CHI_L, 2, 1)) # 2m chi (from across)
        self.state = _replace_state(self.state, melds=test_melds)
        src = jnp.ones((4, 4), dtype=jnp.int32) * -1.0
        src = src.at[0, 0].set(2)
        src = src.at[0, 1].set(1)
        self.expected_src = src.reshape(16,)
        target = jnp.ones((4, 4), dtype=jnp.int32) * -1.0
        target = target.at[0, 0].set(1)
        target = target.at[0, 1].set(2)
        self.expected_target = target.reshape(16,)
        meld_type = jnp.ones((4, 4), dtype=jnp.int32) * -1.0
        meld_type = meld_type.at[0, 0].set(Action.PON)
        meld_type = meld_type.at[0, 1].set(Action.CHI_L)
        meld_type = meld_type / Action.NUM_ACTION
        self.expected_meld_type = meld_type.reshape(16,)
        # --- strategic related ---
        test_riichi = jnp.array([1, 0, 0, 0], dtype=jnp.int32)
        self.state = _replace_state(self.state, riichi=test_riichi)
        test_legal_action_mask_4p = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.int32)
        test_legal_action_mask_4p = test_legal_action_mask_4p.at[0, :].set(1) # player0 all actions are legal
        self.state = _replace_state(self.state, legal_action_mask=test_legal_action_mask_4p)
        expected_legal_state_0p = jnp.ones((11, 34), dtype=jnp.int32).at[2, :].set(0) # cannot add kan


    def test_hand_related(self):
        state = self.state
        state = _replace_state(state, current_player=0)
        # test for 0p
        hand_feature = jnp.zeros((7, 34), dtype=jnp.float32)
        hand_feature = hand_feature.at[:4, :].set(self.expected_0p_hand)
        hand_feature = hand_feature.at[4, :].set(1) # waiting tile
        hand_feature = hand_feature.at[5, :].set(1) # furiten
        hand_feature = hand_feature.at[6, :].set(1 / 6.0) # shanten count
        obs = jitted_observe_2D(state)
        print("obs shape", obs.shape)
        self.assertTrue(jnp.all(hand_feature == obs[:7, :]))
        # test for 1p
        state = _replace_state(state, current_player=1)
        hand_feature = jnp.zeros((7, 34), dtype=jnp.float32).at[6, :].set(1 / 6.0) # shanten count
        obs = jitted_observe_2D(state)
        self.assertTrue(jnp.all(hand_feature == obs[:7, :]))

    def test_game_related(self):
        state = self.state
        # test for 0p
        state = _replace_state(state, current_player=0)
        game_feature = jnp.zeros((15, 34), dtype=jnp.float32)
        game_feature = game_feature.at[:4, :].set((state.round_state.score.reshape(4, 1).repeat(34, axis=1) + 250) / 1250).astype(jnp.float32)
        game_feature = game_feature.at[4, :].set(1 / 3.0) # 2nd place
        game_feature = game_feature.at[5, :].set(0) # 東場
        game_feature = game_feature.at[6, :].set(0) # 0 honba
        game_feature = game_feature.at[7, :].set(1 / 10.0) # 1 kyotaku
        game_feature = game_feature.at[8, :].set(state.round_state.seat_wind[0] / 3.0) # seat wind
        game_feature = game_feature.at[9, :].set(0 % 4 / 3.0) # round wind
        game_feature = game_feature.at[10, :].set((state.round_state.next_deck_ix - state.round_state.last_deck_ix + 1) / 70.0) # remaining tsumo
        game_feature = game_feature.at[11, 0].set(1) # dora display
        game_feature = game_feature.at[12, 1].set(1) # ドラ表示
        obs = jitted_observe_2D(state)
        self.assertTrue(jnp.allclose(game_feature, obs[7:22, :]))


    def test_river_related(self):
        state = self.state
        # test for 0p
        state = _replace_state(state, current_player=0)
        river_feature = jnp.concatenate([self.expected_river_tile_block, self.expected_river_riichi_block, self.expected_river_tsumogiri_block], axis=0)
        obs = jitted_observe_2D(state)
        self.assertTrue(jnp.allclose(river_feature, obs[22:22 + 96 * 3, :]))
        # test for 1p
        state = _replace_state(state, current_player=1)
        river_feature = jnp.concatenate([self.expected_river_tile_block.reshape(4, 24, 34)[jnp.array([1, 2, 3, 0]), :].reshape(96, 34), self.expected_river_riichi_block.reshape(4, 24, 34)[jnp.array([1, 2, 3, 0]), :].reshape(96, 34), self.expected_river_tsumogiri_block.reshape(4, 24, 34)[jnp.array([1, 2, 3, 0]), :].reshape(96, 34)], axis=0)
        obs = jitted_observe_2D(state)
        self.assertTrue(jnp.allclose(river_feature, obs[22:22 + 96 * 3, :]))

    def test_meld_related(self):
        state = self.state
        # test for 0p
        state = _replace_state(state, current_player=0)
        meld_feature = jnp.concatenate([self.expected_src.reshape(16, 1).repeat(34, axis=1), self.expected_target.reshape(16, 1).repeat(34, axis=1), self.expected_meld_type.reshape(16, 1).repeat(34, axis=1)], axis=0)
        obs = jitted_observe_2D(state)
        self.assertTrue(jnp.allclose(meld_feature, obs[22 + 96 * 3:22 + 96 * 3 + 48, :]))
        # test for 1p
        state = _replace_state(state, current_player=1)
        meld_feature = jnp.concatenate([
            self.expected_src.reshape(4, 4)[jnp.array([1, 2, 3, 0])].reshape(16, 1).repeat(34, axis=1),
            self.expected_target.reshape(4, 4)[jnp.array([1, 2, 3, 0])].reshape(16, 1).repeat(34, axis=1),
            self.expected_meld_type.reshape(4, 4)[jnp.array([1, 2, 3, 0])].reshape(16, 1).repeat(34, axis=1),
        ], axis=0)
        obs = jitted_observe_2D(state)
        self.assertTrue(jnp.allclose(meld_feature, obs[22 + 96 * 3:22 + 96 * 3 + 48, :]))


    def test_strategic_related(self):
        state = self.state
        # test for 0p
        test_strategic_feature = jnp.zeros((23, 34), dtype=jnp.float32)
        test_strategic_feature = test_strategic_feature.at[0, :].set(1)  # player 0 is riichi
        test_strategic_feature = test_strategic_feature.at[4, 3].set(1)  # riichi tile is 4m
        test_strategic_feature = test_strategic_feature.at[12:, :].set(1)  # legal actions are all 1
        test_strategic_feature = test_strategic_feature.at[12+2, :].set(0)  # cannot added_kan
        obs = jitted_observe_2D(state)
        self.assertTrue(jnp.allclose(test_strategic_feature, obs[22 + 96 * 3 + 48:22 + 96 * 3 + 48 + 23, :]))
        # test for 1p
        state = _replace_state(state, current_player=1)
        test_strategic_feature = jnp.zeros((23, 34), dtype=jnp.float32)
        test_strategic_feature = test_strategic_feature.at[3, :].set(1)  # player 1 is riichi
        test_strategic_feature = test_strategic_feature.at[7, 3].set(1)  # riichi tile is 4m
        obs = jitted_observe_2D(state)
        self.assertTrue(jnp.allclose(test_strategic_feature, obs[22 + 96 * 3 + 48:22 + 96 * 3 + 48 + 23, :]))


class TestObserveDict(TestCase):
    def setUp(self):
        self.state = State()
        # Collect test items for player 0, change the perspective and test.
        # --- hand related ---
        # hand
        test_hand = jnp.zeros((4, 34), dtype=jnp.int32)
        test_hand = test_hand.at[0, 1].set(1) # 1m 1 tile
        test_hand = test_hand.at[1, 2].set(1) # 2m 1 tile
        test_hand = test_hand.at[2, 3].set(1) # 3m 1 tile
        test_hand = test_hand.at[3, 4].set(1) # 4m 1 tile

        self.state = _replace_state(self.state, hand=test_hand, shanten_current_player=jnp.int8(1))
        # can ron
        test_can_win = jnp.zeros((4, 34), dtype=jnp.int32).at[0, :].set(1)  # player0 only can ron
        self.state = _replace_state(self.state, can_win=test_can_win)
        # furiten
        test_furiten = jnp.zeros((4,), dtype=jnp.int32).at[0].set(1)  # player0 all tiles are furiten
        self.state = _replace_state(self.state, furiten_by_discard=test_furiten)
        # --- game related ---
        # score
        test_score = jnp.array([260, 240, 270, 230], dtype=jnp.int32)
        self.state = _replace_state(self.state, score=test_score)
        # kyotaku
        test_kyotaku = jnp.int8(1)
        self.state = _replace_state(self.state, kyotaku=test_kyotaku)
        # Dora indicators
        test_dora_indicators = jnp.array([0, 1, -1, -1, -1], dtype=jnp.int32)  # dora tiles are 1m, 2m
        self.state = _replace_state(self.state, dora_indicators=test_dora_indicators)
        self.state = _replace_state(self.state, current_player=0)


    def test_hand_related(self):
        state = _replace_state(self.state, current_player=0)
        obs = jitted_observe_dict(state)
        expected_hand = np.array([1] + [-1] * 13, dtype=np.int32)
        np.testing.assert_array_equal(np.array(obs["hand"]), expected_hand)
        self.assertEqual(obs["shanten_count"].item(), 1)
        self.assertEqual(obs["furiten"].item(), 1)

    def test_hand_related_other_player(self):
        state = _replace_state(self.state, current_player=1)
        obs = jitted_observe_dict(state)
        expected_hand = np.array([2] + [-1] * 13, dtype=np.int32)
        np.testing.assert_array_equal(np.array(obs["hand"]), expected_hand)
        self.assertEqual(obs["shanten_count"].item(), 1)
        self.assertEqual(obs["furiten"].item(), 0)

    def test_game_related_fields(self):
        state = _replace_state(self.state,
            round=jnp.int8(3),
            honba=jnp.int8(2),
            kyotaku=jnp.int8(4),
            current_player=jnp.int8(2),
        )
        obs = jitted_observe_dict(state)
        np.testing.assert_array_equal(
            np.array(obs["scores"]), np.array([270, 230, 260, 240], dtype=np.int32)
        )
        self.assertEqual(obs["round"].item(), 3)
        self.assertEqual(obs["honba"].item(), 2)
        self.assertEqual(obs["kyotaku"].item(), 4)
        self.assertEqual(obs["prevalent_wind"].item(), 2)
        self.assertEqual(obs["seat_wind"].item(), 2)
        np.testing.assert_array_equal(
            np.array(obs["dora_indicators"]),
            np.array([0, 1, -1, -1], dtype=np.int32),
        )

    def test_action_history_relative_players(self):
        action_history = self.state.round_state.action_history
        action_history = action_history.at[0, :3].set(
            jnp.array([0, 1, 3], dtype=jnp.int8)
        )
        action_history = action_history.at[1, :3].set(
            jnp.array([10, 11, 12], dtype=jnp.int8)
        )
        action_history = action_history.at[2, :3].set(
            jnp.array([False, False, True], dtype=jnp.bool_)
        )
        state = _replace_state(self.state, action_history=action_history, current_player=jnp.int8(1)) # for player 1
        obs = jitted_observe_dict(state)
        expected_players = np.array([3, 0, 2], dtype=np.int8)
        np.testing.assert_array_equal(
            np.array(obs["action_history"])[0, :3], expected_players
        )
        self.assertEqual(np.array(obs["action_history"])[0, 3], -1)
        np.testing.assert_array_equal(
            np.array(obs["action_history"])[1, :3],
            np.array(action_history)[1, :3],
        )
        np.testing.assert_array_equal(
            np.array(obs["action_history"])[2, :3],
            np.array(action_history)[2, :3],
        )

        state = _replace_state(state, action_history=action_history, current_player=3)
        obs = jitted_observe_dict(state)
        expected_players = np.array([1, 2, 0], dtype=np.int8)
        np.testing.assert_array_equal(
            np.array(obs["action_history"])[0, :3], expected_players
        )
        self.assertEqual(np.array(obs["action_history"])[0, 3], -1)
        np.testing.assert_array_equal(
            np.array(obs["action_history"])[1, :3],
            np.array(action_history)[1, :3],
        )
        np.testing.assert_array_equal(
            np.array(obs["action_history"])[2, :3],
            np.array(action_history)[2, :3],
        )

        state = _replace_state(state, action_history=action_history, current_player=0)
        obs = jitted_observe_dict(state)
        expected_players = np.array([0, 1, 3], dtype=np.int8)
        np.testing.assert_array_equal(
            np.array(obs["action_history"])[0, :3], expected_players
        )
        self.assertEqual(np.array(obs["action_history"])[0, 3], -1)
        np.testing.assert_array_equal(
            np.array(obs["action_history"])[1, :3],
            np.array(action_history)[1, :3],
        )
        np.testing.assert_array_equal(
            np.array(obs["action_history"])[2, :3],
            np.array(action_history)[2, :3],
        )
