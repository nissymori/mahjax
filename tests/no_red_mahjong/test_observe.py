from unittest import TestCase
import jax
import jax.numpy as jnp
from mahjax.no_red_mahjong.observation import _observe_dict
from mahjax.no_red_mahjong.state import State
from mahjax.no_red_mahjong.tile import EMPTY_RIVER, River
from mahjax.no_red_mahjong.meld import Meld, EMPTY_MELD
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.env import _replace_state
import numpy as np

jitted_observe_dict = jax.jit(_observe_dict)


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
