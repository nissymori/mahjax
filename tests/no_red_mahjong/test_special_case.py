import unittest
import jax
import jax.numpy as jnp
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.state import FIRST_DRAW_IDX
from mahjax.no_red_mahjong.env import _init, _step, _make_legal_action_mask_after_draw, _make_legal_action_mask_after_draw_w_riichi, _discard, _next_meld_player, _tsumo, _next_round, _replace_state

STEP_KEY = jax.random.PRNGKey(0)  # wall key for round transitions

jitted_init = jax.jit(_init)
jitted_step = jax.jit(_step)
jitted_make_legal_action_mask_after_draw = jax.jit(_make_legal_action_mask_after_draw)
jitted_make_legal_action_mask_after_draw_w_riichi = jax.jit(_make_legal_action_mask_after_draw_w_riichi)
jitted_discard = jax.jit(_discard)
jitted_next_meld_player = jax.jit(_next_meld_player)
jitted_tsumo = jax.jit(_tsumo)
jitted_next_round = jax.jit(_next_round)
IDX_AFTER_FIRST_DRAW = FIRST_DRAW_IDX - 1


def _advance_after_dummy(state, steps: int = 4):
    """Advance the state after the dummy sharing is complete."""
    # If the dummy count is 0, the dummy sharing is complete, so the next round is called.
    # If the dummy count is not 0, the dummy sharing is not complete, so the next round is called.
    for _ in range(steps):
        state = jitted_next_round(state, STEP_KEY)
        if int(state.round_state.dummy_count) == 0:
            # The dummy sharing is complete, so the next round is called.
            break
    return state


class TestSpecialCase(unittest.TestCase):
    def setUp(self):
        rng = jax.random.PRNGKey(0)
        self.state = jitted_init(rng)


    def set_state(self, state, **kwargs):
        for k, v in kwargs.items():
            state = _replace_state(state,   # type:ignore
                **{k: v}
            )
        return state


    def test_double_ron(self):
        """
        Test if the next meld player is the closest player to the discarded player.
        """
        state = self.state
        legal_action_mask = jnp.zeros((4,Action.DUMMY+1), dtype=jnp.bool_).at[0, Action.RON].set(True).at[1, Action.RON].set(True)
        next_player, _ = jitted_next_meld_player(legal_action_mask, 3)
        self.assertTrue(next_player == 0)
        next_player, _ = jitted_next_meld_player(legal_action_mask, 2)
        self.assertTrue(next_player == 0)

        legal_action_mask = jnp.zeros((4,Action.DUMMY+1), dtype=jnp.bool_).at[0, Action.RON].set(True).at[2, Action.RON].set(True)
        next_player, _ = jitted_next_meld_player(legal_action_mask, 1)
        self.assertTrue(next_player ==2)
        next_player, _ = jitted_next_meld_player(legal_action_mask, 3)
        self.assertTrue(next_player == 0)


    def test_blessings(self):
        """
        Test if the player who wins the game is the player who has the highest score.
        """
        # Test if the player who wins the game is the player who has the highest score.
        state = self.state
        state = _replace_state(state, 
            current_player=jnp.int8(0),
            fan=jnp.zeros((4, 2), dtype=jnp.int8).at[0, 0].set(3),
            fu=jnp.zeros((4, 2), dtype=jnp.int8).at[0, 0].set(30),
            next_deck_ix=FIRST_DRAW_IDX-1,
            meld_counts=jnp.zeros((4,), dtype=jnp.int8),
            dealer=jnp.int8(0),
        )

        state = jitted_tsumo(state)
        print(state.rewards)
        self.assertEqual(jnp.all(state.rewards == jnp.array([480, -160, -160, -160])), True)

        # Test if the player who wins the game is the player who has the highest score.
        state = self.state
        state = _replace_state(state, 
            current_player=jnp.int8(0),
            fan=jnp.zeros((4, 2), dtype=jnp.int8).at[0, 0].set(3),
            fu=jnp.zeros((4, 2), dtype=jnp.int8).at[0, 0].set(30),
            next_deck_ix=FIRST_DRAW_IDX-2,
            meld_counts=jnp.zeros((4,), dtype=jnp.int8),
            dealer=jnp.int8(1),
        )
        state = jitted_tsumo(state)
        self.assertEqual(jnp.all(state.rewards == jnp.array([320, -160, -80, -80])), True)

        # Test if the player who wins the game is the player who has the highest score.
        state = self.state
        state = _replace_state(state, 
            current_player=jnp.int8(0),
            fan=jnp.zeros((4, 2), dtype=jnp.int8).at[0, 0].set(1),
            fu=jnp.zeros((4, 2), dtype=jnp.int8).at[0, 0].set(0),
            next_deck_ix=FIRST_DRAW_IDX-2,
            meld_counts=jnp.zeros((4,), dtype=jnp.int8),
            dealer=jnp.int8(0),
        )
        state = jitted_tsumo(state)
        self.assertEqual(jnp.all(state.rewards == jnp.array([960, -320, -320, -320])), True)

    def test_dealer_can_tsumo_a_complete_first_draw_as_a_single_blessing_of_heaven(self):
        # 123m 33m 444p 666p 77s drawing 7s: TSUMO is legal, and three concealed pons are not four.
        from unittest import mock

        from mahjax.no_red_mahjong import env as m
        from mahjax.no_red_mahjong.state import default_state

        dealt = [0, 4, 8, 9, 10, 48, 49, 50, 56, 57, 58, 96, 97]  # tile ids; the tile is id // 4
        first_draw = 98  # 7s
        ids = [None] * 136
        ids[FIRST_DRAW_IDX] = first_draw  # the dealer's first draw
        ids[-52:-39] = dealt  # player 0's starting hand
        rest = iter(i for i in range(136) if i not in dealt and i != first_draw)
        ids = jnp.array([i if i is not None else next(rest) for i in ids])

        with mock.patch.object(jax.random, "permutation", lambda key, x: ids):
            state = m._init_for_next_round(STEP_KEY, default_state())

        self.assertEqual(int(state.current_player), 0)
        self.assertTrue(bool(state.legal_action_mask[Action.TSUMO]))
        self.assertTrue(bool(jnp.allclose(_tsumo(state).rewards, jnp.array([480.0, -160.0, -160.0, -160.0]))))


    def test_eight_consecutive_deals(self):
        """
        Test if the dealer keeps the deal after eight consecutive deals: there is no renchan limit.
        """
        state = self.state
        state = _replace_state(state, 
            dealer=jnp.int8(0),
            round=jnp.int8(0),
            honba=jnp.int8(8),
            has_won=jnp.array([True, False, False, False], dtype=jnp.bool_),
        )
        state = _advance_after_dummy(state)
        self.assertEqual(state.round_state.round, 0)
        self.assertEqual(state.round_state.honba, 9)
        self.assertEqual(state.round_state.dealer, 0)
