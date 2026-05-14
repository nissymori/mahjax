import unittest
import jax
import jax.numpy as jnp
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.state import FIRST_DRAW_IDX
from mahjax.no_red_mahjong.env import _init, _step, _make_legal_action_mask_after_draw, _make_legal_action_mask_after_draw_w_riichi, _discard, _next_meld_player, _tsumo, _next_round, _replace_state

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
        state = jitted_next_round(state)
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


    def test_eight_consecutive_deals(self):
        """
        Test if the round is terminated when the eight consecutive deals are made.
        """
        state = self.state
        state = _replace_state(state, 
            dealer=jnp.int8(0),
            round=jnp.int8(0),
            honba=jnp.int8(8),
            has_won=jnp.array([True, False, False, False], dtype=jnp.bool_),
        )
        state = _advance_after_dummy(state)
        self.assertEqual(state.round_state.round, 1)
        self.assertEqual(state.round_state.honba, 0)
        self.assertEqual(state.round_state.dealer, 1)
