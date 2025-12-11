import unittest
import jax
import jax.numpy as jnp
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.state import FIRST_DRAW_IDX
from mahjax.no_red_mahjong.env import _init, _step, _make_legal_action_mask_after_draw, _make_legal_action_mask_after_draw_w_riichi, _dora_array, _discard, _make_legal_action_mask_after_discard, _next_meld_player, _selfkan, _closed_kan, _added_kan, _open_kan, _pon, _chi, _pass, _riichi, _ron, _tsumo, _accept_riichi, _append_meld, _draw, _pass, _make_legal_action_mask_after_chi, _abortive_draw_normal, _next_round, _kan

jitted_init = jax.jit(_init)
jitted_step = jax.jit(_step)
jitted_make_legal_action_mask_after_draw = jax.jit(_make_legal_action_mask_after_draw)
jitted_make_legal_action_mask_after_draw_w_riichi = jax.jit(_make_legal_action_mask_after_draw_w_riichi)
jitted_discard = jax.jit(_discard)
jitted_next_meld_player = jax.jit(_next_meld_player)
jitted_abortive_draw_normal = jax.jit(_abortive_draw_normal) # skip test for now
jitted_tsumo = jax.jit(_tsumo)
IDX_AFTER_FIRST_DRAW = FIRST_DRAW_IDX - 1

class TestSpecialCase(unittest.TestCase):
    def setUp(self):
        rng = jax.random.PRNGKey(0)
        self.state = jitted_init(rng)


    def set_state(self, state, **kwargs):
        for k, v in kwargs.items():
            state = state.replace(  # type:ignore
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
        state = state.replace(
            current_player=jnp.int8(0),
            _fan=jnp.zeros((4, 2), dtype=jnp.int8).at[0, 0].set(3),
            _fu=jnp.zeros((4, 2), dtype=jnp.int8).at[0, 0].set(30),
            _next_deck_ix=FIRST_DRAW_IDX-1,
            _n_meld=jnp.zeros((4,), dtype=jnp.int8),
            _dealer=jnp.int8(0),
        )

        state = jitted_tsumo(state)
        print(state.rewards)
        self.assertEqual(jnp.all(state.rewards == jnp.array([480, -160, -160, -160])), True)

        # Test if the player who wins the game is the player who has the highest score.
        state = self.state
        state = state.replace(
            current_player=jnp.int8(0),
            _fan=jnp.zeros((4, 2), dtype=jnp.int8).at[0, 0].set(3),
            _fu=jnp.zeros((4, 2), dtype=jnp.int8).at[0, 0].set(30),
            _next_deck_ix=FIRST_DRAW_IDX-2,
            _n_meld=jnp.zeros((4,), dtype=jnp.int8),
            _dealer=jnp.int8(1),
        )
        state = jitted_tsumo(state)
        self.assertEqual(jnp.all(state.rewards == jnp.array([320, -160, -80, -80])), True)

        # Test if the player who wins the game is the player who has the highest score.
        state = self.state
        state = state.replace(
            current_player=jnp.int8(0),
            _fan=jnp.zeros((4, 2), dtype=jnp.int8).at[0, 0].set(1),
            _fu=jnp.zeros((4, 2), dtype=jnp.int8).at[0, 0].set(0),
            _next_deck_ix=FIRST_DRAW_IDX-2,
            _n_meld=jnp.zeros((4,), dtype=jnp.int8),
            _dealer=jnp.int8(0),
        )
        state = jitted_tsumo(state)
        self.assertEqual(jnp.all(state.rewards == jnp.array([960, -320, -320, -320])), True)




