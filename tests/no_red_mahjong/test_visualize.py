import os
import unittest
import jax
import jax.numpy as jnp
from mahjax.no_red_mahjong.tile import Tile
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.state import FIRST_DRAW_IDX
from mahjax.no_red_mahjong.env import _step, _init, _replace_state
from mahjax._src.visualizer import save_svg, save_svg_animation
from mahjax.no_red_mahjong.players import rule_based_player
from mahjax.no_red_mahjong.yaku import Yaku
from mahjax.no_red_mahjong.env import _dora_array


IDX_AFTER_FIRST_DRAW = FIRST_DRAW_IDX - 1

jitted_step = jax.jit(_step)

def act_randomly(rng: jax.random.PRNGKey, legal_action_mask: jnp.ndarray) -> jnp.ndarray:
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits=logits)


class TestVisualize(unittest.TestCase):
    def setUp(self):
        rng = jax.random.PRNGKey(6)
        self.state = _init(rng)

    def set_state(self, state, **kwargs):
        for k, v in kwargs.items():
            state = _replace_state(state,   # type:ignore
                **{k: v}
            )
        return state


    def test_visualize(self):
        state = self.state
        os.makedirs("fig", exist_ok=True)
        save_svg(state, "fig/test_img.svg")

    def test_animation_random(self):
        state = self.state
        i = 0
        states = []
        rng = jax.random.PRNGKey(1)
        while not state.terminated and i <= 100:
            a = act_randomly(rng, state.legal_action_mask)
            print('step', i, 'current_player', state.current_player, 'action', a)
            state = jitted_step(state, a)
            i += 1
            #print("target", state.round_state.target, "action", a, "current_player", state.current_player, "melds", state.players.melds)
            #save_svg(state, f"fig/test_animation_{i}.svg")
            states.append(state)
        os.makedirs("fig", exist_ok=True)
        save_svg_animation(states, "fig/test_animation_random.svg", frame_duration_seconds=1)

    def test_animation_tsumogiri(self):
        state = self.state
        i = 0
        states = []
        rng = jax.random.PRNGKey(1)
        while not state.round_state.terminated_round and i <= 100:
            a = jnp.where(state.legal_action_mask[Action.DUMMY], Action.DUMMY, Action.TSUMOGIRI)
            print('step', i, 'current_player', state.current_player, 'action', a)
            state = jitted_step(state, a)
            i += 1
            #print("target", state.round_state.target, "action", a, "current_player", state.current_player, "melds", state.players.melds)
            #save_svg(state, f"fig/test_animation_{i}.svg")
            states.append(state)
        os.makedirs("fig", exist_ok=True)
        save_svg_animation(states, "fig/test_animation_tsumogiri.svg", frame_duration_seconds=1)



    def test_animation_rule_based_player(self):
        state = self.state
        i = 0
        states = []
        rng = jax.random.PRNGKey(1)
        while not state.round_state.terminated_round and i <= 100:
            a = jax.jit(rule_based_player)(state, rng)
            if state.legal_action_mask[Action.RIICHI]:
                print("立直できるよ！")
            round_wind = state.round_state.round % 4
            if a == Action.TSUMO:
                yaku, fan, fu = jax.jit(Yaku.judge)(
                    state.players.hand[state.current_player],
                    state.players.melds[state.current_player],
                    state.players.meld_counts[state.current_player],
                    state.round_state.last_draw,
                    state.players.riichi[state.current_player],
                    False,
                    round_wind,
                    state.round_state.seat_wind[state.current_player],
                    _dora_array(state)
                )
                print("立直", state.players.riichi[state.current_player])
                print("ドラ", _dora_array(state))
                print("手牌", state.players.hand[state.current_player])
                print(yaku, fan, fu)
            if a == Action.RON:
                yaku, fan, fu = Yaku.judge(
                    state.players.hand[state.current_player],
                    state.players.melds[state.current_player],
                    state.players.meld_counts[state.current_player],
                    state.round_state.target,
                    state.players.riichi[state.current_player],
                    True,
                    round_wind,
                    state.round_state.seat_wind[state.current_player],
                    _dora_array(state)
                )
                print("立直", state.players.riichi[state.current_player])
                print("ドラ", _dora_array(state))
                print("手牌", state.players.hand[state.current_player])
                print(yaku, fan, fu)

            print('step', i, 'current_player', state.current_player, 'action', a)
            state = jitted_step(state, a)
            i += 1
            states.append(state)
        print(state.round_state.score)
        os.makedirs("fig", exist_ok=True)
        save_svg_animation(states, "fig/test_animation_rule_based_player.svg", frame_duration_seconds=1)