# Copyright 2025 The Mahjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Dict, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp

from mahjax._src.types import Array, PRNGKey
from mahjax.common.config import GameConfig
from mahjax.core import Env

from .config import NO_RED
from .constants import TRUE
from .logic import (
    _start_next_round_or_end_game,
    _init,
    _replace_state,
    _resolve_game_config,
    _step,
)
from .observation import _observe_2D, _observe_dict, _observe_privileged_2D, _observe_privileged_dict
from .state import State, default_state


class Mahjong4p(Env):
    def __init__(
        self,
        round_mode: Literal["single", "east", "half"] = "half",
        observe_type: str = "dict",
        order_points: List[int] = [
            0,
            0,
            0,
            0,
        ],  # No uma by default; in hundreds of points like ``score`` (10-30 uma is [300, 100, -100, -300])
        game_config: Optional[GameConfig] = None,
        round_end: Literal["auto", "confirm"] = "auto",
    ):
        if round_mode not in ("single", "east", "half"):
            raise ValueError(f"round_mode must be one of ('single', 'east', 'half'), got: {round_mode}")
        if observe_type == "2D":
            raise ValueError(f"observe type 2D is not developed yet")
        if round_end not in ("auto", "confirm"):
            raise ValueError(
                f"round_end must be one of ('auto', 'confirm'), got: {round_end}"
            )
        self.round_mode = round_mode
        self.one_round = round_mode == "single"
        self.round_limit = jnp.int8(3 if round_mode == "east" else 7)
        self.observe_func = _observe_dict if observe_type == "dict" else _observe_2D
        self.observe_privileged_func = _observe_privileged_dict if observe_type == "dict" else _observe_privileged_2D
        self.order_points = order_points
        self.game_config = _resolve_game_config(game_config)
        self.round_end = round_end

    def init(self, key: PRNGKey) -> State:
        """Return the initial state. Note that no internal state of
        environment changes.
        Args:
            key: pseudo-random generator key in JAX. Consumed in this function.
        Returns:
            State: initial state of environment
        """
        state = _init(key, self.game_config)
        state = _replace_state(
            state,
            order_points=jnp.array(self.order_points, dtype=jnp.int32),
            round_limit=self.round_limit,
        )
        return state

    def step(
        self,
        state: State,
        action: Array,
        key: Optional[Array] = None,
    ) -> State:
        """Step function.

        ``key`` deals the wall whenever this step ends a round, so it is
        required: the state carries no rng of its own.
        """
        if key is None:
            raise ValueError(
                "step() requires a PRNG key: a step that ends a round deals the "
                "next wall from it, and the state carries no rng of its own."
            )
        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player
        state = _replace_state(
            state,
            order_points=jnp.array(self.order_points, dtype=jnp.int32),
            rewards=jnp.zeros_like(state.rewards),  # Reset rewards to 0 for this step
        )


        stepped_state = _replace_state(
            _step(state, action, key, self.game_config, self.round_end),
            step_count=state.step_count + 1,
        )
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: _replace_state(state, rewards=jnp.zeros_like(state.rewards)),
            lambda: stepped_state,
        )
        state = jax.lax.cond(
            state.round_state.terminated_round & self.one_round,
            lambda: _replace_state(state, terminated=TRUE),
            lambda: state,
        )
        # ``auto`` round_end: when a round ends mid-game (not single mode, not
        # yet game-terminated), advance straight to the next round init state in this
        # same step instead of requiring four DUMMY rotations. If the round-end
        # coincides with the game-end, terminate=True and update the final score
        # with rank points + kyotaku.
        if self.round_end == "auto":
            state = jax.lax.cond(
                state.round_state.terminated_round & ~state.terminated & ~jnp.bool_(self.one_round),
                lambda: _start_next_round_or_end_game(state, key, self.game_config),
                lambda: state,
            )
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )
        state = jax.lax.cond(
            state.terminated,
            lambda: _replace_state(
                state, legal_action_mask=state.players.legal_action_mask.at[state.current_player].set(TRUE)
            ),
            lambda: state,
        )
        return state

    def observe(self, state: State) -> Array:
        assert isinstance(state, State)
        return self.observe_func(state)

    def observe_privileged(self, state: State) -> Dict:
        """``observe()`` plus every other player's concealed hand.

        HIDDEN INFORMATION -- for centralised critics, opponent modelling and
        analysis, never for a policy that will face real opponents. Always returns a
        dict, independent of ``observe_type``: the extra keys have no 2D encoding.
        """
        assert isinstance(state, State)
        return self.observe_privileged_func(state)

    @property
    def id(self) -> str:
        return "no_red_4p" if self.game_config == NO_RED else "red_4p"  # type:ignore

    @property
    def version(self) -> str:
        return "beta"

    @property
    def num_players(self) -> int:
        return 4

    @property
    def num_actions(self) -> int:
        """Return the size of action space (e.g., 9 in Tic-tac-toe)"""
        state = default_state()
        return int(state.legal_action_mask.shape[0])

    @property
    def observation_shape(self) -> Dict[str, Tuple[int, ...]]:
        """Per-key shapes of the observation.

        ``observe_type='dict'`` is the only supported mode, so the observation is a
        dict and there is no single array shape to report -- this used to call
        ``.shape`` on the dict and raise ``AttributeError``. Scalar fields map to
        ``()``.
        """
        state = default_state()
        obs = self.observe(state)
        return {key: jnp.shape(value) for key, value in obs.items()}

    @property
    def _illegal_action_penalty(self) -> float:
        """Negative reward given when illegal action is selected."""
        return -1.0

    def _step_with_illegal_action(self, state: State, loser: Array) -> State:
        penalty = self._illegal_action_penalty
        reward = jnp.ones_like(state.rewards) * (-1 * penalty) * (self.num_players - 1)
        reward = reward.at[loser].set(penalty)
        return _replace_state(state, rewards=reward, terminated=TRUE)
