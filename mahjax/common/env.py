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


from types import ModuleType
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp

from mahjax._src.types import Array, ArrayLike, PRNGKey
from mahjax.common import logic as common_logic
from mahjax.common.config import GameConfig
from mahjax.common.constants import TRUE
from mahjax.common.logic import _init, _replace_state, _start_next_round_or_end_game
from mahjax.core import Env, EnvId

# Any game's state; its ``constants`` set the table.
GameState = Any


class MahjongEnv(Env):
    """The env for any table. A subclass names the table's state class, whose
    ``constants`` set the table, its observation module and its rule presets."""

    State: ClassVar[type[Any]]
    observation: ClassVar[ModuleType]
    logic: ClassVar[ModuleType] = common_logic  # its ``_step`` plays an action
    RED: ClassVar[GameConfig]
    NO_RED: ClassVar[GameConfig]
    ID_SUFFIX: ClassVar[str]  # "4p" makes the ids "red_4p" and "no_red_4p"

    def __init__(
        self,
        round_mode: Literal["single", "east", "half"] = "half",
        observe_type: str = "dict",
        # No uma by default; in hundreds of points like ``score`` (10-30 uma is [300, 100, -100, -300]).
        order_points: Optional[List[int]] = None,
        game_config: Optional[GameConfig] = None,
        round_end: Literal["auto", "confirm"] = "auto",
    ):
        num_players = self.State.constants.NUM_PLAYERS
        if round_mode not in ("single", "east", "half"):
            raise ValueError(f"round_mode must be one of ('single', 'east', 'half'), got: {round_mode}")
        if observe_type == "2D":
            raise ValueError("observe type 2D is not developed yet")
        if round_end not in ("auto", "confirm"):
            raise ValueError(
                f"round_end must be one of ('auto', 'confirm'), got: {round_end}"
            )
        self.round_mode = round_mode
        self.one_round = round_mode == "single"
        # The index of the last regular kyoku: one wind of NUM_PLAYERS kyoku for 'east', two otherwise.
        self.round_limit = jnp.int8(num_players - 1 if round_mode == "east" else 2 * num_players - 1)
        observation = self.observation
        self.observe_func = observation._observe_dict if observe_type == "dict" else observation._observe_2D
        self.observe_privileged_func = (
            observation._observe_privileged_dict if observe_type == "dict" else observation._observe_privileged_2D
        )
        self.order_points = [0] * num_players if order_points is None else order_points
        self.game_config = self.RED if game_config is None else game_config
        self.round_end = round_end

    def init(self, key: PRNGKey) -> GameState:
        """Return the initial state. Note that no internal state of
        environment changes.
        Args:
            key: pseudo-random generator key in JAX. Consumed in this function.
        Returns:
            GameState: initial state of environment
        """
        state = _init(self.State, key, self.game_config)
        state = _replace_state(
            state,
            order_points=jnp.array(self.order_points, dtype=jnp.int32),
            round_limit=self.round_limit,
        )
        return state

    def step(
        self,
        state: GameState,
        action: ArrayLike,
        key: Optional[PRNGKey] = None,
    ) -> GameState:
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
            self.logic._step(state, action, key, self.game_config, self.round_end),
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
        if self.round_end == "auto" and not self.one_round:
            # Dealt on every step and kept only where a round has ended, like the
            # branches of the action dispatch. Not inside the cond: under ``vmap``,
            # with a predicate per env, a cond copies each of its operands once per
            # env, and the deal would make the 22 MiB yaku table one of them.
            next_round_state = _start_next_round_or_end_game(state, key, self.game_config)
            state = jax.lax.cond(
                state.round_state.terminated_round & ~state.terminated,
                lambda: next_round_state,
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

    def observe(self, state: GameState) -> Dict[str, Array]:
        assert isinstance(state, self.State)
        return self.observe_func(state)

    def observe_privileged(self, state: GameState) -> Dict[str, Array]:
        """``observe()`` plus every other player's concealed hand.

        HIDDEN INFORMATION -- for centralised critics, opponent modelling and
        analysis, never for a policy that will face real opponents. Always returns a
        dict, independent of ``observe_type``: the extra keys have no 2D encoding.
        """
        assert isinstance(state, self.State)
        return self.observe_privileged_func(state)

    @property
    def id(self) -> EnvId:
        return ("no_red_" if self.game_config == self.NO_RED else "red_") + self.ID_SUFFIX  # type:ignore

    @property
    def version(self) -> str:
        return "beta"

    @property
    def num_players(self) -> int:
        return self.State.constants.NUM_PLAYERS

    @property
    def num_actions(self) -> int:
        """Return the size of action space (e.g., 9 in Tic-tac-toe)"""
        state = self.State()
        return int(state.legal_action_mask.shape[0])

    @property
    def observation_shape(self) -> Dict[str, Tuple[int, ...]]:
        """Per-key shapes of the observation.

        ``observe_type='dict'`` is the only supported mode, so the observation is a
        dict and there is no single array shape to report -- this used to call
        ``.shape`` on the dict and raise ``AttributeError``. Scalar fields map to
        ``()``.
        """
        state = self.State()
        obs = self.observe(state)
        return {key: jnp.shape(value) for key, value in obs.items()}

    @property
    def _illegal_action_penalty(self) -> float:
        """Negative reward given when illegal action is selected."""
        return -1.0

    def _step_with_illegal_action(self, state: GameState, loser: Array) -> GameState:
        penalty = self._illegal_action_penalty
        reward = jnp.ones_like(state.rewards) * (-1 * penalty) * (self.num_players - 1)
        reward = reward.at[loser].set(penalty)
        return _replace_state(state, rewards=reward, terminated=TRUE)
