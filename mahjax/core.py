# NOTE: This file is copied and modified from Pgx (https://github.com/sotetsuk/pgx).
# Copyright belongs to the original authors.
# We keep tracking the updates of original Pgx implementation.
# We try to minimize the modification to this file. Exceptions includes:
#   - remove observation from the state class since mahjax accept 2 types of observation: dict and 2D array.
#   - fix available environments and make functions
#   - remove unnesesary env implementation since mahjax only support mahjong environment.
#   - function semantics are emptied for flexibility.
#
# Copyright 2023 The Pgx Authors.
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

import abc
import warnings
from typing import Literal, Optional, Tuple, get_args

import jax
import jax.numpy as jnp

from mahjax._src.struct import dataclass
from mahjax._src.types import Array, PRNGKey

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


# Mahjax environments are versioned like OpenAI Gym or Brax.
# OpenAI Gym forces user to specify version (e.g., `MountainCar-v0`); while Brax does not (e.g., `ant`)
# We follow the way of Brax. One can check the environment version by `Env.version`.
# We do not explicitly include version in EnvId for three reasons:
# (1) In game domain, performance measure is not the score in environment but
#     the comparison to other agents (i.e., environment version is less important),
# (2) we do not provide older versions (as with OpenAI Gym), and
# (3) it is tedious to remember and write version numbers.
#
# Naming convention:
# Hyphen - is used to represent that there is a different original game source, and
# Underscore - is used for the other cases.
EnvId = Literal[
    "no_red_mahjong",
    "red_mahjong",
]


@dataclass
class State(abc.ABC):
    """Base state class of all Mahjax game environments. Basically an immutable (frozen) dataclass.
    A basic usage is generating via `Env.init`:

        state = env.init(jax.random.PRNGKey(0))

    and `Env.step` receives and returns this state class:

        state = env.step(state, action, key)

    Serialization via `flax.struct.serialization` is supported.
    There are 6 common attributes over all games:

    Attributes:
        current_player (Array): id of agent to play.
            Note that this does NOT represent the turn (e.g., black/white in Go).
            This ID is consistent over the parallel vmapped states.
        rewards (Array): the `i`-th element indicates the intermediate reward for
            the agent with player-id `i`. If `Env.step` is called for a terminal state,
            the following `state.rewards` is zero for all players.
        terminated (Array): denotes that the state is terminal state. Note that
            some environments (e.g., Go) have an `max_termination_steps` parameter inside
            and will terminate within a limited number of states (following AlphaGo).
        truncated (Array): indicates that the episode ends with the reason other than termination.
            Note that current Mahjax environments do not invoke truncation but users can use `TimeLimit` wrapper
            to truncate the environment. In Mahjax environments, some MinAtar games may not terminate within a finite timestep.
            However, the other environments are supposed to terminate within a finite timestep with probability one.
        legal_action_mask (Array): Boolean array of legal actions. If illegal action is taken,
            the game will terminate immediately with the penalty to the palyer.
    """

    current_player: Array
    rewards: Array
    terminated: Array
    truncated: Array
    legal_action_mask: Array
    _step_count: Array

    @property
    @abc.abstractmethod
    def env_id(self) -> EnvId:
        """Environment id (e.g. "go_19x19")"""
        ...

    def _repr_html_(self) -> str:
        return self.to_svg()

    def to_svg(
        self,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
        use_english: bool = False,
    ) -> str:
        """Return SVG string. Useful for visualization in notebook.

        Args:
            color_theme (Optional[Literal["light", "dark"]]): xxx see also global config.
            scale (Optional[float]): change image size. Default(None) is 1.0

        Returns:
            str: SVG string
        """
        from mahjax._src.visualizer import Visualizer

        v = Visualizer(color_theme=color_theme, scale=scale)
        return v.get_dwg(states=self, use_english=use_english).tostring()

    def save_svg(
        self,
        filename: str,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> None:
        """Save the entire state (not observation) to a file.
        The filename must end with `.svg`

        Args:
            color_theme (Optional[Literal["light", "dark"]]): xxx see also global config.
            scale (Optional[float]): change image size. Default(None) is 1.0

        Returns:
            None
        """
        from mahjax._src.visualizer import save_svg

        save_svg(self, filename, color_theme=color_theme, scale=scale)


class Env(abc.ABC):
    """Environment class API.

    !!! example "Example usage"

        ```py
        env: Env = mahjax.make("no_red_mahjong")
        state = env.init(jax.random.PRNGKey(0))
        action = jax.random.int32(4)
        state = env.step(state, action)
        ```

    """

    def __init__(self): ...

    def init(self, key: PRNGKey) -> State:
        """Return the initial state. Note that no internal state of
        environment changes.

        Args:
            key: pseudo-random generator key in JAX. Consumed in this function.

        Returns:
            State: initial state of environment
        """
        ...

    def step(
        self,
        state: State,
        action: Array,
        key: Optional[Array] = None,
    ) -> State:
        """Step function.

        Args:
            state: State: Current state of the game
            action: Array: Action to be performed
            key: PRNGKey: Pseudo-random generator key in JAX

        Returns:
            State: State after processing the action
        """
        ...

    def observe(self, state: State) -> Array:
        """
        Observation function.

        Args:
            state: State: Current state of the game

        Returns:
            Array: Observation of the state
        """
        ...

    @property
    @abc.abstractmethod
    def id(self) -> EnvId:
        """Environment id."""
        ...

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Environment version. Updated when behavior, parameter, or API is changed.
        Refactoring or speeding up without any expected behavior changes will NOT update the version number.
        """
        ...

    @property
    @abc.abstractmethod
    def num_players(self) -> int:
        """Number of players (3 or 4 in Mahjong)"""
        ...

    @property
    def num_actions(self) -> int:
        """Return the size of action space"""
        ...

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Return the matrix shape of observation"""
        ...

    @property
    def _illegal_action_penalty(self) -> float:
        """Negative reward given when illegal action is selected."""
        ...

    def _step_with_illegal_action(self, state: State, loser: Array) -> State:
        """Step function with illegal action."""
        ...


def available_envs() -> Tuple[EnvId, ...]:
    """List up all environment id available in `mahjax.make` function.

    !!! example "Example usage"

        ```py
        mahjax.available_envs()
        ('no_red_mahjong', 'red_mahjong')
        ```
    """
    games = get_args(EnvId)
    return games


def make(env_id: EnvId, **kwargs):  # noqa: C901
    """Load the specified environment.

    !!! example "Example usage"

        ```py
        env = mahjax.make(
            "no_red_mahjong",
            one_round=True,  # True: Single round, False: Hanchan (East-South game)
            observe_type="dict", # "dict" for Transformer, "2D" for CNN
            order_points=[30, 10, -10, -30],  # Final score bonuses (uma)
        )
        ```
    """
    from mahjax.no_red_mahjong.env import NoRedMahjong

    if env_id == "no_red_mahjong":
        return NoRedMahjong(**kwargs)
    elif env_id == "red_mahjong":
        raise NotImplementedError("Red mahjong is not implemented yet")
    else:
        raise ValueError(
            f"Wrong env_id '{env_id}' is passed. Available ids are: \n{available_envs()}"
        )
