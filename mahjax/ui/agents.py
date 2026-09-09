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

"""Registry of the agents a human can sit down against.

An agent is just ``act(state, key) -> int32`` -- the same shape as
``mahjax.red_mahjong.players.rule_based_player`` -- so a trained policy only has
to be wrapped in an argmax over the legal actions to appear in the UI.

Agents are declared per env: the bundled rule-based players read different hand
representations in the red and no-red envs and are not interchangeable.
"""

from __future__ import annotations

import importlib
import importlib.util
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp

ActFn = Callable[[Any, Any], Any]

ENV_IDS = ("red_mahjong", "no_red_mahjong")


@dataclass(frozen=True)
class Agent:
    agent_id: str
    name: str
    env_ids: Sequence[str]
    act: ActFn
    description: str = ""

    def supports(self, env_id: str) -> bool:
        return env_id in self.env_ids

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.agent_id,
            "name": self.name,
            "envs": list(self.env_ids),
            "description": self.description,
        }


def _jit_once(fn: Callable[..., Any]) -> Callable[..., Any]:
    cache: Dict[int, Callable[..., Any]] = {}

    def wrapped(state: Any, key: Any) -> Any:
        # jax.jit caches per wrapper object, so the wrapper has to outlive the
        # call -- rebuilding it every turn would retrace on every move.
        if 0 not in cache:
            cache[0] = jax.jit(fn)
        return cache[0](state, key)

    return wrapped


def _rule_based(env_id: str) -> ActFn:
    module = importlib.import_module(f"mahjax.{env_id}.players")
    inner = _jit_once(module.rule_based_player)

    def act(state: Any, key: Any) -> Any:
        return jnp.asarray(inner(state, key), dtype=jnp.int32)

    return act


def _random_action(state: Any, key: Any) -> Any:
    # Uniform over the legal actions, written as a masked argmax so it stays
    # jittable (the number of legal actions is not known at trace time).
    noise = jax.random.uniform(key, state.legal_action_mask.shape)
    scored = jnp.where(state.legal_action_mask, noise, -jnp.inf)
    return jnp.argmax(scored).astype(jnp.int32)


_RANDOM = _jit_once(_random_action)


def _random(state: Any, key: Any) -> Any:
    return jnp.asarray(_RANDOM(state, key), dtype=jnp.int32)


@dataclass
class AgentRegistry:
    """Ordered registry; the first agent supporting an env is its default."""

    _agents: Dict[str, Agent] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self._agents:
            self._register_builtins()

    def _register_builtins(self) -> None:
        for env_id in ENV_IDS:
            self.add_agent(
                agent_id=f"rule_based_{'red' if env_id == 'red_mahjong' else 'no_red'}",
                name="Rule-based",
                env_ids=(env_id,),
                act=_rule_based(env_id),
                description=f"Heuristic player bundled with mahjax ({env_id}).",
            )
        self.add_agent(
            agent_id="random",
            name="Random",
            env_ids=ENV_IDS,
            act=_random,
            description="Uniformly random legal action.",
        )

    def add_agent(
        self,
        *,
        name: str,
        act: ActFn,
        env_ids: Sequence[str] = ENV_IDS,
        agent_id: Optional[str] = None,
        description: str = "",
        first: bool = False,
    ) -> Agent:
        """Register an agent. ``first=True`` makes it the default for its envs."""
        if isinstance(env_ids, str):
            env_ids = (env_ids,)
        unknown = [e for e in env_ids if e not in ENV_IDS]
        if unknown:
            raise ValueError(f"Unknown env ids: {unknown}")
        agent = Agent(
            agent_id=agent_id or uuid.uuid4().hex,
            name=name,
            env_ids=tuple(env_ids),
            act=act,
            description=description,
        )
        if first:
            self._agents = {agent.agent_id: agent, **self._agents}
        else:
            self._agents[agent.agent_id] = agent
        return agent

    def get(self, agent_id: str) -> Agent:
        if agent_id not in self._agents:
            raise KeyError(f"Unknown agent id: {agent_id}")
        return self._agents[agent_id]

    def all(self) -> List[Agent]:
        return list(self._agents.values())

    def for_env(self, env_id: str) -> List[Agent]:
        return [a for a in self._agents.values() if a.supports(env_id)]

    def default_for(self, env_id: str) -> Agent:
        candidates = self.for_env(env_id)
        if not candidates:
            raise LookupError(f"No agent registered for {env_id}")
        return candidates[0]

    def resolve(self, env_id: str, agent_id: Optional[str]) -> Agent:
        if agent_id is None:
            return self.default_for(env_id)
        agent = self.get(agent_id)
        if not agent.supports(env_id):
            raise ValueError(f"Agent {agent_id} does not support {env_id}")
        return agent

    # ------------------------------------------------------- loading helpers

    def load_callable(
        self,
        *,
        module: str,
        attribute: str,
        name: Optional[str] = None,
        env_ids: Sequence[str] = ENV_IDS,
        description: str = "",
        first: bool = False,
    ) -> Agent:
        mod = importlib.import_module(module)
        return self._add_loaded(
            getattr(mod, attribute), name or attribute, env_ids, description, first
        )

    def load_callable_from_path(
        self,
        *,
        file_path: Path,
        attribute: str,
        name: Optional[str] = None,
        env_ids: Sequence[str] = ENV_IDS,
        description: str = "",
        first: bool = False,
    ) -> Agent:
        spec = importlib.util.spec_from_file_location(Path(file_path).stem, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return self._add_loaded(
            getattr(module, attribute), name or attribute, env_ids, description, first
        )

    def _add_loaded(
        self,
        fn: Any,
        name: str,
        env_ids: Sequence[str],
        description: str,
        first: bool,
    ) -> Agent:
        if not callable(fn):
            raise TypeError(f"{name} is not callable")

        def act(state: Any, key: Any) -> Any:
            return jnp.asarray(fn(state, key), dtype=jnp.int32)

        return self.add_agent(
            name=name,
            act=act,
            env_ids=env_ids,
            description=description,
            first=first,
        )


__all__ = ["Agent", "AgentRegistry", "ActFn", "ENV_IDS"]
