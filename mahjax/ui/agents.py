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


from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import jax
import jax.numpy as jnp

from mahjax.no_red_mahjong.players import rule_based_player
from mahjax.red_mahjong.players import rule_based_player as red_rule_based_player
from mahjax.no_red_mahjong.state import State

AgentFn = Callable[[State, jnp.ndarray], jnp.ndarray]


@dataclass
class Agent:
    agent_id: str
    name: str
    description: str
    act: AgentFn


class AgentRegistry:
    """Keep track of available agents for the UI server."""

    def __init__(self) -> None:
        self._registry: Dict[str, Agent] = {}
        self._register_builtin_agents()

    def _register_builtin_agents(self) -> None:
        self.add_agent(
            agent_id="rule_based",
            name="Rule-based",
            description="Heuristic rule-based agent bundled with MahJax.",
            act=_rule_based_act,
        )
        self.add_agent(
            agent_id="rule_based_red",
            name="Rule-based (Red)",
            description="Heuristic rule-based agent for red_mahjong.",
            act=_rule_based_red_act,
        )
        self.add_agent(
            agent_id="random",
            name="Random",
            description="Selects a uniformly random legal action.",
            act=_random_act,
        )

    def add_agent(
        self,
        *,
        agent_id: Optional[str] = None,
        name: str,
        description: str,
        act: AgentFn,
    ) -> Agent:
        if agent_id is None:
            agent_id = uuid.uuid4().hex
        agent = Agent(agent_id=agent_id, name=name, description=description, act=act)
        self._registry[agent.agent_id] = agent
        return agent

    def get(self, agent_id: str) -> Agent:
        if agent_id not in self._registry:
            raise KeyError(f"Unknown agent id: {agent_id}")
        return self._registry[agent_id]

    def all(self) -> Dict[str, Agent]:
        return dict(self._registry)

    def default_agent(self) -> Agent:
        if not self._registry:
            raise LookupError("No agents registered")
        first_key = next(iter(self._registry))
        return self._registry[first_key]

    def load_callable_agent(
        self, *, module: str, attribute: str, description: Optional[str] = None
    ) -> Agent:
        mod = importlib.import_module(module)
        act_fn = getattr(mod, attribute)
        if not callable(act_fn):
            raise TypeError(f"{module}.{attribute} is not callable")
        agent_name = getattr(act_fn, "__name__", attribute)
        desc = description or f"Callable agent {module}.{attribute}"
        return self.add_agent(
            name=agent_name,
            description=desc,
            act=_wrap_callable(act_fn),
        )

    def load_callable_from_path(
        self,
        *,
        file_path: Path,
        attribute: str,
        description: Optional[str] = None,
    ) -> Agent:
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        assert isinstance(loader, importlib.abc.Loader)
        loader.exec_module(module)  # type: ignore[attr-defined]
        act_fn = getattr(module, attribute)
        if not callable(act_fn):
            raise TypeError(f"{attribute} in {file_path} is not callable")
        agent_name = getattr(act_fn, "__name__", attribute)
        desc = description or f"Callable agent from {file_path}:{attribute}"
        return self.add_agent(
            name=agent_name,
            description=desc,
            act=_wrap_callable(act_fn),
        )


def _wrap_callable(fn: Callable[[State, jnp.ndarray], int]) -> AgentFn:
    def _act(state: State, rng: jnp.ndarray) -> jnp.ndarray:
        result = fn(state, rng)
        return jnp.asarray(result, dtype=jnp.int32)

    return _act


def _rule_based_act(state: State, rng: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(jax.jit(rule_based_player)(state, rng), dtype=jnp.int32)


def _rule_based_red_act(state: State, rng: jnp.ndarray) -> jnp.ndarray:
    return jnp.asarray(jax.jit(red_rule_based_player)(state, rng), dtype=jnp.int32)


def _random_act(state: State, rng: jnp.ndarray) -> jnp.ndarray:
    legal = jnp.where(state.legal_action_mask)[0]
    idx = jax.random.randint(rng, shape=(), minval=0, maxval=legal.shape[0])
    return jnp.asarray(legal[idx], dtype=jnp.int32)


def ensure_valid_action(action: int, mask: jnp.ndarray) -> int:
    if not bool(mask[action]):
        legal = jnp.where(mask)[0]
        legal_str = ", ".join(str(int(a)) for a in legal)
        raise ValueError(f"Illegal action {action}. Legal: [{legal_str}]")
    return int(action)


__all__ = ["Agent", "AgentRegistry", "AgentFn", "ensure_valid_action"]
