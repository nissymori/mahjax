# NOTE: This file is copied and modified from Pgx (https://github.com/sotetsuk/pgx).
# Copyright belongs to the original authors.
# We keep tracking the updates of original Pgx implementation.
# We try to minimize the modification to this file. Exceptions includes:
#   - put the step and init function out of the jax.lax.cond
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


from typing import Optional

import jax
import jax.numpy as jnp

from mahjax._src.types import Array, PRNGKey
from mahjax.core import State

FALSE = jnp.bool_(False)


def auto_reset(step_fn, init_fn):
    """Auto reset wrapper."""

    def wrapped_step_fn(state: State, action: Array, key: Optional[PRNGKey] = None):
        assert key is not None, ()

        key1, key2 = jax.random.split(key)
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(  # type: ignore
                terminated=FALSE,
                truncated=FALSE,
                rewards=jnp.zeros_like(state.rewards),
            ),
            lambda: state,
        )
        state = step_fn(state, action, key1)
        init_state = init_fn(key2)
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            # state is replaced by initial state,
            # but preserve (terminated, truncated, reward)
            lambda: init_state.replace(  # type: ignore
                terminated=state.terminated,
                truncated=state.truncated,
                rewards=state.rewards,
            ),
            lambda: state,
        )
        return state

    return wrapped_step_fn
