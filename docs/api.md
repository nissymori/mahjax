# How to install

```
pip install mahjax
```

# Usage
We follow almost the same api as [pgx](https://github.com/sotetsuk/pgx).
Below is the example of usage of mahjax.


```py
import jax
import jax.numpy as inp
import mahjax
from mahjax.action import Action


batch_size = 10
rng = jax.random.PRNGKey(0)

# Initialize environment and state
env = mahjax.make(
    "no_red_mahjong",
    one_round=True,  # one_round: if False, hanchan game.
    order_points=[30, 10, -10, -30]  # You can specify the order points by thousands.
)
step_fn = jax.jit(jax.vmap(env.step))
obs_fn = jax.jit(jax.vmap(env.observe))

# Initialize state
rng, subrng = jax.random.split(rng)
rngs = jax.random.split(subrng, batch_size)
state = jax.jit(jax.vmap(env.init))(rngs)

# Tsumogiri play
while not state.terminated.all():
    rng, subrng = jax.random.split(rng, batch_size)
    obs = obs_fn(state)  # (batch_size, ...) access to the observation.
    action = jnp.full(mahjax.Action.Tsumogiri, batch_size)
    state = step_fn(state, action, subrng)
    reward = state.rewards  # (batch_size, 4) access to the reward.
```

