<div align="center">
<img src="https://github.com/nissymori/mahjax/blob/main/docs/assets/logo.svg" width="35%">
</div>

<br>

<div align="center">
  <img src="https://github.com/nissymori/mahjax/blob/main/docs/assets/red_mahjong_random_ja.gif" width="23%">
  <img src="https://github.com/nissymori/mahjax/blob/main/docs/assets/red_mahjong_random_en.gif" width="23%">
</div>

# MahJax

[![PyPI](https://img.shields.io/pypi/v/mahjax.svg)](https://pypi.org/project/mahjax)
[![License](https://img.shields.io/pypi/l/mahjax.svg)](https://pypi.org/project/mahjax)
[![Supported Python versions](https://img.shields.io/pypi/pyversions/mahjax.svg)](https://pypi.org/project/mahjax)

**A GPU-Accelerated Mahjong Simulator for Reinforcement Learning in [JAX](https://github.com/google/jax)**

Japanese Riichi Mahjong is a complex board game that presents a unique combination of **imperfect information**, **multi-player (>2) competition**, **stochastic dynamics**, and **high-dimensional inputs**.
MahJax is highly inspired by [Pgx](https://github.com/sotetsuk/pgx), which offers vectorized simulators for a diverse set of board games.
While Pgx includes imperfect information games (such as miniature poker and mahjong), its primary emphasis is on deterministic perfect-information games like Go, Chess, and Shogi.
We aim to complement this by offering a full-scale Japanese Riichi Mahjong environment written entirely in JAX.

## Overview

- **Vectorized Environment:** Fully JIT-compilable and extremely fast (approx. **1.6M steps/sec** on 8x A100 GPUs).
- **Beautiful Visualization:** Like Pgx, we offer SVG-based game visualization. We also provide an English tile version for those unfamiliar with Chinese characters (Kanji).
- **Playable Interface:** A web-based UI allows you to play directly against the agents you train.
- **RL Examples:** We provide simple examples for Behavior Cloning and Reinforcement Learning in the [`examples/`](https://github.com/nissymori/mahjax/tree/main/examples) directory.

For more details, please refer to the [Documentation](https://nissymori.github.io/mahjax/).

## Quick Start
### Install
Mahjax is available on PyPI. Please make sure that your Python environment has jax and jaxlib installed, depending on your hardware specification.
```bash
pip install mahjax
```

📣 Mahjax is currently under active development. If you prefer to use the latest codebase with the newest features, please clone the repository and install it in editable mode:

```bash
git clone https://github.com/nissymori/mahjax.git
cd mahjax
pip install -e .
```

> [!NOTE]
> The current API is still provisional and under active development, so it may change in future releases.

### Basic Usage
We basically follow the [Pgx](https://github.com/sotetsuk/pgx) API design.

```python
import jax
import jax.numpy as jnp
import mahjax

batch_size = 10
rng = jax.random.PRNGKey(0)

# Initialize environment
env = mahjax.make(
    "red_mahjong",
    one_round=True,      # True: Single round, False: Hanchan (East-South game)
    observe_type="dict", # "dict" for Transformer, "2D" for CNN
    order_points=[30, 10, -10, -30] # Final score bonuses (uma)
)

init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))
obs_fn = jax.jit(jax.vmap(env.observe))

# Initialize state
rng, subrng = jax.random.split(rng)
rngs = jax.random.split(subrng, batch_size)
state = init_fn(rngs)

# Step
rng, subrng = jax.random.split(rng)
rngs = jax.random.split(subrng, batch_size)
action = jnp.zeros((batch_size,), dtype=jnp.int8)
state = step_fn(state, action, rngs)

# Get observation
obs = obs_fn(state)
```

## User interface
MahJax includes a web-based UI (FastAPI + JS) that allows you to play against built-in or custom agents directly in your browser.

### Running the UI

Install dependencies and start the server:
```bash
pip install mahjax
uvicorn mahjax.ui.app:create_app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000 to start playing. The default agents are random and rule_based one.

### Playing Against Your Agent
You can register your trained agent to appear in the UI's agent selector.
Create a python script (e.g., `my_app.py`) and register your agent's act function:

```py
### my_app.py
from pathlib import Path
from mahjax.ui.app import create_app

app = create_app()

# Load your custom agent
app.state.manager.registry.load_callable_from_path(
    file_path=Path("path/to/my_agent.py"),
    attribute="act", # The function name to call: act(state, rng) -> action_id
    description="My Custom Agent",
)
```
Run `uvicorn my_ui:app --port 8000`.    

## Supported Rules

Currently, MahJax supports the following rules:

| Rule | id | Status | Code | Speed (steps/sec) |
|------|------|--------|------|--------|
| No-Red Mahjong | `no_red_mahjong` | ✅ | [no_red_mahjong](https://github.com/nissymori/mahjax/tree/main/mahjax/mahjax/no_red_mahjong) | ~1.6M |
| Red Mahjong | `red_mahjong` | ✅ | [red_mahjong](https://github.com/nissymori/mahjax/tree/main/mahjax/mahjax/red_mahjong) | ~9M |
| Selective Rules | - | 🚧 | - | - |
| 3-player Mahjong | - | 🚧 | - | - |

`red_mahjong` implements standard 4-player riichi mahjong with red fives.
Its rules are designed to follow [Tenhou](https://tenhou.net/), one of the most widely used online mahjong platforms in Japan, and we validate the implementation against downloaded Tenhou game logs.
For the detailed rule specification, see the [official Tenhou rules](https://tenhou.net/0/mj/mjlog/en/mjlog-en-rules.html).

`no_red_mahjong` implements 4-player riichi mahjong without red fives.
This variant is intentionally simplified for speed, and excludes some rules such as abortive draws (`特殊流局`), pao, and double ron.
If throughput is your priority, `no_red_mahjong` is the recommended option.

You can configure the environment with:

- `id`: the rule set, such as `red_mahjong` or `no_red_mahjong`
- `round_mode`: `single` for a single round, `half` for tonpuusen (East-only), or `full` for hanchan (East-South)
- `observe_type`: `dict` for transformer-style inputs or `2D` for CNN-style inputs
- `order_points`: final placement bonuses (uma), for example `[30, 10, -10, -30]`

```python
env = mahjax.make(
    id="red_mahjong",
    round_mode="single",
    observe_type="dict",
    order_points=[30, 10, -10, -30],
)
```



## See also

Jax based environments
- [Pgx](https://github.com/sotetsuk/pgx): Boad game environments such as Go, chess, and Shogi.
- [Brax](https://github.com/google/brax): Robotics control.
- [Gymnax](https://github.com/RobertTLange/gymnax): Popular small scale RL environments such as cartpole or bsuite.
- [Jumanji](https://github.com/instadeepai/jumanji): A diverse suite of RL environments (paking, routing, etc).
- [Craftax](https://arxiv.org/abs/2402.16801): JAX-version of (Crafter + Nethack).
- [JaxMARL](https://github.com/FLAIROx/JaxMARL): Multi-agent environments such as Hanabi.
- [Navix](https://github.com/epignatelli/navix): JAX-version of MiniGrid.

## Cite us
Paper comming soon.

## Acknowledgement
- [sotetsuk](https://github.com/sotetsuk): For general advice on the development of mahjax based on his experience of developping pgx
- [habara-k](https://github.com/habara-k): For developing core JAX components such as shanten and Yaku calculation.
- [OkanoShinri](https://github.com/OkanoShinri): For the initial implementation of MahJax and its SVG visualization.
- [easonyu0203](easonyu0203): For advise on PPO implementation in multi-player imperfect information game.
