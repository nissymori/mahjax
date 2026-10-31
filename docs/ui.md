# MahJax UI

MahJax ships a browser UI with a FastAPI backend and a plain JS frontend (no build step). It does three things:

- play against bundled or your own agents;
- save games and replay them;
- play back states from Python with `mahjax.ui.show`, in a notebook or as one HTML file, with no server.

It covers both tables: four players (`red_4p`, `no_red_4p`) and three (`red_3p`, `no_red_3p`). At three players the seat across stays empty, and the norths each player sets aside lie beyond their melds.

## Quickstart

```bash
pip install mahjax
uvicorn mahjax.ui.app:create_app --factory --host 127.0.0.1 --port 8000
```

Open http://localhost:8000 and start a game from the menu (☰).

## Playing

The menu's **新規対局** section sets up a game:

- **Rules:** 赤あり (`red_4p`), 赤なし (`no_red_4p`), 三人麻雀 赤あり (`red_3p`) or 三人麻雀 赤なし (`no_red_3p`).
- **Length:** half (半荘), east (東風), or a single round (一局).
- **Opponent:** any registered agent for those rules.
- **Seat:**
  - a fixed or random seat for you (three players have no north seat);
  - none, which plays the agents against each other and saves the game.
- **Seed:** leave it empty for a random wall.
- **Name**, and whether to **save the record**.

The board offers only legal actions, so every move you make is valid. Calls, riichi and wins appear as buttons next to your hand. Seats are labelled by their index in the env, Player 0 to Player 3.

These settings change during a game:

- hide the opponents' hands;
- **no calls**: pass every call offer automatically, but still offer ron;
- the delay between agent moves;
- the language: 日本語, English, 한국어 or 简体中文. English also switches to tiles with English labels.

A round ends with a summary of the winners, yaku, han and fu, and the score changes. The han add up to the points the env paid.

## Records and replays

A saved game is an [mjai](https://mjai.app)-style event log, `records/<id>.mjson`. Set `MAHJAX_RECORD_DIR` to keep the logs somewhere else.

The menu's **牌譜** section lists the saved games. Opening one replays it from its seed, and the replay controls:

- step through the moves (← →);
- jump from round to round;
- play the game back automatically;
- choose whose view to follow, or show every hand.

A log records the seed, so it replays only under a MahJax version that plays the same game. A log from a version that plays differently is refused with a message that names that version.

## Showing states from Python

`mahjax.ui.show` plays states back on the same replay screen, with no server, so you can look at a rollout:

```python
import jax
import mahjax
from mahjax.ui import save_html, show

env = mahjax.make("red_4p", round_mode="east", round_end="confirm")
step, init, act = jax.jit(env.step), jax.jit(env.init), jax.jit(mahjax.rule_based_player)
key = jax.random.PRNGKey(0)
state = init(key)
states = [state]
while not bool(state.terminated):
    key, k1, k2 = jax.random.split(key, 3)
    state = step(state, act(state, k1), k2)
    states.append(state)

show(states)                    # in a notebook, the cell's output
save_html(states, "game.html")  # one file that any browser opens
```

- **Input:** one unbatched state, a list of them, or a state with a leading time axis. Take a single env out of a batch first.
- **Where it shows:**
  - In a notebook (Jupyter, JupyterLab, VS Code, Colab), the screen is the cell's output. `height` sets its height in pixels.
  - From a script, `show` saves the page to a temporary file and prints its path.
  - `save_html` saves the page where you say.
- **The page:** it holds the script, the styles, the tile art and every frame, so it opens offline and can be shared as it is. An east game comes to about 0.6 MB.
- **What each frame shows:**
  - Where two states are one recorded action apart, the frame names the action.
  - A round whose end is among the states gets its result.
  - States from a `round_end="auto"` env skip the end of each round, so their rounds just follow one another.
- **Replaces the SVG helpers:** `save_svg`, `save_svg_animation` and `State.to_svg` are gone. `save_html(states, path)` takes what `save_svg_animation` took.

## Your own agents

An agent is a callable `act(state, key) -> action`, with the same signature as `mahjax.rule_based_player`. Register it on the app before you serve it:

```python
# my_ui_app.py
from mahjax.ui.app import create_app

from my_agent import act

app = create_app()
app.state.registry.add_agent(name="My agent", act=act, description="A trained policy")
```

```bash
uvicorn my_ui_app:app --host 127.0.0.1 --port 8000
```

`app.state.registry.load_callable_from_path(file_path=..., attribute="act")` loads one from a file instead. Pass `first=True` to make it the default opponent.

## Notes

- There is no authentication. Keep the server on localhost, or behind a VPN or a reverse proxy with authentication.
