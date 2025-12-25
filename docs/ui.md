# MahJax UI Guide

MahJax ships with a FastAPI backend and a static JS frontend, so you can battle the bundled `mahjax.no_red_mahjong` bots in any browser. This guide shows the basics for running the UI in a release setting.

## Quickstart

```bash
pip install mahjax
uvicorn mahjax.ui.app:create_app --host 0.0.0.0 --port 8000
```

You need Python 3.10+, a matching JAX wheel (install it beforehand), and the runtime deps listed in `requirements.txt` (`fastapi`, `uvicorn`, `svgwrite`, `typing_extensions`). Use `--reload` while developing but turn it off in production.

## Core Features

- **Instant play** – the UI only shows legal actions, so every human move is valid.
- **Bilingual board** – the backend sends both Japanese and English SVGs, and the toggle swaps between them without extra requests.
- **Round insight** – modal summaries list winners, yaku, fan/fu, honba, kyotaku, plus a 50-entry action log.
- **Observer toggle** – hide enemy hands until a non-human Ron/Tsumo or the round end reveals them.
- **Pluggable agents** – register Python callables with `AgentRegistry` (ships with `Rule-based` and `Random`) and they show up right away in `/api/agents`.

## Using the UI

- **Header controls**: choose the agent, match length (`hanchan` or `one_round`), seat (fixed or random), optional seed, display names, AI delay in ms, and whether to hide opponent hands. The Start and End buttons spawn or delete sessions.
- **Board & actions**: the human seat stays at the bottom. Tile buttons map to discard IDs (`0–33`). Extra buttons cover riichi, tsumo, ron, pass, kan, pon, and chi. If only a dummy action remains, an “Advance” button sends `Action.DUMMY`.
- **Scoreboard & log**: scores use turn order relative to the human and display current deltas. Logs show discards, calls, and declarations.
- **Round summaries**: when `_terminated_round` flips on, the server returns `roundSummary`, and the UI shows standings plus winner info. Press “Next Round,” or “End Game” if `isGameEnd` is true.

## Extending Agents & UI

- **Custom Agents**: Register new agents using `AgentRegistry`. Your agent must be a callable accepting `(state, rng)` and returning a valid action ID.
- **Frontend Customization**: Edit `mahjax/ui/static/app.js` and `styles.css` directly. No Node.js build step is required.

### Adding Custom Agents

To add your own agent to the dropdown menu, create a wrapper script that registers your agent function before starting the app.

1.  **Define your agent**: Create a function `(state, rng) -> int`.
2.  **Register and Run**:

```python
# my_ui_app.py
from pathlib import Path
from mahjax.ui.app import create_app

app = create_app()

# Register your agent implementation
app.state.manager.registry.load_callable_from_path(
    file_path=Path("path/to/my_agent_impl.py"),
    attribute="act",  # Function name in your file
    description="My Custom Agent"
)
```

3. Run `uvicorn my_ui_app:app --host 0.0.0.0 --port 8000`. The agent now appears in `/api/agents`, so the UI selector lists it right away.

## Troubleshooting & Limits

- Blank board? Confirm `mahjax/ui/static` exists and FastAPI mounts it. Check the console for `/static` 404 errors.
- Buttons stuck? The game is in `awaiting_ai` or `round_end`. Wait for auto-play or dismiss the summary modal.
- AI idle? Check `ai_delay_ms` and watch `/api/game/{id}/auto` responses.
- Hidden hands never reveal? Clear the checkbox or note that only non-human wins unhide opponents.
- The current build only supports 4-player Riichi without red dora and runs without auth. Protect it behind a reverse proxy or VPN before exposing it to the internet.
