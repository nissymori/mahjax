# MahJax UI Guide

MahJax bundles a FastAPI backend plus a static JS frontend so you can fight the packaged `mahjax.no_red_mahjong` agents from any browser. This short guide focuses on what you need to run it for a release.

## Quickstart

```bash
git clone https://github.com/nissymori/mahjax.git
cd mahjax
python -m pip install -e .  # install deps from requirements.txt
uvicorn mahjax.ui.app:create_app --host 0.0.0.0 --port 8000
```

Requirements: Python 3.10+, a matching JAX wheel (install beforehand), and the listed runtime deps (`fastapi`, `uvicorn`, `svgwrite`, `typing_extensions`). `--reload` is handy during development but disable it for production.

## Core Features

- **Instant play** – the UI only exposes legal actions computed by the environment, so human turns are always valid.
- **Bilingual board** – the backend emits both Japanese and English SVG fragments; the language toggle swaps them without network calls.
- **Round insight** – modal summaries list winners, yaku, fan/fu, honba, kyotaku plus a concise action log (latest 50 entries).
- **Observer toggle** – hide opponent hands until someone other than you declares Ron/Tsumo or a round ends.
- **Pluggable agents** – register Python callables with `AgentRegistry` (builtin `Rule-based` + `Random` included) and they appear immediately in `/api/agents`.

## Operating the UI

- **Header controls**: pick the agent, match length (`hanchan` / `one_round`), seat (fixed or random), optional seed, display names, AI delay in ms, and the “hide opponent hands” switch. Start/End buttons create or delete sessions.
- **Board & actions**: the human seat is always oriented to the bottom. Tile buttons correspond to discard IDs (`0–33`); special buttons cover riichi/tsumo/ron/pass/kan/pon/chi. When a dummy-only action is left, an “Advance” button triggers `Action.DUMMY`.
- **Scoreboard & log**: scores are shown in turn order relative to the human, with current deltas. Logs record when each player discarded, called, or declared.
- **Round summaries**: when `_terminated_round` is set, the server returns `roundSummary` and the UI displays standings + winner info. Press “Next Round” (or “End Game” when `isGameEnd`) to continue.

## Game Flow & API

1. `POST /api/game` with fields such as `agent_id`, `mode`, `seed`, `human_seat`, `ai_delay_ms`, `hide_opponent_hands`. The server spins up a `GameSession`, auto-plays AI turns, and returns the first state requiring human input (`phase = awaiting_human`).
2. For every human decision call `POST /api/game/{id}/action` with a legal action number (discard, riichi, ron, etc.). The response always includes the authoritative state (`svgJapanese`, `svgEnglish`, `legalActions`, `hand`, `scores`, `events`).
3. While the AI is thinking, the frontend periodically hits `POST /api/game/{id}/auto` to let the agent continue for a few steps. The delay is purely client-side (`aiDelayMs`).
4. After a round ends use `POST /api/game/{id}/continue` to clear the dummy actions and move to the next round. Delete stale sessions with `DELETE /api/game/{id}`.

Action IDs follow the environment definition: 0–33 are discards, 34–67 are Kans, and constants such as `Action.TSUMOGIRI`, `Action.RIICHI`, `Action.TSUMO`, `Action.RON`, `Action.PON`, `Action.OPEN_KAN`, `Action.CHI_*`, `Action.PASS`, `Action.DUMMY` cover the rest. The server rejects anything not enabled in `legal_action_mask`.

## Extending Agents & UI

- Register agents by calling `AgentRegistry.add_agent`, `load_callable_agent`, or `load_callable_from_path` after `create_app()`. The callable receives `(state: State, rng: jnp.ndarray)` and must return a valid action ID.
- Customize the frontend by editing `mahjax/ui/static/app.js` (localization strings, behavior) and `styles.css` (layout). No Node.js build step is required; FastAPI serves the files directly.

## Troubleshooting & Limits

- Blank board? Ensure `mahjax/ui/static` exists and FastAPI can mount it. Check console for 404s on `/static`.
- Buttons disabled? The game is likely in `awaiting_ai` or `round_end`. Wait for auto-play or close the summary overlay.
- AI idle? Verify `ai_delay_ms` and inspect responses from `/api/game/{id}/auto`.
- Hidden hands stay hidden? Disable the checkbox or note that only non-human wins reveal opponents.
- Current implementation targets 4-player Riichi without red dora and has no authentication. Put the server behind a reverse proxy/VPN before exposing it on the public internet.
