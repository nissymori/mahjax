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

"""HTTP surface for the play and replay screens.

Handlers are synchronous on purpose: stepping the env is CPU work, so FastAPI
runs them in its worker threads and one lock keeps the session tables
consistent, instead of blocking the event loop.
"""

from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import record as record_mod
from .agents import ENV_IDS, AgentRegistry
from .match import Match, MatchConfig, new_seed, shared_env, shared_step_fn
from .record import RecordError, Replay
from .rules import rules_for

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
TILE_DIR = Path(__file__).resolve().parents[1] / "_src" / "assets" / "tiles"

MAX_GAMES = 32
MAX_REPLAYS = 8


class _RevalidatedFiles(StaticFiles):
    """Static files a browser must check with the server on every load.

    Without a Cache-Control header a browser may reuse the page's script,
    stylesheet or tile art for a while without asking, which after an edit
    leaves the player looking at a half-updated UI with no clue why.
    """

    def file_response(self, *args: Any, **kwargs: Any) -> Any:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


class CreateGame(BaseModel):
    env_id: Literal["red_mahjong", "no_red_mahjong"] = "red_mahjong"
    round_mode: Literal["single", "east", "half"] = "half"
    agent_id: Optional[str] = None
    seed: Optional[int] = Field(None, ge=0, lt=2**31)
    # None means an agent-only game: the server plays it out and saves a record.
    human_seat: Optional[int] = Field(0, ge=0, le=3)
    random_seat: bool = False
    human_name: str = "You"
    hide_hands: bool = True
    no_calls: bool = False
    save_record: bool = True


class ActRequest(BaseModel):
    action: int = Field(..., ge=0, lt=128)


class OptionsRequest(BaseModel):
    hide_hands: Optional[bool] = None
    no_calls: Optional[bool] = None


class CreateReplay(BaseModel):
    record_id: str


@dataclass
class _Sessions:
    registry: AgentRegistry = field(default_factory=AgentRegistry)
    games: Dict[str, Match] = field(default_factory=dict)
    replays: Dict[str, Replay] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def game(self, game_id: str) -> Match:
        match = self.games.get(game_id)
        if match is None:
            raise HTTPException(status_code=404, detail=f"No game {game_id}")
        return match

    def replay(self, replay_id: str) -> Replay:
        replay = self.replays.get(replay_id)
        if replay is None:
            raise HTTPException(status_code=404, detail=f"No replay {replay_id}")
        return replay

    @staticmethod
    def evict(table: Dict[str, Any], limit: int) -> None:
        while len(table) > limit:
            table.pop(next(iter(table)))


#: Compiled ahead of the first game so nobody waits on it. Everything else
#: compiles when it is first asked for, which lands on the "starting" screen.
WARM_ENV = ("red_mahjong", "half")


def _warm_in_background() -> None:
    """Compile the env step for the default rules while the player reads the menu."""

    def run() -> None:
        import jax
        import jax.numpy as jnp

        try:
            env_id, round_mode = WARM_ENV
            rules = rules_for(env_id)
            state = jax.device_get(shared_env(env_id, round_mode).init(jax.random.PRNGKey(0)))
            legal = rules.legal_actions(state)
            step = shared_step_fn(env_id, round_mode)
            jax.block_until_ready(step(state, jnp.int32(legal[0]), jax.random.PRNGKey(1)))
        except Exception:  # a failed warm-up must never take the server with it
            log.warning("env warm-up failed; the first game will compile instead", exc_info=True)

    threading.Thread(target=run, name="mahjax-warmup", daemon=True).start()


def _register_pages(app: FastAPI, sessions: _Sessions) -> None:
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404, detail="index.html is missing")
        return page.read_text(encoding="utf-8")

    @app.get("/api/agents")
    def list_agents(env: Optional[str] = Query(None)) -> List[Dict[str, Any]]:
        if env is not None and env not in ENV_IDS:
            raise HTTPException(status_code=400, detail=f"Unknown env {env}")
        chosen = sessions.registry.for_env(env) if env else sessions.registry.all()
        return [agent.to_dict() for agent in chosen]


def _register_games(app: FastAPI, sessions: _Sessions) -> None:
    @app.post("/api/games")
    def create_game(req: CreateGame) -> Dict[str, Any]:
        seed = req.seed if req.seed is not None else new_seed()
        human_seat = req.human_seat
        if req.random_seat and human_seat is not None:
            human_seat = random.Random(seed ^ 0x5A5A).randrange(4)
        try:
            agent = sessions.registry.resolve(req.env_id, req.agent_id)
        except (KeyError, LookupError, ValueError) as err:
            raise HTTPException(status_code=400, detail=str(err)) from err

        config = MatchConfig(
            env_id=req.env_id,
            round_mode=req.round_mode,
            seed=seed,
            human_seat=human_seat,
            human_name=req.human_name,
            agent_id=agent.agent_id,
            hide_hands=req.hide_hands,
            no_calls=req.no_calls,
            save_record=req.save_record,
        )
        with sessions.lock:
            match = Match(config, agent, on_game_over=record_mod.save_match)
            if human_seat is None:
                match.play_out()
                return {
                    "gameId": None,
                    "recordId": match.record_id,
                    "frames": [],
                    "seed": seed,
                }
            frames = match.start()
            sessions.games[match.id] = match
            sessions.evict(sessions.games, MAX_GAMES)
            return {"gameId": match.id, "frames": frames, "seed": seed}

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str) -> Dict[str, Any]:
        with sessions.lock:
            match = sessions.game(game_id)
            # The record id travels with the state so a reloaded page can still
            # offer to open the finished game.
            return {"gameId": match.id, "frames": [match.view], "recordId": match.record_id}

    @app.post("/api/games/{game_id}/act")
    def act(game_id: str, req: ActRequest) -> Dict[str, Any]:
        with sessions.lock:
            match = sessions.game(game_id)
            try:
                frames = match.act(req.action)
            except ValueError as err:
                raise HTTPException(status_code=400, detail=str(err)) from err
            return {"gameId": match.id, "frames": frames, "recordId": match.record_id}

    @app.post("/api/games/{game_id}/next")
    def next_round(game_id: str) -> Dict[str, Any]:
        with sessions.lock:
            match = sessions.game(game_id)
            frames = match.next_round()
            return {"gameId": match.id, "frames": frames, "recordId": match.record_id}

    @app.patch("/api/games/{game_id}")
    def set_options(game_id: str, req: OptionsRequest) -> Dict[str, Any]:
        with sessions.lock:
            match = sessions.game(game_id)
            match.set_options(hide_hands=req.hide_hands, no_calls=req.no_calls)
            return {"gameId": match.id, "frames": [match.view]}

    @app.delete("/api/games/{game_id}")
    def delete_game(game_id: str) -> Dict[str, Any]:
        with sessions.lock:
            match = sessions.games.pop(game_id, None)
            record_id = match.abandon() if match is not None else None
            return {"status": "ok", "recordId": record_id}


def _register_records(app: FastAPI, sessions: _Sessions) -> None:
    @app.get("/api/records")
    def list_records() -> List[Dict[str, Any]]:
        return record_mod.list_records()

    @app.delete("/api/records/{record_id}")
    def delete_record(record_id: str) -> Dict[str, Any]:
        record_mod.delete_record(record_id)
        return {"status": "ok"}

    @app.post("/api/replays")
    def create_replay(req: CreateReplay) -> Dict[str, Any]:
        with sessions.lock:
            try:
                replay = record_mod.load_replay(req.record_id)
            except RecordError as err:
                raise HTTPException(status_code=400, detail=str(err)) from err
            sessions.replays[replay.id] = replay
            sessions.evict(sessions.replays, MAX_REPLAYS)
            return replay.to_dict()

    @app.get("/api/replays/{replay_id}/frames")
    def replay_frames(
        replay_id: str,
        start: int = Query(0, alias="from", ge=0),
        stop: Optional[int] = Query(None, alias="to"),
        viewpoint: Optional[int] = Query(None, ge=0, le=3),
        show_all: bool = Query(True, alias="showAll"),
    ) -> Dict[str, Any]:
        with sessions.lock:
            replay = sessions.replay(replay_id)
            end = replay.total if stop is None else stop
            return {
                "replayId": replay.id,
                "from": start,
                "frames": replay.frames(
                    start, end, viewpoint=viewpoint, show_all=show_all
                ),
            }

    @app.delete("/api/replays/{replay_id}")
    def delete_replay(replay_id: str) -> Dict[str, Any]:
        with sessions.lock:
            sessions.replays.pop(replay_id, None)
            return {"status": "ok"}


def create_app() -> FastAPI:
    app = FastAPI(title="Mahjax UI", version="2")
    sessions = _Sessions()

    app.state.registry = sessions.registry
    app.state.games = sessions.games
    app.state.replays = sessions.replays

    if STATIC_DIR.exists():
        app.mount("/static", _RevalidatedFiles(directory=STATIC_DIR), name="static")
    if TILE_DIR.exists():
        app.mount("/tiles", _RevalidatedFiles(directory=TILE_DIR), name="tiles")

    _warm_in_background()
    _register_pages(app, sessions)
    _register_games(app, sessions)
    _register_records(app, sessions)
    return app


__all__ = ["create_app"]
