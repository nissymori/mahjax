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

import asyncio
import random
import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .game_manager import GameManager

STATIC_DIR = Path(__file__).resolve().parent / "static"


class CreateGameRequest(BaseModel):
    agent_id: Optional[str] = Field(None, description="Agent identifier")
    mode: Literal["one_round", "hanchan"] = Field("hanchan")
    seed: Optional[int] = Field(None, description="RNG seed. Random if omitted.")
    human_seat: Optional[int] = Field(None, ge=0, le=3)
    random_seat: bool = False
    human_name: Optional[str] = Field(None)
    ai_name: Optional[str] = Field(None)
    ai_delay_ms: int = Field(800, ge=0, le=10000)
    hide_opponent_hands: bool = Field(
        False, description="Hide opponent hands from the board visualization"
    )
    auto_pass_calls: bool = Field(
        False, description="Automatically pass Pon/Chi/Open Kan prompts for the human"
    )


class ActionRequest(BaseModel):
    action: int = Field(..., ge=0, le=78)


class AutoRequest(BaseModel):
    steps: int = Field(1, ge=1, le=32)


class VisibilityRequest(BaseModel):
    hide_opponent_hands: Optional[bool] = Field(
        None, description="Hide opponent hands without restarting the game"
    )
    auto_pass_calls: Optional[bool] = Field(
        None, description="Auto-pass Pon/Chi/Open-Kan prompts for the human"
    )


def create_app() -> FastAPI:
    app = FastAPI(title="MahJax Human vs AI UI", version="0.1.0")
    manager = GameManager()
    manager_lock = asyncio.Lock()
    app.state.manager = manager
    app.state.manager_lock = manager_lock

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="index.html not found")
        return index_path.read_text(encoding="utf-8")

    @app.get("/api/agents")
    async def list_agents():
        return [
            {
                "id": agent.agent_id,
                "name": agent.name,
                "description": agent.description,
            }
            for agent in manager.registry.all().values()
        ]

    @app.post("/api/game")
    async def create_game(req: CreateGameRequest):
        async with manager_lock:
            try:
                if req.agent_id:
                    agent = manager.registry.get(req.agent_id)
                else:
                    agent = manager.registry.default_agent()
            except (KeyError, LookupError) as err:
                raise HTTPException(status_code=404, detail=str(err)) from err
            seed = (
                req.seed if req.seed is not None else int(time.time() * 1000) % (2**32)
            )
            rng_for_seat = random.Random(seed ^ 0xA5A5_5A5A)
            if req.random_seat or req.human_seat is None:
                human_seat = rng_for_seat.randint(0, 3)
            else:
                human_seat = req.human_seat
            base_ai_name = req.ai_name or agent.name
            player_names = [f"{base_ai_name} {i+1}" for i in range(4)]
            player_names[human_seat] = req.human_name or "You"
            session = manager.create_session(
                agent_id=agent.agent_id,
                human_seat=human_seat,
                one_round=req.mode == "one_round",
                seed=seed,
                player_names=player_names,
                ai_delay_ms=req.ai_delay_ms,
                hide_opponent_hands=req.hide_opponent_hands,
                auto_pass_calls=req.auto_pass_calls,
            )
            return session.to_view()

    @app.get("/api/game/{game_id}")
    async def get_game(game_id: str):
        async with manager_lock:
            try:
                session = manager.get(game_id)
            except KeyError as err:
                raise HTTPException(status_code=404, detail=str(err)) from err
            return session.to_view()

    @app.post("/api/game/{game_id}/action")
    async def post_action(game_id: str, req: ActionRequest):
        async with manager_lock:
            try:
                session = manager.get(game_id)
            except KeyError as err:
                raise HTTPException(status_code=404, detail=str(err)) from err
            if session.state.terminated:
                raise HTTPException(status_code=400, detail="Game already finished")
            if session.round_summary is not None:
                raise HTTPException(status_code=400, detail="Round summary pending")
            if int(session.state.current_player) != session.human_seat:
                raise HTTPException(status_code=400, detail="Not human turn")
            try:
                session.apply_action(req.action, actor="human")
            except ValueError as err:
                raise HTTPException(status_code=400, detail=str(err)) from err
            return session.to_view()

    @app.post("/api/game/{game_id}/auto")
    async def auto_step(game_id: str, req: AutoRequest):
        async with manager_lock:
            try:
                session = manager.get(game_id)
            except KeyError as err:
                raise HTTPException(status_code=404, detail=str(err)) from err
            for _ in range(req.steps):
                if session.state.terminated or session.round_summary is not None:
                    break
                if int(session.state.current_player) == session.human_seat:
                    break
                result = session.ai_step()
                if result is None:
                    break
            return session.to_view()

    @app.post("/api/game/{game_id}/continue")
    async def continue_round(game_id: str):
        async with manager_lock:
            try:
                session = manager.get(game_id)
            except KeyError as err:
                raise HTTPException(status_code=404, detail=str(err)) from err
            session.continue_after_round()
            return session.to_view()

    @app.post("/api/game/{game_id}/visibility")
    async def update_visibility(game_id: str, req: VisibilityRequest):
        async with manager_lock:
            try:
                session = manager.get(game_id)
            except KeyError as err:
                raise HTTPException(status_code=404, detail=str(err)) from err
            if req.hide_opponent_hands is not None:
                session.set_hide_opponent_hands(req.hide_opponent_hands)
            if req.auto_pass_calls is not None:
                session.set_auto_pass_calls(req.auto_pass_calls)
            return session.to_view()

    @app.delete("/api/game/{game_id}")
    async def delete_game(game_id: str):
        async with manager_lock:
            manager.remove(game_id)
            return {"status": "ok"}

    return app


__all__ = ["create_app"]
