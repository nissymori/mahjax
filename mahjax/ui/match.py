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

"""One live game: the env, the seats, and the rule for when to stop for a human.

The server drives the game. A client sends one decision and gets back every
board position between that decision and the next one it has to make, so the
browser animates from a list it already holds instead of polling per turn.
"""

from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence

import jax
import jax.numpy as jnp

from . import mjai, view
from .agents import Agent
from .rules import Rules, rules_for
from .view import SeatInfo

NUM_PLAYERS = 4
MAX_DUMMY_STEPS = 4
#: A round that needs more steps than this is a bug, not a long hand.
MAX_STEPS_PER_ADVANCE = 4000


@dataclass
class MatchConfig:
    env_id: str = "red_mahjong"
    round_mode: str = "half"
    seed: int = 0
    human_seat: Optional[int] = 0
    human_name: str = "You"
    agent_id: Optional[str] = None
    hide_hands: bool = True
    no_calls: bool = False
    save_record: bool = True


def new_seed() -> int:
    return uuid.uuid4().int % (2**31)


def _fold(root: Any, index: int) -> Any:
    return jax.random.fold_in(root, index)


@lru_cache(maxsize=8)
def shared_env(env_id: str, round_mode: str) -> Any:
    """Envs are configuration only, so games of the same shape can share one.

    Sharing matters: ``jax.jit`` caches per callable, and a fresh env per game
    would re-trace ``step`` -- seconds of dead time at every new game.
    """
    return rules_for(env_id).make_env(round_mode)


@lru_cache(maxsize=8)
def shared_step_fn(env_id: str, round_mode: str) -> Any:
    return jax.jit(shared_env(env_id, round_mode).step)


@dataclass
class _RoundTrack:
    """Everything about the current round that the result payload needs."""

    score_start: List[int] = field(default_factory=list)
    wins: List[Dict[str, Any]] = field(default_factory=list)
    step_start: int = 0


class Match:
    """A game in progress.

    Steps are numbered from zero and each one consumes ``fold_in(root, i + 1)``,
    with ``fold_in(root, 0)`` dealing the first hand. That makes the whole game a
    function of the seed and the action list, which is what replay relies on.
    """

    def __init__(
        self,
        config: MatchConfig,
        agent: Agent,
        *,
        on_game_over: Optional[Callable[["Match"], Optional[str]]] = None,
    ) -> None:
        self.id = uuid.uuid4().hex
        self.config = config
        self.agent = agent
        self.rules: Rules = rules_for(config.env_id)
        self.env = shared_env(config.env_id, config.round_mode)
        self._step_fn = shared_step_fn(config.env_id, config.round_mode)
        self._on_game_over = on_game_over

        self._root = jax.random.PRNGKey(config.seed)
        self.step_index = 0
        self.state = jax.device_get(self.env.init(_fold(self._root, 0)))

        self.seats: List[SeatInfo] = self._build_seats()
        self.actions: List[int] = []
        self.events: List[Dict[str, Any]] = []
        self.created_at = _dt.datetime.now().astimezone().isoformat(timespec="seconds")

        self.result: Optional[Dict[str, Any]] = None
        self.game_over = False
        self.record_id: Optional[str] = None
        self._round = _RoundTrack(score_start=self.rules.scores(self.state))
        self._round_display_state: Optional[Any] = None

        self.events.append(self._start_game_event())
        self.events.append(mjai.start_kyoku_event(self.rules, self.state))
        self._warm_up()

    def _warm_up(self) -> None:
        """Compile the step function and the agent before anyone has to wait.

        Whoever moves first decides where that cost lands: when the human is
        the dealer nothing has been stepped yet, so without this the player's
        very first discard pays for the whole compile. Both calls are thrown
        away -- they exist only to fill the jit caches, which are shared by
        every later game of the same shape.
        """
        legal = self.rules.legal_actions(self.state)
        if not legal:
            return
        key = _fold(self._root, 0)
        jax.block_until_ready(self._step_fn(self.state, jnp.int32(legal[0]), key))
        jax.block_until_ready(self.agent.act(self.state, key))

    # ------------------------------------------------------------------ setup

    def _build_seats(self) -> List[SeatInfo]:
        """Number the agent seats. Three opponents all called "Rule-based" are
        impossible to tell apart, and the number leads because a long agent name
        gets clipped on the seat plate."""
        seats: List[SeatInfo] = []
        nth = 0
        for i in range(NUM_PLAYERS):
            if i == self.config.human_seat:
                seats.append(SeatInfo(name=self.config.human_name, kind="human"))
            else:
                nth += 1
                seats.append(SeatInfo(name=f"{nth}. {self.agent.name}", kind="agent"))
        return seats

    def _start_game_event(self) -> Dict[str, Any]:
        players = []
        for i in range(NUM_PLAYERS):
            if i == self.config.human_seat:
                players.append({"kind": "human"})
            else:
                players.append({"kind": "agent", "agent_id": self.agent.agent_id})
        import jax as _jax

        import mahjax

        return {
            "type": "start_game",
            "names": [s.name for s in self.seats],
            "mahjax": {
                "format": 1,
                "mahjax_version": mahjax.__version__,
                "jax": {
                    "version": _jax.__version__,
                    "threefry_partitionable": bool(
                        _jax.config.jax_threefry_partitionable
                    ),
                },
                "env_id": self.config.env_id,
                "round_mode": self.config.round_mode,
                "seed": int(self.config.seed),
                "players": players,
                "human_seat": self.config.human_seat,
                "created_at": self.created_at,
                "complete": False,
            },
        }

    # ------------------------------------------------------------ visibility

    @property
    def human_seat(self) -> Optional[int]:
        return self.config.human_seat

    def _reveal(self) -> List[bool]:
        if not self.config.hide_hands or self.result is not None or self.game_over:
            return [True] * NUM_PLAYERS
        return [i == self.config.human_seat for i in range(NUM_PLAYERS)]

    def _frame(
        self,
        state: Any,
        *,
        prompt: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return view.build_view(
            self.rules,
            state,
            self.seats,
            reveal=self._reveal(),
            prompt=prompt,
            result=result,
        )

    @property
    def view(self) -> Dict[str, Any]:
        """The current position, for a client that reloaded mid-game."""
        state = self._round_display_state if self.result is not None else self.state
        prompt = None
        if self.result is None and not self.game_over:
            prompt = self._human_prompt()
        return self._frame(state, prompt=prompt, result=self.result)

    def _human_prompt(self) -> Optional[Dict[str, Any]]:
        if self.config.human_seat is None:
            return None
        if int(self.state.current_player) != self.config.human_seat:
            return None
        return view.build_prompt(self.rules, self.state, self.config.human_seat)

    # ---------------------------------------------------------------- driving

    def _step(self, action: int) -> Any:
        """Apply one env action, recording it and the mjai events it produced."""
        pre = self.state
        key = _fold(self._root, self.step_index + 1)
        post = jax.device_get(self._step_fn(pre, jnp.int32(action), key))
        self.step_index += 1
        self.actions.append(int(action))

        win = None
        seat = int(pre.current_player)
        if action in (self.rules.RON, self.rules.TSUMO):
            is_ron = action == self.rules.RON
            win = view.build_win(self.rules, pre, post, seat, is_ron)
            self._round.wins.append(win)
        self.events.extend(
            mjai.encode_step(self.rules, pre, int(action), post, win=win)
        )
        self.state = post
        return post

    def start(self) -> List[Dict[str, Any]]:
        frames = [self._frame(self.state, prompt=self._human_prompt())]
        if frames[0].get("prompt") is not None:
            return frames
        return frames[:1] + self.advance()

    def act(self, action: int) -> List[Dict[str, Any]]:
        """Apply the human's decision, then run on to their next one."""
        if self.game_over:
            raise ValueError("Game is already over")
        if self.result is not None:
            raise ValueError("Round result is pending; call next_round()")
        if self.config.human_seat is None:
            raise ValueError("This game has no human seat")
        if int(self.state.current_player) != self.config.human_seat:
            raise ValueError("Not your turn")
        if not bool(self.state.legal_action_mask[int(action)]):
            legal = ", ".join(str(a) for a in self.rules.legal_actions(self.state))
            raise ValueError(f"Illegal action {action}. Legal: [{legal}]")
        frames: List[Dict[str, Any]] = []
        self._apply_and_frame(int(action), frames)
        return frames + self.advance()

    def next_round(self) -> List[Dict[str, Any]]:
        """Dismiss the round result and play on into the next round."""
        if self.result is None:
            return []
        if self.game_over:
            return []
        self.result = None
        self._round_display_state = None
        self._round = _RoundTrack(
            score_start=self.rules.scores(self.state), step_start=self.step_index
        )
        self.events.append(mjai.start_kyoku_event(self.rules, self.state))
        return self.advance()

    def set_options(
        self, *, hide_hands: Optional[bool] = None, no_calls: Optional[bool] = None
    ) -> None:
        if hide_hands is not None:
            self.config.hide_hands = bool(hide_hands)
        if no_calls is not None:
            self.config.no_calls = bool(no_calls)

    def _apply_and_frame(self, action: int, frames: List[Dict[str, Any]]) -> None:
        pre_visible = action not in (self.rules.PASS, self.rules.DUMMY)
        self._step(action)
        if pre_visible:
            frames.append(self._frame(self.state))

    # --------------------------------------------------------------- the loop

    def advance(self) -> List[Dict[str, Any]]:
        """Play until a human has to decide, the round ends, or the game does."""
        frames: List[Dict[str, Any]] = []
        if self.game_over or self.result is not None:
            return frames
        for _ in range(MAX_STEPS_PER_ADVANCE):
            # is_round_over() also covers a terminal state, so a single-round
            # game still gets its result overlay instead of ending silently.
            if self.rules.is_round_over(self.state):
                self._settle_round(frames)
                return frames

            legal = self.rules.legal_actions(self.state)
            if not legal:
                return frames
            seat = int(self.state.current_player)

            if self.rules.KYUUSHU is not None and legal == [self.rules.KYUUSHU]:
                self._settle_abortive(frames)
                return frames

            if seat == self.config.human_seat:
                if legal == [self.rules.TSUMOGIRI]:
                    self._apply_and_frame(self.rules.TSUMOGIRI, frames)
                    continue
                if self.config.no_calls and self._is_skippable_claim(legal):
                    self._apply_and_frame(self.rules.PASS, frames)
                    continue
                prompt = view.build_prompt(self.rules, self.state, seat)
                if prompt is None:
                    self._apply_and_frame(legal[0], frames)
                    continue
                frames.append(self._frame(self.state, prompt=prompt))
                return frames

            key = _fold(self._root, self.step_index + 1)
            action = int(jax.device_get(self.agent.act(self.state, key)))
            self._apply_and_frame(action, frames)
        raise RuntimeError("advance() exceeded its step budget; the env is looping")

    def _is_skippable_claim(self, legal: Sequence[int]) -> bool:
        """A call prompt with nothing but calls in it -- safe to auto-pass.

        Ron is never auto-passed: passing on a winning tile costs the player
        temporary furiten, and permanent furiten while in riichi.
        """
        if self.rules.PASS not in legal:
            return False
        if self.rules.RON in legal:
            return False
        return all(
            a == self.rules.PASS
            or self.rules.is_chi(a)
            or self.rules.is_pon(a)
            or a == self.rules.OPEN_KAN
            for a in legal
        )

    # ------------------------------------------------------------ round ends

    def _settle_round(self, frames: List[Dict[str, Any]]) -> None:
        """A win or an exhaustive draw: show this board, then share via DUMMY.

        The four DUMMY steps are applied straight away because the game-over flag
        only appears during them, and the result overlay has to know whether this
        was the last round. The board the client sees is the one captured here,
        before those steps reset it.
        """
        end_state = self.state
        for _ in range(MAX_DUMMY_STEPS):
            if bool(self.state.terminated):
                break
            if self.rules.legal_actions(self.state) != [self.rules.DUMMY]:
                break
            self._step(self.rules.DUMMY)
        game_over = bool(self.state.terminated)

        wins = self._round.wins
        if wins:
            result_type = "ron" if any(w["from"] is not None for w in wins) else "tsumo"
        else:
            result_type = "draw"
        self._emit_result(end_state, result_type, None, wins, game_over, frames)

    def _settle_abortive(self, frames: List[Dict[str, Any]]) -> None:
        """A forced abortive draw: one KYUUSHU action jumps straight to the next
        round, so the board to show is the one from before that action."""
        end_state = self.state
        reason = self._abortive_reason(end_state)
        assert self.rules.KYUUSHU is not None
        self._step(self.rules.KYUUSHU)
        game_over = bool(self.state.terminated)
        self._emit_result(end_state, "abortive", reason, [], game_over, frames)

    def _emit_result(
        self,
        end_state: Any,
        result_type: str,
        reason: Optional[str],
        wins: List[Dict[str, Any]],
        game_over: bool,
        frames: List[Dict[str, Any]],
    ) -> None:
        self.result = view.build_round_result(
            self.rules,
            type=result_type,
            reason=reason,
            state=end_state,
            winners=wins,
            score_start=self._round.score_start,
            game_over=game_over,
        )
        self._round_display_state = end_state
        self.events.append(mjai.end_kyoku_event())
        self.game_over = game_over
        if game_over:
            self._write_end_game(end_state)
        frames.append(self._frame(end_state, result=self.result))

    def _abortive_reason(self, state: Any) -> str:
        players = state.players
        if int(sum(bool(x) for x in players.has_won)) >= 2:
            return "triple_ron"
        if int(sum(int(x) for x in players.riichi)) == NUM_PLAYERS:
            return "four_riichi"
        if int(sum(int(x) for x in players.n_kan)) >= 4:
            return "four_kans"
        return "four_winds"

    def _write_end_game(self, end_state: Any) -> None:
        final = view.final_standings(self.rules, end_state)
        self.events.append(
            mjai.end_game_event([entry["score"] for entry in final])
        )
        self.events[0]["mahjax"]["complete"] = True
        if self.config.save_record and self._on_game_over is not None:
            self.record_id = self._on_game_over(self)

    # ------------------------------------------------------------- self-play

    def play_out(self) -> None:
        """Run an agent-only game to its end (no human seat, no frames kept)."""
        if self.config.human_seat is not None:
            raise ValueError("play_out() is for agent-only games")
        while not self.game_over:
            self.advance()
            if self.result is None:
                break
            if self.game_over:
                break
            self.next_round()

    def abandon(self) -> Optional[str]:
        """Give up on an unfinished game, saving what was played."""
        if self.game_over:
            return self.record_id
        if self.config.save_record and self._on_game_over is not None and self.actions:
            self.record_id = self._on_game_over(self)
        return self.record_id


__all__ = ["Match", "MatchConfig", "new_seed"]
