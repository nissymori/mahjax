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

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np

from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.env import _dora_array
from mahjax.no_red_mahjong.state import DORA_ARRAY, FIRST_DRAW_IDX, State
from mahjax.no_red_mahjong.tile import Tile
from mahjax.no_red_mahjong.yaku import Yaku
from mahjax._src.visualizer import _to_red_env_state
from mahjax.red_mahjong.action import Action as RedAction
from mahjax.red_mahjong.env import RedMahjong
from mahjax.red_mahjong.env import _dora_array as _red_dora_array
from mahjax.red_mahjong.state import GameConfig as RedGameConfig
from mahjax.red_mahjong.tile import Tile as RedTile
from mahjax.red_mahjong.visualization import render_round_svg
from mahjax.red_mahjong.yaku import Yaku as RedYaku

from .agents import Agent, AgentRegistry, ensure_valid_action
from .utils import tile_label, tile_labels

WIND_NAMES = ["東", "南", "西", "北"]
DORA_TILE_LOOKUP = np.array(DORA_ARRAY)
YAKU_NAMES_EN = [
    "Pinfu",
    "Pure Double Chis",
    "Twice Pure Double Chis",
    "Outside Hand",
    "Terminals in All Sets",
    "Pure Straight",
    "Mixed Triple Chis",
    "Triple Pons",
    "All Pons",
    "Three Concealed Pons",
    "Three Kans",
    "Seven Pairs",
    "All Simples",
    "Half Flush",
    "Full Flush",
    "All Terminals and Honors",
    "Little Three Dragons",
    "White Dragon",
    "Green Dragon",
    "Red Dragon",
    "Prevalent Wind",
    "Seat Wind",
    "Fully Concealed Hand",
    "Riichi",
    "Big Three Dragons",
    "Little Four Winds",
    "Big Four Winds",
    "Nine Gates",
    "Thirteen Orphans",
    "All Terminals",
    "All Honors",
    "All Green",
    "Four Concealed Pons",
    "Four Kans",
]
YAKU_NAMES_JA = [
    "平和",
    "一盃口",
    "二盃口",
    "混全帯么九",
    "純全帯么九",
    "一気通貫",
    "三色同順",
    "三色同刻",
    "対々和",
    "三暗刻",
    "三槓子",
    "七対子",
    "断么九",
    "混一色",
    "清一色",
    "混老頭",
    "小三元",
    "白",
    "發",
    "中",
    "場風",
    "自風",
    "門前清自摸和",
    "立直",
    "大三元",
    "小四喜",
    "大四喜",
    "九蓮宝燈",
    "国士無双",
    "清老頭",
    "字一色",
    "緑一色",
    "四暗刻",
    "四槓子",
]
RED_YAKU_NAMES_EN = [
    "Fully Concealed Hand",
    "Riichi",
    "Ippatsu",
    "Robbing a Kan",
    "Rinshan Kaihou",
    "Haitei Raoyue",
    "Houtei Raoyui",
    "Pinfu",
    "All Simples",
    "Pure Double Chis",
    "Seat Wind East",
    "Seat Wind South",
    "Seat Wind West",
    "Seat Wind North",
    "Prevalent Wind East",
    "Prevalent Wind South",
    "Prevalent Wind West",
    "Prevalent Wind North",
    "White Dragon",
    "Green Dragon",
    "Red Dragon",
    "Double Riichi",
    "Seven Pairs",
    "Outside Hand",
    "Pure Straight",
    "Mixed Triple Chis",
    "Triple Pons",
    "Three Kans",
    "All Pons",
    "Three Concealed Pons",
    "Little Three Dragons",
    "All Terminals and Honors",
    "Twice Pure Double Chis",
    "Terminals in All Sets",
    "Half Flush",
    "Full Flush",
    "Renhou",
    "Heavenly Hand",
    "Earthly Hand",
    "Big Three Dragons",
    "Four Concealed Pons",
    "Four Concealed Pons Single Wait",
    "All Honors",
    "All Green",
    "All Terminals",
    "Nine Gates",
    "Pure Nine Gates",
    "Thirteen Orphans",
    "Thirteen Orphans 13-Wait",
    "Big Four Winds",
    "Little Four Winds",
    "Four Kans",
]
RED_YAKU_NAMES_JA = [
    "門前清自摸和",
    "立直",
    "一発",
    "槍槓",
    "嶺上開花",
    "海底摸月",
    "河底撈魚",
    "平和",
    "断么九",
    "一盃口",
    "自風 東",
    "自風 南",
    "自風 西",
    "自風 北",
    "場風 東",
    "場風 南",
    "場風 西",
    "場風 北",
    "白",
    "發",
    "中",
    "ダブル立直",
    "七対子",
    "混全帯么九",
    "一気通貫",
    "三色同順",
    "三色同刻",
    "三槓子",
    "対々和",
    "三暗刻",
    "小三元",
    "混老頭",
    "二盃口",
    "純全帯么九",
    "混一色",
    "清一色",
    "人和",
    "天和",
    "地和",
    "大三元",
    "四暗刻",
    "四暗刻単騎",
    "字一色",
    "緑一色",
    "清老頭",
    "九蓮宝燈",
    "純正九蓮宝燈",
    "国士無双",
    "国士無双十三面待ち",
    "大四喜",
    "小四喜",
    "四槓子",
]


class ExtraYakuDefinition(NamedTuple):
    english: str
    japanese: str
    attr: str


EXTRA_RON_YAKU: List[ExtraYakuDefinition] = [
    ExtraYakuDefinition("Ippatsu", "一発", "_ippatsu"),
    ExtraYakuDefinition("Double Riichi", "ダブル立直", "_double_riichi"),
    ExtraYakuDefinition("Robbing a Kan", "槍槓", "_kan_declared"),
    ExtraYakuDefinition("Houtei Raoyui", "河底撈魚", "_is_haitei"),
]
EXTRA_TSUMO_YAKU: List[ExtraYakuDefinition] = [
    ExtraYakuDefinition("Rinshan Kaihou", "嶺上開花", "_can_after_kan"),
    ExtraYakuDefinition("Ippatsu", "一発", "_ippatsu"),
    ExtraYakuDefinition("Double Riichi", "ダブル立直", "_double_riichi"),
    ExtraYakuDefinition("Haitei Raoyue", "海底摸月", "_is_haitei"),
]


@dataclass
class ActionEvent:
    step: int
    player: int
    actor: str
    action: int
    description: str
    timestamp: float = field(default_factory=lambda: time.time())

    def to_dict(self, player_names: List[str]) -> Dict[str, Any]:
        return {
            "step": self.step,
            "player": self.player,
            "playerName": player_names[self.player],
            "actor": self.actor,
            "action": self.action,
            "description": self.description,
            "timestamp": self.timestamp,
        }


@dataclass
class WinnerSummary:
    player: int
    name: str
    points_delta: int
    fan: int
    fu: int
    yaku: List[str]
    yaku_japanese: List[str]
    dora_count: int
    ura_dora_count: int
    dora_tiles: List[int]
    ura_dora_tiles: List[int]
    is_riichi: bool
    yakuman: int
    winning_tile: Optional[int]
    from_player: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player": self.player,
            "name": self.name,
            "pointsDelta": self.points_delta,
            "fan": self.fan,
            "fu": self.fu,
            "yakuman": self.yakuman,
            "yaku": self.yaku,
            "yakuEnglish": self.yaku,
            "yakuJapanese": self.yaku_japanese,
            "yakuLocalized": {
                "en": self.yaku,
                "ja": self.yaku_japanese,
            },
            "dora": self.dora_count,
            "uraDora": self.ura_dora_count,
            "doraTiles": self.dora_tiles,
            "doraTileLabels": tile_labels(self.dora_tiles),
            "uraDoraTiles": self.ura_dora_tiles,
            "uraDoraTileLabels": tile_labels(self.ura_dora_tiles),
            "isRiichi": self.is_riichi,
            "winningTile": self.winning_tile,
            "winningTileLabel": (
                tile_label(self.winning_tile) if self.winning_tile is not None else None
            ),
            "fromPlayer": self.from_player,
        }


@dataclass
class RoundSummary:
    reason: str
    rewards: List[int]
    winners: List[WinnerSummary]
    round_count: int
    honba: int
    kyotaku: int
    is_game_end: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reason": self.reason,
            "rewards": self.rewards,
            "winners": [w.to_dict() for w in self.winners],
            "round": self.round_count,
            "honba": self.honba,
            "kyotaku": self.kyotaku,
            "isGameEnd": self.is_game_end,
        }


class GameSession:
    def __init__(
        self,
        *,
        env_id: str,
        env: Any,
        state: Any,
        rng: jnp.ndarray,
        agent: Agent,
        human_seat: int,
        player_names: List[str],
        ai_delay_ms: int = 1000,
        hide_opponent_hands: bool = False,
        auto_pass_calls: bool = False,
    ) -> None:
        self.id = uuid.uuid4().hex
        self.env_id = env_id
        self.is_red = env_id == "red_mahjong"
        # The action space (and a few action enum values) differs between red
        # and no-red. Use env_id to distinguish, not state attribute presence
        # — both state classes now expose ``players``/``round_state``.
        self._uses_red_state = self.is_red
        self.env = env
        self.state = jax.device_get(state)
        self.rng = rng
        self.agent = agent
        self.human_seat = int(human_seat)
        self.player_names = player_names
        self.ai_delay_ms = ai_delay_ms
        self.hide_opponent_hands = hide_opponent_hands
        self.auto_pass_calls = auto_pass_calls
        self._reveal_hidden_hands = False
        self._step_fn = jax.jit(self.env.step)
        self.step_counter = 0
        self.events: List[ActionEvent] = []
        self._pending_events: List[ActionEvent] = []
        self.round_summary: Optional[RoundSummary] = None
        self.round_history: List[RoundSummary] = []
        self.last_action: Optional[ActionEvent] = None
        self._auto_pass_lock = False

    # -------------------- Core stepping --------------------
    def apply_action(self, action: int, actor: str) -> ActionEvent:
        event = self._apply_action(action, actor)
        self._maybe_auto_pass_calls()
        return event

    def _apply_action(self, action: int, actor: str) -> ActionEvent:
        mask = self.state.legal_action_mask
        ensure_valid_action(action, mask)
        prev_state = self.state
        next_state = self._step_fn(prev_state, jnp.int32(action))
        next_state = jax.device_get(next_state)
        self.state = next_state
        self.step_counter += 1
        description = (
            describe_action_red(action, prev_state)
            if self._uses_red_state
            else describe_action(action, prev_state)
        )
        event = ActionEvent(
            step=self.step_counter,
            player=int(prev_state.current_player),
            actor=actor,
            action=int(action),
            description=description,
        )
        self.events.append(event)
        self._pending_events.append(event)
        self.last_action = event
        self._maybe_reveal_after_win(action, int(prev_state.current_player))
        if self._is_terminated_round(next_state) and not self._is_terminated_round(
            prev_state
        ):
            if self._uses_red_state:
                summary = build_round_summary_red(
                    prev_state, next_state, event, self.player_names
                )
            else:
                summary = build_round_summary(
                    prev_state, next_state, event, self.player_names
                )
            self.round_summary = summary
            self.round_history.append(summary)
        return event

    def ai_step(self) -> Optional[ActionEvent]:
        if self.state.terminated:
            return None
        if self._is_terminated_round(self.state):
            return None
        if int(self.state.current_player) == self.human_seat:
            return None
        self.rng, subkey = jax.random.split(self.rng)
        action = self.agent.act(self.state, subkey)
        action_int = int(jax.device_get(action))
        return self.apply_action(action_int, actor="ai")

    def auto_play_until_interrupt(self, max_steps: int = 200) -> None:
        steps = 0
        while steps < max_steps:
            if self.state.terminated or self.round_summary is not None:
                break
            if int(self.state.current_player) == self.human_seat:
                break
            if not bool(self.state.legal_action_mask.any()):
                break
            event = self.ai_step()
            if event is None:
                break
            steps += 1
        if steps >= max_steps:
            raise RuntimeError("Exceeded auto-play step limit; possible loop detected")

    def continue_after_round(self) -> None:
        if self.round_summary is None:
            return
        self.round_summary = None
        self._reveal_hidden_hands = False
        steps = 0
        while steps < 8 and self._is_terminated_round(self.state):
            mask = self.state.legal_action_mask
            dummy_action = RedAction.DUMMY if self._uses_red_state else Action.DUMMY
            if not bool(mask[dummy_action]):
                break
            self.apply_action(dummy_action, actor="system")
            steps += 1
        self.auto_play_until_interrupt()

    def _maybe_reveal_after_win(self, action: int, actor: int) -> None:
        if not self.hide_opponent_hands:
            return
        if self._reveal_hidden_hands:
            return
        if actor == self.human_seat:
            return
        ron_action = RedAction.RON if self._uses_red_state else Action.RON
        tsumo_action = RedAction.TSUMO if self._uses_red_state else Action.TSUMO
        if action in (ron_action, tsumo_action):
            self._reveal_hidden_hands = True

    def _should_reveal_hidden_hands(self) -> bool:
        if not self.hide_opponent_hands:
            return True
        if self._reveal_hidden_hands:
            return True
        if self.round_summary:
            for winner in self.round_summary.winners:
                if winner.player != self.human_seat:
                    return True
        return False

    def set_hide_opponent_hands(self, hide: bool) -> None:
        """Toggle whether opponent hands are hidden in subsequent views."""
        self.hide_opponent_hands = hide
        if hide:
            # Re-apply masking immediately until a new reveal condition is met.
            self._reveal_hidden_hands = False

    def set_auto_pass_calls(self, enabled: bool) -> None:
        """Toggle auto-pass behavior for Pon/Chi/Open Kan prompts."""
        self.auto_pass_calls = enabled
        self._maybe_auto_pass_calls()

    def _maybe_auto_pass_calls(self) -> None:
        if (
            not self.auto_pass_calls
            or self.state.terminated
            or self._is_terminated_round(self.state)
        ):
            return
        if self._auto_pass_lock:
            return
        try:
            self._auto_pass_lock = True
            while self._should_auto_pass_current():
                pass_action = RedAction.PASS if self._uses_red_state else Action.PASS
                self._apply_action(pass_action, actor="auto_pass_call")
                if self.state.terminated or self._is_terminated_round(self.state):
                    break
        finally:
            self._auto_pass_lock = False

    def _should_auto_pass_current(self) -> bool:
        if self._uses_red_state:
            return False
        if (
            not self.auto_pass_calls
            or self.state.terminated
            or bool(self.state.round_state.terminated_round)
        ):
            return False
        if int(self.state.current_player) != self.human_seat:
            return False
        target = int(self.state.round_state.target)
        if target < 0:
            return False
        current = int(self.state.current_player)
        mask_source = self.state.players.legal_action_mask[
            current
        ]  # includes pending meld prompts
        mask = np.array(mask_source).astype(bool)
        if Action.RON < mask.size and mask[Action.RON]:
            return False
        if Action.TSUMO < mask.size and mask[Action.TSUMO]:
            return False
        call_actions = [
            Action.PON,
            Action.OPEN_KAN,
            Action.CHI_L,
            Action.CHI_M,
            Action.CHI_R,
        ]
        has_call = any(action < mask.size and mask[action] for action in call_actions)
        if not has_call:
            return False
        return bool(Action.PASS < mask.size and mask[Action.PASS])

    # -------------------- View helpers --------------------
    def consume_events(self) -> List[Dict[str, Any]]:
        events = [e.to_dict(self.player_names) for e in self._pending_events]
        self._pending_events.clear()
        return events

    def phase(self) -> str:
        if self.state.terminated:
            return "finished"
        if self.round_summary is not None:
            return "round_end"
        if int(self.state.current_player) == self.human_seat:
            return "awaiting_human"
        return "awaiting_ai"

    def to_view(self) -> Dict[str, Any]:
        if self._uses_red_state:
            return self._to_view_red()
        state = self.state
        scores = [int(s) * 100 for s in np.array(state.round_state.score)]
        rewards = [int(np.round(r * 100)) for r in np.array(state.rewards)]
        winds = [WIND_NAMES[int(w)] for w in np.array(state.round_state.seat_wind)]
        rank_order = [int(i) for i in np.argsort([-s for s in scores])]
        # Keep player (human) at the bottom by rotating seat-indexed arrays.
        oriented = orient_state_for_player(state, self.human_seat)
        red_state = _to_red_env_state(oriented)
        reveal_all_hands = (not self.hide_opponent_hands) or self._should_reveal_hidden_hands()
        # Export both language variants so the frontend can switch without
        # requesting a new state from the server.
        svg_japanese = render_round_svg(
            red_state,
            show_all_hands=reveal_all_hands,
            visible_player=0,
            language="ja",
        )
        svg_english = render_round_svg(
            red_state,
            show_all_hands=reveal_all_hands,
            visible_player=0,
            language="en",
        )
        legal_view = None
        advance_view: Optional[Dict[str, Any]] = None
        is_human_turn = int(state.current_player) == self.human_seat
        if is_human_turn:
            mask_np = np.array(state.legal_action_mask).astype(bool)
            advance_view = build_advance_info_from_mask(state, mask_np)
        # ラウンドサマリが保留中の場合は、手番に関わらず UI に「次の局へ/終局」を提示し、
        # 数値アクション送信ではなくサマリ表示に誘導する（/continue で進行）
        if self.round_summary is not None:
            is_final = is_game_end_pending(state)
            advance_view = {
                "enabled": True,
                # 人間の手番ではないので、数値アクションは送らずに UI 側でサマリを表示する
                "action": None,
                "label": "終局" if is_final else "次の局へ",
                "isFinal": is_final,
                "dummyOnly": True,
            }
        if is_human_turn and not state.terminated and self.round_summary is None:
            legal_view = build_legal_actions_view(state, self.human_seat)
        hand_view = build_hand_view(state, self.human_seat)
        return {
            "gameId": self.id,
            "envId": self.env_id,
            "phase": self.phase(),
            "currentPlayer": int(state.current_player),
            "humanSeat": self.human_seat,
            "playerNames": self.player_names,
            "winds": winds,
            "scores": scores,
            "rewards": rewards,
            "rankOrder": rank_order,
            "svg": svg_japanese,
            "svgJapanese": svg_japanese,
            "svgEnglish": svg_english,
            "legalActions": legal_view,
            "advanceAction": advance_view,
            "hand": hand_view,
            "roundSummary": (
                self.round_summary.to_dict() if self.round_summary else None
            ),
            "events": self.consume_events(),
            "terminated": bool(state.terminated),
            "aiDelayMs": self.ai_delay_ms,
            "step": self.step_counter,
            "hideOpponentHands": self.hide_opponent_hands,
            "autoPassCalls": self.auto_pass_calls,
        }

    def _to_view_red(self) -> Dict[str, Any]:
        state = self.state
        score_raw = np.array(state.round_state.score)
        scores = [int(s) * 100 for s in score_raw]
        rewards = [int(np.round(r * 100)) for r in np.array(state.rewards)]
        winds = [WIND_NAMES[int(w)] for w in np.array(state.round_state.seat_wind)]
        rank_order = [int(i) for i in np.argsort([-s for s in scores])]
        oriented = orient_red_state_for_player(state, self.human_seat)
        reveal_all_hands = (not self.hide_opponent_hands) or self._should_reveal_hidden_hands()
        svg_ja = render_round_svg(
            oriented,
            show_all_hands=reveal_all_hands,
            visible_player=0,
            language="ja",
        )
        svg_en = render_round_svg(
            oriented,
            show_all_hands=reveal_all_hands,
            visible_player=0,
            language="en",
        )
        legal_view = None
        advance_view: Optional[Dict[str, Any]] = None
        is_human_turn = int(state.current_player) == self.human_seat
        if is_human_turn:
            mask_np = np.array(state.legal_action_mask).astype(bool)
            if bool(mask_np[RedAction.DUMMY]) and int(mask_np.sum()) == 1:
                advance_view = {
                    "enabled": True,
                    "action": RedAction.DUMMY,
                    "label": "次の局へ",
                    "isFinal": bool(state.terminated),
                    "dummyOnly": True,
                }
        if self.round_summary is not None:
            advance_view = {
                "enabled": True,
                "action": None,
                "label": "終局" if bool(state.terminated) else "次の局へ",
                "isFinal": bool(state.terminated),
                "dummyOnly": True,
            }
        if is_human_turn and not state.terminated and self.round_summary is None:
            legal_view = build_legal_actions_view_red(state, self.human_seat)
        hand_view = build_hand_view_red(state, self.human_seat)
        return {
            "gameId": self.id,
            "envId": self.env_id,
            "phase": self.phase(),
            "currentPlayer": int(state.current_player),
            "humanSeat": self.human_seat,
            "playerNames": self.player_names,
            "winds": winds,
            "scores": scores,
            "rewards": rewards,
            "rankOrder": rank_order,
            "svg": svg_ja,
            "svgJapanese": svg_ja,
            "svgEnglish": svg_en,
            "legalActions": legal_view,
            "advanceAction": advance_view,
            "hand": hand_view,
            "roundSummary": (
                self.round_summary.to_dict() if self.round_summary else None
            ),
            "events": self.consume_events(),
            "terminated": bool(state.terminated),
            "aiDelayMs": self.ai_delay_ms,
            "step": self.step_counter,
            "hideOpponentHands": self.hide_opponent_hands,
            "autoPassCalls": self.auto_pass_calls,
        }

    def _is_terminated_round(self, state: Any) -> bool:
        return bool(state.round_state.terminated_round)


class GameManager:
    def __init__(self) -> None:
        self.registry = AgentRegistry()
        self.sessions: Dict[str, GameSession] = {}

    def create_session(
        self,
        *,
        env_id: str,
        agent_id: str,
        human_seat: int,
        round_mode: str,
        seed: int,
        player_names: Optional[List[str]] = None,
        ai_delay_ms: int = 1000,
        hide_opponent_hands: bool = False,
        auto_pass_calls: bool = False,
    ) -> GameSession:
        agent = self.registry.get(agent_id)
        if env_id in ("red_mahjong", "no_red_mahjong") and agent.agent_id == "rule_based":
            agent = self.registry.get("rule_based_red")
        if env_id == "red_mahjong":
            env = RedMahjong(round_mode=round_mode)
        else:
            env = RedMahjong(
                round_mode=round_mode,
                game_config=RedGameConfig(use_red_fives=jnp.bool_(False)),
            )
        rng = jax.random.PRNGKey(seed)
        rng, init_key = jax.random.split(rng)
        state = env.init(init_key)
        names = player_names or [f"Player {i+1}" for i in range(4)]
        session = GameSession(
            env_id=env_id,
            env=env,
            state=state,
            rng=rng,
            agent=agent,
            human_seat=human_seat,
            player_names=names,
            ai_delay_ms=ai_delay_ms,
            hide_opponent_hands=hide_opponent_hands,
            auto_pass_calls=auto_pass_calls,
        )
        session.auto_play_until_interrupt()
        self.sessions[session.id] = session
        return session

    def get(self, session_id: str) -> GameSession:
        if session_id not in self.sessions:
            raise KeyError(f"Unknown game id: {session_id}")
        return self.sessions[session_id]

    def remove(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)


# ------------------- Helper functions -------------------


def orient_state_for_player(state: State, player: int) -> State:
    """Return a view of ``state`` where ``player`` becomes seat 0.

    This keeps the underlying state immutable while rotating all player-indexed
    arrays used for SVG rendering so the board is shown from the requested
    perspective.
    """

    seat = int(player) % 4
    if seat == 0:
        return state

    order = jnp.array([(seat + i) % 4 for i in range(4)], dtype=jnp.int32)

    def _reorder(arr):
        return jnp.take(arr, order, axis=0)

    return state.replace(  # type: ignore[arg-type]
        current_player=jnp.int8((int(state.current_player) - seat) % 4),
        players=state.players.replace(
            hand=_reorder(state.players.hand),
            melds=_reorder(state.players.melds),
            riichi=_reorder(state.players.riichi),
            river=_reorder(state.players.river),
        ),
        round_state=state.round_state.replace(
            score=_reorder(state.round_state.score),
            dealer=jnp.int8((int(state.round_state.dealer) - seat) % 4),
        ),
    )


def mask_opponent_hands(state: State, visible_player: int = 0) -> State:
    """Return a copy of ``state`` where opponent tiles are marked as hidden.

    Tiles are negated so downstream SVG rendering can display tile backs while
    preserving the total count.
    """
    hands = jnp.asarray(state.players.hand)
    seat_mask = (jnp.arange(hands.shape[0]) != int(visible_player))[:, None]
    masked_hands = jnp.where(seat_mask, -hands, hands)
    return state.replace(players=state.players.replace(hand=masked_hands))  # type: ignore[arg-type]


def orient_red_state_for_player(state: Any, player: int) -> Any:
    """Return a red EnvState view where `player` becomes seat 0."""
    seat = int(player) % 4
    if seat == 0:
        return state

    order = jnp.array([(seat + i) % 4 for i in range(4)], dtype=jnp.int32)

    def _reorder(arr):
        return jnp.take(arr, order, axis=0)

    return state.replace(
        current_player=jnp.int8((int(state.current_player) - seat) % 4),
        players=state.players.replace(
            hand=_reorder(state.players.hand),
            hand_with_red=_reorder(state.players.hand_with_red),
            hand_ids=_reorder(state.players.hand_ids),
            hand_counts=_reorder(state.players.hand_counts),
            drawn_tile=_reorder(state.players.drawn_tile),
            legal_action_mask=_reorder(state.players.legal_action_mask),
            can_win=_reorder(state.players.can_win),
            has_yaku=_reorder(state.players.has_yaku),
            fan=_reorder(state.players.fan),
            fu=_reorder(state.players.fu),
            melds=_reorder(state.players.melds),
            meld_tiles=_reorder(state.players.meld_tiles),
            meld_info=_reorder(state.players.meld_info),
            meld_counts=_reorder(state.players.meld_counts),
            river=_reorder(state.players.river),
            discards=_reorder(state.players.discards),
            discard_info=_reorder(state.players.discard_info),
            discard_counts=_reorder(state.players.discard_counts),
            riichi=_reorder(state.players.riichi),
            riichi_declared=_reorder(state.players.riichi_declared),
            riichi_step=_reorder(state.players.riichi_step),
            double_riichi=_reorder(state.players.double_riichi),
            ippatsu=_reorder(state.players.ippatsu),
            furiten_by_discard=_reorder(state.players.furiten_by_discard),
            furiten_by_pass=_reorder(state.players.furiten_by_pass),
            is_hand_concealed=_reorder(state.players.is_hand_concealed),
            pon=_reorder(state.players.pon),
            has_won=_reorder(state.players.has_won),
            n_kan=_reorder(state.players.n_kan),
            has_nagashi_mangan=_reorder(state.players.has_nagashi_mangan),
        ),
        round_state=state.round_state.replace(
            seat_wind=_reorder(state.round_state.seat_wind),
            init_wind=_reorder(state.round_state.init_wind),
            score=_reorder(state.round_state.score),
            dealer=jnp.int8((int(state.round_state.dealer) - seat) % 4),
            shanten_current_player=jnp.int8(
                (int(state.round_state.shanten_current_player) - seat) % 4
            ),
            last_player=jnp.int8(
                jnp.where(
                    state.round_state.last_player < 0,
                    state.round_state.last_player,
                    (state.round_state.last_player - seat) % 4,
                )
            ),
        ),
    )


def describe_action(action: int, prev: State) -> str:
    if 0 <= action < Tile.NUM_TILE_TYPE:
        return f"打 {tile_label(action)}"
    if 34 <= action < 34 + Tile.NUM_TILE_TYPE:
        tile = action - Tile.NUM_TILE_TYPE
        return f"カン {tile_label(tile)}"
    if action == Action.TSUMOGIRI:
        return "ツモ切り"
    if action == Action.RIICHI:
        return "立直宣言"
    if action == Action.TSUMO:
        return "自摸"
    if action == Action.RON:
        target = int(prev.round_state.target)
        tile = tile_label(target) if target >= 0 else ""
        from_player = int(prev.round_state.last_player)
        return f"ロン {tile} ← {from_player}"
    if action == Action.PON:
        return f"ポン {tile_label(int(prev.round_state.target))}"
    if action == Action.OPEN_KAN:
        return f"明槓 {tile_label(int(prev.round_state.target))}"
    if action in (Action.CHI_L, Action.CHI_M, Action.CHI_R):
        target = int(prev.round_state.target)
        chi_tiles = compute_chi_tiles(action, target)
        return "チー " + "".join(tile_labels(chi_tiles))
    if action == Action.PASS:
        return "パス"
    if action == Action.DUMMY:
        return "進行"
    return f"アクション {action}"


def compute_chi_tiles(action: int, target: int) -> List[int]:
    if target < 0:
        return []
    if action == Action.CHI_L:
        return [target, target + 1, target + 2]
    if action == Action.CHI_M:
        return [target - 1, target, target + 1]
    if action == Action.CHI_R:
        return [target - 2, target - 1, target]
    return []


def describe_action_red(action: int, prev: Any) -> str:
    if 0 <= action < RedTile.NUM_TILE_TYPE_WITH_RED:
        return f"打 {tile_label(action)}"
    if action == RedAction.TSUMOGIRI:
        return "ツモ切り"
    if action == RedAction.RIICHI:
        return "立直宣言"
    if action == RedAction.TSUMO:
        return "自摸"
    if action == RedAction.RON:
        return "ロン"
    if action == RedAction.PASS:
        return "パス"
    if action == RedAction.DUMMY:
        return "進行"
    return f"アクション {action}"


def build_hand_view_red(state: Any, player: int) -> Dict[str, Any]:
    hand_counts = np.array(state.players.hand_with_red[player])
    sequence: List[int] = []
    for tile, entry in enumerate(hand_counts):
        sequence.extend([tile] * int(entry))
    sequence.sort()
    base_sequence: List[int] = list(sequence)
    draw_tile: Optional[int] = None
    is_current = player == int(state.current_player)
    last_draw = int(state.round_state.last_draw) if is_current else -1
    should_separate = False
    legal_mask = np.array(state.legal_action_mask).astype(bool)
    in_draw_phase = (
        is_current
        and 0 <= int(RedAction.TSUMOGIRI) < int(legal_mask.size)
        and bool(legal_mask[int(RedAction.TSUMOGIRI)])
    )
    if in_draw_phase and last_draw >= 0:
        last_draw_idx: Optional[int] = None
        for idx in range(len(base_sequence) - 1, -1, -1):
            if base_sequence[idx] == last_draw:
                last_draw_idx = idx
                break
        if last_draw_idx is not None:
            draw_tile = base_sequence.pop(last_draw_idx)
            should_separate = True
    return {
        "tiles": [],
        "sequence": base_sequence,
        "lastDraw": last_draw,
        "drawTile": draw_tile,
        "separateLastDraw": should_separate,
    }


def build_legal_actions_view_red(state: Any, player: int) -> Dict[str, Any]:
    mask = np.array(state.legal_action_mask).astype(bool)
    def enabled(action: int) -> bool:
        return 0 <= int(action) < mask.size and bool(mask[int(action)])

    hand_counts = np.array(state.players.hand_with_red[player])
    discard_tiles = []
    for tile, count in enumerate(hand_counts):
        can_discard = bool(mask[tile]) if tile < mask.size else False
        if count <= 0 and not can_discard:
            continue
        discard_tiles.append(
            {
                "tile": tile,
                "label": tile_label(tile),
                "count": int(count),
                "action": tile,
                "enabled": can_discard,
            }
        )
    target = int(state.round_state.target)
    target_type = int(RedTile.to_tile_type(target)) if target >= 0 else -1

    kan_actions = []
    # Closed/Added Kan actions in red rules are contiguous (37..70) for 34 tile-types.
    for tile in range(RedTile.NUM_TILE_TYPE):
        action = RedTile.NUM_TILE_TYPE_WITH_RED + tile
        if action >= mask.size:
            break
        if enabled(action):
            kind = "暗槓"
            if 0 <= target_type < RedTile.NUM_TILE_TYPE:
                target_matches = (
                    tile == target_type
                    or (
                        int(RedTile.is_tile_type_five(tile))
                        and target_type == int(RedTile.to_tile_type(tile))
                    )
                )
                if target_matches:
                    kind = "加槓"
            kan_actions.append(
                {
                    "tile": tile,
                    "label": tile_label(tile),
                    "action": action,
                    "kind": kind,
                }
            )

    call_options: Dict[str, Any] = {}
    if target >= 0 and enabled(RedAction.OPEN_KAN):
        call_options["open_kan"] = {
            "action": RedAction.OPEN_KAN,
            "tiles": [target] * 4,
            "labels": tile_labels([target] * 4),
        }

    if target >= 0 and (enabled(RedAction.PON) or enabled(RedAction.PON_RED)):
        pon_action = RedAction.PON_RED if enabled(RedAction.PON_RED) else RedAction.PON
        call_options["pon"] = {
            "action": pon_action,
            "tiles": [target] * 3,
            "labels": tile_labels([target] * 3),
        }

    chi_list = []
    chi_actions = (
        RedAction.CHI_L,
        RedAction.CHI_L_RED,
        RedAction.CHI_M,
        RedAction.CHI_M_RED,
        RedAction.CHI_R,
        RedAction.CHI_R_RED,
    )
    for action in chi_actions:
        if not enabled(action) or target < 0:
            continue
        base = int(RedTile.to_tile_type(target))
        if action in (RedAction.CHI_L, RedAction.CHI_L_RED):
            tiles = [base, base + 1, base + 2]
        elif action in (RedAction.CHI_M, RedAction.CHI_M_RED):
            tiles = [base - 1, base, base + 1]
        else:
            tiles = [base - 2, base - 1, base]
        # Red chi variants include a red five in the sequence.
        if action in (RedAction.CHI_L_RED, RedAction.CHI_M_RED, RedAction.CHI_R_RED):
            tiles = [int(RedTile.to_red(t)) if int(RedTile.is_tile_type_five(t)) else t for t in tiles]
        chi_list.append(
            {
                "action": action,
                "tiles": tiles,
                "labels": tile_labels(tiles),
            }
        )
    if chi_list:
        call_options["chi"] = chi_list

    return {
        "discardTiles": discard_tiles,
        "tsumogiri": {"enabled": enabled(RedAction.TSUMOGIRI), "action": RedAction.TSUMOGIRI},
        "riichi": {"enabled": enabled(RedAction.RIICHI), "action": RedAction.RIICHI},
        "tsumo": {"enabled": enabled(RedAction.TSUMO), "action": RedAction.TSUMO},
        "ron": {
            "enabled": enabled(RedAction.RON),
            "action": RedAction.RON,
            "target": target if target >= 0 else None,
            "targetLabel": tile_label(target) if target >= 0 else None,
        },
        "kan": kan_actions,
        "call": call_options,
        "pass": {"enabled": enabled(RedAction.PASS), "action": RedAction.PASS},
        "dummyOnly": False,
        "advance": None,
    }


def build_hand_view(state: State, player: int) -> Dict[str, Any]:
    hand_counts = np.array(state.players.hand[player])
    tiles = []
    for tile, count in enumerate(hand_counts):
        if count <= 0:
            continue
        tiles.append(
            {
                "tile": tile,
                "label": tile_label(tile),
                "count": int(count),
            }
        )
    sequence: List[int] = []
    for tile, entry in enumerate(hand_counts):
        sequence.extend([tile] * int(entry))
    sequence.sort()

    base_sequence: List[int] = list(sequence)
    draw_tile: Optional[int] = None
    is_current = player == int(state.current_player)
    last_draw = int(state.round_state.last_draw) if is_current else -1
    should_separate = False
    if is_current and last_draw >= 0 and len(base_sequence) % 3 == 2:
        last_draw_idx: Optional[int] = None
        for idx in range(len(base_sequence) - 1, -1, -1):
            if base_sequence[idx] == last_draw:
                last_draw_idx = idx
                break
        if last_draw_idx is not None:
            draw_tile = base_sequence.pop(last_draw_idx)
            should_separate = True
    return {
        "tiles": tiles,
        "sequence": base_sequence,
        "lastDraw": last_draw,
        "drawTile": draw_tile,
        "separateLastDraw": should_separate,
    }


def build_legal_actions_view(state: State, player: int) -> Dict[str, Any]:
    mask = np.array(state.legal_action_mask).astype(bool)
    hand_counts = np.array(state.players.hand[player])
    discard_tiles = []
    for tile, count in enumerate(hand_counts):
        can_discard = bool(mask[tile])
        if count <= 0 and not can_discard:
            continue
        discard_tiles.append(
            {
                "tile": tile,
                "label": tile_label(tile),
                "count": int(count),
                "action": tile,
                "enabled": can_discard,
            }
        )
    tsumogiri_available = bool(mask[Action.TSUMOGIRI])
    kan_actions = []
    for tile in range(Tile.NUM_TILE_TYPE):
        action = Tile.NUM_TILE_TYPE + tile
        if action >= mask.size:
            break
        if bool(mask[action]):
            pon_info = int(state.players.pon[(player, tile)])
            kind = "加槓" if pon_info > 0 else "暗槓"
            kan_actions.append(
                {
                    "tile": tile,
                    "label": tile_label(tile),
                    "action": action,
                    "kind": kind,
                }
            )
    target = int(state.round_state.target)
    call_options: Dict[str, Any] = {}
    if bool(mask[Action.PON]) and target >= 0:
        call_options["pon"] = {
            "action": Action.PON,
            "tiles": [target] * 3,
            "labels": tile_labels([target] * 3),
        }
    if bool(mask[Action.OPEN_KAN]) and target >= 0:
        call_options["open_kan"] = {
            "action": Action.OPEN_KAN,
            "tiles": [target] * 4,
            "labels": tile_labels([target] * 4),
        }
    chi_list = []
    for action in (Action.CHI_L, Action.CHI_M, Action.CHI_R):
        if bool(mask[action]) and target >= 0:
            tiles = compute_chi_tiles(action, target)
            chi_list.append(
                {
                    "action": action,
                    "tiles": tiles,
                    "labels": tile_labels(tiles),
                }
            )
    if chi_list:
        call_options["chi"] = chi_list
    advance = build_advance_info_from_mask(state, mask)
    return {
        "discardTiles": discard_tiles,
        "tsumogiri": {"enabled": tsumogiri_available, "action": Action.TSUMOGIRI},
        "riichi": {
            "enabled": bool(mask[Action.RIICHI]),
            "action": Action.RIICHI,
        },
        "tsumo": {
            "enabled": bool(mask[Action.TSUMO]),
            "action": Action.TSUMO,
        },
        "ron": {
            "enabled": bool(mask[Action.RON]),
            "action": Action.RON,
            "target": target if target >= 0 else None,
            "targetLabel": tile_label(target) if target >= 0 else None,
        },
        "kan": kan_actions,
        "call": call_options,
        "pass": {
            "enabled": bool(mask[Action.PASS]),
            "action": Action.PASS,
        },
        "dummyOnly": bool(advance),
        "advance": advance,
    }


def build_advance_info_from_mask(
    state: State, mask: np.ndarray
) -> Optional[Dict[str, Any]]:
    if mask.size <= Action.DUMMY or not bool(mask[Action.DUMMY]):
        return None
    if int(mask.sum()) != 1:
        return None
    is_final = is_game_end_pending(state)
    return {
        "enabled": True,
        "action": Action.DUMMY,
        "label": "終局" if is_final else "次の局へ",
        "isFinal": is_final,
        "dummyOnly": True,
    }


def build_round_summary(
    prev_state: State,
    next_state: State,
    event: ActionEvent,
    player_names: List[str],
) -> RoundSummary:
    rewards = [int(np.round(r * 100)) for r in np.array(next_state.rewards)]
    reason = "abortive_draw_normal"
    winners: List[WinnerSummary] = []
    if event.action == Action.TSUMO:
        reason = "tsumo"
        winners.append(
            summarise_winner(
                prev_state,
                next_state,
                player=event.player,
                player_names=player_names,
                winning_tile=int(prev_state.round_state.last_draw),
                from_player=None,
                is_ron=False,
            )
        )
    elif event.action == Action.RON:
        reason = "ron"
        winners.append(
            summarise_winner(
                prev_state,
                next_state,
                player=event.player,
                player_names=player_names,
                winning_tile=int(prev_state.round_state.target),
                from_player=int(prev_state.round_state.last_player),
                is_ron=True,
            )
        )
    is_game_end = is_game_end_pending(next_state)
    return RoundSummary(
        reason=reason,
        rewards=rewards,
        winners=winners,
        round_count=int(prev_state.round_state.round),
        honba=int(prev_state.round_state.honba),
        kyotaku=int(prev_state.round_state.kyotaku),
        is_game_end=is_game_end,
    )


def build_round_summary_red(
    prev_state: Any,
    next_state: Any,
    event: ActionEvent,
    player_names: List[str],
) -> RoundSummary:
    rewards = [int(np.round(r * 100)) for r in np.array(next_state.rewards)]
    reason = "abortive_draw_normal"
    winners: List[WinnerSummary] = []
    if event.action == RedAction.TSUMO:
        reason = "tsumo"
        winners.append(
            summarise_winner_red(
                prev_state,
                next_state,
                player=event.player,
                player_names=player_names,
                winning_tile=int(prev_state.round_state.last_draw),
                from_player=None,
                is_ron=False,
            )
        )
    elif event.action == RedAction.RON:
        reason = "ron"
        winners.append(
            summarise_winner_red(
                prev_state,
                next_state,
                player=event.player,
                player_names=player_names,
                winning_tile=int(prev_state.round_state.target),
                from_player=int(prev_state.round_state.last_player),
                is_ron=True,
            )
        )
    return RoundSummary(
        reason=reason,
        rewards=rewards,
        winners=winners,
        round_count=int(prev_state.round_state.round),
        honba=int(prev_state.round_state.honba),
        kyotaku=int(prev_state.round_state.kyotaku),
        is_game_end=bool(next_state.terminated),
    )


def is_game_end_pending(state: State) -> bool:
    score = np.array(state.round_state.score, dtype=np.int32)
    if score.size == 0:
        return False
    dealer = int(state.round_state.dealer)
    hora = np.array(state.players.has_won, dtype=bool)
    can_ron = np.array(state.players.can_win, dtype=bool)
    is_tempai = can_ron.any(axis=-1)
    will_dealer_continue = bool(is_tempai[dealer] or hora[dealer])
    is_final_round = int(state.round_state.round) == int(state.round_state.round_limit)
    has_dealer_end = not will_dealer_continue
    init_order = np.array(state.round_state.init_wind, dtype=int)
    order = np.argsort(score[init_order])
    score_sorted = score[order]
    top_idx = order[int(np.argmax(score_sorted))]
    is_dealer_top = int(top_idx) == dealer
    has_minus_score = bool((score < 0).any())
    return bool(
        (is_final_round and has_dealer_end)
        or has_minus_score
        or (is_final_round and is_dealer_top)
    )


def summarise_winner(
    prev_state: State,
    next_state: State,
    *,
    player: int,
    player_names: List[str],
    winning_tile: int,
    from_player: Optional[int],
    is_ron: bool,
) -> WinnerSummary:
    hand = jnp.asarray(prev_state.players.hand[player])
    melds = jnp.asarray(prev_state.players.melds[player])
    n_meld = jnp.int32(prev_state.players.meld_counts[player])
    riichi = jnp.bool_(prev_state.players.riichi[player])
    riichi_flag_state = bool(np.array(riichi))
    prevalent_wind = jnp.int32(prev_state.round_state.round % 4)
    seat_wind = jnp.int32(prev_state.round_state.seat_wind[player])
    dora = jnp.asarray(_dora_array(prev_state))
    hand_for_count = hand
    if is_ron and 0 <= winning_tile < Tile.NUM_TILE_TYPE:
        hand_for_count = hand_for_count.at[winning_tile].add(1)
    flatten = Yaku.flatten(hand_for_count, melds, n_meld)
    flatten_np = np.array(flatten, dtype=np.int32)
    dora_np = np.array(dora, dtype=np.int32)
    visible_dora = int(np.dot(flatten_np, dora_np[0]))
    ura_dora = int(np.dot(flatten_np, dora_np[1])) if riichi_flag_state else 0
    dora_tile_list = resolve_dora_tiles(prev_state.round_state.dora_indicators)
    ura_dora_tile_list = resolve_dora_tiles(prev_state.round_state.ura_dora_indicators)
    yaku_mask, fan, fu = Yaku.judge(
        hand,
        melds,
        n_meld,
        jnp.int32(winning_tile),
        riichi,
        jnp.bool_(is_ron),
        prevalent_wind,
        seat_wind,
        dora,
    )
    yaku_mask_np = np.array(yaku_mask)
    fan_val = int(np.array(fan))
    fu_val = int(np.array(fu))
    yakuman = 0
    indices = [i for i, flag in enumerate(yaku_mask_np) if flag]
    yaku_english = [YAKU_NAMES_EN[i] for i in indices]
    yaku_japanese = [YAKU_NAMES_JA[i] for i in indices]
    extra_definitions = EXTRA_RON_YAKU if is_ron else EXTRA_TSUMO_YAKU
    extras_english = list_extra_yaku(
        prev_state, player, extra_definitions, use_english=True
    )
    extras_japanese = list_extra_yaku(
        prev_state, player, extra_definitions, use_english=False
    )
    if not is_ron:
        if is_first_turn(prev_state) and int(prev_state.players.meld_counts.sum()) == 0:
            if player == int(prev_state.round_state.dealer):
                extras_english.append("Heavenly Hand")
                extras_japanese.append("天和")
            else:
                extras_english.append("Earthly Hand")
                extras_japanese.append("地和")
    yaku_english.extend(extras_english)
    yaku_japanese.extend(extras_japanese)
    riichi_yaku_flag = any(
        name in yaku_english for name in ("Riichi", "Double Riichi")
    )
    if fu_val == 0 and fan_val > 0:
        yakuman = fan_val
    points_delta = int(np.round(np.array(next_state.rewards[player]) * 100))
    return WinnerSummary(
        player=player,
        name=player_names[player],
        points_delta=points_delta,
        fan=fan_val,
        fu=fu_val,
        yaku=yaku_english,
        yaku_japanese=yaku_japanese,
        dora_count=visible_dora,
        ura_dora_count=ura_dora,
        dora_tiles=dora_tile_list,
        ura_dora_tiles=ura_dora_tile_list,
        is_riichi=riichi_yaku_flag,
        yakuman=yakuman,
        winning_tile=winning_tile if winning_tile >= 0 else None,
        from_player=from_player,
    )


def summarise_winner_red(
    prev_state: Any,
    next_state: Any,
    *,
    player: int,
    player_names: List[str],
    winning_tile: int,
    from_player: Optional[int],
    is_ron: bool,
) -> WinnerSummary:
    hand = jnp.asarray(prev_state.players.hand_with_red[player])
    melds = jnp.asarray(prev_state.players.melds[player])
    n_meld = jnp.int32(prev_state.players.meld_counts[player])
    riichi = bool(np.array(prev_state.players.riichi[player]))
    flatten = RedYaku.flatten(hand, melds, n_meld)
    if is_ron and 0 <= winning_tile < RedTile.NUM_TILE_TYPE_WITH_RED:
        flatten = flatten.at[int(RedTile.to_tile_type(winning_tile))].add(1)
    dora = _red_dora_array(prev_state)
    flatten_np = np.array(flatten, dtype=np.int32)
    dora_np = np.array(dora, dtype=np.int32)
    visible_dora = int(np.dot(flatten_np, dora_np[0]))
    ura_dora = int(np.dot(flatten_np, dora_np[1])) if riichi else 0
    dora_tile_list = resolve_dora_tiles(prev_state.round_state.dora_indicators)
    ura_dora_tile_list = resolve_dora_tiles(prev_state.round_state.ura_dora_indicators)
    yaku_mask, fan, fu = RedYaku.judge(hand, jnp.bool_(is_ron), jnp.int32(player), prev_state)
    yaku_mask_np = np.array(yaku_mask, dtype=bool)
    fan_val = int(np.array(fan))
    fu_val = int(np.array(fu))
    indices = [i for i, flag in enumerate(yaku_mask_np) if flag]
    yaku_english = [RED_YAKU_NAMES_EN[i] for i in indices]
    yaku_japanese = [RED_YAKU_NAMES_JA[i] for i in indices]
    extra_definitions = EXTRA_RON_YAKU if is_ron else EXTRA_TSUMO_YAKU
    yaku_english.extend(list_extra_yaku(prev_state, player, extra_definitions, use_english=True))
    yaku_japanese.extend(list_extra_yaku(prev_state, player, extra_definitions, use_english=False))
    if not is_ron and is_first_turn(prev_state) and int(np.array(prev_state.players.meld_counts).sum()) == 0:
        if player == int(prev_state.round_state.dealer):
            yaku_english.append("Heavenly Hand")
            yaku_japanese.append("天和")
        else:
            yaku_english.append("Earthly Hand")
            yaku_japanese.append("地和")
    yakuman = 0
    if fu_val == 0 and fan_val > 0:
        yakuman = fan_val
    points_delta = int(np.round(np.array(next_state.rewards[player]) * 100))
    return WinnerSummary(
        player=player,
        name=player_names[player],
        points_delta=points_delta,
        fan=fan_val,
        fu=fu_val,
        yaku=yaku_english,
        yaku_japanese=yaku_japanese,
        dora_count=visible_dora,
        ura_dora_count=ura_dora,
        dora_tiles=dora_tile_list,
        ura_dora_tiles=ura_dora_tile_list,
        is_riichi=any(name in yaku_english for name in ("Riichi", "Double Riichi")),
        yakuman=yakuman,
        winning_tile=winning_tile if winning_tile >= 0 else None,
        from_player=from_player,
    )


def resolve_dora_tiles(indicators: jnp.ndarray) -> List[int]:
    indices = np.array(indicators, dtype=int)
    tiles: List[int] = []
    for idx in indices:
        if idx >= 0:
            tile_type = int(RedTile.to_tile_type(idx))
            tiles.append(int(DORA_TILE_LOOKUP[tile_type]))
    return tiles


def list_extra_yaku(
    state: Any,
    player: int,
    definitions: List[ExtraYakuDefinition],
    *,
    use_english: bool,
) -> List[str]:
    names: List[str] = []
    for definition in definitions:
        value = _resolve_state_flag_value(state, definition.attr)
        if isinstance(value, (np.ndarray, jnp.ndarray)):
            array_value = np.array(value)
            if array_value.ndim == 0:
                flag = bool(array_value)
            elif player < array_value.shape[0]:
                flag = bool(array_value[player])
            else:
                flag = False
        else:
            flag = bool(value)
        if flag:
            names.append(definition.english if use_english else definition.japanese)
    return names


def _resolve_state_flag_value(state: Any, attr: str) -> Any:
    # ``EXTRA_*_YAKU`` definitions use legacy ``_field`` names; strip the
    # leading underscore and look it up under ``players`` / ``round_state``.
    nested_attr = attr[1:] if attr.startswith("_") else attr
    if hasattr(state.players, nested_attr):
        return getattr(state.players, nested_attr)
    if hasattr(state.round_state, nested_attr):
        return getattr(state.round_state, nested_attr)
    raise AttributeError(f"State has no attribute {attr}")


def is_first_turn(state: Any) -> bool:
    return bool(int(state.round_state.next_deck_ix) == FIRST_DRAW_IDX)


__all__ = ["GameManager", "GameSession", "ActionEvent", "RoundSummary"]
