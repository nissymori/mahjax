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

"""Env state -> the JSON the browser draws (spec sections 6.1 and 6.2).

Three jobs, no state of its own:

``build_view``
    one table frame, in absolute seat order.
``build_prompt``
    ``legal_action_mask`` -> the choices a human is offered. The mask is
    mirrored, never re-derived: every legal action ends up in exactly one of
    ``discardable`` / ``tsumogiri`` / ``options``.
``build_win`` / ``build_round_result`` / ``final_standings``
    the round-result overlay, with the winning hand's han broken down so that
    the parts add up to the number the env actually paid out.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .rules import NUM_PLAYERS, NUM_TILE_TYPES, Rules, tile_type

__all__ = [
    "SeatInfo",
    "YAKU_NAMES",
    "build_prompt",
    "build_view",
    "describe_action",
    "build_win",
    "build_round_result",
    "final_standings",
]


@dataclass(frozen=True)
class SeatInfo:
    """Who sits at one seat, for the seat label."""

    name: str
    kind: str  # "human" | "agent"


#: Display names, keyed by the env-independent yaku key used below.
YAKU_NAMES: Dict[str, Dict[str, str]] = {
    "menzen_tsumo": {"ja": "門前清自摸和", "en": "Fully Concealed Hand"},
    "riichi": {"ja": "立直", "en": "Riichi"},
    "ippatsu": {"ja": "一発", "en": "Ippatsu"},
    "chankan": {"ja": "搶槓", "en": "Robbing a Kan"},
    "rinshan": {"ja": "嶺上開花", "en": "After a Kan"},
    "haitei": {"ja": "海底摸月", "en": "Under the Sea"},
    "houtei": {"ja": "河底撈魚", "en": "Under the River"},
    "pinfu": {"ja": "平和", "en": "Pinfu"},
    "tanyao": {"ja": "断么九", "en": "All Simples"},
    "iipeiko": {"ja": "一盃口", "en": "Pure Double Chow"},
    "seat_wind": {"ja": "自風", "en": "Seat Wind"},
    "seat_wind_east": {"ja": "自風 東", "en": "Seat Wind East"},
    "seat_wind_south": {"ja": "自風 南", "en": "Seat Wind South"},
    "seat_wind_west": {"ja": "自風 西", "en": "Seat Wind West"},
    "seat_wind_north": {"ja": "自風 北", "en": "Seat Wind North"},
    "round_wind": {"ja": "場風", "en": "Round Wind"},
    "round_wind_east": {"ja": "場風 東", "en": "Round Wind East"},
    "round_wind_south": {"ja": "場風 南", "en": "Round Wind South"},
    "round_wind_west": {"ja": "場風 西", "en": "Round Wind West"},
    "round_wind_north": {"ja": "場風 北", "en": "Round Wind North"},
    "haku": {"ja": "白", "en": "White Dragon"},
    "hatsu": {"ja": "發", "en": "Green Dragon"},
    "chun": {"ja": "中", "en": "Red Dragon"},
    "double_riichi": {"ja": "ダブル立直", "en": "Double Riichi"},
    "chiitoitsu": {"ja": "七対子", "en": "Seven Pairs"},
    "chanta": {"ja": "混全帯么九", "en": "Outside Hand"},
    "ittsu": {"ja": "一気通貫", "en": "Pure Straight"},
    "sanshoku": {"ja": "三色同順", "en": "Mixed Triple Chow"},
    "sanshoku_doukou": {"ja": "三色同刻", "en": "Triple Pung"},
    "sankantsu": {"ja": "三槓子", "en": "Three Kans"},
    "toitoi": {"ja": "対々和", "en": "All Pungs"},
    "sanankou": {"ja": "三暗刻", "en": "Three Concealed Pungs"},
    "shousangen": {"ja": "小三元", "en": "Little Three Dragons"},
    "honroutou": {"ja": "混老頭", "en": "All Terminals and Honors"},
    "ryanpeikou": {"ja": "二盃口", "en": "Twice Pure Double Chow"},
    "junchan": {"ja": "純全帯么九", "en": "Terminals in All Sets"},
    "honitsu": {"ja": "混一色", "en": "Half Flush"},
    "chinitsu": {"ja": "清一色", "en": "Full Flush"},
    "renhou": {"ja": "人和", "en": "Hand of Man"},
    "tenhou": {"ja": "天和", "en": "Blessing of Heaven"},
    "chiihou": {"ja": "地和", "en": "Blessing of Earth"},
    "daisangen": {"ja": "大三元", "en": "Big Three Dragons"},
    "suuankou": {"ja": "四暗刻", "en": "Four Concealed Pungs"},
    "suuankou_tanki": {"ja": "四暗刻単騎", "en": "Four Concealed Pungs Single Wait"},
    "tsuuiisou": {"ja": "字一色", "en": "All Honors"},
    "ryuuiisou": {"ja": "緑一色", "en": "All Green"},
    "chinroutou": {"ja": "清老頭", "en": "All Terminals"},
    "chuuren": {"ja": "九蓮宝燈", "en": "Nine Gates"},
    "junsei_chuuren": {"ja": "純正九蓮宝燈", "en": "True Nine Gates"},
    "kokushi": {"ja": "国士無双", "en": "Thirteen Orphans"},
    "kokushi_13": {"ja": "国士無双十三面待ち", "en": "Thirteen Orphans 13-Wait"},
    "daisuushii": {"ja": "大四喜", "en": "Big Four Winds"},
    "shousuushii": {"ja": "小四喜", "en": "Little Four Winds"},
    "suukantsu": {"ja": "四槓子", "en": "Four Kans"},
}

# Yaku index -> key, in the order each env's ``Yaku`` module numbers them.
_RED_YAKU_KEYS: Tuple[str, ...] = (
    "menzen_tsumo", "riichi", "ippatsu", "chankan", "rinshan", "haitei", "houtei",
    "pinfu", "tanyao", "iipeiko",
    "seat_wind_east", "seat_wind_south", "seat_wind_west", "seat_wind_north",
    "round_wind_east", "round_wind_south", "round_wind_west", "round_wind_north",
    "haku", "hatsu", "chun", "double_riichi", "chiitoitsu", "chanta", "ittsu",
    "sanshoku", "sanshoku_doukou", "sankantsu", "toitoi", "sanankou", "shousangen",
    "honroutou", "ryanpeikou", "junchan", "honitsu", "chinitsu", "renhou",
    "tenhou", "chiihou", "daisangen", "suuankou", "suuankou_tanki", "tsuuiisou",
    "ryuuiisou", "chinroutou", "chuuren", "junsei_chuuren", "kokushi",
    "kokushi_13", "daisuushii", "shousuushii", "suukantsu",
)
_NO_RED_YAKU_KEYS: Tuple[str, ...] = (
    "pinfu", "iipeiko", "ryanpeikou", "chanta", "junchan", "ittsu", "sanshoku",
    "sanshoku_doukou", "toitoi", "sanankou", "sankantsu", "chiitoitsu", "tanyao",
    "honitsu", "chinitsu", "honroutou", "shousangen", "haku", "hatsu", "chun",
    "round_wind", "seat_wind", "menzen_tsumo", "riichi",
    "daisangen", "shousuushii", "daisuushii", "chuuren", "kokushi", "chinroutou",
    "tsuuiisou", "ryuuiisou", "suuankou", "suukantsu",
)

_TILE_NAMES: Tuple[str, ...] = tuple(
    [f"{n}m" for n in range(1, 10)]
    + [f"{n}p" for n in range(1, 10)]
    + [f"{n}s" for n in range(1, 10)]
    + ["東", "南", "西", "北", "白", "發", "中"]
)


def _tile_name(tile: int) -> str:
    return _TILE_NAMES[tile_type(tile)]


# --------------------------------------------------------------------- yaku

@dataclass(frozen=True)
class _YakuBackend:
    """The one env module the win breakdown has to reach into."""

    #: ``(hand, melds, n_meld, last_tile, riichi, is_ron, round_wind, seat_wind,
    #: dora[batch]) -> (mask, fan, fu)``, batched over the dora argument.
    judge: Any
    fan: np.ndarray  # (2, n_yaku), indexed [1 if concealed else 0]
    yakuman: np.ndarray  # (n_yaku,) yakuman multiples
    keys: Tuple[str, ...]
    dora_next: np.ndarray  # indicator tile type -> dora tile type
    first_turn_deck_ix: int  # 天和 / 地和 hold while next_deck_ix is at least this


@functools.lru_cache(maxsize=None)
def _yaku_backend(env_id: str) -> _YakuBackend:
    if env_id == "red_mahjong":
        from mahjax.red_mahjong.constants import DORA_ARRAY, FIRST_DRAW_IDX
        from mahjax.red_mahjong.yaku import Yaku, _Internal

        raw = Yaku.judge_hand_related
        fan, yakuman, keys = _Internal.FAN, _Internal.YAKUMAN, _RED_YAKU_KEYS
    else:
        from mahjax.no_red_mahjong.state import DORA_ARRAY, FIRST_DRAW_IDX
        from mahjax.no_red_mahjong.yaku import Yaku  # type: ignore[assignment]

        raw = Yaku.judge  # type: ignore[assignment]
        fan, yakuman, keys = Yaku.FAN, Yaku.YAKUMAN, _NO_RED_YAKU_KEYS  # type: ignore[attr-defined]

    def one(hand, melds, n_meld, last_tile, riichi, is_ron, pw, sw, dora):
        return raw(hand, melds, n_meld, last_tile, riichi, is_ron, pw, sw, dora)

    return _YakuBackend(
        judge=jax.jit(jax.vmap(one, in_axes=(None,) * 8 + (0,))),
        fan=np.asarray(fan, dtype=np.int32),
        yakuman=np.asarray(yakuman, dtype=np.int32),
        keys=keys,
        # The env's own table, not ``rules.DORA_NEXT``: no_red_mahjong's differs
        # for 9p/1s/9s, and the win must add up to what that env actually paid.
        dora_next=np.asarray(DORA_ARRAY, dtype=np.int32),
        first_turn_deck_ix=int(FIRST_DRAW_IDX) - 4,
    )


def _dora_counts(backend: _YakuBackend, indicators: np.ndarray) -> np.ndarray:
    counts = np.zeros(NUM_TILE_TYPES, dtype=np.int32)
    for indicator in indicators:
        if int(indicator) >= 0:
            counts[backend.dora_next[tile_type(int(indicator))]] += 1
    return counts


@dataclass(frozen=True)
class _Judged:
    """One win as the env's yaku module scores it, with the han split apart."""

    mask: np.ndarray
    fu: int
    fan: int
    hand_han: int
    aka_han: int
    dora_han: int
    ura_han: int
    dora: List[int]
    ura_dora: List[int]
    concealed: bool
    #: Total fan per dora snapshot; the chosen one is the entry the env agrees with.
    candidates: Tuple[int, ...]


def _judge_win(rules: Rules, state: Any, seat: int, is_ron: bool, cached_fan: int) -> _Judged:
    """Re-run the env's yaku judge on a pre-win state and split its ``fan``.

    The judge is run against a growing dora array -- none, front, front+ura --
    so that its one opaque total falls apart into aka / dora / ura counts
    without any of them being re-derived here.
    """
    backend = _yaku_backend(rules.env_id)
    rs = state.round_state
    seat = int(seat)

    hand = np.asarray(
        state.players.hand_with_red[seat] if rules.has_red else state.players.hand[seat]
    ).copy()
    if is_ron:
        last_tile = int(rs.target)
    else:
        last_tile = int(rs.last_draw)
        hand[last_tile] -= 1  # ``judge`` adds the winning tile back itself

    indicators = np.asarray(rs.dora_indicators, dtype=np.int32)
    ura_indicators = np.asarray(rs.ura_dora_indicators, dtype=np.int32)
    n_kan_doras = int(rs.n_kan_doras)
    snapshots = [(indicators, ura_indicators)]
    if n_kan_doras > 0:
        # A rinshan win is cached in ``_kan``, before ``_draw_after_kan`` turns
        # that kan's dora indicator over, so the newest one may not be part of
        # the number the env paid. Offer both snapshots and keep the one that
        # reproduces it.
        dropped = indicators.copy()
        dropped_ura = ura_indicators.copy()
        dropped[n_kan_doras] = -1
        dropped_ura[n_kan_doras] = -1
        snapshots.append((dropped, dropped_ura))

    zero = np.zeros(NUM_TILE_TYPES, dtype=np.int32)
    batch = []
    for front, back in snapshots:
        counts, ura_counts = _dora_counts(backend, front), _dora_counts(backend, back)
        batch += [[zero, zero], [counts, zero], [counts, ura_counts]]

    mask, fan, fu = backend.judge(
        jnp.asarray(hand),
        state.players.melds[seat],
        jnp.int32(state.players.meld_counts[seat]),
        jnp.int32(last_tile),
        jnp.bool_(state.players.riichi[seat]),
        jnp.bool_(is_ron),
        jnp.int32(int(rs.round) // 4),
        jnp.int32(int(rs.seat_wind[seat])),
        jnp.asarray(np.asarray(batch, dtype=np.int8)),
    )
    mask = np.asarray(mask)
    fan = np.asarray(fan, dtype=np.int32)

    chosen = 0
    for k in range(len(snapshots)):
        if int(fan[3 * k + 2]) == cached_fan:
            chosen = k
            break
    base = 3 * chosen
    front, back = snapshots[chosen]

    melds = rules.melds(state, seat)
    concealed = all(meld["kind"] == "kan_closed" for meld in melds)
    row = backend.fan[1 if concealed else 0]
    bits = np.flatnonzero(mask[base])
    hand_han = int(row[bits].sum())
    has_riichi = bool(state.players.riichi[seat])
    return _Judged(
        mask=mask[base],
        fu=int(fu[base]),
        fan=int(fan[base + 2]),
        hand_han=hand_han,
        aka_han=int(fan[base]) - hand_han,
        dora_han=int(fan[base + 1]) - int(fan[base]),
        ura_han=int(fan[base + 2]) - int(fan[base + 1]),
        dora=[int(t) for t in front if int(t) >= 0],
        ura_dora=[int(t) for t in back if int(t) >= 0] if has_riichi else [],
        concealed=concealed,
        candidates=tuple(int(fan[3 * k + 2]) for k in range(len(snapshots))),
    )


# ------------------------------------------------------------------- prompt

_ACTION_LABELS = {
    "riichi": "立直",
    "tsumo": "ツモ",
    "ron": "ロン",
    "pon": "ポン",
    "chi": "チー",
    "kyuushu": "九種九牌",
    "pass": "パス",
}
_OPTION_ORDER = ("ron", "tsumo", "riichi", "kan", "pon", "chi", "kyuushu", "pass")


def _is_added_kan(state: Any, seat: int, tile_type_: int) -> bool:
    """A self-kan upgrades a pon iff one is still standing.

    The env zeroes ``players.pon`` for the tile as it folds the pon into the
    kan, so a live entry is exactly what tells the two kinds of self-kan apart.
    """
    return int(state.players.pon[int(seat), int(tile_type_)]) > 0


def _option(rules: Rules, state: Any, seat: int, action: int) -> Dict[str, Any]:
    target = int(state.round_state.target)
    if action == rules.RIICHI:
        return {"action": action, "kind": "riichi", "tiles": None, "label": _ACTION_LABELS["riichi"]}
    if action == rules.TSUMO:
        return {"action": action, "kind": "tsumo", "tiles": None, "label": _ACTION_LABELS["tsumo"]}
    if action == rules.RON:
        return {"action": action, "kind": "ron", "tiles": None, "label": _ACTION_LABELS["ron"]}
    if action == rules.PASS:
        return {"action": action, "kind": "pass", "tiles": None, "label": _ACTION_LABELS["pass"]}
    if action == rules.KYUUSHU:
        return {"action": action, "kind": "kyuushu", "tiles": None, "label": _ACTION_LABELS["kyuushu"]}
    if rules.is_pon(action):
        return {
            "action": action,
            "kind": "pon",
            "tiles": rules.pon_tiles(action, target),
            "label": _ACTION_LABELS["pon"],
        }
    if rules.is_chi(action):
        return {
            "action": action,
            "kind": "chi",
            "tiles": rules.chi_tiles(action, target),
            "label": _ACTION_LABELS["chi"],
        }
    if action == rules.OPEN_KAN:
        return {"action": action, "kind": "kan", "tiles": rules.open_kan_tiles(target), "label": "明槓"}
    if rules.is_kan(action):
        tile = rules.kan_tile_type(action)
        added = _is_added_kan(state, seat, tile)
        return {
            "action": action,
            "kind": "kan",
            "tiles": rules.closed_kan_tiles(tile),
            "label": f"{'加槓' if added else '暗槓'} {_TILE_NAMES[tile]}",
        }
    raise ValueError(f"{rules.env_id}: action {action} is not an offerable choice")


def build_prompt(rules: Rules, state: Any, seat: int) -> Optional[Dict[str, Any]]:
    """The choices ``seat`` is being asked to make, or ``None``.

    ``None`` covers three things: it is not this seat's decision, the round is
    already settled, or the env left exactly one forced action that the server
    plays by itself (spec section 3.3: riichi tsumogiri, the nine-terminals
    confirmation of an abortive draw).
    """
    seat = int(seat)
    if int(state.current_player) != seat or rules.is_round_over(state):
        return None
    legal = rules.legal_actions(state)
    if not legal or (len(legal) == 1 and legal[0] in (rules.TSUMOGIRI, rules.KYUUSHU, rules.DUMMY)):
        return None

    rs = state.round_state
    target_tile = int(rs.target)
    if target_tile >= 0:
        target = {
            "seat": int(rs.last_player),
            "tile": target_tile,
            "kind": "kan" if bool(rs.kan_declared) else "discard",
        }
    else:
        target = None

    discardable = [a for a in legal if rules.is_discard(a)]
    tsumogiri = rules.TSUMOGIRI if rules.TSUMOGIRI in legal else None
    options = [
        _option(rules, state, seat, a)
        for a in legal
        if not rules.is_discard(a) and a != rules.TSUMOGIRI
    ]
    options.sort(key=lambda o: (_OPTION_ORDER.index(o["kind"]), o["action"]))
    return {
        "kind": "claim" if target is not None else "turn",
        "target": target,
        "discardable": discardable,
        "tsumogiri": tsumogiri,
        "options": options,
    }


def describe_action(rules: Rules, pre_state: Any, action: int) -> Dict[str, Any]:
    """What one step did, read off the state it was taken from.

    Used for the replay bar's "南家: 打 5m" line, so it names the tiles the
    action moves rather than the action id.
    """
    action = int(action)
    seat = int(pre_state.current_player)
    rs = pre_state.round_state
    if rules.is_discard(action):
        return {"seat": seat, "kind": "discard", "tiles": [action]}
    if action == rules.TSUMOGIRI:
        return {"seat": seat, "kind": "discard", "tiles": [int(rs.last_draw)]}
    if action == rules.RIICHI:
        return {"seat": seat, "kind": "riichi", "tiles": None}
    if action == rules.TSUMO:
        return {"seat": seat, "kind": "tsumo", "tiles": [int(rs.last_draw)]}
    if action == rules.RON:
        return {"seat": seat, "kind": "ron", "tiles": [int(rs.target)]}
    if action == rules.PASS:
        return {"seat": seat, "kind": "pass", "tiles": None}
    if action == rules.DUMMY:
        return {"seat": seat, "kind": "dummy", "tiles": None}
    if action == rules.KYUUSHU:
        return {"seat": seat, "kind": "kyuushu", "tiles": None}
    if rules.is_pon(action):
        return {"seat": seat, "kind": "pon", "tiles": rules.pon_tiles(action, int(rs.target))}
    if rules.is_chi(action):
        return {"seat": seat, "kind": "chi", "tiles": rules.chi_tiles(action, int(rs.target))}
    if action == rules.OPEN_KAN:
        return {"seat": seat, "kind": "kan_open", "tiles": rules.open_kan_tiles(int(rs.target))}
    if rules.is_kan(action):
        tile = rules.kan_tile_type(action)
        return {
            "seat": seat,
            "kind": "kan_added" if _is_added_kan(pre_state, seat, tile) else "kan_closed",
            "tiles": rules.closed_kan_tiles(tile),
        }
    raise ValueError(f"{rules.env_id}: unknown action {action}")


# --------------------------------------------------------------------- view


def _seat_view(
    rules: Rules, state: Any, seat: int, info: SeatInfo, reveal: bool, score: int
) -> Dict[str, Any]:
    return {
        "name": info.name,
        "kind": info.kind,
        "wind": int(state.round_state.seat_wind[seat]),
        "score": score,
        "riichi": rules.riichi_state(state, seat),
        "hand": rules.hand_tiles(state, seat) if reveal else None,
        "handCount": rules.hand_count(state, seat),
        "drawn": rules.drawn_tile(state, seat) if reveal else None,
        "melds": rules.melds(state, seat),
        "river": [
            {
                "tile": d["tile"],
                "tsumogiri": d["tsumogiri"],
                "riichi": d["riichi"],
                "called": d["called"],
            }
            for d in rules.river(state, seat)
        ],
    }


def build_view(
    rules: Rules,
    state: Any,
    seats: Sequence[SeatInfo],
    *,
    reveal: Sequence[bool],
    prompt: Optional[Dict[str, Any]] = None,
    result: Optional[Dict[str, Any]] = None,
    step: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """One table frame in absolute seat order (spec section 6.1)."""
    rs = state.round_state
    indicators = np.asarray(rs.dora_indicators)
    scores = rules.scores(state)
    over = rules.is_round_over(state)

    # Ura dora stays secret until a riichi win puts it in the overlay.
    ura: Optional[List[int]] = None
    if result is not None:
        for winner in result.get("winners") or []:
            if winner.get("uraDora"):
                ura = list(winner["uraDora"])
                break

    return {
        "env": rules.env_id,
        "round": {
            "index": int(rs.round),
            "honba": int(rs.honba),
            "kyotaku": int(rs.kyotaku),
            "remaining": rules.remaining_draws(state),
            "dora": [int(t) if int(t) >= 0 else None for t in indicators],
            "uraDora": ura,
        },
        "dealer": int(rs.dealer),
        "current": None if over else int(state.current_player),
        "seats": [
            _seat_view(rules, state, i, seats[i], bool(reveal[i]), scores[i])
            for i in range(NUM_PLAYERS)
        ],
        "lastDiscard": rules.last_discard(state),
        "prompt": prompt,
        "result": result,
        # The board shown at a round end is the one from before the sharing
        # steps, and the env only flips `terminated` during those -- so a
        # finished game is announced by the result, not by this state.
        "gameOver": bool(state.terminated)
        or bool(result is not None and result.get("gameOver")),
        "step": step,
    }


# ---------------------------------------------------------------- win / result


def _situational(rules: Rules, state: Any, seat: int, is_ron: bool) -> List[Tuple[str, bool]]:
    """The yaku the env adds outside the judge mask, in the env's own order."""
    rs = state.round_state
    players = state.players
    ippatsu = bool(players.ippatsu[seat]) and bool(players.riichi[seat])
    double_riichi = bool(players.double_riichi[seat])
    if is_ron:
        chankan = bool(rs.kan_declared)
        houtei = bool(rs.is_haitei) and (not chankan or not rules.has_red)
        return [
            ("ippatsu", ippatsu),
            ("double_riichi", double_riichi),
            ("chankan", chankan),
            ("houtei", houtei),
        ]
    rinshan = bool(rs.can_after_kan)
    haitei = bool(rs.is_haitei) and (not rinshan or not rules.has_red)
    return [
        ("rinshan", rinshan),
        ("ippatsu", ippatsu),
        ("double_riichi", double_riichi),
        ("haitei", haitei),
    ]


def build_win(
    rules: Rules, pre_state: Any, post_state: Any, seat: int, is_ron: bool
) -> Dict[str, Any]:
    """One winner's overlay entry, scored exactly as the env scored it.

    Everything but ``points`` comes from ``pre_state``: the winning step has
    already folded the hand away.
    """
    seat = int(seat)
    rs = pre_state.round_state
    backend = _yaku_backend(rules.env_id)
    cached_fan = int(pre_state.players.fan[seat, 0])
    cached_fu = int(pre_state.players.fu[seat, 0])
    is_hand_yakuman = cached_fu == 0  # the judge reports a yakuman as fu 0
    situational = _situational(rules, pre_state, seat, is_ron)
    bonus = sum(1 for _, flag in situational if flag)

    if is_ron:
        winning_tile = int(rs.target)
        pure_first_turn = False
        han = cached_fan + (0 if is_hand_yakuman else bonus)
        fu = cached_fu
    else:
        winning_tile = int(rs.last_draw)
        pure_first_turn = (
            int(rs.next_deck_ix) >= backend.first_turn_deck_ix
            and int(np.asarray(pre_state.players.meld_counts).sum()) == 0
        )
        rinshan = bool(rs.can_after_kan)
        if is_hand_yakuman:
            han = cached_fan + int(pure_first_turn)
        elif pure_first_turn:
            han = 1  # 天和 / 地和 stands alone as a single yakuman
        else:
            han = cached_fan + bonus
        if is_hand_yakuman or pure_first_turn:
            fu = 0
        else:
            fu = cached_fu + (0 if rules.has_red else 2 * int(rinshan))
    is_yakuman = is_hand_yakuman or pure_first_turn

    judged = _judge_win(rules, pre_state, seat, is_ron, cached_fan)

    yaku: List[Dict[str, Any]] = []
    if is_yakuman:
        if is_hand_yakuman:
            for i in np.flatnonzero(judged.mask):
                value = int(backend.yakuman[i])
                if value:
                    yaku.append({"name": YAKU_NAMES[backend.keys[i]]["ja"], "han": value})
        if pure_first_turn:
            key = "tenhou" if seat == int(rs.dealer) else "chiihou"
            yaku.append({"name": YAKU_NAMES[key]["ja"], "han": 1})
    else:
        row = backend.fan[1 if judged.concealed else 0]
        for i in np.flatnonzero(judged.mask):
            value = int(row[i])
            if value:
                yaku.append({"name": YAKU_NAMES[backend.keys[i]]["ja"], "han": value})
        yaku += [
            {"name": YAKU_NAMES[key]["ja"], "han": 1} for key, flag in situational if flag
        ]

    hand = rules.hand_tiles(pre_state, seat)
    if not is_ron:
        hand.remove(winning_tile)
    return {
        "seat": seat,
        "from": int(rs.last_player) if is_ron else None,
        "hand": hand,
        "melds": rules.melds(pre_state, seat),
        "winningTile": winning_tile,
        "yaku": yaku,
        "doraHan": 0 if is_yakuman else judged.dora_han,
        "akaHan": 0 if is_yakuman else judged.aka_han,
        "uraHan": 0 if is_yakuman else judged.ura_han,
        "han": han,
        "fu": fu,
        "yakuman": han if is_yakuman else 0,
        "dora": judged.dora,
        "uraDora": judged.ura_dora,
        "points": rules.rewards(post_state)[seat],
    }


def final_standings(rules: Rules, state: Any) -> List[Dict[str, int]]:
    """Final standings in seat order, with the points and the rank bonus apart.

    ``uma`` keeps the env's own units (the +30 / +10 / -10 / -30 it carries) and
    is deliberately not folded into ``score``: the two are read side by side, not
    added. The scores themselves are recomputed here rather than read back from
    the state because the env only folds uma in on one of its two game-end paths
    (spec section 9.3).
    """
    rs = state.round_state
    scores = np.asarray(rs.score, dtype=np.int64)
    order_points = np.asarray(rs.order_points, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")  # ties go to the earlier seat
    rank_of = np.zeros(NUM_PLAYERS, dtype=np.int64)
    uma_of = np.zeros(NUM_PLAYERS, dtype=np.int64)
    for rank, seat in enumerate(order):
        rank_of[seat] = rank + 1
        uma_of[seat] = order_points[rank]
    final = scores.copy()
    final[int(order[0])] += 10 * int(rs.kyotaku)  # sticks left on the table
    return [
        {
            "seat": seat,
            "rank": int(rank_of[seat]),
            "score": int(final[seat]) * 100,
            "uma": int(uma_of[seat]),
        }
        for seat in range(NUM_PLAYERS)
    ]


def abortive_reason(rules: Rules, state: Any) -> str:
    """Why the env forced an abortive draw on ``state``.

    The env reports a table condition only as a ``KYUUSHU``-only mask, so the
    condition itself has to be read back off the board. Play and replay both
    call this, which is what keeps a saved game's overlay saying the same thing
    the live one said.
    """
    players = state.players
    if int(sum(bool(x) for x in players.has_won)) >= 2:
        return "triple_ron"
    if int(sum(int(x) for x in players.riichi)) == NUM_PLAYERS:
        return "four_riichi"
    if int(sum(int(x) for x in players.n_kan)) >= 4:
        return "four_kans"
    return "four_winds"


def abortive_cause(rules: Rules, state: Any) -> Tuple[str, Optional[int]]:
    """``(reason, seat)`` for the abortive draw about to be applied to ``state``.

    Nine terminals is one player's call, so it names the seat that made it; a
    table condition belongs to nobody and leaves the seat unset.
    """
    if rules.KYUUSHU is None:
        raise ValueError(f"{rules.env_id} has no abortive draws")
    if rules.legal_actions(state) != [rules.KYUUSHU]:
        return "kyuushu", int(state.current_player)
    return abortive_reason(rules, state), None


def build_round_result(
    rules: Rules,
    *,
    type: str,
    reason: Optional[str],
    state: Any,
    winners: Sequence[Dict[str, Any]],
    score_start: Sequence[int],
    game_over: bool,
    final: Optional[Sequence[Dict[str, int]]] = None,
    abort_seat: Optional[int] = None,
) -> Dict[str, Any]:
    """The round-result overlay (spec section 6.2).

    ``state`` is the round-ending state and ``score_start`` the scores this
    round opened with, both in points.
    """
    rs = state.round_state
    scores = rules.scores(state)
    is_draw = type == "draw"
    nagashi = rules.nagashi_mangan(state) if is_draw else None
    if final is None and game_over:
        final = final_standings(rules, state)
    return {
        "type": type,
        "reason": reason,
        # Who declared it, for the abortive draw that one player calls.
        "abortSeat": None if abort_seat is None else int(abort_seat),
        "round": {"index": int(rs.round), "honba": int(rs.honba), "kyotaku": int(rs.kyotaku)},
        "winners": list(winners),
        "tenpai": rules.tenpai(state) if is_draw else None,
        # Only meaningful when nagashi mangan is what the env paid out on.
        "nagashiMangan": nagashi if nagashi is not None and any(nagashi) else None,
        # Not the sum of ``rewards``: the score difference also carries the
        # riichi sticks paid mid-round and a double ron's second payment.
        "deltas": [scores[i] - int(score_start[i]) for i in range(NUM_PLAYERS)],
        "scores": scores,
        "gameOver": bool(game_over),
        "final": list(final) if final is not None else None,
    }
