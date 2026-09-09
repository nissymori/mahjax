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

"""mjai event log <-> env actions (spec section 5.3).

Encoding is a state diff: one ``env.step`` can produce several protocol events
because the env folds the following draw (and a kan's replacement draw, and the
riichi stick being taken) into the step that caused it. :func:`encode_step`
emits them in the order they logically happened.

Decoding cannot be a pure function of the log, because a log records what
players *did*, not what they were *asked*. Two things are missing:

* ``PASS`` -- the env asks exactly one claimant at a time for a discard, and a
  seat that declines leaves no trace at all.
* ``DUMMY`` -- the four information-sharing steps between rounds.

So :func:`decode_events` replays the log against a fresh env and reads the
question from ``legal_action_mask`` at every decision point: if the log's next
event is not this seat taking this claim, the seat passed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .rules import RED_FIVE_TYPES, RED_TILE_BASE, Rules, red_of, rules_for, tile_type

Event = Dict[str, Any]

_SUITS = "mps"
_HONORS = ("E", "S", "W", "N", "P", "F", "C")
_HONOR_INDEX = {name: i for i, name in enumerate(_HONORS)}
_WINDS = ("E", "S", "W", "N")

#: The exhaustive draw (荒牌平局). Every other reason is an abortive draw and
#: is replayed as the ``KYUUSHU`` action.
EXHAUSTIVE_DRAW = "howanpai"
ABORTIVE_REASONS = (
    "kyushukyuhai",
    "suufonrenda",
    "suuchariichi",
    "suukaikan",
    "sanchaho",
)

#: Event types that stand for an env action. Everything else in a log
#: (``tsumo``, ``dora``, ``reach_accepted``, ``start_*``, ``end_*``) is
#: derived state and is skipped while decoding.
_ACTION_TYPES = frozenset(
    {"dahai", "reach", "pon", "chi", "daiminkan", "ankan", "kakan", "hora", "ryukyoku"}
)


class MjaiError(Exception):
    """A log could not be decoded against the env."""


# ------------------------------------------------------------------ notation


def tile_to_mjai(tile: int) -> str:
    """Red-aware tile id -> mjai notation (``0`` -> ``"1m"``, ``34`` -> ``"5mr"``)."""
    tile = int(tile)
    if tile >= RED_TILE_BASE:
        base = tile_type(tile)
        return f"{base % 9 + 1}{_SUITS[base // 9]}r"
    if tile >= 27:
        return _HONORS[tile - 27]
    return f"{tile % 9 + 1}{_SUITS[tile // 9]}"


def mjai_to_tile(s: str) -> int:
    """mjai notation -> red-aware tile id. Inverse of :func:`tile_to_mjai`."""
    if s in _HONOR_INDEX:
        return 27 + _HONOR_INDEX[s]
    if len(s) in (2, 3) and s[0] in "123456789" and s[1] in _SUITS:
        base = _SUITS.index(s[1]) * 9 + int(s[0]) - 1
        if len(s) == 2:
            return base
        if s[2] == "r" and base in RED_FIVE_TYPES:
            return red_of(base)
    raise ValueError(f"Not an mjai tile: {s!r}")


def event_actor(event: Event) -> Optional[int]:
    """The seat an event is attributed to, or ``None`` for table-wide events."""
    actor = event.get("actor")
    return None if actor is None else int(actor)


# ------------------------------------------------------------------ encoding


def start_kyoku_event(rules: Rules, state: Any) -> Event:
    """``start_kyoku`` for a freshly dealt round.

    The dealer's 14th tile is not part of ``tehais``; it is the ``tsumo`` that
    :func:`start_kyoku_events` appends after this one.
    """
    rs = state.round_state
    round_index = int(rs.round)
    dealer = int(rs.dealer)
    drawn = int(rs.last_draw)

    tehais: List[List[str]] = []
    for seat in range(4):
        tiles = rules.hand_tiles(state, seat)
        if seat == dealer and drawn >= 0:
            tiles.remove(drawn)
        tehais.append([tile_to_mjai(t) for t in tiles])

    return {
        "type": "start_kyoku",
        "bakaze": _WINDS[round_index // 4 % 4],
        "kyoku": round_index % 4 + 1,
        "honba": int(rs.honba),
        "kyotaku": int(rs.kyotaku),
        "oya": dealer,
        "dora_marker": tile_to_mjai(rules.dora_indicators(state)[0]),
        "scores": rules.scores(state),
        "tehais": tehais,
    }


def start_kyoku_events(rules: Rules, state: Any) -> List[Event]:
    """``start_kyoku`` plus the dealer's opening draw."""
    events = [start_kyoku_event(rules, state)]
    drawn = int(state.round_state.last_draw)
    if drawn >= 0:
        events.append(
            {"type": "tsumo", "actor": int(state.round_state.dealer), "pai": tile_to_mjai(drawn)}
        )
    return events


def end_kyoku_event() -> Event:
    return {"type": "end_kyoku"}


def end_game_event(scores: Sequence[int]) -> Event:
    return {"type": "end_game", "scores": [int(s) for s in scores]}


def encode_step(
    rules: Rules,
    pre_state: Any,
    action: int,
    post_state: Any,
    *,
    win: Optional[Dict[str, Any]] = None,
) -> List[Event]:
    """The mjai events one ``env.step`` produced, in order.

    ``win`` is the matching winner dict from ``view.build_win`` and is only
    read on a ``RON`` / ``TSUMO`` step, to fill in the scoring fields.
    """
    action = int(action)
    if action == rules.DUMMY:
        return []

    events: List[Event] = []
    actor = int(pre_state.current_player)

    if rules.is_discard(action) or action == rules.TSUMOGIRI:
        events.append(_dahai_event(rules, pre_state, action, actor))

    accepted = _reach_accepted_event(rules, pre_state, post_state)
    if accepted is not None:
        events.append(accepted)

    if action == rules.RIICHI:
        events.append({"type": "reach", "actor": actor})
    elif rules.is_pon(action) or rules.is_chi(action) or action == rules.OPEN_KAN:
        events.append(_call_event(rules, pre_state, action, actor))
    elif rules.is_kan(action):
        events.append(_self_kan_event(rules, pre_state, action, actor))

    is_kyuushu = rules.KYUUSHU is not None and action == rules.KYUUSHU
    if not is_kyuushu:
        events.extend(_dora_events(rules, pre_state, post_state))
        drawn = _drawn_tile(pre_state, post_state)
        if drawn is not None:
            events.append(
                {"type": "tsumo", "actor": int(post_state.current_player), "pai": tile_to_mjai(drawn)}
            )

    if action in (rules.RON, rules.TSUMO):
        events.append(_hora_event(rules, pre_state, post_state, action, win))
    elif is_kyuushu:
        events.append(
            _ryukyoku_event(
                rules, pre_state, post_state, _abortive_reason(rules, pre_state), pre_state
            )
        )
    elif _is_exhaustive_draw(rules, pre_state, post_state):
        events.append(
            _ryukyoku_event(rules, pre_state, post_state, EXHAUSTIVE_DRAW, post_state)
        )
    return events


def _dahai_event(rules: Rules, pre_state: Any, action: int, actor: int) -> Event:
    tsumogiri = action == rules.TSUMOGIRI
    tile = int(pre_state.round_state.last_draw) if tsumogiri else action
    if tile < 0:
        raise MjaiError("TSUMOGIRI with no drawn tile")
    return {"type": "dahai", "actor": actor, "pai": tile_to_mjai(tile), "tsumogiri": tsumogiri}


def _call_event(rules: Rules, pre_state: Any, action: int, actor: int) -> Event:
    target = int(pre_state.round_state.target)
    from_seat = int(pre_state.round_state.last_player)
    if rules.is_pon(action):
        tiles = rules.pon_tiles(action, target)
        kind, called, consumed = "pon", tiles[2], tiles[:2]
    elif rules.is_chi(action):
        tiles = rules.chi_tiles(action, target)
        at = tiles.index(target)
        kind, called, consumed = "chi", tiles[at], tiles[:at] + tiles[at + 1 :]
    else:
        tiles = rules.open_kan_tiles(target)
        kind, called, consumed = "daiminkan", tiles[3], tiles[:3]
    return {
        "type": kind,
        "actor": actor,
        "target": from_seat,
        "pai": tile_to_mjai(called),
        "consumed": [tile_to_mjai(t) for t in consumed],
    }


def _self_kan_event(rules: Rules, pre_state: Any, action: int, actor: int) -> Event:
    kan_type = rules.kan_tile_type(action)
    tiles = rules.closed_kan_tiles(kan_type)
    if int(pre_state.players.pon[actor, kan_type]) == 0:
        return {"type": "ankan", "actor": actor, "consumed": [tile_to_mjai(t) for t in tiles]}
    # The three tiles of the pon are already melded, so the copy still in hand
    # is the one being added -- that is what tells a red five from a black one.
    added = kan_type
    if rules.has_red and kan_type in RED_FIVE_TYPES:
        if int(rules.hand(pre_state, actor)[red_of(kan_type)]) > 0:
            added = red_of(kan_type)
    tiles.remove(added)
    return {
        "type": "kakan",
        "actor": actor,
        "pai": tile_to_mjai(added),
        "consumed": [tile_to_mjai(t) for t in tiles],
    }


def _reach_accepted_event(rules: Rules, pre_state: Any, post_state: Any) -> Optional[Event]:
    """The riichi stick is taken one step after the declaration, at the next draw
    or at the call that ate the declaration discard."""
    was = np.asarray(pre_state.players.riichi, dtype=bool)
    now = np.asarray(post_state.players.riichi, dtype=bool)
    newly = np.flatnonzero(now & ~was)
    if newly.size == 0:
        return None
    return {
        "type": "reach_accepted",
        "actor": int(newly[0]),
        "deltas": _deltas(rules, pre_state, post_state),
        "scores": rules.scores(post_state),
    }


def _dora_events(rules: Rules, pre_state: Any, post_state: Any) -> List[Event]:
    before = len(rules.dora_indicators(pre_state))
    after = rules.dora_indicators(post_state)
    if len(after) <= before:
        return []
    return [{"type": "dora", "dora_marker": tile_to_mjai(t)} for t in after[before:]]


def _drawn_tile(pre_state: Any, post_state: Any) -> Optional[int]:
    """The tile drawn during this step, or ``None``.

    ``next_deck_ix`` only ever moves on a live-wall draw and ``n_kan`` only ever
    moves when the replacement tile is taken, so the two counters separate a
    draw from a call that merely reshuffled a hand.
    """
    from_wall = int(post_state.round_state.next_deck_ix) < int(pre_state.round_state.next_deck_ix)
    from_dead_wall = int(np.asarray(post_state.players.n_kan).sum()) > int(
        np.asarray(pre_state.players.n_kan).sum()
    )
    if not (from_wall or from_dead_wall):
        return None
    tile = int(post_state.round_state.last_draw)
    return tile if tile >= 0 else None


def _hora_event(
    rules: Rules,
    pre_state: Any,
    post_state: Any,
    action: int,
    win: Optional[Dict[str, Any]],
) -> Event:
    actor = int(pre_state.current_player)
    if action == rules.TSUMO:
        target, tile = actor, int(pre_state.round_state.last_draw)
    else:
        target, tile = int(pre_state.round_state.last_player), int(pre_state.round_state.target)
    event: Event = {
        "type": "hora",
        "actor": actor,
        "target": target,
        "pai": tile_to_mjai(tile),
    }
    if win is not None:
        yakus = [[str(y["name"]), int(y["han"])] for y in win.get("yaku", [])]
        for name, count in (
            ("dora", win.get("doraHan", 0)),
            ("aka dora", win.get("akaHan", 0)),
            ("ura dora", win.get("uraHan", 0)),
        ):
            if count:
                yakus.append([name, int(count)])
        event["yakus"] = yakus
        event["fu"] = int(win.get("fu", 0))
        event["fan"] = int(win.get("han", 0))
        event["hora_points"] = int(win.get("points", 0))
    event["deltas"] = _deltas(rules, pre_state, post_state)
    event["scores"] = rules.scores(post_state)
    return event


def _ryukyoku_event(
    rules: Rules, pre_state: Any, post_state: Any, reason: str, hands_from: Any
) -> Event:
    tenpai = rules.tenpai(hands_from)
    return {
        "type": "ryukyoku",
        "reason": reason,
        "tenpais": tenpai,
        "tehais": [
            [tile_to_mjai(t) for t in rules.hand_tiles(hands_from, seat)] if tenpai[seat] else None
            for seat in range(4)
        ],
        "deltas": _deltas(rules, pre_state, post_state),
        "scores": rules.scores(post_state),
    }


def _is_exhaustive_draw(rules: Rules, pre_state: Any, post_state: Any) -> bool:
    if not bool(post_state.round_state.is_abortive_draw_normal):
        return False
    return rules.is_round_over(post_state) and not rules.is_round_over(pre_state)


def _abortive_reason(rules: Rules, pre_state: Any) -> str:
    """Why the env offered a ``KYUUSHU``-only mask.

    The env collapses every abortive draw onto the same action, so the cause has
    to be read back off the state that triggered it.
    """
    mask = np.asarray(pre_state.legal_action_mask, dtype=bool)
    if int(mask.sum()) > 1:
        return "kyushukyuhai"  # discards were still on offer: a voluntary declaration
    players = pre_state.players
    if bool(np.asarray(players.has_won).any()):
        return "sanchaho"
    if int(np.asarray(players.riichi).sum()) == 4:
        return "suuchariichi"
    n_kan = np.asarray(players.n_kan)
    if int(n_kan.sum()) >= 4 and int((n_kan > 0).sum()) >= 2:
        return "suukaikan"
    if _is_four_winds(rules, pre_state):
        return "suufonrenda"
    return "kyushukyuhai"


def _is_four_winds(rules: Rules, state: Any) -> bool:
    firsts = []
    for seat in range(4):
        river = rules.river(state, seat)
        if not river:
            return False
        firsts.append(river[0]["tile"])
    return all(27 <= t <= 30 and t == firsts[0] for t in firsts)


def _deltas(rules: Rules, pre_state: Any, post_state: Any) -> List[int]:
    """Score movement across one step.

    Read off ``score`` rather than ``rewards`` because ``rewards`` is stale on
    every step that does not end a round, and a riichi stick moves on steps that
    do not.
    """
    before = rules.scores(pre_state)
    after = rules.scores(post_state)
    return [a - b for a, b in zip(after, before)]


# ------------------------------------------------------------------ decoding


def decode_events(
    rules: Rules, events: Sequence[Event], *, env: Any = None, step_fn: Any = None
) -> List[int]:
    """The env action sequence that reproduces ``events``.

    Driven by the env rather than read straight off the log, because ``PASS`` and
    ``DUMMY`` are not written down. ``env`` / ``step_fn`` let a caller that
    already holds a traced env avoid a second ``jax.jit`` trace.
    """
    return [action for _, action in decode_steps(rules, events, env=env, step_fn=step_fn)]


def decode_steps(
    rules: Rules, events: Sequence[Event], *, env: Any = None, step_fn: Any = None
) -> Iterator[Tuple[Any, int]]:
    """Yield ``(state, action)`` for every step of the replay.

    ``state`` is the numpy-backed state the action was chosen in, so a caller
    that wants the whole state sequence gets it without replaying twice.
    """
    import jax
    import jax.numpy as jnp

    events = list(events)
    meta = mahjax_meta(events)
    if "seed" not in meta:
        raise MjaiError("start_game carries no mahjax.seed; the wall cannot be reproduced")
    round_mode = str(meta.get("round_mode", "half"))
    if env is None or step_fn is None:
        shared_env, shared_step = _replay_env(rules.env_id, round_mode)
        env = env or shared_env
        step_fn = step_fn or shared_step

    root = jax.random.PRNGKey(int(meta["seed"]))
    state = jax.device_get(env.init(jax.random.fold_in(root, 0)))

    tail = _last_replayable_index(events)
    cursor = 0
    step = 0
    while not bool(state.terminated):
        if cursor > tail:
            return
        if rules.is_round_over(state):
            action, consumed_at = rules.DUMMY, None
        else:
            action, consumed_at = _decide(rules, state, events, cursor)
        yield state, action
        if consumed_at is not None:
            cursor = consumed_at + 1
        state = jax.device_get(step_fn(state, jnp.int32(action), jax.random.fold_in(root, step + 1)))
        step += 1

    if _next_action_index(events, cursor) < len(events):
        raise MjaiError("the env ended the game with log events still unconsumed")


@lru_cache(maxsize=8)
def _replay_env(env_id: str, round_mode: str) -> Tuple[Any, Any]:
    """One traced env per shape: ``env.step`` is not jitted, and tracing it per
    replay costs seconds."""
    import jax

    env = rules_for(env_id).make_env(round_mode)
    return env, jax.jit(env.step)


def mahjax_meta(events: Sequence[Event]) -> Dict[str, Any]:
    """The ``mahjax`` extension block of the log's ``start_game`` event."""
    for event in events:
        if event.get("type") == "start_game":
            return dict(event.get("mahjax", {}))
    return {}


def _decide(
    rules: Rules, state: Any, events: Sequence[Event], cursor: int
) -> Tuple[int, Optional[int]]:
    legal = set(rules.legal_actions(state))
    at = _next_action_index(events, cursor)
    action = _event_action(rules, state, events[at]) if at < len(events) else None
    if action is not None and action in legal:
        return action, at
    # The env is asking this seat about someone else's tile and the log does not
    # show them taking it -- that silence is the PASS.
    if rules.PASS in legal:
        return rules.PASS, None
    raise MjaiError(
        f"seat {int(state.current_player)} must act but the log offers nothing legal "
        f"(next event: {events[at] if at < len(events) else None})"
    )


def _event_action(rules: Rules, state: Any, event: Event) -> Optional[int]:
    """The env action ``event`` asks of the seat on turn, or ``None``."""
    kind = event.get("type")
    if kind == "ryukyoku":
        if event.get("reason") in ABORTIVE_REASONS and rules.KYUUSHU is not None:
            return rules.KYUUSHU
        return None
    if kind not in _ACTION_TYPES or event_actor(event) != int(state.current_player):
        return None
    if kind == "reach":
        return rules.RIICHI
    if kind == "dahai":
        return _dahai_action(rules, state, event)
    if kind == "hora":
        return rules.TSUMO if event_actor(event) == int(event["target"]) else rules.RON
    if kind == "daiminkan":
        return rules.OPEN_KAN
    if kind == "ankan":
        return rules.kan_action(tile_type(mjai_to_tile(event["consumed"][0])))
    if kind == "kakan":
        return rules.kan_action(tile_type(mjai_to_tile(event["pai"])))
    if kind == "pon":
        return _call_action(rules, state, event, rules.pon_actions, rules.pon_tiles)
    if kind == "chi":
        return _call_action(
            rules, state, event, [a for a, _, _ in rules.chi_actions], rules.chi_tiles
        )
    return None


def _dahai_action(rules: Rules, state: Any, event: Event) -> int:
    tile = mjai_to_tile(event["pai"])
    if bool(event.get("tsumogiri")):
        return rules.TSUMOGIRI
    # A one-of-a-kind drawn tile has no discard action of its own: the env only
    # offers TSUMOGIRI for it.
    legal = np.asarray(state.legal_action_mask, dtype=bool)
    if not legal[tile] and int(state.round_state.last_draw) == tile:
        return rules.TSUMOGIRI
    return tile


def _call_action(rules: Rules, state: Any, event: Event, candidates: Sequence[int], tiles_of) -> int:
    """Pick the call variant whose tiles match the log (red fives differ)."""
    target = int(state.round_state.target)
    legal = np.asarray(state.legal_action_mask, dtype=bool)
    wanted = sorted([mjai_to_tile(event["pai"])] + [mjai_to_tile(t) for t in event["consumed"]])
    fallback = None
    for action in candidates:
        if not legal[action]:
            continue
        if sorted(tiles_of(action, target)) == wanted:
            return action
        if fallback is None:
            fallback = action
    if fallback is None:
        raise MjaiError(f"no legal call matches {event}")
    return fallback


def _is_action_event(event: Event) -> bool:
    kind = event.get("type")
    if kind == "ryukyoku":
        return event.get("reason") in ABORTIVE_REASONS
    return kind in _ACTION_TYPES


def _next_action_index(events: Sequence[Event], cursor: int) -> int:
    while cursor < len(events) and not _is_action_event(events[cursor]):
        cursor += 1
    return cursor


def _last_replayable_index(events: Sequence[Event]) -> int:
    """Index of the last event that keeps the replay going, or ``-1``.

    PASS and the four DUMMYs have no event of their own, so nothing runs out to
    stop a record that was cut short mid-game (``mahjax.complete: false``) --
    this does. ``end_game`` counts, because reaching it is what the closing
    DUMMYs are for.
    """
    last = -1
    for i, event in enumerate(events):
        if _is_action_event(event) or event.get("type") == "end_game":
            last = i
    return last


__all__ = [
    "ABORTIVE_REASONS",
    "EXHAUSTIVE_DRAW",
    "Event",
    "MjaiError",
    "decode_events",
    "decode_steps",
    "encode_step",
    "end_game_event",
    "end_kyoku_event",
    "event_actor",
    "mahjax_meta",
    "mjai_to_tile",
    "start_kyoku_event",
    "start_kyoku_events",
    "tile_to_mjai",
]
