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

"""Tests for ``mahjax.ui.view``.

Env construction and the first ``jit`` trace cost several seconds each, so both
envs are rolled out once in a module-scoped fixture and every test reads from
that recording.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mahjax.ui.rules import Rules, rules_for, tile_type
from mahjax.ui.view import (
    SeatInfo,
    _judge_win,
    build_prompt,
    build_round_result,
    build_view,
    build_win,
    describe_action,
    final_standings,
)

ENV_IDS = ["red_mahjong", "no_red_mahjong"]
NUM_GAMES = 6
VIEW_EVERY = 25  # views are the expensive part; sample them
SEATS = [SeatInfo(name=n, kind=k) for n, k in
         [("You", "human"), ("Bot A", "agent"), ("Bot B", "agent"), ("Bot C", "agent")]]
ALL_REVEAL = [True] * 4


@dataclass
class Rollout:
    rules: Rules
    env: Any
    init: Any
    step: Any
    prompts: int = 0
    forced: Dict[int, int] = field(default_factory=dict)
    option_kinds: Dict[str, int] = field(default_factory=dict)
    kan_labels: Dict[str, int] = field(default_factory=dict)
    prompt_failures: List[str] = field(default_factory=list)
    wins: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Dict[str, Any]] = field(default_factory=list)
    views: List[Dict[str, Any]] = field(default_factory=list)
    melds: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    uneven: Any = None
    kan_dora_wins: int = 0
    fallbacks: int = 0


def _score(rules: Rules, fan: int, fu: int) -> int:
    if rules.has_red:
        from mahjax.red_mahjong.yaku import Yaku
    else:
        from mahjax.no_red_mahjong.yaku import Yaku
    return int(Yaku.score(jnp.int32(fan), jnp.int32(fu)))


def _ceil100(value: int) -> int:
    return -(-value // 100)


def _winner_gain(rules: Rules, pre: Any, seat: int, is_ron: bool, han: int, fu: int) -> List[int]:
    """What the env should hand the winner, in hundreds; pao gives a 2nd answer."""
    rs = pre.round_state
    dealer, honba, kyotaku = int(rs.dealer), int(rs.honba), int(rs.kyotaku)
    basic = _score(rules, han, fu)
    full = _ceil100(basic * (6 if seat == dealer else 4))
    if is_ron:
        # Honba is paid once per discard, so a double ron's second winner gets none.
        paid = 0 if bool(np.asarray(pre.players.has_won).any()) else honba * 3
        return [full + paid + 10 * kyotaku]
    s1, s2 = _ceil100(basic), _ceil100(basic * 2)
    normal = (s2 * 3 if seat == dealer else s1 * 2 + s2) + 3 * honba
    return [normal + 10 * kyotaku, full + 3 * honba + 10 * kyotaku]


def _check_prompt(roll: Rollout, state: Any) -> Optional[Dict[str, Any]]:
    rules = roll.rules
    seat = int(state.current_player)
    legal = set(rules.legal_actions(state))
    prompt = build_prompt(rules, state, seat)
    if prompt is None:
        if not rules.is_round_over(state) and legal:
            key = legal.pop() if len(legal) == 1 else -1
            roll.forced[key] = roll.forced.get(key, 0) + 1
        return None
    roll.prompts += 1
    offered = list(prompt["discardable"])
    if prompt["tsumogiri"] is not None:
        offered.append(prompt["tsumogiri"])
    offered += [o["action"] for o in prompt["options"]]
    if len(offered) != len(set(offered)):
        roll.prompt_failures.append(f"duplicate action in prompt: {offered}")
    if set(offered) != legal:
        roll.prompt_failures.append(f"prompt {sorted(offered)} != legal {sorted(legal)}")
    want_claim = int(state.round_state.target) >= 0
    if (prompt["kind"] == "claim") != want_claim:
        roll.prompt_failures.append(f"kind {prompt['kind']} for target {int(state.round_state.target)}")
    if want_claim and prompt["target"]["tile"] != int(state.round_state.target):
        roll.prompt_failures.append("claim target tile mismatch")
    for option in prompt["options"]:
        roll.option_kinds[option["kind"]] = roll.option_kinds.get(option["kind"], 0) + 1
        if option["kind"] in ("pon", "chi", "kan") and not option["tiles"]:
            roll.prompt_failures.append(f"{option['kind']} option without tiles")
        if option["kind"] in ("riichi", "tsumo", "ron", "kyuushu", "pass") and option["tiles"]:
            roll.prompt_failures.append(f"{option['kind']} option carries tiles")
        if option["kind"] == "kan":
            _check_kan_label(roll, state, seat, option)
    return prompt


def _check_kan_label(roll: Rollout, state: Any, seat: int, option: Dict[str, Any]) -> None:
    """The three kan words, checked against the hand rather than against ``pon``."""
    rules = roll.rules
    label = option["label"].split()[0]
    roll.kan_labels[label] = roll.kan_labels.get(label, 0) + 1
    tile = rules.kan_tile_type(option["action"]) if option["action"] != rules.OPEN_KAN else None
    if option["action"] == rules.OPEN_KAN:
        if label != "明槓":
            roll.prompt_failures.append(f"open kan labelled {label}")
        return
    held = int(sum(n for t, n in enumerate(rules.hand(state, seat)) if tile_type(t) == tile))
    ponned = any(m["kind"] == "pon" and tile_type(m["tiles"][0]) == tile
                 for m in rules.melds(state, seat))
    want = "加槓" if ponned else "暗槓"
    if label != want:
        roll.prompt_failures.append(f"self kan of {tile} labelled {label}, held {held}")
    if want == "暗槓" and held != 4:
        roll.prompt_failures.append(f"closed kan of {tile} with {held} in hand")
    if want == "加槓" and held < 1:
        roll.prompt_failures.append(f"added kan of {tile} with none in hand")


def _record_meld(roll: Rollout, pre: Any, action: int, post: Any) -> None:
    event = describe_action(roll.rules, pre, action)
    if not event["kind"].startswith(("pon", "chi", "kan")):
        return
    if event["kind"] in roll.melds:
        return
    seat = event["seat"]
    before = roll.rules.melds(pre, seat)
    after = roll.rules.melds(post, seat)
    roll.melds[event["kind"]] = {
        "event": event,
        "before": before,
        "after": after,
        "discarder": int(pre.round_state.last_player),
        "river": roll.rules.river(post, int(pre.round_state.last_player)),
    }


def _check_kan_dora_fallback(roll: Rollout, pre: Any, seat: int, is_ron: bool) -> None:
    """A rinshan win is cached before its own kan dora turns over (view.py).

    That case is far too rare to hit by rolling out, so instead every win with a
    kan dora on the table is asked what it would look like had the env cached
    the smaller total: the reduced snapshot has to answer, and add up.
    """
    if int(pre.round_state.n_kan_doras) <= 0:
        return
    roll.kan_dora_wins += 1
    cached = int(pre.players.fan[seat, 0])
    judged = _judge_win(roll.rules, pre, seat, is_ron, cached)
    if judged.fu != int(pre.players.fu[seat, 0]):
        roll.prompt_failures.append(f"judged fu {judged.fu} != cached {int(pre.players.fu[seat, 0])}")
    if judged.candidates[0] != cached:
        roll.prompt_failures.append(f"live dora snapshot gives {judged.candidates} not {cached}")
        return
    reduced_fan = judged.candidates[1]
    if reduced_fan == cached:
        return  # that indicator was not a dora for this hand; nothing to choose
    roll.fallbacks += 1
    reduced = _judge_win(roll.rules, pre, seat, is_ron, reduced_fan)
    parts = reduced.hand_han + reduced.aka_han + reduced.dora_han + reduced.ura_han
    if reduced.fan != reduced_fan or parts != reduced_fan:
        roll.prompt_failures.append(f"reduced snapshot {reduced} != {reduced_fan}")
    if len(reduced.dora) != len(judged.dora) - 1:
        roll.prompt_failures.append("reduced snapshot kept every dora indicator")


def _run(env_id: str) -> Rollout:
    rules = rules_for(env_id)
    env = rules.make_env("half")
    if rules.has_red:
        from mahjax.red_mahjong.players import rule_based_player
    else:
        from mahjax.no_red_mahjong.players import rule_based_player
    roll = Rollout(rules=rules, env=env, init=jax.jit(env.init), step=jax.jit(env.step))
    act = jax.jit(rule_based_player)

    for game in range(NUM_GAMES):
        key = jax.random.PRNGKey(game)
        live = roll.init(jax.random.fold_in(key, 0))
        state = jax.device_get(live)
        score_start = rules.scores(state)
        round_wins: List[Dict[str, Any]] = []
        for i in range(3000):
            if bool(state.terminated):
                break
            legal = rules.legal_actions(state)
            if not legal:
                break
            prompt = _check_prompt(roll, state)
            if i % VIEW_EVERY == 0:
                roll.views.append(
                    build_view(rules, state, SEATS, reveal=ALL_REVEAL, prompt=prompt)
                )
            action = int(act(live, jax.random.fold_in(key, 1000 + i)))
            if game % 2 == 0:
                # Kans are what the rule-based player almost never does, and they
                # drive rinshan draws, kan dora and the added/closed kan words.
                kans = [a for a in legal if rules.is_kan(a) or a == rules.OPEN_KAN]
                if kans:
                    action = kans[0]
            if action not in legal:
                action = legal[int(np.random.RandomState(game * 9973 + i).randint(len(legal)))]
            pre = state
            live = roll.step(live, jnp.int32(action), jax.random.fold_in(key, i + 1))
            state = jax.device_get(live)
            _record_meld(roll, pre, action, state)

            if action in (rules.TSUMO, rules.RON):
                seat = int(pre.current_player)
                _check_kan_dora_fallback(roll, pre, seat, action == rules.RON)
                win = build_win(rules, pre, state, seat, action == rules.RON)
                round_wins.append(win)
                roll.wins.append(
                    {
                        "win": win,
                        "is_ron": action == rules.RON,
                        "seat": seat,
                        "cached_fan": int(pre.players.fan[seat, 0]),
                        "cached_fu": int(pre.players.fu[seat, 0]),
                        "gain": _winner_gain(
                            rules, pre, seat, action == rules.RON, win["han"], win["fu"]
                        ),
                    }
                )
            abortive = action == rules.KYUUSHU
            if rules.is_round_over(state) or abortive:
                end = pre if abortive else state
                kind = "abortive" if abortive else ("draw" if not round_wins else
                                                    ("ron" if round_wins[0]["from"] is not None else "tsumo"))
                result = build_round_result(
                    rules,
                    type=kind,
                    reason="kyuushu" if abortive else None,
                    state=end,
                    winners=round_wins,
                    score_start=score_start,
                    game_over=bool(state.terminated),
                )
                roll.results.append({"result": result, "start": list(score_start)})
                if roll.uneven is None and len(set(rules.scores(end))) == 4:
                    roll.uneven = end
                roll.views.append(
                    build_view(rules, end, SEATS, reveal=ALL_REVEAL, result=result,
                               step={"index": i, "total": 0, "roundIndex": int(end.round_state.round),
                                     "event": describe_action(rules, pre, action)})
                )
                round_wins = []
                score_start = rules.scores(state)
    return roll


@pytest.fixture(scope="module", params=ENV_IDS)
def roll(request) -> Rollout:
    return _run(request.param)


def test_prompt_is_a_partition_of_the_legal_mask(roll: Rollout) -> None:
    assert roll.prompt_failures == []
    assert roll.prompts > 500
    # The rollout has to have exercised the interesting option kinds.
    for kind in ("pon", "chi", "kan", "riichi", "pass", "tsumo", "ron"):
        assert roll.option_kinds.get(kind, 0) > 0, f"{kind} never offered"
    assert {"暗槓", "明槓"} <= set(roll.kan_labels), roll.kan_labels


def test_prompt_only_skipped_when_forced(roll: Rollout) -> None:
    # build_prompt returns None mid-round only for the single forced actions
    # the server plays by itself.
    forced = {a for a in roll.forced if a >= 0}
    assert forced <= {roll.rules.TSUMOGIRI, roll.rules.KYUUSHU, roll.rules.DUMMY}
    assert roll.forced.get(-1, 0) == 0


def test_kan_dora_snapshot_is_the_one_the_env_scored(roll: Rollout) -> None:
    assert roll.prompt_failures == []
    assert roll.kan_dora_wins > 0
    assert roll.fallbacks > 0


def test_yaku_han_reconciles_with_the_env(roll: Rollout) -> None:
    assert len(roll.wins) > 20
    for record in roll.wins:
        win = record["win"]
        total = sum(y["han"] for y in win["yaku"])
        where = f"{roll.rules.env_id} seat {record['seat']} {win}"
        if win["yakuman"]:
            assert total == win["han"], where
            assert win["fu"] == 0, where
        else:
            assert total + win["doraHan"] + win["akaHan"] + win["uraHan"] == win["han"], where
            assert win["doraHan"] >= 0 and win["akaHan"] >= 0 and win["uraHan"] >= 0, where
            assert win["fu"] > 0, where
        if not roll.rules.has_red:
            assert win["akaHan"] == 0, where
        if win["uraHan"]:
            assert win["uraDora"], where


def test_ura_dora_is_only_shown_to_a_riichi_winner(roll: Rollout) -> None:
    """The under-dora is face down unless the winner declared riichi, so a hand
    that did not must not report a single ura tile or a single ura han."""
    plain = 0
    for record in roll.wins:
        win = record["win"]
        declared = {"立直", "ダブル立直"} & {y["name"] for y in win["yaku"]}
        where = f"{roll.rules.env_id} seat {record['seat']} {win['yaku']}"
        if declared:
            continue
        plain += 1
        assert win["uraDora"] == [], where
        assert win["uraHan"] == 0, where
    assert plain > 0, "no win without riichi in this rollout"


def test_win_han_and_fu_are_what_the_env_paid(roll: Rollout) -> None:
    for record in roll.wins:
        win = record["win"]
        assert win["points"] in [g * 100 for g in record["gain"]], (
            f"{roll.rules.env_id} han={win['han']} fu={win['fu']} "
            f"points={win['points']} expected={record['gain']}"
        )


def test_views_and_results_are_json(roll: Rollout) -> None:
    assert roll.views and roll.results
    seat_keys = {"name", "kind", "wind", "score", "riichi", "hand", "handCount",
                 "drawn", "melds", "river"}
    prompted = ura_shown = 0
    for view in roll.views:
        text = json.dumps(view, ensure_ascii=False)
        assert json.loads(text)["env"] == roll.rules.env_id
        assert set(view) == {"env", "round", "dealer", "current", "seats",
                             "lastDiscard", "prompt", "result", "gameOver", "step"}
        assert len(view["seats"]) == 4
        assert all(set(s) == seat_keys for s in view["seats"])
        assert len(view["round"]["dora"]) == 5
        prompted += view["prompt"] is not None
        ura_shown += view["round"]["uraDora"] is not None
    assert prompted > 0 and ura_shown > 0
    winner_keys = {"seat", "from", "hand", "melds", "winningTile", "yaku", "doraHan",
                   "akaHan", "uraHan", "han", "fu", "yakuman", "dora", "uraDora", "points"}
    assert all(set(w["win"]) == winner_keys for w in roll.wins)
    for record in roll.results:
        result = record["result"]
        json.dumps(result, ensure_ascii=False)
        assert set(result) == {
            "type", "reason", "abortSeat", "round", "winners", "tenpai",
            "nagashiMangan", "deltas", "scores", "gameOver", "final",
        }
        assert [s - d for s, d in zip(result["scores"], result["deltas"])] == record["start"]


def test_round_result_fields_follow_the_kind(roll: Rollout) -> None:
    kinds = {record["result"]["type"] for record in roll.results}
    assert {"draw", "ron", "tsumo"} <= kinds
    for record in roll.results:
        result = record["result"]
        if result["type"] == "draw":
            assert result["tenpai"] is not None and len(result["tenpai"]) == 4
            assert result["winners"] == []
        else:
            assert result["tenpai"] is None
            assert result["nagashiMangan"] is None
        assert (result["final"] is not None) == result["gameOver"]

    state = jax.device_get(roll.init(jax.random.fold_in(jax.random.PRNGKey(5), 0)))
    over = build_round_result(
        roll.rules, type="draw", reason=None, state=state, winners=[],
        score_start=[25000] * 4, game_over=True,
    )
    json.dumps(over, ensure_ascii=False)
    assert [f["seat"] for f in over["final"]] == [0, 1, 2, 3]
    assert over["final"] == final_standings(roll.rules, state)


def test_melds_reach_the_view_with_the_right_shape(roll: Rollout) -> None:
    seen = roll.melds
    assert {"pon", "chi", "kan_closed", "kan_open"} <= set(seen)
    called_index = {"pon": 2, "chi": None, "kan_open": 3, "kan_added": 3, "kan_closed": None}
    for kind, record in seen.items():
        after, before = record["after"], record["before"]
        view_kind = {"pon": "pon", "chi": "chi", "kan_open": "kan_open",
                     "kan_added": "kan_added", "kan_closed": "kan_closed"}[kind]
        if kind == "kan_added":
            assert len(after) == len(before)  # the added kan replaces its pon
            meld = next(m for m in after if m["kind"] == "kan_added")
        else:
            assert len(after) == len(before) + 1
            meld = after[-1]
        assert meld["kind"] == view_kind
        assert meld["tiles"] == record["event"]["tiles"]
        if kind == "kan_closed":
            assert meld["from"] is None and meld["called"] is None
        elif kind != "kan_added":
            assert meld["from"] == record["discarder"]
            assert record["river"][-1]["called"] is True
        if called_index[kind] is not None:
            assert meld["called"] == called_index[kind]


def test_initial_and_first_discard_snapshot(roll: Rollout) -> None:
    rules = roll.rules
    state = jax.device_get(roll.init(jax.random.fold_in(jax.random.PRNGKey(7), 0)))
    dealer = int(state.round_state.dealer)
    drawn = int(state.round_state.last_draw)
    view = build_view(rules, state, SEATS, reveal=[i == dealer for i in range(4)])

    assert view["dealer"] == dealer == int(view["current"])
    assert view["lastDiscard"] is None
    assert sum(1 for d in view["round"]["dora"] if d is not None) == 1
    assert view["round"]["uraDora"] is None
    assert view["round"]["honba"] == 0 and view["round"]["kyotaku"] == 0
    for seat, entry in enumerate(view["seats"]):
        assert entry["score"] == 25000
        assert entry["melds"] == [] and entry["river"] == []
        assert entry["riichi"] == "none"
        if seat == dealer:
            assert entry["handCount"] == 14 and len(entry["hand"]) == 14
            assert entry["drawn"] == drawn and drawn in entry["hand"]
        else:
            assert entry["handCount"] == 13 and entry["hand"] is None
            assert entry["drawn"] is None

    live = roll.init(jax.random.fold_in(jax.random.PRNGKey(7), 0))
    live = roll.step(live, jnp.int32(rules.TSUMOGIRI), jax.random.fold_in(jax.random.PRNGKey(7), 1))
    after = jax.device_get(live)
    view = build_view(rules, after, SEATS, reveal=[True] * 4)
    river = view["seats"][dealer]["river"]
    assert river == [{"tile": drawn, "tsumogiri": True, "riichi": False, "called": False}]
    assert view["lastDiscard"] == {"seat": dealer, "index": 0}
    assert view["seats"][dealer]["handCount"] == 13
    assert view["seats"][dealer]["drawn"] is None
    assert view["round"]["remaining"] == rules.remaining_draws(after)


def test_final_standings_keep_points_and_uma_apart(roll: Rollout) -> None:
    rules = roll.rules
    state = jax.device_get(roll.init(jax.random.fold_in(jax.random.PRNGKey(11), 0)))
    standings = final_standings(rules, state)
    assert [s["seat"] for s in standings] == [0, 1, 2, 3]
    assert [s["rank"] for s in standings] == [1, 2, 3, 4]
    # Everyone starts level, so seat order breaks the tie and nothing moves the
    # points; the rank bonus is reported on its own, in the env's own units.
    assert [s["score"] for s in standings] == [25000] * 4
    assert [s["uma"] for s in standings] == np.asarray(state.round_state.order_points).tolist()

    end = roll.uneven
    assert end is not None
    standings = final_standings(rules, end)
    scores = rules.scores(end)
    assert [s["seat"] for s in standings] == [0, 1, 2, 3]
    by_rank = sorted(standings, key=lambda s: s["rank"])
    assert [s["rank"] for s in by_rank] == [1, 2, 3, 4]
    assert [scores[s["seat"]] for s in by_rank] == sorted(scores, reverse=True)
    # Only the riichi sticks move the points; uma is never added into them.
    sticks = 1000 * int(end.round_state.kyotaku)
    assert sum(s["score"] for s in standings) == sum(scores) + sticks
    assert sum(s["uma"] for s in standings) == 0
    assert by_rank[0]["uma"] == max(s["uma"] for s in standings)


def test_describe_action_names_the_tiles(roll: Rollout) -> None:
    rules = roll.rules
    state = jax.device_get(roll.init(jax.random.fold_in(jax.random.PRNGKey(3), 0)))
    drawn = int(state.round_state.last_draw)
    seat = int(state.current_player)
    assert describe_action(rules, state, rules.TSUMOGIRI) == {
        "seat": seat, "kind": "discard", "tiles": [drawn]
    }
    other = next(a for a in rules.legal_actions(state) if rules.is_discard(a) and a != drawn)
    assert describe_action(rules, state, other) == {
        "seat": seat, "kind": "discard", "tiles": [other]
    }
