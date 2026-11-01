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

"""Tests for showing states handed in from Python (``mahjax.ui.show``)."""

from __future__ import annotations

import base64
import functools
import gzip
import html
import json
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mahjax.ui import save_html, show, standalone, view
from mahjax.ui.agents import _rule_based
from mahjax.ui.record import StateReplay
from mahjax.ui.rules import rules_for, shared_env, shared_step_fn


@functools.lru_cache(maxsize=None)
def _rollout() -> List[Any]:
    """The first round of a rule-based east game and the opening of the second, every state kept.

    The env and its traced step are the ones the other UI tests use.
    """
    env, step, act = shared_env("red_4p", "east"), shared_step_fn("red_4p", "east"), _rule_based()
    key = jax.random.PRNGKey(0)
    state = jax.device_get(env.init(key))
    first_deal = (int(state.round_state.round), int(state.round_state.honba))
    states = [state]
    for _ in range(400):
        key, k1, k2 = jax.random.split(key, 3)
        state = jax.device_get(step(state, act(state, k1), k2))
        states.append(state)
        deal = (int(state.round_state.round), int(state.round_state.honba))
        if deal != first_deal and int(state.round_state.round_step) > 0:
            break
    return states


def _as_auto(states: List[Any]) -> List[Any]:
    """The states an ``round_end="auto"`` env shows: it deals the next round in the
    step that ends one, so the round's end and the sharing steps are never seen."""
    rules = rules_for("red_4p")
    return [state for state in states if not rules.is_round_over(state)]


def test_a_confirm_game_names_each_action_and_shows_the_result() -> None:
    # The game is made to end in the second round, so only that round's result ends it.
    states = _rollout()
    replay = StateReplay(states + [states[-1].replace(terminated=np.bool_(True))])
    assert replay.rules.env_id == "red_4p"
    assert sum(1 for step in replay._steps if step["event"]) > 0
    first, last = sorted(replay._results)
    assert replay.frame(first, viewpoint=None, show_all=True)["result"] is not None
    assert [replay._results[i]["gameOver"] for i in (first, last)] == [False, True]
    # Two rounds tile the frames; the sharing steps after a result are not frames.
    assert [span.start for span in replay.rounds] == [0, first + 1]
    assert replay.rounds[0].end == first and replay.rounds[-1].end == replay.total - 1 == last


def test_an_auto_game_plays_back_every_state() -> None:
    states = _as_auto(_rollout())
    replay = StateReplay(states)
    assert replay.total == len(states)
    assert len(replay.rounds) == 2 and not replay._results
    assert len(replay.frames(0, replay.total)) == replay.total


def test_a_stacked_state_is_a_sequence() -> None:
    states = _rollout()[:10]
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
    assert StateReplay(stacked).total == StateReplay(states).total
    assert StateReplay(states[0]).total == 1


def test_a_batch_of_sequences_is_refused() -> None:
    states = _rollout()[:4]
    batched = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *states)
    trajectory = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), batched, batched)
    with pytest.raises(ValueError, match="unbatched"):
        StateReplay(trajectory)


@functools.lru_cache(maxsize=None)
def _three_player_game() -> List[Any]:
    """A rule-based 3-player east game, every state kept."""
    env, step, act = shared_env("red_3p", "east"), shared_step_fn("red_3p", "east"), _rule_based()
    key = jax.random.PRNGKey(1)
    state = jax.device_get(env.init(key))
    states = [state]
    while not bool(state.terminated):
        key, k1, k2 = jax.random.split(key, 3)
        state = jax.device_get(step(state, act(state, k1), k2))
        states.append(state)
    return states


def test_a_three_player_game_plays_back_with_its_norths() -> None:
    states = _three_player_game()
    replay = StateReplay(states)
    assert replay.rules.env_id == "red_3p"
    assert [seat.name for seat in replay.seats] == ["Player 0", "Player 1", "Player 2"]
    # A wind lasts three rounds.
    labels = [span.label for span in replay.rounds]
    assert "東3局" in labels and not any("4局" in label for label in labels)
    frames = replay.frames(0, replay.total)
    assert all(len(frame["seats"]) == 3 for frame in frames)
    assert any(frame["step"]["event"] and frame["step"]["event"]["kind"] == "kita" for frame in frames)
    assert any(seat["kita"] for frame in frames for seat in frame["seats"])
    # Every win's breakdown adds up to the han the env paid, the norths included.
    winners = [w for result in replay._results.values() for w in result["winners"]]
    assert any(w["kitaHan"] for w in winners)
    for w in winners:
        if not w["yakuman"]:
            parts = sum(y["han"] for y in w["yaku"]) + w["doraHan"] + w["akaHan"] + w["uraHan"] + w["kitaHan"]
            assert parts == w["han"]
    # A north can be set aside from the buttons too.
    rules = replay.rules
    offered = next(s for s in states if bool(np.asarray(s.legal_action_mask)[rules.KITA]))
    prompt = view.build_prompt(rules, offered, int(offered.current_player))
    kita = [o for o in prompt["options"] if o["kind"] == "kita"]
    assert [(o["label"], o["labelEn"]) for o in kita] == [("北抜き", "Kita")]


def _carried(page: str) -> Dict[str, Any]:
    """The data a saved page carries, unpacked as its script unpacks it."""
    (packed,) = re.findall(r'<script id="mahjax-data" type="application/octet-stream">([^<]*)</script>', page)
    return json.loads(gzip.decompress(base64.b64decode(packed)))


def test_outside_a_notebook_show_saves_a_page_that_carries_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    show(_rollout())
    (path,) = tmp_path.glob("mahjax-*.html")
    assert str(path) in capsys.readouterr().out
    page = path.read_text(encoding="utf-8")
    assert "/static/" not in page  # the script and the styles are inline
    data = _carried(page)
    replay = StateReplay(_rollout())
    assert data["frames"] == json.loads(json.dumps(replay.frames(0, replay.total)))
    assert [seat["name"] for seat in data["replay"]["seats"]] == ["Player 0", "Player 1", "Player 2", "Player 3"]
    assert {name: len(files) for name, files in data["tiles"].items()} == {"ja": 40, "en": 40}


def test_save_html_writes_the_page(tmp_path: Path) -> None:
    save_html(_rollout()[:3], tmp_path / "game.html")
    assert _carried((tmp_path / "game.html").read_text(encoding="utf-8"))["replay"]["total"] == 3


def test_a_notebook_gets_the_page_in_an_iframe() -> None:
    page = "<!DOCTYPE html><p class=\"x\">a & 'b' < c</p>"
    found = re.fullmatch(r'<iframe srcdoc="([^"]*)" style="([^"]*)"></iframe>', standalone._iframe(page, 500))
    assert html.unescape(found.group(1)) == page
    assert "height: 500px" in found.group(2)


def test_every_fixed_word_on_the_page_is_in_every_language() -> None:
    page = (standalone.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    tags = [dict(re.findall(r'([\w-]+)="([^"]*)"', tag)) for tag in re.findall(r"<\w[^>]*>", page)]
    translated = [attrs for attrs in tags if any(name.startswith("data-en") for name in attrs)]
    assert len(translated) > 40
    for attrs in translated:
        for name in [name for name in attrs if name.startswith("data-en")]:
            assert attrs.get(name.replace("data-en", "data-ko")), attrs
            assert attrs.get(name.replace("data-en", "data-zh")), attrs
    menu = re.search(r'<select id="optLang">(.*?)</select>', page, re.S).group(1)
    assert re.findall(r'<option value="(\w+)"', menu) == ["ja", "en", "ko", "zh"]
