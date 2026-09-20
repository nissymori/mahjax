"""Saving a game as an mjai log and replaying it back."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from mahjax.ui import record as record_mod
from mahjax.ui.agents import AgentRegistry
from mahjax.ui.match import Match, MatchConfig
from mahjax.ui.record import Replay
from mahjax.ui.rules import rules_for

ENV_IDS = ("red_mahjong", "no_red_mahjong")


@pytest.fixture(scope="module")
def registry() -> AgentRegistry:
    return AgentRegistry()


@pytest.fixture(autouse=True)
def _records_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setenv("MAHJAX_RECORD_DIR", str(tmp_path / "records"))


def _self_play(registry: AgentRegistry, env_id: str, seed: int, round_mode: str = "east") -> Match:
    config = MatchConfig(
        env_id=env_id,
        round_mode=round_mode,
        seed=seed,
        human_seat=None,
        save_record=False,
    )
    match = Match(config, registry.default_for(env_id))
    match.play_out()
    return match


#: A red east game in which the agent below is offered nine terminals and takes it.
NINE_TERMINALS_SEED = 14


def _nine_terminals_game(registry: AgentRegistry) -> Match:
    """Self-play with a random agent that always declares nine terminals when offered.

    A legal declaration needs nine terminal and honor types on a player's first
    draw, so a plain random agent almost never makes one, and the bundled
    heuristic never does.
    """
    random_agent = registry.get("random")
    kyuushu = rules_for("red_mahjong").KYUUSHU

    def act(state, key):
        mask = state.legal_action_mask
        if bool(mask[kyuushu]) and int(mask.sum()) > 1:
            return kyuushu
        return random_agent.act(state, key)

    config = MatchConfig(
        env_id="red_mahjong", round_mode="east", seed=NINE_TERMINALS_SEED, human_seat=None, save_record=False
    )
    match = Match(config, replace(random_agent, act=act))
    match.play_out()
    return match


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_record_round_trips_through_the_log(registry: AgentRegistry, env_id: str) -> None:
    match = _self_play(registry, env_id, seed=21)
    record_id = record_mod.save_events(match.events)
    replay = record_mod.load_replay(record_id)

    assert replay.rules.env_id == env_id
    assert replay.total > 0
    assert replay.rounds
    # Round spans must tile the frame list without gaps or overlaps.
    assert replay.rounds[0].start == 0
    for previous, nxt in zip(replay.rounds, replay.rounds[1:]):
        assert nxt.start == previous.end + 1
    assert replay.rounds[-1].end == replay.total - 1


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_replay_reaches_the_same_scores(registry: AgentRegistry, env_id: str) -> None:
    match = _self_play(registry, env_id, seed=34)
    replay = Replay(match.events)
    played = match.rules.scores(match.state)
    last_result = None
    for index in sorted(replay._results):  # noqa: SLF001 - the result map is the point
        last_result = replay._results[index]
    assert last_result is not None
    assert last_result["gameOver"] is True
    assert last_result["scores"] == match.result["scores"]
    assert sum(played) == sum(last_result["scores"])


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_every_replay_frame_is_serialisable(registry: AgentRegistry, env_id: str) -> None:
    match = _self_play(registry, env_id, seed=7, round_mode="single")
    replay = Replay(match.events)
    for index in range(replay.total):
        frame = replay.frame(index, viewpoint=2, show_all=False)
        json.dumps(frame)
        assert frame["step"]["index"] == index
        assert frame["step"]["total"] == replay.total
    # A viewpoint-only replay hides the other three hands until the result.
    opening = replay.frame(0, viewpoint=2, show_all=False)
    assert opening["seats"][2]["hand"] is not None
    assert all(opening["seats"][i]["hand"] is None for i in (0, 1, 3))
    everyone = replay.frame(0, viewpoint=2, show_all=True)
    assert all(seat["hand"] is not None for seat in everyone["seats"])


def test_listing_summarises_saved_records(registry: AgentRegistry) -> None:
    match = _self_play(registry, "red_mahjong", seed=15, round_mode="single")
    record_id = record_mod.save_events(match.events)
    listed = record_mod.list_records()
    assert [r["id"] for r in listed] == [record_id]
    entry = listed[0]
    assert entry["env"] == "red_mahjong"
    assert entry["roundMode"] == "single"
    assert entry["complete"] is True
    assert len(entry["players"]) == 4

    record_mod.delete_record(record_id)
    assert record_mod.list_records() == []


def test_a_chosen_nine_terminals_still_breaks_the_record(registry: AgentRegistry) -> None:
    """Nine terminals declared by choice ends the round, so the log has to carry
    a boundary there; without one the replay drifts a round out of step.
    """
    match = _nine_terminals_game(registry)
    kinds = [e["type"] for e in match.events]
    assert kinds.count("start_kyoku") == kinds.count("end_kyoku")
    reasons = [e.get("reason") for e in match.events if e["type"] == "ryukyoku"]
    assert "kyushukyuhai" in reasons, "this seed no longer produces a chosen abort"
    for i, event in enumerate(match.events):
        if event["type"] in ("hora", "ryukyoku"):
            rest = [e["type"] for e in match.events[i + 1 :]]
            assert "end_kyoku" in rest, f"no round boundary after {event['type']}"
    Replay(match.events)  # would raise if the two streams disagreed


def test_a_tampered_log_is_refused(registry: AgentRegistry) -> None:
    """A record that no longer matches the env must fail loudly, not quietly."""
    match = _self_play(registry, "red_mahjong", seed=44, round_mode="single")
    events = [dict(e) for e in match.events]
    for event in events:
        if event.get("type") == "tsumo":
            event["pai"] = "1m" if event["pai"] != "1m" else "9s"
            break
    else:
        pytest.skip("no tsumo event to tamper with")
    with pytest.raises(record_mod.RecordError):
        Replay(events)


def test_a_replay_keeps_the_name_of_the_abortive_draw(registry: AgentRegistry) -> None:
    """The overlay a replay shows must say what the live board said.

    The env reports an abortive draw only as a mask, so the reason has to be
    read off the board before the step is applied; the replay used to leave it
    blank and label every abortive draw with the generic title.
    """
    match = _nine_terminals_game(registry)

    replay = Replay(match.events)
    abortive = [r for r in replay._results.values() if r["type"] == "abortive"]  # noqa: SLF001
    assert abortive, "this seed no longer produces an abortive draw"
    assert any(r["reason"] == "kyuushu" for r in abortive)
    for result in abortive:
        assert result["reason"] is not None
        if result["reason"] == "kyuushu":
            assert result["abortSeat"] is not None
        else:
            assert result["abortSeat"] is None


def test_a_voided_round_lists_no_winners(registry: AgentRegistry) -> None:
    """An abortive draw is a draw. The log still carries the wins that were
    declared before the round was voided, and a replay must not put them up as
    results the live board never showed."""
    match = _nine_terminals_game(registry)

    replay = Replay(match.events)
    abortive = [r for r in replay._results.values() if r["type"] == "abortive"]  # noqa: SLF001
    assert abortive, "this seed no longer produces an abortive draw"
    for result in abortive:
        assert result["winners"] == []



def test_a_record_saved_on_a_result_still_replays(registry: AgentRegistry) -> None:
    """Quitting while a round's result is on screen leaves a log that ends on
    that result. The replay stops there instead of dealing a round the log never
    reached."""
    match = _self_play(registry, "red_mahjong", seed=1)
    cut = next(i for i, e in enumerate(match.events) if e["type"] == "end_kyoku")
    replay = Replay(match.events[: cut + 1])
    results = [replay._results[k] for k in sorted(replay._results)]  # noqa: SLF001
    assert len(results) == 1
    assert results[0]["gameOver"] is False
    assert len(replay.rounds) == 1


def test_a_record_saved_as_the_next_round_is_dealt_still_replays(registry: AgentRegistry) -> None:
    """Quitting straight after moving on leaves a log that ends on a fresh
    start_kyoku. The sharing steps before it were played, so the replay plays
    them too and checks the deal."""
    match = _self_play(registry, "red_mahjong", seed=1)
    second = [i for i, e in enumerate(match.events) if e["type"] == "start_kyoku"][1]
    replay = Replay(match.events[: second + 1])
    opening = replay._states[-1].round_state  # noqa: SLF001
    assert len(replay._results) == 1  # noqa: SLF001
    assert int(opening.honba) == match.events[second]["honba"]
    assert int(opening.round) % 4 + 1 == match.events[second]["kyoku"]
    # The dealt round has a span of its own, so the replay bar does not file it
    # under the first round.
    last = len(replay._states) - 1  # noqa: SLF001
    assert [(r.start, r.end) for r in replay.rounds] == [(0, last - 1), (last, last)]


def test_a_record_from_another_version_says_so(registry: AgentRegistry) -> None:
    """The replay re-runs the env, so a log written by a version that plays
    differently cannot be followed. The error names the version, not the step."""
    match = _self_play(registry, "red_mahjong", seed=44, round_mode="single")
    events = [dict(e) for e in match.events]
    events[0]["mahjax"] = {**events[0]["mahjax"], "mahjax_version": "0.0.1"}
    for event in events:
        if event.get("type") == "tsumo":
            event["pai"] = "1m" if event["pai"] != "1m" else "9s"
            break
    else:
        pytest.skip("no tsumo event to change")
    with pytest.raises(record_mod.RecordError, match="mahjax 0.0.1"):
        Replay(events)
