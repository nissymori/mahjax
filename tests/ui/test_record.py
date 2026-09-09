"""Saving a game as an mjai log and replaying it back."""

from __future__ import annotations

import json

import pytest

from mahjax.ui import record as record_mod
from mahjax.ui.agents import AgentRegistry
from mahjax.ui.match import Match, MatchConfig
from mahjax.ui.record import Replay

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
