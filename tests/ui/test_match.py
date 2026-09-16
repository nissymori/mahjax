"""End-to-end checks on the live-game driver."""

from __future__ import annotations

import json

import pytest

from mahjax.ui.agents import AgentRegistry
from mahjax.ui.match import Match, MatchConfig

ENV_IDS = ("red_mahjong", "no_red_mahjong")


@pytest.fixture(scope="module")
def registry() -> AgentRegistry:
    return AgentRegistry()


def _match(registry: AgentRegistry, **kwargs) -> Match:
    config = MatchConfig(save_record=False, **kwargs)
    agent = registry.default_for(config.env_id)
    return Match(config, agent)


def _assert_frames_stop_somewhere(frames, match: Match) -> None:
    """Every batch of frames has to leave the client with something to do."""
    assert frames, "advance() returned no frames"
    last = frames[-1]
    assert (
        last["prompt"] is not None or last["result"] is not None or last["gameOver"]
    ), "frames ended without a prompt, a result or the game being over"
    for frame in frames:
        json.dumps(frame)  # no numpy scalars may leak into the payload


def _play(match: Match, *, max_rounds: int = 30) -> int:
    """Drive a human seat with the bundled heuristic until the game ends."""
    import jax

    frames = match.start()
    _assert_frames_stop_somewhere(frames, match)
    rounds = 0
    while not match.game_over and rounds < max_rounds:
        if match.result is not None:
            frames = match.next_round()
            rounds += 1
            if match.game_over:
                break
            _assert_frames_stop_somewhere(frames, match)
            continue
        prompt = frames[-1]["prompt"]
        assert prompt is not None
        key = jax.random.PRNGKey(match.step_index)
        action = int(jax.device_get(match.agent.act(match.state, key)))
        frames = match.act(action)
        _assert_frames_stop_somewhere(frames, match)
    return rounds


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_human_game_reaches_game_over(registry: AgentRegistry, env_id: str) -> None:
    match = _match(registry, env_id=env_id, round_mode="east", seed=11, human_seat=0)
    _play(match)
    assert match.game_over
    assert match.result is not None
    assert match.result["gameOver"] is True
    assert match.result["final"] is not None
    ranks = sorted(entry["rank"] for entry in match.result["final"])
    assert ranks == [1, 2, 3, 4]


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_final_frame_announces_the_game_is_over(registry: AgentRegistry, env_id: str) -> None:
    """The env only sets `terminated` during the sharing steps, so the frame the
    player is looking at has to take that flag from the result."""
    match = _match(registry, env_id=env_id, round_mode="east", seed=2024, human_seat=0)
    _play(match)
    assert match.game_over
    frame = match.view
    assert frame["result"]["gameOver"] is True
    assert frame["gameOver"] is True, "the last frame did not report the game as over"
    assert match.next_round() == []


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_agent_only_game_plays_itself_out(registry: AgentRegistry, env_id: str) -> None:
    match = _match(registry, env_id=env_id, round_mode="east", seed=5, human_seat=None)
    match.play_out()
    assert match.game_over
    assert match.events[0]["type"] == "start_game"
    assert match.events[0]["mahjax"]["complete"] is True
    assert match.events[-1]["type"] == "end_game"


def test_prompt_never_offers_a_lone_tsumogiri(registry: AgentRegistry) -> None:
    """Riichi auto-discard: a forced tsumogiri must not stop for the human."""
    match = _match(registry, env_id="red_mahjong", round_mode="half", seed=3, human_seat=0)
    rules = match.rules
    seen_prompts = 0

    import jax

    frames = match.start()
    for _ in range(4000):
        if match.game_over:
            break
        if match.result is not None:
            match.next_round()
            frames = [match.view]
            continue
        prompt = frames[-1]["prompt"]
        if prompt is None:
            break
        seen_prompts += 1
        legal = rules.legal_actions(match.state)
        assert legal != [rules.TSUMOGIRI], "stopped for a forced tsumogiri"
        key = jax.random.PRNGKey(match.step_index)
        frames = match.act(int(jax.device_get(match.agent.act(match.state, key))))
    assert seen_prompts > 20


def test_no_calls_auto_passes_but_keeps_ron(registry: AgentRegistry) -> None:
    match = _match(
        registry,
        env_id="red_mahjong",
        round_mode="half",
        seed=8,
        human_seat=0,
        no_calls=True,
    )
    import jax

    frames = match.start()
    for _ in range(4000):
        if match.game_over:
            break
        if match.result is not None:
            match.next_round()
            frames = [match.view]
            continue
        prompt = frames[-1]["prompt"]
        if prompt is None:
            break
        if prompt["kind"] == "claim":
            kinds = {opt["kind"] for opt in prompt["options"]}
            assert "ron" in kinds, f"a call-only prompt survived no_calls: {kinds}"
        key = jax.random.PRNGKey(match.step_index)
        frames = match.act(int(jax.device_get(match.agent.act(match.state, key))))


def test_hidden_hands_open_up_at_the_result(registry: AgentRegistry) -> None:
    match = _match(
        registry, env_id="red_mahjong", round_mode="east", seed=2, human_seat=1
    )
    frames = match.start()
    first = frames[0]
    hidden = [i for i in range(4) if i != 1]
    assert all(first["seats"][i]["hand"] is None for i in hidden)
    assert first["seats"][1]["hand"] is not None

    import jax

    for _ in range(4000):
        if match.result is not None or match.game_over:
            break
        prompt = frames[-1]["prompt"]
        if prompt is None:
            break
        key = jax.random.PRNGKey(match.step_index)
        frames = match.act(int(jax.device_get(match.agent.act(match.state, key))))
    assert match.result is not None
    result = match.result
    frame = frames[-1]
    # Only the hands a real table would turn over: the winners, or everyone who
    # was tenpai at an exhaustive draw. Your own hand is always visible.
    if result["type"] in ("ron", "tsumo"):
        expected = {w["seat"] for w in result["winners"]} | {1}
    elif result["type"] == "draw":
        expected = {i for i, t in enumerate(result["tenpai"]) if t} | {1}
    else:
        expected = {1}
    shown = {i for i, seat in enumerate(frame["seats"]) if seat["hand"] is not None}
    assert shown == expected, f"{result['type']}: showed {shown}, expected {expected}"


def test_illegal_action_is_refused(registry: AgentRegistry) -> None:
    match = _match(registry, env_id="red_mahjong", round_mode="east", seed=1, human_seat=0)
    match.start()
    while match.result is None and int(match.state.current_player) != 0:
        break
    illegal = next(
        a
        for a in range(match.rules.NUM_ACTION)
        if not bool(match.state.legal_action_mask[a])
    )
    if int(match.state.current_player) == 0 and match.result is None:
        with pytest.raises(ValueError):
            match.act(illegal)
