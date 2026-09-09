"""mjai event log round trip: env steps -> events -> env actions -> same table."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mahjax.ui import mjai
from mahjax.ui.rules import Rules, rules_for

ENV_IDS = ("red_mahjong", "no_red_mahjong")
ROUND_MODE = "east"
MAX_STEPS = 4000

# Seeds found by scanning under the policy in ``_choose``; the situations each
# one is here for are pinned by ``test_the_seeds_still_cover_the_hard_cases``.
GAMES = (
    ("red_mahjong", 88),
    ("red_mahjong", 105),
    ("red_mahjong", 69),
    ("red_mahjong", 60),
    ("no_red_mahjong", 3),
    ("no_red_mahjong", 20),
)

COVERAGE = {
    ("red_mahjong", 88): {
        "double_ron",
        "ryukyoku:suuchariichi",
        "ryukyoku:howanpai",
        "ankan",
        "kakan",
        "daiminkan",
        "pon",
        "chi",
        "reach",
        "reach_accepted",
        "hora",
    },
    ("red_mahjong", 105): {"double_ron", "ryukyoku:kyushukyuhai", "ankan", "daiminkan"},
    ("red_mahjong", 69): {"ryukyoku:suufonrenda", "kakan", "daiminkan"},
    ("red_mahjong", 60): {"ryukyoku:suukaikan", "ankan", "daiminkan", "reach", "hora"},
    ("no_red_mahjong", 3): {
        "ryukyoku:howanpai",
        "ankan",
        "daiminkan",
        "pon",
        "chi",
        "reach",
        "reach_accepted",
        "hora",
    },
    ("no_red_mahjong", 20): {"hora", "ankan", "daiminkan", "reach"},
}


@dataclass
class Played:
    rules: Rules
    seed: int
    events: List[Dict[str, Any]]
    actions: List[int]
    final: Any
    hora_step: Optional[Tuple[Any, int, Any]]


# --------------------------------------------------------------------- driver


def _rule_based(env_id: str) -> Any:
    if env_id == "red_mahjong":
        from mahjax.red_mahjong.players import rule_based_player
    else:
        from mahjax.no_red_mahjong.players import rule_based_player
    return jax.jit(rule_based_player)


def _choose(rules: Rules, act: Any, state: Any, rng: Any, key: Any) -> int:
    """Win, declare, kan; otherwise let the built-in rule-based player discard.

    Uniformly random legal play almost never reaches tenpai, so it yields logs
    with no riichi and no win at all -- exactly the events whose encoding spans
    several steps and is worth testing.
    """
    legal = rules.legal_actions(state)
    for wanted in (rules.RON, rules.TSUMO, rules.RIICHI, rules.KYUUSHU):
        if wanted is not None and wanted in legal:
            return wanted
    kans = [a for a in legal if rules.is_kan(a) or a == rules.OPEN_KAN]
    if kans:
        return kans[0]
    action = int(act(state, key))
    return action if action in legal else int(rng.choice(legal))


def _is_kyuushu(rules: Rules, action: int) -> bool:
    return rules.KYUUSHU is not None and action == rules.KYUUSHU


def _play(env_id: str, seed: int) -> Played:
    """Play a whole game, writing the log the way ``record.py`` will."""
    rules = rules_for(env_id)
    env, step_fn = mjai._replay_env(env_id, ROUND_MODE)  # noqa: SLF001 - share the trace
    act = _rule_based(env_id)
    rng = np.random.default_rng(seed)
    root = jax.random.PRNGKey(seed)

    state = jax.device_get(env.init(jax.random.fold_in(root, 0)))
    events: List[Dict[str, Any]] = [
        {
            "type": "start_game",
            "names": ["P0", "P1", "P2", "P3"],
            "mahjax": {"format": 1, "env_id": env_id, "round_mode": ROUND_MODE, "seed": seed},
        }
    ]
    events += mjai.start_kyoku_events(rules, state)
    actions: List[int] = []
    hora_step: Optional[Tuple[Any, int, Any]] = None

    for step in range(MAX_STEPS):
        if bool(state.terminated):
            break
        pre = state
        action = _choose(rules, act, pre, rng, jax.random.fold_in(root, 10_000 + step))
        state = jax.device_get(
            step_fn(pre, jnp.int32(action), jax.random.fold_in(root, step + 1))
        )
        events += mjai.encode_step(rules, pre, action, state)
        actions.append(action)
        if hora_step is None and action in (rules.RON, rules.TSUMO):
            hora_step = (pre, action, state)

        was_over, is_over = rules.is_round_over(pre), rules.is_round_over(state)
        if _is_kyuushu(rules, action) or (is_over and not was_over):
            events.append(mjai.end_kyoku_event())
        if _is_kyuushu(rules, action) or (was_over and not is_over):
            events += mjai.start_kyoku_events(rules, state)

    assert bool(state.terminated), f"{env_id} seed {seed} did not finish in {MAX_STEPS} steps"
    events.append(mjai.end_game_event(rules.scores(state)))
    return Played(
        rules=rules,
        seed=seed,
        events=events,
        actions=actions,
        final=state,
        hora_step=hora_step,
    )


def _replay(rules: Rules, seed: int, actions: List[int]) -> Any:
    env, step_fn = mjai._replay_env(rules.env_id, ROUND_MODE)  # noqa: SLF001
    root = jax.random.PRNGKey(seed)
    state = jax.device_get(env.init(jax.random.fold_in(root, 0)))
    for i, action in enumerate(actions):
        state = jax.device_get(
            step_fn(state, jnp.int32(action), jax.random.fold_in(root, i + 1))
        )
    return state


def _table(rules: Rules, state: Any) -> Dict[str, Any]:
    return {
        "scores": rules.scores(state),
        "round": int(state.round_state.round),
        "honba": int(state.round_state.honba),
        "kyotaku": int(state.round_state.kyotaku),
        "dealer": int(state.round_state.dealer),
        "hands": [rules.hand(state, seat).tolist() for seat in range(4)],
        "terminated": bool(state.terminated),
    }


def _situations(played: Played) -> Set[str]:
    """What a game actually contained, as flat tags."""
    found: Set[str] = {event["type"] for event in played.events}
    found |= {f"ryukyoku:{e['reason']}" for e in played.events if e["type"] == "ryukyoku"}
    horas = 0
    for event in played.events:
        if event["type"] == "hora":
            horas += 1
            if horas >= 2:
                found.add("double_ron")
        elif event["type"] == "end_kyoku":
            horas = 0
    if played.rules.PASS in played.actions:
        found.add("pass")
    if played.rules.DUMMY in played.actions:
        found.add("dummy")
    return found


@pytest.fixture(scope="module")
def games() -> Dict[Tuple[str, int], Played]:
    return {(env_id, seed): _play(env_id, seed) for env_id, seed in GAMES}


# ---------------------------------------------------------------------- tests


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_tile_notation_round_trips(env_id: str) -> None:
    rules = rules_for(env_id)
    for tile in range(rules.n_tiles):
        assert mjai.mjai_to_tile(mjai.tile_to_mjai(tile)) == tile
    assert mjai.tile_to_mjai(0) == "1m"
    assert mjai.tile_to_mjai(26) == "9s"
    assert mjai.tile_to_mjai(27) == "E"
    assert mjai.tile_to_mjai(33) == "C"
    assert mjai.tile_to_mjai(34) == "5mr"
    assert (rules.n_tiles == 37) == rules.has_red  # no_red never names a red five
    with pytest.raises(ValueError):
        mjai.mjai_to_tile("1mr")


@pytest.mark.parametrize(("env_id", "seed"), GAMES)
def test_the_log_decodes_to_the_actions_that_were_played(
    games: Dict[Tuple[str, int], Played], env_id: str, seed: int
) -> None:
    played = games[(env_id, seed)]
    assert mjai.decode_events(played.rules, played.events) == played.actions


@pytest.mark.parametrize(("env_id", "seed"), GAMES)
def test_replaying_the_decoded_actions_reaches_the_same_table(
    games: Dict[Tuple[str, int], Played], env_id: str, seed: int
) -> None:
    played = games[(env_id, seed)]
    decoded = mjai.decode_events(played.rules, played.events)
    assert _table(played.rules, _replay(played.rules, seed, decoded)) == _table(
        played.rules, played.final
    )


def test_every_event_is_plain_json(games: Dict[Tuple[str, int], Played]) -> None:
    for played in games.values():
        for event in played.events:
            json.dumps(event)
            _assert_plain(event)


def _assert_plain(value: Any) -> None:
    """No numpy scalars: they survive ``json.dumps`` on some versions and not others."""
    if isinstance(value, dict):
        for key, item in value.items():
            assert type(key) is str, (key, type(key))
            _assert_plain(item)
    elif isinstance(value, list):
        for item in value:
            _assert_plain(item)
    else:
        assert type(value) in (str, int, bool, type(None)), (value, type(value))


def test_the_seeds_still_cover_the_hard_cases(games: Dict[Tuple[str, int], Played]) -> None:
    everything: Set[str] = set()
    for key, played in games.items():
        found = _situations(played)
        assert COVERAGE[key] <= found, f"{key} lost {COVERAGE[key] - found}"
        everything |= found
    # Neither of these is written in a log; both are inferred while decoding.
    assert {"pass", "dummy"} <= everything


def test_a_reach_is_accepted_on_a_later_step(games: Dict[Tuple[str, int], Played]) -> None:
    """The declaration and the stick are different events on different steps."""
    played = games[("red_mahjong", 88)]
    events = played.events
    reach = next(i for i, e in enumerate(events) if e["type"] == "reach")
    actor = events[reach]["actor"]
    assert events[reach + 1] == {
        "type": "dahai",
        "actor": actor,
        "pai": events[reach + 1]["pai"],
        "tsumogiri": events[reach + 1]["tsumogiri"],
    }
    accepted = next(
        e for e in events[reach:] if e["type"] == "reach_accepted" and e["actor"] == actor
    )
    assert accepted["deltas"][actor] == -1000
    assert sum(accepted["deltas"]) == -1000


def test_a_win_carries_the_scoring_fields(games: Dict[Tuple[str, int], Played]) -> None:
    """``win`` comes from ``view.build_win``; these are the keys read out of it."""
    played = games[("red_mahjong", 88)]
    pre, action, post = played.hora_step
    win = {
        "yaku": [{"name": "riichi", "han": 1}, {"name": "tanyao", "han": 1}],
        "doraHan": 2,
        "akaHan": 1,
        "uraHan": 0,
        "han": 5,
        "fu": 40,
        "points": 8000,
    }
    events = mjai.encode_step(played.rules, pre, action, post, win=win)
    hora = next(e for e in events if e["type"] == "hora")
    assert hora["yakus"] == [["riichi", 1], ["tanyao", 1], ["dora", 2], ["aka dora", 1]]
    assert (hora["fu"], hora["fan"], hora["hora_points"]) == (40, 5, 8000)
    assert hora["actor"] == int(pre.current_player)
    assert (hora["target"] == hora["actor"]) is (action == played.rules.TSUMO)
    assert hora["deltas"][hora["actor"]] > 0
    assert hora["scores"] == played.rules.scores(post)


def test_a_truncated_record_replays_as_far_as_it_goes(
    games: Dict[Tuple[str, int], Played]
) -> None:
    """An unfinished game has no ``end_game``, so the replay must stop, not guess."""
    played = games[("red_mahjong", 105)]
    decoded = mjai.decode_events(played.rules, played.events[:250])
    assert 0 < len(decoded) < len(played.actions)
    assert decoded == played.actions[: len(decoded)]
