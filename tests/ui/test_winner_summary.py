import math

import jax
import jax.numpy as jnp
import numpy as np

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.env import RedMahjong, _pao
from mahjax.red_mahjong.players import rule_based_player
from mahjax.red_mahjong.yaku import Yaku
from mahjax.ui.game_manager import summarise_winner_red


def _expected_gain(prev_state, summary, is_ron: bool) -> int:
    # What the env pays the winner (100-point units) for the han/fu the summary shows.
    basic = int(Yaku.score(jnp.int32(summary.fan), jnp.int32(summary.fu)))
    rs = prev_state.round_state
    honba, kyotaku = int(rs.honba), int(rs.kyotaku)
    dealer = int(rs.dealer) == summary.player
    if is_ron:
        first = not bool(np.asarray(prev_state.players.has_won).any())
        gain = math.ceil(basic * (6 if dealer else 4) / 100)
        return gain + (3 * honba + 10 * kyotaku if first else 0)
    s1, s2 = math.ceil(basic / 100), math.ceil(basic * 2 / 100)
    gain = 3 * s2 if dealer else 2 * s1 + s2
    return gain + 3 * honba + 10 * kyotaku


def _summary(prev_state, next_state, player: int, is_ron: bool):
    rs = prev_state.round_state
    return summarise_winner_red(
        prev_state,
        next_state,
        player=player,
        player_names=["a", "b", "c", "d"],
        winning_tile=int(rs.target) if is_ron else int(rs.last_draw),
        from_player=int(rs.last_player) if is_ron else None,
        is_ron=is_ron,
    )


def _collect_wins(n: int):
    env = RedMahjong(round_mode="single")
    step, act = jax.jit(env.step), jax.jit(rule_based_player)
    wins = []
    for seed in range(200):
        key = jax.random.PRNGKey(seed)
        state = env.init(key)
        while not bool(state.terminated):
            key, k1, k2 = jax.random.split(key, 3)
            action = int(act(state, k1))
            if action in (Action.RON, Action.TSUMO):
                player = int(state.current_player)
                if not bool(_pao(state, player)[0]):
                    wins.append((state, action, step(state, jnp.int32(action), k2), player))
            state = step(state, jnp.int32(action), k2)
        if len(wins) >= n:
            break
    return env, step, wins


def test_red_winner_summary_han_and_fu_explain_the_payment() -> None:
    env, step, wins = _collect_wins(12)
    assert len(wins) >= 12
    riichi_win = None
    for prev, action, nxt, player in wins:
        is_ron = action == Action.RON
        summary = _summary(prev, nxt, player, is_ron)
        assert summary.points_delta // 100 == _expected_gain(prev, summary, is_ron)
        if riichi_win is None and bool(prev.players.riichi[player]):
            riichi_win = (prev, action, player)

    # Situational yaku are rare in a short rollout, so force them on a riichi win and
    # let the env pay it again: the summary must count them the way the env does.
    assert riichi_win is not None
    prev, action, player = riichi_win
    is_ron = action == Action.RON
    forced = prev.replace(  # type: ignore[attr-defined]
        players=prev.players.replace(
            ippatsu=prev.players.ippatsu.at[player].set(True),
            double_riichi=prev.players.double_riichi.at[player].set(True),
        ),
        round_state=prev.round_state.replace(is_haitei=jnp.bool_(True)),
    )
    nxt = step(forced, jnp.int32(action), jax.random.PRNGKey(0))
    summary = _summary(forced, nxt, player, is_ron)
    assert summary.points_delta // 100 == _expected_gain(forced, summary, is_ron)
    assert "Ippatsu" in summary.yaku and "Double Riichi" in summary.yaku
    assert "Riichi" not in summary.yaku  # shown as Double Riichi, not Riichi + Double Riichi
