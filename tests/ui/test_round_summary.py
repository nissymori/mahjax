import jax

from mahjax.ui.game_manager import GameManager


def test_red_round_summary_flags_only_the_last_round_as_game_end() -> None:
    session = GameManager().create_session(
        env_id="red_mahjong", agent_id="rule_based", human_seat=0, round_mode="east", seed=3
    )
    key = jax.random.PRNGKey(0)
    flags = []
    for _ in range(3000):
        if session.state.terminated:
            break
        if session.round_summary is not None:
            flags.append(session.round_summary.is_game_end)
            view = session.to_view()
            assert view["roundSummary"]["isGameEnd"] == flags[-1]
            assert view["advanceAction"]["isFinal"] == flags[-1]
            assert view["advanceAction"]["label"] == ("終局" if flags[-1] else "次の局へ")
            session.continue_after_round()
            continue
        if int(session.state.current_player) == session.human_seat:
            key, sub = jax.random.split(key)
            session.apply_action(int(jax.device_get(session.agent.act(session.state, sub))), actor="human")
        else:
            session.ai_step()
    assert session.state.terminated
    assert len(flags) >= 4  # an east game has at least four rounds
    assert flags[-1] is True
    assert not any(flags[:-1])
