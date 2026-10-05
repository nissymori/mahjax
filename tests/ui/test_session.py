import jax
import pytest

from mahjax.red_mahjong.action import Action as RedAction
from mahjax.ui.game_manager import GameManager


def _play(session, max_actions: int) -> int:
    key = jax.random.PRNGKey(0)
    rounds = 0
    for _ in range(max_actions):
        if session.state.terminated:
            break
        if session.round_summary is not None:
            rounds += 1
            session.to_view()
            session.continue_after_round()
            continue
        if int(session.state.current_player) == session.human_seat:
            key, sub = jax.random.split(key)
            action = int(jax.device_get(session.agent.act(session.state, sub)))
            session.apply_action(action, actor="human")
        else:
            session.ai_step()
        session.to_view()
    return rounds


@pytest.mark.parametrize("env_id", ["no_red_mahjong", "red_mahjong"])
def test_session_handles_call_prompts_and_round_end(env_id: str) -> None:
    # The UI builds a RedMahjong env for both env ids ("no red" only turns red
    # fives off), so both sessions must read actions in the red action space.
    session = GameManager().create_session(
        env_id=env_id, agent_id="rule_based", human_seat=0, round_mode="east", seed=5
    )
    rounds = _play(session, max_actions=300)
    passes = [e for e in session.events if e.action == RedAction.PASS]
    assert passes, "the rollout never reached a call prompt"
    assert all(e.description == "パス" for e in passes)
    assert rounds > 0
