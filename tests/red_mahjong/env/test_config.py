import dataclasses

import jax
import jax.numpy as jnp
import pytest

from mahjax.red_mahjong.config import NO_RED, RED, GameConfig
from mahjax.red_mahjong.env import RedMahjong
from mahjax.red_mahjong.tile import Tile


def test_presets_turn_every_optional_rule_on_or_off() -> None:
    assert GameConfig() == RED
    assert dataclasses.asdict(RED) == {
        "use_red_fives": True,
        "allow_double_ron": True,
        "enable_special_abortive_draw": True,
        "enable_pao": True,
    }
    assert dataclasses.asdict(NO_RED) == {
        "use_red_fives": False,
        "allow_double_ron": False,
        "enable_special_abortive_draw": False,
        "enable_pao": False,
    }


def test_game_config_is_static() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        RED.enable_pao = False
    assert hash(GameConfig(enable_pao=False)) == hash(dataclasses.replace(RED, enable_pao=False))
    with pytest.raises(TypeError, match="use_red_fives"):
        GameConfig(use_red_fives=jnp.bool_(False))


def test_env_holds_its_config() -> None:
    assert RedMahjong().game_config is RED
    assert RedMahjong(game_config=NO_RED).game_config is NO_RED


def test_no_red_deals_no_red_fives() -> None:
    key = jax.random.PRNGKey(0)
    red = RedMahjong(game_config=RED).init(key)
    no_red = RedMahjong(game_config=NO_RED).init(key)

    assert int((red.round_state.deck >= Tile.NUM_TILE_TYPE).sum()) == 3
    assert int((no_red.round_state.deck >= Tile.NUM_TILE_TYPE).sum()) == 0
    assert not bool(no_red.round_state.use_red_fives)


def test_red_use_red_fives_survives_round_transitions() -> None:
    from mahjax.red_mahjong.config import GameConfig as RedGameConfig

    env = RedMahjong(
        round_mode="half",
        game_config=RedGameConfig(use_red_fives=False),
        next_round_style="dummy_share",
    )
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(3)
    key, sub = jax.random.split(key)
    state = env.init(sub)
    rounds = set()

    for _ in range(4000):
        assert not bool(state.round_state.use_red_fives)
        rounds.add(int(state.round_state.round))
        key, action_key, step_key = jax.random.split(key, 3)
        action = jax.random.categorical(
            action_key, jnp.log(jnp.where(state.legal_action_mask, 1.0, 0.0))
        )
        state = step(state, action.astype(jnp.int8), step_key)
        if bool(state.terminated) or bool(state.truncated):
            break

    assert len(rounds) > 1
