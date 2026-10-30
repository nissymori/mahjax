import dataclasses

import jax
import jax.numpy as jnp
import pytest

from mahjax.mahjong_4p.action import Action
from mahjax.mahjong_4p.config import NO_RED, RED, GameConfig
from mahjax.mahjong_4p.env import Mahjong4p
from mahjax.mahjong_4p.tile import Tile

from ..helpers import compiled_env


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
    assert Mahjong4p().game_config is RED
    assert Mahjong4p(game_config=NO_RED).game_config is NO_RED


def test_no_red_deals_no_red_fives() -> None:
    key = jax.random.PRNGKey(0)
    red = compiled_env(game_config=RED).init(key)
    no_red = compiled_env(game_config=NO_RED).init(key)

    assert int((red.round_state.deck >= Tile.NUM_TILE_TYPE).sum()) == 3
    assert int((no_red.round_state.deck >= Tile.NUM_TILE_TYPE).sum()) == 0


def test_red_use_red_fives_survives_round_transitions() -> None:
    from mahjax.mahjong_4p.config import GameConfig as RedGameConfig

    env = compiled_env(
        round_mode="half",
        game_config=RedGameConfig(use_red_fives=False),
        round_end="confirm",
    )
    step = env.step
    key = jax.random.PRNGKey(3)
    key, sub = jax.random.split(key)
    state = env.init(sub)
    rounds = set()

    for _ in range(4000):
        assert int(state.round_state.deck.max()) < Tile.NUM_TILE_TYPE  # no red five in any wall
        rounds.add(int(state.round_state.round))
        key, action_key, step_key = jax.random.split(key, 3)
        action = jax.random.categorical(
            action_key, jnp.log(jnp.where(state.legal_action_mask, 1.0, 0.0))
        )
        state = step(state, action.astype(jnp.int8), step_key)
        if bool(state.terminated) or bool(state.truncated):
            break

    assert len(rounds) > 1


def test_no_red_never_offers_a_red_five_or_kyuushu() -> None:
    """Random play under NO_RED: no red five is dealt, and no red-five discard or
    call, and no KYUUSHU, is ever legal."""
    env = compiled_env(game_config=NO_RED)
    red_only = jnp.array(
        [*Tile.RED_FIVE.values(), Action.PON_RED, Action.CHI_L_RED, Action.CHI_M_RED, Action.CHI_R_RED, Action.KYUUSHU]
    )
    key = jax.random.PRNGKey(1)
    key, sub = jax.random.split(key)
    state = env.init(sub)
    rounds = set()

    for _ in range(1500):
        if bool(state.terminated):
            break
        assert not bool(state.legal_action_mask[red_only].any())
        assert int(state.players.hand_with_red[:, Tile.NUM_TILE_TYPE :].sum()) == 0
        rounds.add(int(state.round_state.round))
        key, action_key, step_key = jax.random.split(key, 3)
        action = jax.random.categorical(action_key, jnp.where(state.legal_action_mask, 0.0, -jnp.inf))
        state = env.step(state, action.astype(jnp.int32), step_key)

    assert len(rounds) > 1
