"""JAX-compatible Hong Kong mahjong state."""

import jax
import jax.numpy as jnp

from mahjax._src.struct import dataclass
from mahjax._src.types import Array, PRNGKey
from mahjax.core import EnvId
from mahjax.hong_kong_mahjong.action import Action
from mahjax.hong_kong_mahjong.meld import EMPTY_MELD

NUM_PLAYERS = 4
NUM_STANDARD_TILE_TYPES = 34
NUM_PHYSICAL_TILES = 144
MAX_MELDS = 4
MAX_DISCARDS = 40


@dataclass
class PlayerStateArrays:
    hand: Array = jnp.zeros((NUM_PLAYERS, NUM_STANDARD_TILE_TYPES), dtype=jnp.int8)
    flowers: Array = jnp.zeros((NUM_PLAYERS, 8), dtype=jnp.bool_)
    legal_action_mask: Array = jnp.zeros((NUM_PLAYERS, Action.NUM_ACTION), dtype=jnp.bool_)
    melds: Array = jnp.full((NUM_PLAYERS, MAX_MELDS), EMPTY_MELD, dtype=jnp.uint16)
    meld_counts: Array = jnp.zeros(NUM_PLAYERS, dtype=jnp.int8)
    river: Array = jnp.full((NUM_PLAYERS, MAX_DISCARDS), -1, dtype=jnp.int8)
    discard_counts: Array = jnp.zeros(NUM_PLAYERS, dtype=jnp.int8)
    pon: Array = jnp.zeros((NUM_PLAYERS, NUM_STANDARD_TILE_TYPES), dtype=jnp.int8)
    has_won: Array = jnp.zeros(NUM_PLAYERS, dtype=jnp.bool_)


@dataclass
class RoundState:
    rng_key: PRNGKey = jax.random.PRNGKey(0)
    deck: Array = jnp.zeros(NUM_PHYSICAL_TILES, dtype=jnp.int8)
    wall_index: Array = jnp.int16(0)
    dealer: Array = jnp.int8(0)
    round: Array = jnp.int8(0)
    round_limit: Array = jnp.int8(7)
    score: Array = jnp.zeros(NUM_PLAYERS, dtype=jnp.int32)
    seat_wind: Array = jnp.arange(NUM_PLAYERS, dtype=jnp.int8)
    prevalent_wind: Array = jnp.int8(0)
    last_draw: Array = jnp.int8(-1)
    last_player: Array = jnp.int8(-1)
    target: Array = jnp.int8(-1)
    draw_next: Array = jnp.bool_(False)
    after_kong: Array = jnp.bool_(False)
    robbing_kong: Array = jnp.bool_(False)
    pending_kong_player: Array = jnp.int8(-1)
    terminated_round: Array = jnp.bool_(False)


@dataclass
class State:
    current_player: Array = jnp.int8(0)
    legal_action_mask: Array = jnp.zeros(Action.NUM_ACTION, dtype=jnp.bool_)
    players: PlayerStateArrays = PlayerStateArrays()
    round_state: RoundState = RoundState()
    step_count: Array = jnp.int32(0)
    rewards: Array = jnp.zeros(NUM_PLAYERS, dtype=jnp.float32)
    terminated: Array = jnp.bool_(False)
    truncated: Array = jnp.bool_(False)

    @property
    def env_id(self) -> EnvId:
        return "hong_kong_mahjong"


def default_state() -> State:
    return State()
