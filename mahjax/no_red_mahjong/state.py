# Copyright 2025 The Mahjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import jax
import jax.numpy as jnp

from mahjax._src.struct import dataclass
from mahjax._src.types import Array, PRNGKey
from mahjax.core import EnvId
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.meld import EMPTY_MELD
from mahjax.no_red_mahjong.tile import EMPTY_RIVER, Tile

NUM_PLAYERS = 4
NUM_TILE_TYPES = Tile.NUM_TILE_TYPE
NUM_PHYSICAL_TILES = 136
MAX_HAND_TILES = 14
MAX_MELDS_PER_PLAYER = 4
MAX_DISCARDS_PER_PLAYER = 24
MAX_DORA_INDICATORS = 5
LEGAL_ACTION_SIZE = Action.NUM_ACTION

STARTING_POINTS = 250
HONBA_BONUS = 300
RIICHI_BET = 1_000

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
FIRST_DRAW_IDX = (
    135 - 13 * 4
)  # The index of the first drawn tile after drawing 13*4 tiles from the deck
DORA_ARRAY = jnp.array(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        0,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        9,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        19,
        28,
        29,
        30,
        27,
        32,
        33,
        31,
    ]
)


@dataclass
class PlayerStateArrays:
    hand: Array = jnp.zeros((NUM_PLAYERS, NUM_TILE_TYPES), dtype=jnp.int8)
    legal_action_mask: Array = jnp.zeros((NUM_PLAYERS, LEGAL_ACTION_SIZE), dtype=jnp.bool_)
    can_win: Array = jnp.zeros((NUM_PLAYERS, NUM_TILE_TYPES), dtype=jnp.bool_)
    has_yaku: Array = jnp.zeros((NUM_PLAYERS, 2), dtype=jnp.bool_)
    fan: Array = jnp.zeros((NUM_PLAYERS, 2), dtype=jnp.int32)
    fu: Array = jnp.zeros((NUM_PLAYERS, 2), dtype=jnp.int32)
    melds: Array = jnp.full((NUM_PLAYERS, MAX_MELDS_PER_PLAYER), EMPTY_MELD, dtype=jnp.uint16)
    meld_counts: Array = jnp.zeros((NUM_PLAYERS,), dtype=jnp.int8)
    river: Array = jnp.full((NUM_PLAYERS, MAX_DISCARDS_PER_PLAYER), EMPTY_RIVER, dtype=jnp.uint16)
    discard_counts: Array = jnp.zeros((NUM_PLAYERS,), dtype=jnp.int8)
    riichi: Array = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    riichi_declared: Array = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    double_riichi: Array = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    ippatsu: Array = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    furiten_by_discard: Array = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    furiten_by_pass: Array = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    is_hand_concealed: Array = jnp.ones((NUM_PLAYERS,), dtype=jnp.bool_)
    pon: Array = jnp.zeros((NUM_PLAYERS, NUM_TILE_TYPES), dtype=jnp.int32)
    has_won: Array = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    n_kan: Array = jnp.zeros((NUM_PLAYERS,), dtype=jnp.int8)


@dataclass
class RoundState:
    rng_key: PRNGKey = jax.random.PRNGKey(0)
    # action history (player, action, is_tsumogiri), 70 (discard) + 70 (every pass) + 16 (four players meld 4 times) + 16 (discard for the melds) + 4 (dummy actions) + 20 (buffer)
    action_history: Array = jnp.full((3, 200), -1, dtype=jnp.int8)
    shanten_current_player: Array = jnp.int8(0)
    round: Array = jnp.int8(0)
    round_limit: Array = jnp.int8(7)
    terminated_round: Array = FALSE
    honba: Array = jnp.int8(0)
    kyotaku: Array = jnp.int8(0)
    init_wind: Array = jnp.array([0, 1, 2, 3], dtype=jnp.int8)
    seat_wind: Array = jnp.array([0, 1, 2, 3], dtype=jnp.int8)
    dealer: Array = jnp.int8(0)
    order_points: Array = jnp.array([30, 10, -10, -30], dtype=jnp.int32)
    score: Array = jnp.full((NUM_PLAYERS,), 250, dtype=jnp.int32)
    deck: Array = jnp.zeros((NUM_PHYSICAL_TILES,), dtype=jnp.int8)
    next_deck_ix: Array = jnp.int32(FIRST_DRAW_IDX)
    last_deck_ix: Array = jnp.int8(14)
    draw_next: Array = FALSE
    last_draw: Array = jnp.int8(-1)
    last_player: Array = jnp.int8(0)
    dora_indicators: Array = jnp.full((MAX_DORA_INDICATORS,), -1, dtype=jnp.int8)
    ura_dora_indicators: Array = jnp.full((MAX_DORA_INDICATORS,), -1, dtype=jnp.int8)
    is_abortive_draw_normal: Array = FALSE
    dummy_count: Array = jnp.int8(0)
    is_haitei: Array = FALSE
    target: Array = jnp.int8(-1)
    n_kan_doras: Array = jnp.int8(0)
    kan_declared: Array = FALSE
    can_after_kan: Array = FALSE
    can_robbing_kan: Array = FALSE


@dataclass
class EnvState:
    current_player: Array = jnp.int8(0)
    legal_action_mask: Array = jnp.zeros(LEGAL_ACTION_SIZE, dtype=jnp.bool_)
    players: PlayerStateArrays = PlayerStateArrays()
    round_state: RoundState = RoundState()
    step_count: Array = jnp.int32(0)
    rewards: Array = jnp.zeros(NUM_PLAYERS, dtype=jnp.float32)
    terminated: Array = FALSE
    truncated: Array = FALSE

    @property
    def env_id(self) -> EnvId:
        return "no_red_mahjong"


State = EnvState


def default_state() -> EnvState:
    return EnvState()
