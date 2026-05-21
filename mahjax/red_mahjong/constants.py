from __future__ import annotations

from typing import Final

import jax.numpy as jnp

from .action import Action

NUM_PLAYERS: Final[int] = 4
NUM_TILE_TYPES: Final[int] = 34
NUM_TILE_TYPES_WITH_RED: Final[int] = 37
NUM_PHYSICAL_TILES: Final[int] = 136
COPIES_PER_TILE: Final[int] = 4

MAX_HAND_TILES: Final[int] = 14
MAX_MELDS_PER_PLAYER: Final[int] = 4
MAX_DISCARDS_PER_PLAYER: Final[int] = 24
MAX_DORA_INDICATORS: Final[int] = 5
DEAD_WALL_TILES: Final[int] = 14

STARTING_POINTS: Final[int] = 25_000
TARGET_POINTS: Final[int] = 30_000
HONBA_BONUS: Final[int] = 300
RIICHI_BET: Final[int] = 1_000

RED_FIVE_TILE_TYPES: Final[tuple[int, int, int]] = (4, 13, 22)
RED_FIVE_TILE_IDS: Final[tuple[int, int, int]] = tuple(tile_type * 4 for tile_type in RED_FIVE_TILE_TYPES)

SENTINEL_TILE_ID: Final[int] = -1
SENTINEL_MELD_VALUE: Final[int] = -1
SENTINEL_DISCARD_VALUE: Final[int] = -1

NO_WINNER_NONE: Final[int] = 0
NO_WINNER_NORMAL: Final[int] = 1

LEGAL_ACTION_SIZE: Final[int] = Action.NUM_ACTION

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

TILE_RANGE = jnp.arange(NUM_TILE_TYPES, dtype=jnp.int32)
ZERO_MASK_1D = jnp.zeros(Action.NUM_ACTION, dtype=jnp.bool_)
ZERO_MASK_2D = jnp.zeros((NUM_PLAYERS, Action.NUM_ACTION), dtype=jnp.bool_)

FIRST_DRAW_IDX = 135 - 13 * 4
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
        9,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        18,
        28,
        29,
        30,
        27,
        32,
        33,
        31,
    ],
    dtype=jnp.int8,
)
