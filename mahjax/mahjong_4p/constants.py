from __future__ import annotations

from typing import Final

import jax.numpy as jnp

from mahjax.common.action import Action
from mahjax.common.constants import INITIAL_HAND_TILES

NUM_PLAYERS: Final[int] = 4
NUM_PHYSICAL_TILES: Final[int] = 136

MAX_DISCARDS_PER_PLAYER: Final[int] = 24
MAX_DORA_INDICATORS: Final[int] = 5
DEAD_WALL_TILES: Final[int] = 14

STARTING_POINTS: Final[int] = 25_000
TARGET_POINTS: Final[int] = 30_000
HONBA_BONUS: Final[int] = 300
RIICHI_BET: Final[int] = 1_000

NO_WINNER_NONE: Final[int] = 0
NO_WINNER_NORMAL: Final[int] = 1

# ``score`` is in units of 100 points, ``TARGET_POINTS`` in points.
TARGET_SCORE: Final[int] = TARGET_POINTS // 100
SUDDEN_DEATH_ROUNDS: Final[int] = 4

LEGAL_ACTION_SIZE: Final[int] = Action.NUM_ACTION

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

ZERO_MASK_1D = jnp.zeros(Action.NUM_ACTION, dtype=jnp.bool_)
ZERO_MASK_2D = jnp.zeros((NUM_PLAYERS, Action.NUM_ACTION), dtype=jnp.bool_)

# The wall is ``deck``. The hands are dealt from its end and the draws go down from
# ``FIRST_DRAW_IDX``. ``deck[:DEAD_WALL_TILES]`` is the dead wall: the n-th kan's
# replacement tile is ``deck[RINSHAN_IX + n]``, the k-th dora indicator is
# ``deck[FIRST_DORA_INDICATOR_IX - 2 * k]`` and its ura is the tile below it.
FIRST_DRAW_IDX = NUM_PHYSICAL_TILES - 1 - INITIAL_HAND_TILES * NUM_PLAYERS
RINSHAN_IX: Final[int] = 10
FIRST_DORA_INDICATOR_IX: Final[int] = 9
