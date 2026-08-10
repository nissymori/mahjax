"""Tile constants for a 144-tile Hong Kong mahjong set."""

import jax.numpy as jnp

from mahjax._src.types import Array


class Tile:
    """Tile types 0-33 are standard tiles; 34-41 are unique flowers."""

    NUM_STANDARD_TILE_TYPES = 34
    NUM_FLOWER_TYPES = 8
    NUM_TILE_TYPE = 42
    NUM_TILE_ID = 144

    EAST = 27
    SOUTH = 28
    WEST = 29
    NORTH = 30
    WHITE = 31
    GREEN = 32
    RED = 33
    FLOWER_START = 34
    FLOWER_END = 42

    # Four copies of each standard tile followed by one of each flower/season.
    FROM_TILE_ID_TO_TILE = jnp.concatenate(
        [
            (jnp.arange(136) // 4).astype(jnp.int8),
            jnp.arange(FLOWER_START, FLOWER_END, dtype=jnp.int8),
        ]
    )

    @staticmethod
    def from_tile_id_to_tile(tile_id: Array) -> Array:
        return Tile.FROM_TILE_ID_TO_TILE[tile_id]

    @staticmethod
    def is_flower(tile: Array) -> Array:
        return (tile >= Tile.FLOWER_START) & (tile < Tile.FLOWER_END)

    @staticmethod
    def flower_seat(tile: Array) -> Array:
        """Return the matching seat (east=0 through north=3), or -1."""
        return jnp.where(Tile.is_flower(tile), (tile - Tile.FLOWER_START) % 4, -1)

    @staticmethod
    def is_standard(tile: Array) -> Array:
        return (tile >= 0) & (tile < Tile.NUM_STANDARD_TILE_TYPES)
