from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import (
    COPIES_PER_TILE,
    HONBA_BONUS,
    LEGAL_ACTION_SIZE,
    MAX_DISCARDS_PER_PLAYER,
    MAX_DORA_INDICATORS,
    MAX_HAND_TILES,
    MAX_MELDS_PER_PLAYER,
    NUM_PHYSICAL_TILES,
    NUM_PLAYERS,
    NUM_TILE_TYPES,
    NUM_TILE_TYPES_WITH_RED,
    RIICHI_BET,
    SENTINEL_DISCARD_VALUE,
    SENTINEL_MELD_VALUE,
    SENTINEL_TILE_ID,
    STARTING_POINTS,
    TARGET_POINTS,
)
from .meld import EMPTY_MELD
from .struct import dataclass
from .tile import EMPTY_RIVER
from .types import PRNGKey


FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)


@dataclass
class GameConfig:
    allow_open_tanyao: jnp.bool_ = TRUE
    allow_kuikae: jnp.bool_ = FALSE
    use_red_fives: jnp.bool_ = TRUE
    allow_double_ron: jnp.bool_ = TRUE
    enable_special_abortive_draw: jnp.bool_ = TRUE
    enable_pao: jnp.bool_ = TRUE
    seed_wall_from_key: jnp.bool_ = TRUE
    starting_points: jnp.int32 = jnp.int32(STARTING_POINTS)
    target_points: jnp.int32 = jnp.int32(TARGET_POINTS)
    honba_bonus: jnp.int32 = jnp.int32(HONBA_BONUS)
    riichi_bet: jnp.int32 = jnp.int32(RIICHI_BET)


def default_game_config() -> GameConfig:
    return GameConfig()


@dataclass
class PlayerStateArrays:
    hand: jnp.ndarray = jnp.zeros((NUM_PLAYERS, NUM_TILE_TYPES), dtype=jnp.int8)
    hand_with_red: jnp.ndarray = jnp.zeros((NUM_PLAYERS, NUM_TILE_TYPES_WITH_RED), dtype=jnp.int8)
    hand_ids: jnp.ndarray = jnp.full((NUM_PLAYERS, MAX_HAND_TILES), SENTINEL_TILE_ID, dtype=jnp.int16)
    hand_counts: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.int8)
    drawn_tile: jnp.ndarray = jnp.full((NUM_PLAYERS,), SENTINEL_TILE_ID, dtype=jnp.int16)
    legal_action_mask: jnp.ndarray = jnp.zeros((NUM_PLAYERS, LEGAL_ACTION_SIZE), dtype=jnp.bool_)
    can_win: jnp.ndarray = jnp.zeros((NUM_PLAYERS, NUM_TILE_TYPES), dtype=jnp.bool_)
    has_yaku: jnp.ndarray = jnp.zeros((NUM_PLAYERS, 2), dtype=jnp.bool_)
    fan: jnp.ndarray = jnp.zeros((NUM_PLAYERS, 2), dtype=jnp.int32)
    fu: jnp.ndarray = jnp.zeros((NUM_PLAYERS, 2), dtype=jnp.int32)
    melds: jnp.ndarray = jnp.full((NUM_PLAYERS, MAX_MELDS_PER_PLAYER), EMPTY_MELD, dtype=jnp.uint16)
    meld_tiles: jnp.ndarray = jnp.full((NUM_PLAYERS, MAX_MELDS_PER_PLAYER, 4), SENTINEL_TILE_ID, dtype=jnp.int16)
    meld_info: jnp.ndarray = jnp.full((NUM_PLAYERS, MAX_MELDS_PER_PLAYER, 3), SENTINEL_MELD_VALUE, dtype=jnp.int8)
    meld_counts: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.int8)
    river: jnp.ndarray = jnp.full((NUM_PLAYERS, MAX_DISCARDS_PER_PLAYER), EMPTY_RIVER, dtype=jnp.uint16)
    discards: jnp.ndarray = jnp.full((NUM_PLAYERS, MAX_DISCARDS_PER_PLAYER), SENTINEL_DISCARD_VALUE, dtype=jnp.int16)
    discard_info: jnp.ndarray = jnp.full((NUM_PLAYERS, MAX_DISCARDS_PER_PLAYER, 4), SENTINEL_MELD_VALUE, dtype=jnp.int8)
    discard_counts: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.int8)
    riichi: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    riichi_declared: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    riichi_step: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.int8)
    double_riichi: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    ippatsu: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    furiten_by_discard: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    furiten_by_pass: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    is_hand_concealed: jnp.ndarray = jnp.ones((NUM_PLAYERS,), dtype=jnp.bool_)
    pon: jnp.ndarray = jnp.zeros((NUM_PLAYERS, NUM_TILE_TYPES), dtype=jnp.int32)
    has_won: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.bool_)
    n_kan: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.int8)
    has_nagashi_mangan: jnp.ndarray = jnp.ones((NUM_PLAYERS,), dtype=jnp.bool_)


@dataclass
class RoundState:
    rng_key: PRNGKey = jax.random.PRNGKey(0)
    action_history: jnp.ndarray = jnp.full((3, 200), -1, dtype=jnp.int8)
    shanten_current_player: jnp.int8 = jnp.int8(0)
    round: jnp.int8 = jnp.int8(0)
    round_limit: jnp.int8 = jnp.int8(7)
    terminated_round: jnp.bool_ = FALSE
    honba: jnp.int8 = jnp.int8(0)
    kyotaku: jnp.int8 = jnp.int8(0)
    init_wind: jnp.ndarray = jnp.array([0, 1, 2, 3], dtype=jnp.int8)
    seat_wind: jnp.ndarray = jnp.array([0, 1, 2, 3], dtype=jnp.int8)
    dealer: jnp.int8 = jnp.int8(0)
    order_points: jnp.ndarray = jnp.array([30, 10, -10, -30], dtype=jnp.int32)
    score: jnp.ndarray = jnp.full((NUM_PLAYERS,), 250, dtype=jnp.int32)
    deck: jnp.ndarray = jnp.zeros((NUM_PHYSICAL_TILES,), dtype=jnp.int8)
    next_deck_ix: jnp.int32 = jnp.int32(83)
    last_deck_ix: jnp.int8 = jnp.int8(14)
    draw_next: jnp.bool_ = FALSE
    last_draw: jnp.int8 = jnp.int8(-1)
    last_player: jnp.int8 = jnp.int8(-1)
    dora_indicators: jnp.ndarray = jnp.full((MAX_DORA_INDICATORS,), -1, dtype=jnp.int8)
    ura_dora_indicators: jnp.ndarray = jnp.full((MAX_DORA_INDICATORS,), -1, dtype=jnp.int8)
    is_abortive_draw_normal: jnp.bool_ = FALSE
    dummy_count: jnp.int8 = jnp.int8(0)
    is_haitei: jnp.bool_ = FALSE
    target: jnp.int8 = jnp.int8(-1)
    n_kan_doras: jnp.int8 = jnp.int8(0)
    kan_declared: jnp.bool_ = FALSE
    can_after_kan: jnp.bool_ = FALSE
    can_robbing_kan: jnp.bool_ = FALSE


@dataclass
class EnvState:
    current_player: jnp.int8 = jnp.int8(0)
    legal_action_mask: jnp.ndarray = jnp.zeros((LEGAL_ACTION_SIZE,), dtype=jnp.bool_)
    players: PlayerStateArrays = PlayerStateArrays()
    round_state: RoundState = RoundState()
    step_count: jnp.int32 = jnp.int32(0)
    rewards: jnp.ndarray = jnp.zeros((NUM_PLAYERS,), dtype=jnp.float32)
    terminated: jnp.bool_ = FALSE
    truncated: jnp.bool_ = FALSE

    @property
    def env_id(self) -> str:
        return "red_mahjong"


State = EnvState


def default_state() -> EnvState:
    return EnvState()
