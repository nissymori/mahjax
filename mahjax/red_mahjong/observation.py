from typing import Dict
import jax.numpy as jnp
from mahjax.red_mahjong.env import State
from mahjax.red_mahjong.env import Array
import jax


@jax.jit
def hand_counts_to_idx(counts: Array, fill: int = -1, hand_size: int = 14) -> Array:
    # Check the input in the JIT outer loop, but keep the minimum guard
    counts = counts.astype(jnp.int32)
    # Each column of (37,4) is 0,1,2,3, and if (col_index < count) is True, then the tile is selected
    col = jnp.arange(4)[None, :]  # (1,4)
    mask = col < counts[:, None]  # (37,4) bool

    # Value table: if selected, the tile index, if not selected, fill
    tile_ids = jnp.tile(jnp.arange(37, dtype=jnp.int32)[:, None], (1, 4))  # (37,4)
    vals = jnp.where(mask, tile_ids, fill)  # (37,4) The contents are i or -1
    vals = vals.reshape(-1)  # (148,)

    # Sort the mask by True(=1) to the front to move the True to the front
    key = mask.reshape(-1).astype(jnp.int32)  # (148,)
    # argsort is ascending, so -key moves True to the front
    order = jnp.argsort(-key, stable=True)
    sorted_vals = vals[order]

    # Extract the top hand_size (the rest should be fill, but just in case, use where)
    out = sorted_vals[:hand_size]
    out = jnp.where(out == fill, fill, out).astype(jnp.int32)
    return out

def _observe_dict(state: State) -> Dict:
    """
    - hand: (14,) player's hand [0-36], -1 means empty
    - last_draw: (1,) The last drawn tile [0-36], -1 means no draw
    - action history: (3, 200) action history [player, action(tile), tsumogiri], player index is relative to the current player in [0, 3], discards store the actual tile in [0, 36], non-discard actions store the raw action id in [0, 86], and tsumogiri is 0/1 for discards and -1 otherwise
    - shanten count: (1,) The number of shanten (0-6)
    - furiten: (1,) Whether the player is in furiten [True/False]
    - scores: (4,) The scores of the players ordered from the current player's perspective (c_p, right, across, left)
    - round: (1,) The round number (0-12)
    - honba: (1,) The honba number
    - kyotaku: (1,) The kyotaku number
    - round wind: (1,) The round wind [0-3]
    - seat wind: (1,) The seat wind [0-3]
    - dora indicators: (4,) The dora indicators [0-36] (red-aware), -1 means no dora
    """
    c_p = state.current_player
    c_p_based_order = (jnp.arange(4) + c_p) % 4
    # hand features
    hand_c_p_37 = state.players.hand_with_red[c_p]
    hand_c_p_14 = hand_counts_to_idx(hand_c_p_37)
    # action histories
    player_history = state.round_state.action_history[0, :].astype(jnp.int32)  # (200,)
    valid_history = player_history >= 0  # default value is -1, so we need to mask it
    relative_player_history = jnp.mod(player_history - jnp.int32(c_p), 4).astype(
        state.round_state.action_history.dtype
    )  # translate the player index to the relative index. e.g. if the original player index is 1, and the current player index is 3, then the relative player index is 2.
    relative_player_history = jnp.where(
        valid_history, relative_player_history, state.round_state.action_history[0, :]
    )
    action_history = state.round_state.action_history.at[0, :].set(relative_player_history)
    # game features
    shanten_c_p = state.round_state.shanten_current_player
    furiten = state.players.furiten_by_discard[c_p] | state.players.furiten_by_pass[c_p]
    scores = state.round_state.score[c_p_based_order]
    _round = state.round_state.round
    honba = state.round_state.honba
    kyotaku = state.round_state.kyotaku
    prevalent_wind = jnp.int8(state.round_state.round // 4)
    seat_wind = state.round_state.seat_wind[c_p]
    # Keep red-aware [0-36]: a red five revealed as an indicator is dead, which is
    # information-bearing for remaining-aka availability and opponent value inference,
    # even though red/black does not change which tile is dora.
    dora_indicators = state.round_state.dora_indicators[:4]  # (4,)
    return {
        "hand": hand_c_p_14,
        "last_draw": state.round_state.last_draw,
        "action_history": action_history,
        "shanten_count": shanten_c_p,
        "furiten": furiten,
        "scores": scores,
        "round": _round,
        "honba": honba,
        "kyotaku": kyotaku,
        "prevalent_wind": prevalent_wind,
        "seat_wind": seat_wind,
        "dora_indicators": dora_indicators,
    }


def _observe_2D(state: State) -> Array:
    """
    TBD
    """
    pass
  