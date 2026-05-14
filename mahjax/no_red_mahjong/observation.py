from typing import Dict

import jax
import jax.numpy as jnp

from mahjax._src.types import Array
from mahjax.no_red_mahjong.state import State
from mahjax.no_red_mahjong.tile import Tile
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.meld import Meld
from mahjax.no_red_mahjong.tile import River

def hand_counts_to_idx(counts: Array, fill: int = -1, hand_size: int = 14) -> Array:
    # Check the input in the JIT outer loop, but keep the minimum guard
    counts = counts.astype(jnp.int32)
    # Each column of (34,4) is 0,1,2,3, and if (col_index < count) is True, then the tile is selected
    col = jnp.arange(4)[None, :]  # (1,4)
    mask = col < counts[:, None]  # (34,4) bool

    # Value table: if selected, the tile index, if not selected, fill
    tile_ids = jnp.tile(jnp.arange(34, dtype=jnp.int32)[:, None], (1, 4))  # (34,4)
    vals = jnp.where(mask, tile_ids, fill)  # (34,4) The contents are i or -1
    vals = vals.reshape(-1)  # (136,)

    # Sort the mask by True(=1) to the front to move the True to the front
    key = mask.reshape(-1).astype(jnp.int32)  # (136,)
    # argsort is ascending, so -key moves True to the front
    order = jnp.argsort(-key, stable=True)
    sorted_vals = vals[order]

    # Extract the top hand_size (the rest should be fill, but just in case, use where)
    out = sorted_vals[:hand_size]
    out = jnp.where(out == fill, fill, out).astype(jnp.int32)
    return out
    
def _observe_dict(state: State) -> Dict:
    """
    - hand: (14,) player's hand [0-33], -1 means empty
    - last_draw: (1,) The last drawn tile [0-33], -1 means no draw
    - action history: (3, 200) action history [player, action, is_tsumogiri], player index is relative to the current player in [0, 3], discards store the discarded tile, non-discard actions store the raw action id, and is_tsumogiri is 0/1 for discards and -1 otherwise
    - shanten count: (1,) The number of shanten (0-6)
    - furiten: (1,) Whether the player is in furiten [True/False]
    - scores: (4,) The scores of the players ordered from the current player's perspective (c_p, right, across, left)
    - round: (1,) The round number (0-12)
    - honba: (1,) The honba number
    - kyotaku: (1,) The kyotaku number
    - round wind: (1,) The round wind [0-3]
    - seat wind: (1,) The seat wind [0-3]
    - dora indicators: (4,) The dora indicators [0-33], -1 means no dora
    """
    c_p = state.current_player
    c_p_based_order = (jnp.arange(4) + c_p) % 4
    # hand features
    hand_c_p_34 = state.players.hand[c_p]
    hand_c_p_14 = hand_counts_to_idx(hand_c_p_34)
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

    last_draw = state.round_state.last_draw

    # game features
    shanten_c_p = state.round_state.shanten_current_player
    furiten = state.players.furiten_by_discard[c_p] | state.players.furiten_by_pass[c_p]
    scores = state.round_state.score[c_p_based_order]
    _round = state.round_state.round
    honba = state.round_state.honba
    kyotaku = state.round_state.kyotaku
    prevalent_wind = state.round_state.seat_wind[c_p]
    seat_wind = state.round_state.init_wind[c_p]
    dora_indicators = state.round_state.dora_indicators[:4]  # (4,)
    return {
        "hand": hand_c_p_14,
        "last_draw": last_draw,
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
    Basically based on moral's observation: https://github.com/nissymori/Mahjoong/blob/main/CLAUDE.md
    But slightly modified for ease of implementation and memory efficiency.
    All the features are sorted from the current player's perspective.
    Observation: (299, 34)
    - Hand Features (7 channels)
        - Tiles in hand: 4 channels
        - Waiting tile: 1 channel
        - Furiten: 1 channel [binary]
        - Shanten count (0-6): 1 channels [normalized to 0-1]
    - Game Features (15 channels)
        - Scores: 4 channels [normalized to 0-1]
        - Rank of Current Player: 1 channel [normalized to 0-1]
        - Round: 1 channel [normalized to 0-1]
        - Honba: 1 channel [normalized to 0-1]
        - Kyotaku: 1 channel [normalized to 0-1]
        - Wind: 2 channels (seat and round winds) [normalized to 0-1]
        - Tiles remaining: 1 channel [normalized to 0-1]
        - Dora indicators: 4 channels
    - River Features (216 channels)
        - Tile: 4(players) * 18 = 72 channel TODO: mortalはrecent 6も追加で入れている.
        - Discard flags: (Kan, Pon, Chi(L,M,R), Riichi) 4(player) * 18(river_length) = 72 channels [normalized to 0-1]
        - Tedashi/Tsumogiri: 4(player) * 18(river_length) = 72 channels [normalized to 0-1]
    - Meld Features (48 channels)
        - Src Player: 4(player) * 4(possible melds) = 16 channels [normalized to 0-1]
        - Target tile: 4 channels * 4(possible melds) = 16 channels [normalized to 0-1]
        - Meld type: 4(player) * 4(possible melds) = 16 channels [normalized to 0-1]
    - Strategic State Features (23 channels)
        - Riichi states: 4 channels [binary]
        - Riichi discarded tiles: 4 channels
        - Last tedashi: 4 channels  # TODO: placeholder for now
        - Legal actions: 11 channels (discard(1), closed_kan(1), added_kan(1), open_kan(1, binary), pon(1, binary), chi(1, binary), ron(1, binary), pass(1, binary), tsumo(1, binary), riichi(1, binary), dummy(1, binary))
    """
    c_p = state.current_player
    c_p_based_order = (jnp.arange(4) + c_p) % 4
    # ---------- Hand Features (7 ch) ----------
    hand_c = state.players.hand[c_p]  # (34,)
    # Number of tiles: >=1, >=2, >=3, ==4
    thresholds = jnp.array([1, 2, 3, 4], dtype=jnp.int32)[:, None]  # (4,1)
    hand_bins = (hand_c[None, :] >= thresholds).astype(jnp.float32)  # (4,34)
    # Waiting tile (can ron)
    wait_feat = state.players.can_win[c_p][None, :].astype(jnp.float32)  # (1,34)
    # Furiten (broadcast scalar)
    is_furiten = state.players.furiten_by_discard[c_p] | state.players.furiten_by_pass[c_p]
    furiten_feat = jnp.full((1, 34), is_furiten, dtype=jnp.float32)
    # Shanten (0..6 to 0..1 normalized)
    shanten_val = state.round_state.shanten_current_player  # Shanten count is pre-calculated
    shanten_feat = jnp.full((1, 34), (shanten_val / 6.0), dtype=jnp.float32)
    hand_block = jnp.concatenate(
        [hand_bins, wait_feat, furiten_feat, shanten_feat], axis=0
    )  # (7,34)

    # ---------- Game Features (15 ch) ----------
    # The highest score is 100000, the lowest score is -250, so normalize to 0-1 by adding 250
    score_norm = ((state.round_state.score + 250) / 1250.0).astype(jnp.float32)[
        c_p_based_order, None
    ]  # (4,1)
    score_feat = jnp.repeat(score_norm, 34, axis=1)  # (4,34)
    # Rank (higher score is higher rank): 0..3 to 0..1
    # rank_idx = 0: highest rank, 3: lowest rank
    order = jnp.argsort(-state.round_state.score)
    inv = jnp.argsort(order)
    rank_idx = inv[c_p].astype(jnp.float32)
    rank_feat = jnp.full((1, 34), rank_idx / 3.0, dtype=jnp.float32)
    # Round/Honba/Kyotaku normalization
    round_feat = jnp.full(
        (1, 34),
        (
            state.round_state.round.astype(jnp.float32)
            / jnp.maximum(1.0, state.round_state.round_limit.astype(jnp.float32))
        ),
        dtype=jnp.float32,
    )
    honba_feat = jnp.full(
        (1, 34), (state.round_state.honba.astype(jnp.float32) / 10.0), dtype=jnp.float32
    )
    kyotaku_feat = jnp.full(
        (1, 34), (state.round_state.kyotaku.astype(jnp.float32) / 10.0), dtype=jnp.float32
    )
    # Wind (seat wind and prevalent wind) 0..3 to 0..1
    seat_wind = state.round_state.seat_wind[c_p].astype(jnp.float32) / 3.0
    prevalent_wind = (state.round_state.round % 4).astype(jnp.float32) / 3.0
    wind_feat = jnp.stack(
        [
            jnp.full((34,), seat_wind, dtype=jnp.float32),
            jnp.full((34,), prevalent_wind, dtype=jnp.float32),
        ],
        axis=0,
    )  # (2,34)
    # Remaining tsumo (approximately): (_next_deck_ix - _last_deck_ix + 1) / 70
    tiles_rem = (
        state.round_state.next_deck_ix.astype(jnp.int32)
        - state.round_state.last_deck_ix.astype(jnp.int32)
        + 1
    )
    tiles_rem = jnp.clip(tiles_rem, 0, 70).astype(jnp.float32) / 70.0
    tiles_rem_feat = jnp.full((1, 34), tiles_rem, dtype=jnp.float32)

    # Dora display (maximum 4 tiles to 4ch one-hot)
    # -1 is ignored (zero)
    dora_inds = state.round_state.dora_indicators[:4].astype(jnp.int32)  # (4,)
    valid = (dora_inds >= 0) & (dora_inds < 34)
    # (4,34) one-hot then mask
    dora_oh = (dora_inds[:, None] == jnp.arange(34)[None, :]).astype(
        jnp.float32
    ) * valid[:, None]
    game_block = jnp.concatenate(
        [
            score_feat,
            rank_feat,
            round_feat,
            honba_feat,
            kyotaku_feat,
            wind_feat,
            tiles_rem_feat,
            dora_oh,
        ],
        axis=0,
    )  # (15,34)

    # ---------- River Features (96 * 3 ch) ----------
    # decode: [tile(0..33|-1), riichi(0/1), gray(0/1), tsumogiri(0/1), src(0..3/3), mt(0..5)]
    river = state.players.river[c_p_based_order]
    rdec = River.decode_river(river)  # (6,4,24)
    r_tile = rdec[0]  # (4,24) int32 ( -1 if empty )
    r_riichi = rdec[1].astype(jnp.float32)  # (4,24)
    r_tsumogiri = rdec[3].astype(jnp.float32)  # (4,24)
    # Tile type one-hot: (4,24,34)
    # Empty (-1) is all-zero by clip→one_hot→mask
    t_clip = jnp.clip(r_tile, 0, 33)
    t_oh = (t_clip[..., None] == jnp.arange(34)[None, None, :]).astype(
        jnp.float32
    )  # (4,24,34)
    t_oh = t_oh * (r_tile[..., None] >= 0)  # Empty is 0
    # Interpret each slot as "1 channel" and convert to 72ch
    river_tile_block = t_oh.reshape(4 * 24, 34)  # (96,34)
    river_riichi_block = jnp.repeat(r_riichi.reshape(-1, 1), 34, axis=1)  # (96,34)
    river_tsumogiri_block = jnp.repeat(
        r_tsumogiri.reshape(-1, 1), 34, axis=1
    )  # (96,34)
    river_block = jnp.concatenate(
        [river_tile_block, river_riichi_block, river_tsumogiri_block], axis=0
    )  # (384,34)

    # ---------- Meld Features (48 ch) ----------
    melds = state.players.melds[c_p_based_order].reshape(-1)  # (16,)
    src = Meld.src(melds)  # (16,) # TODO: -1 may exist
    target = Meld.target(melds)  # (16,) # TODO: -1 may exist
    meld_type = Meld.action(melds) / Action.NUM_ACTION  # (16,) # TODO: -1 may exist
    meld_block = jnp.concatenate([src, target, meld_type], axis=0)  # (48,)
    meld_block = jnp.repeat(meld_block[:, None], 34, axis=1).astype(
        jnp.float32
    )  # (48,34)

    # ---------- Strategic Features (23 ch) ----------
    # 4.1: Riichi state
    riichi_states = jnp.repeat(
        state.players.riichi[c_p_based_order].astype(jnp.float32)[:, None], 34, axis=1
    )  # (4,34)
    # 4.2: Riichi declared tile (tile in the slot where riichi==1) /33
    riichi_mask = r_riichi  # (4,24)
    has_riichi = riichi_mask.sum(axis=1) > 0  # (4,)
    riichi_tile = (riichi_mask * r_tile).sum(axis=1)  # (4,)
    riichi_tile = jnp.where(has_riichi, riichi_tile, -1)  # (4,)
    # one-hot representation
    riichi_tile_feat = (riichi_tile[:, None] == jnp.arange(34)[None, :]).astype(
        jnp.float32
    )  # (4,34)
    riichi_tile_block = jnp.where(
        (riichi_tile == -1)[:, None], jnp.zeros((4, 34)), riichi_tile_feat
    )
    # 4.3: Last hand out
    last_tedashi_block = jnp.zeros(
        (4, 34), dtype=jnp.float32
    )  # TODO: placeholder for now

    # 4.4: Action summary (for the current player)
    lam = state.players.legal_action_mask[c_p]  # (NUM_ACTION,)
    # discard: 0..33 any
    feat_discard = lam[: Tile.NUM_TILE_TYPE].any()
    # selfkan: 34..67 any (from here, closed_kan/added_kan is estimated)
    selfkan_mask = lam[Tile.NUM_TILE_TYPE : Action.TSUMOGIRI]
    has_selfkan = selfkan_mask.any()
    # added_kan estimation: If there is a tile in the selfkan candidate that has _pon information, then "added kan"
    # Extract one tile idx weighted by the sum (multiple can be set, use the maximum idx)
    idx34 = jnp.arange(Tile.NUM_TILE_TYPE)
    cand_idx = jnp.argmax(selfkan_mask * idx34)  # 0..33 any
    has_added_kan = has_selfkan & (state.players.pon[(c_p, cand_idx)] > 0)
    has_closed_kan = has_selfkan & jnp.logical_not(has_added_kan)

    feat_open_kan = lam[Action.OPEN_KAN]
    feat_pon = lam[Action.PON]
    feat_chi = lam[Action.CHI_L : Action.CHI_R + 1].any()
    feat_ron = lam[Action.RON]
    feat_pass = lam[Action.PASS]
    feat_tsumo = lam[Action.TSUMO]
    feat_riichi = lam[Action.RIICHI]
    feat_dummy = lam[Action.DUMMY]

    strat_vec = jnp.stack(
        [
            feat_discard,
            has_closed_kan,
            has_added_kan,
            feat_open_kan,
            feat_pon,
            feat_chi,
            feat_ron,
            feat_pass,
            feat_tsumo,
            feat_riichi,
            feat_dummy,
        ],
        axis=0,
    ).astype(
        jnp.float32
    )  # (11,)
    strat_block = jnp.repeat(strat_vec[:, None], 34, axis=1)  # (11,34)
    strategic_block = jnp.concatenate(
        [riichi_states, riichi_tile_block, last_tedashi_block, strat_block], axis=0
    )  # (23,34)

    # ---------- Concatenate all ----------
    obs = jnp.concatenate(
        [hand_block, game_block, river_block, meld_block, strategic_block], axis=0
    ).astype(
        jnp.float32
    )  # (299,34)

    return obs
