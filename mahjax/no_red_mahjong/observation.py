from typing import Dict

import jax
import jax.numpy as jnp

from mahjax._src.types import Array
from mahjax.no_red_mahjong.state import State
from mahjax.no_red_mahjong.tile import Tile
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.meld import EMPTY_MELD, Meld
from mahjax.no_red_mahjong.shanten import Shanten
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
    """Observation for the current player. Mirrors ``red_mahjong/observation.py``;
    see that module for the reasoning behind each field. Every field is either the
    current player's own private information or public information.

    The no-red variant differs from red only where red fives force a distinction:
    tiles are plain types [0-33] everywhere (no 37-wide ids, no red-aware ``hand``,
    no ``Tile.to_tile_type`` mapping), and the action space is correspondingly
    narrower (discards 0-33, self-kan 34-67, no ``*_RED`` call variants).

    Shapes marked ``()`` are 0-d arrays, not length-1 vectors.

    - hand: (14,) int32, concealed hand as tile types [0-33], -1 padding. Melded
      tiles are NOT here.
    - last_draw: () int8, the observer's own most recent draw [0-33], -1 when they
      are not currently holding a draw. Masked rather than passed through: the
      underlying state field is round-level, and during a robbing-a-kan window it
      still holds the kan declarer's private draw while the responder is to act.
    - action_history: (3, 200) int8, this round's actions in order
      [player, action, tsumogiri]; player is relative to the observer (0 == me),
      discards store the tile and other actions the raw action id, told apart by the
      tsumogiri channel (0/1 for discards, -1 otherwise). Unused slots are -1.
    - shanten_count: () int8, shanten of the concealed hand, [-1, 6]
    - discard_shanten: (34, 3) int8, shanten after discarding each tile type, split
      into [normal, seven pairs, thirteen orphans], per-column ranges
      [0, 8] / [0, 13] / [0, 13]. ``Shanten.NOT_IN_HAND`` (14) marks types not held.
    - is_discard_node: () bool, whether a discard is legal (3n+2 hand)
    - furiten: () bool
    - target: () int8, the tile a pending call/ron decision is about, -1 if none
    - last_player: () int8, relative seat of whoever acted last, -1 if none
    - river: (6, 4, 24) int8, [tile, riichi, called_away, tsumogiri, src, meld_type],
      rows (me, right, across, left); ``src`` is ``(discarder - caller) mod 4`` on the
      DISCARDER's row, so the caller is ``(row - src) mod 4``
    - melds: (3, 4, 4) int8, [action, called_tile, src], rows as above; ``src`` here is
      ``(discarder - owner) mod 4`` -- the OPPOSITE frame from ``river`` -- and 0 means
      a closed kan
    - tiles_seen: (34,) int8, copies of each tile type already visible, [0, 4]
    - scores: (4,) int32, seat-rotated
    - round / honba / kyotaku: () int8
    - wall_remaining: () int32, drawable tiles left in the live wall, [0, 70]
    - prevalent_wind: () int8, round // 4
    - seat_wind: () int8, the current player's seat wind [0-3]. NOTE: unlike red,
      this is NOT 0 for the dealer whenever the dealer sits at seat 1 or 3 --
      ``no_red``'s ``_calc_wind`` returns ``[(e + i) % 4]`` where red returns
      ``[(i - e) % 4]``, and every consumer indexes it by player. That is a
      pre-existing no_red defect (it also feeds yaku.py:352 seat-wind yakuhai and
      fu), not a property of this observation; fixing it there fixes this key too.
    - dora_indicators: (5,) int8, [0-33], -1 for unrevealed slots
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
    )
    relative_player_history = jnp.where(
        valid_history, relative_player_history, state.round_state.action_history[0, :]
    )
    action_history = state.round_state.action_history.at[0, :].set(relative_player_history)

    # game features
    # Computed here rather than cached in ``RoundState``: it is a pure function of the
    # hand being observed, and this function already defines the observer as ``c_p``.
    # As a cached field it had to be re-established by every round-construction path,
    # and the one that rebuilt RoundState from defaults after the step had finalized
    # silently reset it to 0 (== tenpai) for the first observation of every new round.
    shanten_c_p = Shanten.number(hand_c_p_34).astype(jnp.int8)
    # Per-discard shanten. Computed unconditionally on purpose: under vmap each lane
    # sits at a different node type, so "is this a discard node" is a per-lane tracer
    # and both sides of any branch are evaluated anyway. A ``lax.cond`` here is not
    # merely useless but catastrophic -- it makes XLA materialize the ~70MiB shanten
    # CACHE once per lane (PR #65).
    discard_shanten = Shanten.detailed_discard_shanten(hand_c_p_34).astype(jnp.int8)
    # Discard actions are 0..33; 34..67 are self-kan, so the slice stops at 34.
    # ``~state.terminated`` is load-bearing: ``step`` replaces the mask with all-True
    # on termination, which would otherwise report every terminal observation as a
    # discard node.
    is_discard_node = ~state.terminated & (
        state.legal_action_mask[: Tile.NUM_TILE_TYPE].any()
        | state.legal_action_mask[Action.TSUMOGIRI]
    )
    # ``last_draw`` is round-level, not per-player. ``_discard`` clears it to -1, so at
    # an ordinary call node it is already empty; the exception is the robbing-a-kan
    # window, where the declarer keeps their draw while ``current_player`` switches to
    # the responder. Report it only when the observer can still act on it.
    observer_holds_draw = is_discard_node | state.legal_action_mask[Action.TSUMO]
    last_draw = jnp.where(
        observer_holds_draw, state.round_state.last_draw, jnp.int8(-1)
    ).astype(jnp.int8)
    furiten = state.players.furiten_by_discard[c_p] | state.players.furiten_by_pass[c_p]
    target = state.round_state.target
    last_player_abs = state.round_state.last_player
    last_player = jnp.where(
        last_player_abs >= 0,
        jnp.mod(last_player_abs.astype(jnp.int32) - jnp.int32(c_p), 4),
        jnp.int32(-1),
    ).astype(jnp.int8)

    # Public table state, rotated so row 0 is the observer. Both ``src`` fields are
    # DIFFERENCES of absolute seats, so rotating the rows leaves them valid unchanged.
    river = River.decode_river(state.players.river[c_p_based_order]).astype(jnp.int8)
    melds_rotated = state.players.melds[c_p_based_order]
    melds = jnp.stack(
        [
            Meld.action(melds_rotated),
            # No red fives here, so ``target`` IS the tile; red needs ``target_tile``
            # to re-attach the red bit.
            Meld.target(melds_rotated),
            Meld.src(melds_rotated),
        ],
        axis=0,
    ).astype(jnp.int8)

    # Visible/dead tile counts. The called tile must not be counted twice:
    # ``River.add_meld`` leaves it in the DISCARDER's river with only a ``gray`` bit
    # set while the caller's meld also contains it, so rivers are counted EXCLUDING
    # called-away slots and melds are counted in full.
    seen = hand_c_p_34.astype(jnp.int32)

    def _add(counts: Array, idx: Array, addend: Array, keep: Array) -> Array:
        # Guard the scatter: a raw -1 would WRAP to bin 33 rather than be dropped.
        return counts.at[jnp.where(keep, idx, 0)].add(jnp.where(keep, addend, 0))

    river_tile = river[0].astype(jnp.int32).reshape(-1)  # (96,) tile types, -1 empty
    river_called_away = river[2].astype(jnp.int32).reshape(-1)
    seen = _add(
        seen, river_tile, jnp.int32(1), (river_tile >= 0) & (river_called_away == 0)
    )

    # Melds store only (action, target, src), never the tile list, so the tile set is
    # reconstructed from the action id.
    melds_flat = state.players.melds.reshape(-1)  # (16,) all seats; rotation irrelevant
    meld_valid = melds_flat != EMPTY_MELD
    meld_action = Meld.action(melds_flat)
    meld_target = Meld.target(melds_flat)
    meld_is_chi = Meld.is_chi(melds_flat)
    # Pon is 3 copies, any kan (open / closed / added) is 4.
    seen = _add(
        seen,
        meld_target,
        jnp.where(Meld.is_kan(melds_flat), jnp.int32(4), jnp.int32(3)),
        meld_valid & ~meld_is_chi,
    )
    # CHI_L/CHI_M/CHI_R are consecutive ids here (no interleaved ``*_RED`` variants),
    # so ``action - CHI_L`` is directly the called tile's position in the run and
    # subtracting it gives the run's lowest tile. Red needs ``Meld._chi_index`` for
    # the same thing because its chi ids are interleaved with red variants.
    chi_low = meld_target - (meld_action - Action.CHI_L)
    for offset in range(3):
        seen = _add(seen, chi_low + offset, jnp.int32(1), meld_valid & meld_is_chi)

    dora_tile = state.round_state.dora_indicators.astype(jnp.int32)
    seen = _add(seen, dora_tile, jnp.int32(1), dora_tile >= 0)
    tiles_seen = seen.astype(jnp.int8)

    scores = state.round_state.score[c_p_based_order]
    _round = state.round_state.round
    honba = state.round_state.honba
    kyotaku = state.round_state.kyotaku
    # Live wall remaining. Use the between-turns formula (``+ 1``): ``next_deck_ix``
    # indexes the next drawable tile and ``last_deck_ix`` is the haitei line, itself
    # the last drawable index, so the count is inclusive of both ends. Do NOT copy the
    # ``next - last`` form used inside the draw path, which is deliberately off by one
    # because it runs before ``next_deck_ix`` is decremented.
    wall_remaining = jnp.maximum(
        state.round_state.next_deck_ix.astype(jnp.int32)
        - state.round_state.last_deck_ix.astype(jnp.int32)
        + 1,
        0,
    )
    prevalent_wind = jnp.int8(state.round_state.round // 4)
    seat_wind = state.round_state.seat_wind[c_p]
    dora_indicators = state.round_state.dora_indicators
    return {
        "hand": hand_c_p_14,
        "last_draw": last_draw,
        "action_history": action_history,
        "shanten_count": shanten_c_p,
        "discard_shanten": discard_shanten,
        "is_discard_node": is_discard_node,
        "furiten": furiten,
        "target": target,
        "last_player": last_player,
        "river": river,
        "melds": melds,
        "tiles_seen": tiles_seen,
        "scores": scores,
        "round": _round,
        "honba": honba,
        "kyotaku": kyotaku,
        "wall_remaining": wall_remaining,
        "prevalent_wind": prevalent_wind,
        "seat_wind": seat_wind,
        "dora_indicators": dora_indicators,
    }


def _observe_2D(state: State) -> Array:
    """
    TBD
    """
    pass