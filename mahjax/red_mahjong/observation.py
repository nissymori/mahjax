from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp

# Import from the leaf modules rather than from ``env``: ``env`` imports this
# module, so ``import mahjax.red_mahjong.observation`` used to raise a circular
# ImportError unless ``env`` happened to have been imported first.
from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.constants import (
    MAX_DORA_INDICATORS,
    NUM_TILE_TYPES_WITH_RED,
    RED_FIVE_TILE_TYPES,
)
from mahjax.red_mahjong.hand import Hand
from mahjax.red_mahjong.meld import EMPTY_MELD, Meld
from mahjax.red_mahjong.shanten import Shanten
from mahjax.red_mahjong.state import State
from mahjax.red_mahjong.tile import River, Tile
from mahjax.red_mahjong.types import Array


# ``fill`` and ``hand_size`` must be static: ``hand_size`` is a slice bound, so a
# traced value fails outright, and a traced ``fill`` silently changes what the
# padding compares equal to. Without this only the defaults ever worked.
@partial(jax.jit, static_argnames=("fill", "hand_size"))
def hand_counts_to_idx(counts: Array, fill: int = -1, hand_size: int = 14) -> Array:
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
    """Observation for the current player. Every field is either the current
    player's own private information or public information; nothing here is
    hidden from a human sitting in that seat.

    Shapes marked ``()`` are 0-d arrays, not length-1 vectors.

    HAND-RELATED -- what I am holding and what I can do with it.
    - hand: (34,) int8, COUNT of each tile TYPE in the concealed hand, [0, 4]. Red
      fives are folded into their type (5m red and 5m black both count at index 4);
      ``is_red`` carries the redness. Melded tiles are NOT here -- they left the
      concealed hand -- so the counts sum to fewer than 13/14 when melds exist.
    - is_red: (34,) bool, whether the concealed hand holds the RED five of that type.
      Only indices 4 / 13 / 22 (5m / 5p / 5s) can ever be True, since there is exactly
      one red five per suit. Together with ``hand`` this is lossless: it is the same
      information the old (14,) red-aware id list carried, in the shape consumers
      actually wanted.
    - last_draw: () int8, the observer's own most recent draw [0-36], -1 when they
      are not currently holding a draw. Masked rather than passed through: the
      underlying state field is round-level, and during a robbing-a-kan window it
      still holds the kan declarer's private draw while the responder is to act.
    - shanten_count: () int8, shanten of the concealed hand, [-1, 6]
      (-1 == complete hand, 0 == tenpai)
    - discard_shanten: (34, 3) int8, shanten after discarding each tile TYPE,
      split into [normal, seven pairs, thirteen orphans], standard notation with
      per-column ranges [0, 8] / [0, 13] / [0, 13] (the min across the three is
      the ordinary shanten and is in [0, 6]).
      ``Shanten.NOT_IN_HAND`` (14) marks tile types the hand does not hold.
      Indexed by tile type, not by hand slot, and not red-aware: discarding a red
      five and a black five give the same shanten. At a call/response node the
      hand is 3n+1, so these are the values for the 3n hands one further discard
      away -- see ``is_discard_node``.
    - can_win: (34,) bool, SELF ONLY -- for each tile type, whether it completes the
      observer's hand. Meaningful at tenpai (it is the wait table). Opponents' rows
      exist in state but are hidden information and are deliberately not exposed.
      STALE AFTER PON/CHI: the env refreshes it in _init, _discard and _draw_after_kan
      but not in _pon/_chi, so at a post-call discard node it still describes the
      pre-call hand and may disagree with ``shanten_count`` (which is recomputed here
      on every call). Prefer ``shanten_count`` where they conflict.
    - discard_can_win: (34, 34) int8, per-discard wait table. Row t is "if I discard
      tile type t, which tiles then complete my hand". `-1` fills rows that have no
      answer: not a discard node, or a tile the hand does not hold. Elsewhere 0/1.
      This is what separates two discards that both reach tenpai -- ``discard_shanten``
      only says both are tenpai, not which wait is better. Derived at observe time,
      not cached in state.
    - furiten: () bool, whether the player is currently furiten. A property of this
      hand's wait set, not of the table.
    - is_discard_node: () bool, True when a discard (or tsumogiri) is legal, i.e.
      the concealed hand is 3n+2 and ``discard_shanten`` answers "what if I
      discard this". False at ron/pon/chi/kan/pass nodes. It says which question
      the other hand features are answering, so it is read alongside them.

    MELD-RELATED -- the open sets, mine and everyone else's.
    - melds: (3, 4, 4) int8, every player's melds, rows ordered as above and one
      column per meld slot. Channels are [action, called_tile, src]:
        * action is the raw action id that formed the meld (-1 for an empty slot,
          which is the mask); 37-70 are self-kan, 75-83 are pon/kan/chi calls
        * called_tile is red-aware [0-36]
        * src here is ``(discarder - meld owner) mod 4``, i.e. the OPPOSITE frame
          from the river channel: the discarder's row is ``(row + src) mod 4``.
          0 means a closed kan (no discarder). Also a seat difference, so it
          survives the rotation unchanged.
      A player's own melds are here too -- they are absent from ``hand``, which
      holds only the concealed tiles.

    ACTION HISTORY -- the round as a sequence.
    - action_history: (3, 200) int8, this round's actions in order
      [player, action, tsumogiri]:
        * player is relative to the current player, 0 == me, in [0, 3]
        * action stores the discarded tile in [0, 36] for discards (TSUMOGIRI is
          resolved to the tile it discards) and the raw action id in [0, 86]
          otherwise; the two are told apart by the tsumogiri channel
        * tsumogiri is 0/1 for discards and -1 for everything else
      Unused slots are -1. The buffer is per-round and is cleared at every round
      boundary.

    GLOBAL -- table context that is not about any one tile or event.
    - scores: (4,) int32, scores ordered from the current player's seat
      (me, right, across, left)
    - round: () int8, the kyoku counter in [0, round_limit]. ``RedMahjong`` sets
      round_limit to 4 for 'east' and 8 for 'single'/'half', so the widest range is
      [0, 8] -- not the [0, 12] an older docstring claimed.
    - round_limit: () int8, last kyoku index of the game: 4 for 'east', 8 for
      'single'/'half'. With ``round`` it gives how much game is left.
    - honba: () int8
    - kyotaku: () int8
    - prevalent_wind: () int8, the round wind, round // 4. Reaches 2 (West), not
      just {0 East, 1 South}: ``round`` runs to ``round_limit`` == 8 for
      'single'/'half', and 8 // 4 == 2 on that last kyoku.
    - seat_wind: () int8, the current player's seat wind [0-3]; 0 is the dealer
    - dora_indicators: (MAX_DORA_INDICATORS,) int8, indicators [0-36] (red-aware),
      -1 for slots not yet revealed
    - ippatsu: (4,) bool, seat-rotated. Not derivable from anything else here.
    - riichi: (4,) bool, seat-rotated. Public for every seat.
    - is_hand_concealed: (4,) bool, seat-rotated. NOT derivable from `melds`: a closed
      kan leaves the hand concealed, so deriving it would mean reimplementing that
      rule. Gates riichi legality and menzen tsumo.
    - wall_remaining: () int32, tiles still drawable from the live wall, [0, 70].
      The round's clock: it drives haitei, the exhaustive draw, and the env's own
      riichi precondition. Counted inclusively between ``next_deck_ix`` and the
      haitei line, matching ``visualization.remaining_tiles``.
    - tiles_seen: (34,) int8, how many copies of each tile TYPE are already visible
      from this seat, [0, 4]. Own concealed hand + every river + every meld +
      revealed dora indicators. Red fives fold into their type. A called tile is
      counted once, not twice: rivers contribute only their non-called-away slots
      (the tile stays in the discarder's river with a `called_away` bit while the
      caller's meld also holds it). Derivable from `hand`/`river`/`melds` -- it is a
      scatter-add, not a walk -- but provided directly since it costs nothing.
    - target: () int8, the tile a pending call/ron decision is about, -1 if none
    - last_player: () int8, relative seat of whoever acted last, -1 if none
    """
    c_p = state.current_player
    c_p_based_order = (jnp.arange(4) + c_p) % 4
    # hand features
    hand_c_p_37 = state.players.hand_with_red[c_p]
    # Counts + a red flag, not an id list: consumers want "how many of type t" and
    # were all redoing the same scatter. Reds fold into their type.
    hand_c_p_34 = state.players.hand[c_p].astype(jnp.int8)
    is_red_c_p = (
        jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.bool_)
        .at[jnp.asarray(RED_FIVE_TILE_TYPES)]
        .set(hand_c_p_37[Tile.NUM_TILE_TYPE : NUM_TILE_TYPES_WITH_RED] > 0)
    )
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
    # Derived here, not cached: as a RoundState field every round-construction path had
    # to re-establish it, and the one rebuilding from defaults reset it to 0 (tenpai)
    # for the first observation of each round.
    shanten_c_p = Shanten.number(state.players.hand[c_p]).astype(jnp.int8)
    # Unconditional on purpose. ``is_discard_node`` is a per-lane tracer under vmap, so
    # a lax.cond evaluates both sides anyway AND makes XLA materialize the ~70MiB
    # shanten CACHE per lane (7x slower at batch 256, OOM at 1024).
    discard_shanten = Shanten.detailed_discard_shanten(state.players.hand[c_p]).astype(jnp.int8)
    # Per-discard wait table. ``discard_shanten`` says WHETHER a discard reaches tenpai;
    # this says WHAT the wait is -- the difference between two tenpai discards. Only
    # meaningful at 3n+2 for held tiles: at 3n+1 the post-discard hand is 3n and
    # ``can_ron`` is structurally False. Unconditional then masked, same lax.cond trap
    # as above. Not cached in state: (34,34) per player, derivable from the hand.
    _cand = jnp.maximum(hand_c_p_34[None, :] - jnp.eye(Tile.NUM_TILE_TYPE, dtype=hand_c_p_34.dtype), 0)
    _dcw = jax.vmap(
        lambda h: jax.vmap(lambda u: Hand.can_ron(h, u))(jnp.arange(Tile.NUM_TILE_TYPE))
    )(_cand)

    # Discard actions are 0..36; 37..70 are self-kan, so the slice stops at 37.
    # ``~state.terminated`` is load-bearing: ``step`` replaces the mask with all-True on
    # termination, which would otherwise report every terminal observation as a discard
    # node while ``discard_shanten`` describes a hand nobody will play.
    is_discard_node = ~state.terminated & (
        state.legal_action_mask[:NUM_TILE_TYPES_WITH_RED].any()
        | state.legal_action_mask[Action.TSUMOGIRI]
    )
    # -1 means "no answer here": either not a discard node, or a tile the hand does
    # not hold so there is nothing to discard. Signed, so -1 stays distinct from a
    # genuine "discarding t leaves me waiting on nothing" (all zeros).
    discard_can_win = jnp.where(
        is_discard_node & (hand_c_p_34 > 0)[:, None],
        _dcw.astype(jnp.int8),
        jnp.int8(-1),
    )
    # ``last_draw`` is round-level, not per-player. ``_discard`` clears it, so ordinary
    # response nodes are already empty. The exception is robbing-a-kan: the declarer
    # keeps their draw while ``current_player`` switches to the responder. Gate on "can
    # the observer act on it", not on ``kan_declared`` (also set when the declarer
    # legitimately keeps their rinshan).
    # ``~state.terminated`` guards BOTH terms: ``step`` sets the mask all-True on
    # termination, so an unguarded TSUMO bit would re-expose the tile this mask hides.
    observer_holds_draw = ~state.terminated & (
        is_discard_node | state.legal_action_mask[Action.TSUMO]
    )
    last_draw = jnp.where(
        observer_holds_draw, state.round_state.last_draw, jnp.int8(-1)
    ).astype(jnp.int8)
    furiten = state.players.furiten_by_discard[c_p] | state.players.furiten_by_pass[c_p]
    # The tile a pending call/ron is about, and who let it go. Relative seat, same
    # convention as the history's player channel.
    target = state.round_state.target
    last_player_abs = state.round_state.last_player
    last_player = jnp.where(
        last_player_abs >= 0,
        jnp.mod(last_player_abs.astype(jnp.int32) - jnp.int32(c_p), 4),
        jnp.int32(-1),
    ).astype(jnp.int8)
    # Public table state, rotated so row 0 is the observer (same frame as ``scores``).
    # Derivable from ``action_history`` but only by learning a parser: linking a call to
    # the discard it consumed is a non-adjacent lookup. Decoding it here is free (state
    # reads, no shanten) and removes that sub-problem.
    # Decoded but NOT emitted: every discard is already in ``action_history`` (tile in
    # channel 1, tsumogiri in channel 2) and every call in ``melds``
    # ((action, called_tile, src)), so a river plane would ship the same events twice.
    # It is still needed here to build ``tiles_seen``.
    river = River.decode_river(state.players.river[c_p_based_order]).astype(jnp.int8)
    melds_rotated = state.players.melds[c_p_based_order]
    melds = jnp.stack(
        [
            Meld.action(melds_rotated),
            Meld.target_tile(melds_rotated),
            Meld.src(melds_rotated),
        ],
        axis=0,
    ).astype(jnp.int8)

    # Visible/dead tile counts. The called tile must not be counted twice:
    # ``River.add_meld`` leaves it in the DISCARDER's river with only a ``gray`` bit
    # set while the caller's meld also contains it, so rivers are counted EXCLUDING
    # called-away slots and melds are counted in full. Red fives fold into their type.
    seen = state.players.hand[c_p].astype(jnp.int32)  # (34,) own concealed hand

    def _add(counts: Array, idx: Array, addend: Array, keep: Array) -> Array:
        # Guard the scatter: a raw -1 would WRAP to bin 33 rather than be dropped.
        return counts.at[jnp.where(keep, idx, 0)].add(jnp.where(keep, addend, 0))

    river_tile = river[0].astype(jnp.int32).reshape(-1)  # (96,) red-aware, -1 empty
    river_called_away = river[2].astype(jnp.int32).reshape(-1)
    seen = _add(
        seen,
        Tile.to_tile_type(river_tile),
        jnp.int32(1),
        (river_tile >= 0) & (river_called_away == 0),
    )

    # Melds store only (action, target, src), never the tile list, so the tile set is
    # reconstructed from the action id.
    melds_flat = state.players.melds.reshape(-1)  # (16,) all seats; rotation is irrelevant here
    meld_valid = melds_flat != EMPTY_MELD
    meld_action = Meld.action(melds_flat)
    meld_target = Meld.target(melds_flat)  # already a tile TYPE [0-33]
    meld_is_chi = Meld.is_chi(melds_flat)
    # Pon is 3 copies, any kan (open / closed / added) is 4.
    seen = _add(
        seen,
        meld_target,
        jnp.where(Meld.is_kan(melds_flat), jnp.int32(4), jnp.int32(3)),
        meld_valid & ~meld_is_chi,
    )
    # A chi is three consecutive types; ``_chi_index`` says where the called tile sits
    # in the run, so subtracting it gives the run's lowest tile.
    chi_low = meld_target - Meld._chi_index(meld_action)
    for offset in range(3):
        seen = _add(seen, chi_low + offset, jnp.int32(1), meld_valid & meld_is_chi)

    dora_tile = state.round_state.dora_indicators.astype(jnp.int32)
    seen = _add(seen, Tile.to_tile_type(dora_tile), jnp.int32(1), dora_tile >= 0)
    tiles_seen = seen.astype(jnp.int8)

    # Own wait table, SELF ONLY -- state holds all 4 rows but 3 are hidden. Read from
    # state rather than recomputed; the cache exists to avoid solving completion twice.
    #
    # STALE AFTER PON/CHI: refreshed in _init, _discard and _draw_after_kan but not in
    # _pon/_chi, which rewrite the hand. At a post-call discard node it can claim a wait
    # the player no longer has. ``shanten_count`` is recomputed every call -- trust it.
    can_win = state.players.can_win[c_p]
    # Ippatsu is not derivable from anything else exposed. ``is_hand_concealed`` is NOT
    # derivable from ``melds`` either: a closed kan leaves the hand concealed, so a
    # consumer would have to reimplement that rule to get riichi legality and menzen
    # tsumo right. Both are public for all four seats.
    # Riichi is public for all four seats. It was previously only recoverable from
    # the river plane's riichi channel, which is going away.
    riichi = state.players.riichi[c_p_based_order]
    ippatsu = state.players.ippatsu[c_p_based_order]
    is_hand_concealed = state.players.is_hand_concealed[c_p_based_order]
    scores = state.round_state.score[c_p_based_order]
    _round = state.round_state.round
    round_limit = state.round_state.round_limit
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
    # Red-aware [0-36]: a revealed red five is dead, which matters for remaining-aka
    # availability even though redness does not change which tile is dora. Slice with
    # MAX_DORA_INDICATORS, not a literal, so raising it widens rather than truncates.
    dora_indicators = state.round_state.dora_indicators[:MAX_DORA_INDICATORS]
    return {
        # ---- hand-related: what I am holding and what I can do with it ----------
        "hand": hand_c_p_34,
        "is_red": is_red_c_p,
        "last_draw": last_draw,
        "shanten_count": shanten_c_p,
        "discard_shanten": discard_shanten,
        "can_win": can_win,
        "discard_can_win": discard_can_win,
        "furiten": furiten,
        "is_discard_node": is_discard_node,
        # ---- meld-related: the open sets, mine and everyone else's --------------
        "melds": melds,
        # ---- action history: the round as a sequence, and the pending decision --
        "action_history": action_history,
        # ---- global: table context that is not about any one tile or event ------
        "scores": scores,
        "round": _round,
        "round_limit": round_limit,
        "honba": honba,
        "kyotaku": kyotaku,
        "prevalent_wind": prevalent_wind,
        "seat_wind": seat_wind,
        "dora_indicators": dora_indicators,
        "ippatsu": ippatsu,
        "riichi": riichi,
        "is_hand_concealed": is_hand_concealed,
        "wall_remaining": wall_remaining,
        "tiles_seen": tiles_seen,
        "target": target,  
        "last_player": last_player,
    }


def _observe_2D(state: State) -> Array:
    """
    TBD
    """
    pass
  

def _observe_privileged_dict(state: State) -> Dict:
    """``_observe_dict`` plus the OTHER players' concealed hands.

    This is HIDDEN INFORMATION and must never reach a policy that will be evaluated
    or deployed against real opponents. It exists for centralised-critic training,
    opponent modelling, dataset analysis and debugging, where seeing all four hands
    is legitimate. Everything in the base observation is unchanged, so anything
    consuming ``observe()`` also consumes this.

    Added keys, both in the same seat-relative frame as ``scores`` -- row 0 is the
    player to the current player's RIGHT, row 1 across, row 2 left. The observer's
    own hand is NOT repeated here; it is already ``hand`` / ``is_red``.
    - others_hand: (3, 34) int8, per-type counts of each other player's CONCEALED
      hand, [0, 4]. Melded tiles are excluded, exactly as for ``hand``.
    - others_is_red: (3, 34) bool, whether that player holds the RED five of the type.
    """
    obs = _observe_dict(state)
    # Rows 1..3 of the observer-rooted rotation: right, across, left.
    others = (jnp.arange(1, 4) + state.current_player) % 4
    obs["others_hand"] = state.players.hand[others].astype(jnp.int8)
    obs["others_is_red"] = (
        jnp.zeros((3, Tile.NUM_TILE_TYPE), dtype=jnp.bool_)
        .at[:, jnp.asarray(RED_FIVE_TILE_TYPES)]
        .set(state.players.hand_with_red[others][:, Tile.NUM_TILE_TYPE : NUM_TILE_TYPES_WITH_RED] > 0)
    )
    return obs

def _observe_privileged_2D(state: State) -> Array:
    """
    TBD
    """
    pass