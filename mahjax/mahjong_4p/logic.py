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


import dataclasses
from typing import Literal, Optional, Tuple

import jax
import jax.numpy as jnp

from mahjax._src.types import Array, PRNGKey
from mahjax.common.action import Action
from mahjax.common.config import GameConfig
from mahjax.common.constants import DORA_ARRAY, TILE_RANGE
from mahjax.common.hand import Hand
from mahjax.common.meld import Meld
from mahjax.common.tile import River, Tile
from mahjax.common.yaku import Yaku

from .config import RED
from .constants import (
    FALSE,
    FIRST_DRAW_IDX,
    SUDDEN_DEATH_ROUNDS,
    TARGET_SCORE,
    TRUE,
    ZERO_MASK_1D,
    ZERO_MASK_2D,
)
from .state import PlayerStateArrays, RoundState, State, default_state

v_can_win = jax.vmap(
    jax.vmap(Hand.can_ron, in_axes=(None, 0)), in_axes=(0, None)
)  # For each player and tile, check if the player can win by RON


ACTION_KIND = jnp.zeros(Action.NUM_ACTION, dtype=jnp.int32)
ACTION_KIND = ACTION_KIND.at[: Tile.NUM_TILE_TYPE_WITH_RED].set(0)  # discard
ACTION_KIND = ACTION_KIND.at[Action.TSUMOGIRI].set(0)
ACTION_KIND = ACTION_KIND.at[Tile.NUM_TILE_TYPE_WITH_RED : Action.TSUMOGIRI].set(
    1
)  # closed_kan/added_kan
ACTION_KIND = ACTION_KIND.at[Action.RIICHI].set(2)
ACTION_KIND = ACTION_KIND.at[Action.RON].set(3)
ACTION_KIND = ACTION_KIND.at[Action.TSUMO].set(4)
ACTION_KIND = ACTION_KIND.at[Action.PON].set(5)
ACTION_KIND = ACTION_KIND.at[Action.PON_RED].set(5)
ACTION_KIND = ACTION_KIND.at[Action.OPEN_KAN].set(1)
ACTION_KIND = ACTION_KIND.at[Action.CHI_L : Action.CHI_R_RED + 1].set(6)
ACTION_KIND = ACTION_KIND.at[Action.PASS].set(7)
ACTION_KIND = ACTION_KIND.at[Action.KYUUSHU].set(8)
ACTION_KIND = ACTION_KIND.at[Action.DUMMY].set(9)

_PLAYER_FIELDS = frozenset(field.name for field in dataclasses.fields(PlayerStateArrays))
_ROUND_FIELDS = frozenset(field.name for field in dataclasses.fields(RoundState))


def _replace_state(state: State, **updates) -> State:
    """Replace fields of ``state`` by name, wherever they live.

    ``legal_action_mask`` is every player's mask, shaped (NUM_PLAYERS, NUM_ACTION).
    When it or ``current_player`` changes, the env-level mask becomes the current
    player's row; ``_advance_to_next_decision`` then narrows it to the claims open
    to them.
    """
    env_updates = {}
    player_updates = {}
    round_updates = {}
    for key, value in updates.items():
        if key in _PLAYER_FIELDS:
            player_updates[key] = value
        elif key in _ROUND_FIELDS:
            round_updates[key] = value
        else:
            env_updates[key] = value
    if "legal_action_mask" in player_updates and jnp.ndim(player_updates["legal_action_mask"]) != 2:
        raise ValueError("legal_action_mask takes every player's mask, shaped (NUM_PLAYERS, NUM_ACTION)")

    players = state.players if not player_updates else state.players.replace(**player_updates)
    round_state = state.round_state if not round_updates else state.round_state.replace(**round_updates)
    if "legal_action_mask" in player_updates or "current_player" in env_updates:
        current_player = env_updates.get("current_player", state.current_player)
        env_updates["legal_action_mask"] = players.legal_action_mask[current_player]

    env_updates.setdefault("players", players)
    env_updates.setdefault("round_state", round_state)
    return state.replace(**env_updates)


def _make_state(**updates) -> State:
    return _replace_state(default_state(), **updates)


def _resolve_game_config(game_config: Optional[GameConfig]) -> GameConfig:
    return RED if game_config is None else game_config


def _apply_red_five_config(deck: Array, game_config: Optional[GameConfig] = None) -> Array:
    if _resolve_game_config(game_config).use_red_fives:
        return deck
    return Tile.to_tile_type(deck).astype(jnp.int8)


def _live_wall_end_ix(state: State) -> jnp.ndarray:
    """Haitei line: last drawable wall index; ``last_deck_ix`` advances per kan (王牌繰り)."""
    return state.round_state.last_deck_ix.astype(jnp.int32)


def _set_tile_type_action(mask: Array, tile_type: Array, value: Array) -> Array:
    tile_type = Tile.to_tile_type(tile_type)
    mask = mask.at[tile_type].set(value)
    return jax.lax.cond(
        Tile.is_tile_type_five(tile_type),
        lambda: mask.at[Tile.to_red(tile_type)].set(value),
        lambda: mask,
    )


def _set_hands(state: State, hand_with_red: Array) -> State:
    """Write every player's hand. ``hand`` is always ``hand_with_red`` with red fives counted as fives."""
    return _replace_state(state, hand=jax.vmap(Hand.to_34)(hand_with_red), hand_with_red=hand_with_red)


def _set_player_hand(state: State, player: Array, hand_with_red_row: Array) -> State:
    """Write one player's hand; see ``_set_hands``."""
    hand_with_red = state.players.hand_with_red.at[player].set(hand_with_red_row)
    hand = state.players.hand.at[player].set(Hand.to_34(hand_with_red_row))
    return _replace_state(state, hand=hand, hand_with_red=hand_with_red)


CHI_ACTIONS = jnp.array(
    [
        Action.CHI_L,
        Action.CHI_L_RED,
        Action.CHI_M,
        Action.CHI_M_RED,
        Action.CHI_R,
        Action.CHI_R_RED,
    ],
    dtype=jnp.int32,
)


def _special_abortive_draw_mask() -> Array:
    return ZERO_MASK_2D.at[:, Action.KYUUSHU].set(TRUE)


def _trigger_special_abortive_draw(state: State) -> State:
    return _replace_state(
        state,
        legal_action_mask=_special_abortive_draw_mask(),
        draw_next=FALSE,
        kan_declared=FALSE,
        is_abortive_draw_normal=FALSE,
    )


@jax.jit
def yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile(
    state: State,
    tile: Array,
    next_tile: Array,
    ron_dora_indicators: Optional[Array] = None,
    ron_ura_dora_indicators: Optional[Array] = None,
    use_red_fives: Array = jnp.bool_(True),
) -> Tuple[Array, Array, Array]:
    """
    Calculate YAKU for the discarded tile and the next drawn tile.
    Returns per-player cached values for RON on ``tile`` and TSUMO on ``next_tile``.
    ``ron_(ura_)dora_indicators`` score the RON with other indicators than the TSUMO:
    a robbed added kan is scored before the previous kan's indicator turns over.
    """
    ron_state = _replace_state(state, target=jnp.int8(tile))
    if ron_dora_indicators is not None:
        ron_state = _replace_state(
            ron_state,
            dora_indicators=ron_dora_indicators,
            ura_dora_indicators=ron_ura_dora_indicators,
        )
    tsumo_state = _replace_state(state, last_draw=jnp.int8(next_tile))
    is_rons2 = jnp.array([True, False], dtype=jnp.bool_)
    idx = jnp.arange(8)
    player_idx = idx // 2
    is_ron_idx = idx % 2

    hand_b = state.players.hand_with_red[player_idx]
    is_ron_b = is_rons2[is_ron_idx]

    def f(hand, player, is_ron):
        rs = jax.lax.cond(is_ron, lambda: ron_state, lambda: tsumo_state)
        return Yaku.judge(hand, is_ron, player, rs, use_red_fives=use_red_fives)

    yaku8, fan8, fu8 = jax.vmap(f)(hand_b, player_idx, is_ron_b)
    yaku42 = yaku8.reshape(4, 2, -1)
    fan42 = fan8.reshape(4, 2)
    fu42 = fu8.reshape(4, 2)
    has_yaku = yaku42.any(axis=-1)
    return has_yaku, fan42.astype(jnp.int32), fu42.astype(jnp.int32)


def _init(rng: PRNGKey, game_config: Optional[GameConfig] = None) -> State:
    """
    Initialize the state.

    The state carries no rng: later rounds are dealt from the key passed to
    ``step``. Dealer and wall draw from independent subkeys, so the dealer is
    not a deterministic function of the wall.
    """
    dealer_key, wall_key = jax.random.split(rng)
    dealer = jnp.int8(jax.random.randint(dealer_key, (), 0, 4))
    state = _make_state(
        current_player=dealer,
        dealer=dealer,
        init_wind=_calc_wind(dealer),
        seat_wind=_calc_wind(dealer),
    )
    return _init_for_next_round(wall_key, state, game_config)


def _init_for_next_round(
    rng: PRNGKey,
    state: State,
    game_config: Optional[GameConfig] = None,
) -> State:
    """
    Initialize the state for the next round.
    """
    prepared = _prepare_next_round_assets(rng, game_config)
    return _init_for_next_round_from_prepared(state, prepared, game_config)


def _prepare_next_round_assets(
    rng: PRNGKey,
    game_config: Optional[GameConfig] = None,
) -> Tuple[Array, Array, Array, Array, Array]:
    """Deal the next round's wall from ``rng`` (the key given to ``step``)."""
    deck = Tile.from_tile_id_to_tile(jax.random.permutation(rng, jnp.arange(136))).astype(jnp.int8)
    deck = _apply_red_five_config(deck, game_config)
    init_hand_with_red = Hand.make_init_hand(deck)
    init_hand = jax.vmap(Hand.to_34)(init_hand_with_red)
    dora_indicators = jnp.array([deck[9], -1, -1, -1, -1], dtype=jnp.int8)
    ura_dora_indicators = jnp.array([deck[8], -1, -1, -1, -1], dtype=jnp.int8)
    can_ron = v_can_win(init_hand, TILE_RANGE)
    return deck, dora_indicators, ura_dora_indicators, init_hand_with_red, can_ron


def _init_for_next_round_from_prepared(
    state: State,
    prepared: Tuple[Array, Array, Array, Array, Array],
    game_config: Optional[GameConfig] = None,
) -> State:
    last_player = jnp.int8(-1)
    deck, dora_indicators, ura_dora_indicators, init_hand_with_red, can_ron = prepared
    state = _replace_state(
        state,
        last_player=last_player,
        deck=deck,
        dora_indicators=dora_indicators,
        ura_dora_indicators=ura_dora_indicators,
    )
    state = _set_hands(state, init_hand_with_red)
    c_p = state.current_player
    new_tile = state.round_state.deck[state.round_state.next_deck_ix]
    new_tile_type = Tile.to_tile_type(new_tile)
    next_deck_ix = state.round_state.next_deck_ix - 1
    eval_state = _replace_state(state, last_draw=new_tile)
    # The full judge: ``judge_yakuman`` has no meld decomposition and counts every
    # three of a kind as a concealed pon. ``_tsumo`` adds the Blessing of Heaven
    # to the cached yakuman count (fu stays 0, as before) and pays it alone otherwise.
    _, fan, fu = Yaku.judge(
        state.players.hand_with_red[c_p],
        FALSE,
        c_p,
        eval_state,
        use_red_fives=_resolve_game_config(game_config).use_red_fives,
    )
    hand_with_red = state.players.hand_with_red.at[c_p].set(Hand.add(state.players.hand_with_red[c_p], new_tile))
    # The mask reads ``can_win``, which is only written below.
    legal_action_mask_c_p = _mask_after_draw(
        _replace_state(state, can_win=can_ron), hand_with_red, c_p, new_tile, game_config
    )
    legal_action_mask_4p = ZERO_MASK_2D.at[c_p, :].set(legal_action_mask_c_p)
    state = _replace_state(
        state,
        has_yaku=state.players.has_yaku.at[c_p, 0].set(can_ron[c_p, new_tile_type]),
        fan=state.players.fan.at[c_p, 0].set(jnp.where(fu == 0, fan, 0)),
        fu=state.players.fu.at[c_p, 0].set(jnp.int32(0)),
        can_win=can_ron,
        legal_action_mask=legal_action_mask_4p,
        next_deck_ix=next_deck_ix,
        last_draw=new_tile,
        target=jnp.int8(-1),
    )
    return _set_player_hand(state, c_p, hand_with_red[c_p])


def _calc_wind(east_player: Array) -> Array:
    east_player = jnp.asarray(east_player, dtype=jnp.int32)
    players = jnp.arange(4, dtype=jnp.int32)
    return ((players - east_player) % 4).astype(jnp.int8)


def _is_first_turn(next_deck_ix: Array) -> Array:
    return next_deck_ix >= FIRST_DRAW_IDX - 4


def _append_action_history(state: State, action: Array) -> State:
    """Record ``action`` in this round's action history and advance the cursor.

    The cursor is ``round_state.round_step``, which resets with the rest of
    ``RoundState`` at every round boundary. It must not be ``state.step_count``:
    that counter is incremented one layer up (``Mahjong4p.step``) and is carried
    across rounds by ``_start_next_round_or_end_game``, while ``action_history``
    itself is rebuilt empty each round. Indexing a per-round buffer with a
    hanchan-global counter walked past the end of the buffer, and JAX drops
    out-of-bounds scatters silently, so the observation saw an all -1 history for
    most of a game.

    PASS is not recorded. A declined call or ron is invisible at a real table, and
    a recorded PASS would tell the others that its seat could have called or won.

    Nor is RON, yet: while others who can ron the same tile are still to answer, a
    recorded RON would tell them how the ones before them did. ``_record_rons``
    writes the RONs once they all have.
    """
    action_i32 = jnp.asarray(action, dtype=jnp.int32)
    is_recorded = (action_i32 != Action.PASS) & (action_i32 != Action.RON)
    return _write_action_history(state, state.current_player, action_i32, is_recorded)


def _write_action_history(state: State, player: Array, action_i32: Array, is_recorded: Array) -> State:
    is_tsumogiri = action_i32 == Action.TSUMOGIRI
    is_discard = ((0 <= action_i32) & (action_i32 < Tile.NUM_TILE_TYPE_WITH_RED)) | is_tsumogiri
    history_action = jnp.where(is_tsumogiri, state.round_state.last_draw, action_i32)
    history_action = jnp.where(is_discard, history_action, action_i32).astype(jnp.int8)
    history_tsumogiri = jnp.where(
        is_discard,
        is_tsumogiri.astype(jnp.int8),
        jnp.int8(-1),
    )

    capacity = jnp.int32(state.round_state.action_history.shape[1])
    cursor = jnp.asarray(state.round_state.round_step, dtype=jnp.int32)
    # Clamp rather than let the scatter vanish: on overflow the newest action
    # overwrites the last slot and ``history_overflow`` records that it happened.
    idx = jnp.minimum(cursor, capacity - 1)

    action_history = state.round_state.action_history.at[0, idx].set(player)
    action_history = action_history.at[1, idx].set(history_action)
    action_history = action_history.at[2, idx].set(history_tsumogiri)
    return _replace_state(
        state,
        action_history=jnp.where(is_recorded, action_history, state.round_state.action_history),
        round_step=jnp.where(is_recorded, jnp.minimum(cursor + 1, capacity), cursor),
        history_overflow=state.round_state.history_overflow | (is_recorded & (cursor >= capacity)),
    )


def _step(
    state: State,
    action: Array,
    key: PRNGKey,
    game_config: Optional[GameConfig] = None,
    round_end: Literal["auto", "confirm"] = "auto",
) -> State:
    """Play ``action`` and advance to the next decision.

    ``round_end`` is the env's. Under ``"confirm"`` every seat plays DUMMY to end a
    round, so the dispatch table has the ``_next_round`` branch. Under ``"auto"``
    ``Mahjong4p.step`` deals the next round itself and DUMMY is never legal, so
    that branch is left out.
    """
    state = _append_action_history(state, action)
    state = _apply_action(state, action, key, game_config, round_end)
    return _advance_to_next_decision(state, game_config)


def _apply_action(
    state: State,
    action: Array,
    key: PRNGKey,
    game_config: Optional[GameConfig] = None,
    round_end: Literal["auto", "confirm"] = "auto",
) -> State:
    """Compute the state after each kind of action, then pick ``action``'s."""
    discard_state = _discard(state, action, game_config)
    kan_state = _kan(state, action, game_config)
    riichi_state = _riichi(state)
    ron_state = _ron(state, game_config)
    tsumo_state = _tsumo(state, game_config)
    pon_state = _pon(state, action)
    chi_state = _chi(state, action)
    pass_state = _pass(state, game_config)
    prepared_next_round = _prepare_next_round_assets(key, game_config)
    special_next_round_state = _special_next_round(
        state,
        key,
        game_config,
        prepared_next_round=prepared_next_round,
    )
    if round_end == "confirm":
        next_round_state = _next_round(
            state,
            key,
            game_config,
            prepared_next_round=prepared_next_round,
        )
    else:
        next_round_state = state  # DUMMY: no-op; illegal under ``auto``, env.step catches it.
    fn_idx = ACTION_KIND[action]
    return jax.lax.switch(
        fn_idx,
        [
            lambda: discard_state,
            lambda: kan_state,
            lambda: riichi_state,
            lambda: ron_state,
            lambda: tsumo_state,
            lambda: pon_state,
            lambda: chi_state,
            lambda: pass_state,
            lambda: special_next_round_state,
            lambda: next_round_state,
        ],
    )


def _advance_to_next_decision(
    state: State,
    game_config: Optional[GameConfig] = None,
) -> State:
    state = jax.lax.cond(
        state.round_state.draw_next & ~state.round_state.is_abortive_draw_normal,
        lambda: _draw(state, game_config),
        lambda: state,
    )
    state = jax.lax.cond(
        state.round_state.kan_declared
        & ~state.round_state.is_abortive_draw_normal
        & ~state.players.legal_action_mask[:, Action.RON].any(),
        lambda: _draw_rinshan(state, game_config),
        lambda: state,
    )
    state = jax.lax.cond(
        state.round_state.is_abortive_draw_normal & (state.round_state.dummy_count == 0) & ~state.terminated,
        lambda: _abortive_draw_normal(state),
        lambda: state,
    )
    # ``state.replace`` rather than ``_replace_state``: only the env-level mask is
    # narrowed, and the player's row keeps the calls still waiting there.
    return state.replace(
        legal_action_mask=_claims_open_to(state.players.legal_action_mask, state.current_player),
    )


def _draw(state: State, game_config: Optional[GameConfig] = None) -> State:
    """
    Draw a tile from the deck
    - Update the next drawn tile
    - Generate the legal action for the player who drew the tile
    - Accept the riichi
    - Update the furiten by pass
    - Update the is haitei flag
    """
    state = _accept_riichi(state)
    config = _resolve_game_config(game_config)
    c_p = state.current_player
    next_deck_ix = state.round_state.next_deck_ix - 1
    first_discards = River.decode_tile(state.players.river[:, 0])
    all_first_discards_exist = jnp.all(state.players.discard_counts > 0)
    is_four_wind = (
        all_first_discards_exist
        & jnp.all(Tile.is_tile_four_wind(first_discards))
        & jnp.all(first_discards == first_discards[0])
    )
    is_pure_first_turn = (next_deck_ix >= FIRST_DRAW_IDX - 5) & (
        state.players.meld_counts.sum() == 0
    )
    is_four_wind_draw = is_four_wind & is_pure_first_turn
    is_four_riichi_draw = state.players.riichi.sum() == 4
    is_special_abortive_draw = config.enable_special_abortive_draw & (
        is_four_wind_draw | is_four_riichi_draw
    )
    special_state = _trigger_special_abortive_draw(state)
    is_haitei = state.round_state.next_deck_ix == _live_wall_end_ix(state)
    new_tile = state.round_state.deck[state.round_state.next_deck_ix]
    hand_with_red = state.players.hand_with_red.at[c_p].set(
        Hand.add(state.players.hand_with_red[c_p], new_tile)
    )
    # ``is_haitei`` is a property of the newly drawn state. The action mask must
    # see that updated flag so last-live-wall tsumo is legal even for open hands.
    draw_eval_state = _replace_state(state, is_haitei=is_haitei)
    legal_action_mask_c_p = jax.lax.select(
        state.players.riichi[c_p],
        _mask_after_draw_in_riichi(draw_eval_state, hand_with_red, c_p, new_tile),
        _mask_after_draw(draw_eval_state, hand_with_red, c_p, new_tile, game_config),
    )
    legal_action_mask_4p = state.players.legal_action_mask.at[c_p, :].set(legal_action_mask_c_p)
    normal_state = _replace_state(
        state,
        target=jnp.int8(-1),
        has_yaku=state.players.has_yaku.at[c_p, 0].set(state.players.has_yaku[c_p, 1]),
        fan=state.players.fan.at[c_p, 0].set(state.players.fan[c_p, 1]),
        fu=state.players.fu.at[c_p, 0].set(state.players.fu[c_p, 1]),
        next_deck_ix=next_deck_ix,
        last_draw=new_tile,
        legal_action_mask=legal_action_mask_4p,
        furiten_by_pass=state.players.furiten_by_pass.at[c_p].set(
            state.players.furiten_by_pass[c_p] & state.players.riichi[c_p]
        ),
        is_haitei=is_haitei,
        draw_next=FALSE,
    )
    normal_state = _set_player_hand(normal_state, c_p, hand_with_red[c_p])
    return jax.lax.cond(
        is_special_abortive_draw,
        lambda: special_state,
        lambda: normal_state,
    )


def _mask_after_draw(
    state: State,
    hand: Array,
    c_p: Array,
    new_tile: Array,
    game_config: Optional[GameConfig] = None,
    *,
    live_draws_left: Optional[Array] = None,
) -> Array:
    """
    Legal action mask for the player who drew a tile
    - Set discardable tiles
    - Set if the player can play CLOSED_KAN or ADDED_KAN
    - Set if the player can declare RIICHI
    - Set if the player can win by TSUMO

    ``live_draws_left`` is the number of live-wall draws still available
    AFTER the tile being decided on. Default None derives it for the normal
    ``_draw`` call site, which builds this mask with the PRE-decrement
    ``next_deck_ix`` (the drawn tile is already in hand), so the count is
    ``next_deck_ix - live_wall_end``. The rinshan call site draws from the
    dead wall with ``last_deck_ix`` already advanced, so it passes its own
    count explicitly.
    """
    config = _resolve_game_config(game_config)
    new_tile_type = Tile.to_tile_type(new_tile)
    tiles_ok = (hand[c_p] > 0).astype(jnp.bool_)
    tiles_ok = tiles_ok.at[new_tile].set(hand[c_p, new_tile] >= 2)
    mask = ZERO_MASK_1D.at[: Tile.NUM_TILE_TYPE_WITH_RED].set(tiles_ok)
    mask = mask.at[Action.TSUMOGIRI].set(TRUE)
    cannot_kan = state.players.n_kan.sum() >= 4
    tile_types = jnp.arange(Tile.NUM_TILE_TYPE, dtype=jnp.int32)
    can_kan = jax.vmap(
        lambda tile_type: (
            Hand.can_closed_kan(hand[c_p], tile_type)
            | (
                Hand.can_added_kan(hand[c_p], tile_type)
                & (state.players.pon[(c_p, tile_type)] > 0)
            )
        )
        & ~state.round_state.is_haitei
        & ~cannot_kan
    )(tile_types)
    mask = mask.at[Tile.NUM_TILE_TYPE_WITH_RED : Action.TSUMOGIRI].set(can_kan)
    # 残りツモが 4 回**未満**のときは立直不可。この判定は宣言者の摸打**後**に
    # 山へ残る枚数で数える。normal draw ではこのマスクが next_deck_ix 減算前に
    # 作られる（摸った牌はもう手牌にある）ので残りは ``next - live_wall_end``
    # ちょうど——``visualization.remaining_tiles`` の ``next - last + 1``（局間
    # 用の式）をここに使うと摸った牌を二重に数えて残り 3 枚でも立直が通り、
    # mjai 互換エンジン（libriichi / riichienv）は全てそれをチョンボにする。
    if live_draws_left is None:
        live_draws_left = state.round_state.next_deck_ix - _live_wall_end_ix(state)
    no_next_draw = live_draws_left < 4
    # 供託を払えない持ち点 1000 点未満も立直不可（score は 100 点単位）。
    cannot_afford_stick = state.round_state.score[c_p] < 10
    can_riichi = jnp.where(
        state.players.riichi[c_p]
        | ~state.players.is_hand_concealed[c_p]
        | no_next_draw
        | cannot_afford_stick,
        FALSE,
        Hand.can_riichi(hand[c_p]),
    )
    mask = mask.at[Action.RIICHI].set(can_riichi)
    can_tsumo = state.players.can_win[c_p, new_tile_type]
    _can_after_kan = state.round_state.can_after_kan
    _is_haitei = state.round_state.is_haitei
    _has_yaku = state.players.has_yaku[c_p, 1]
    mask = mask.at[Action.TSUMO].set(
        can_tsumo
        & (state.players.is_hand_concealed[c_p] | _can_after_kan | _is_haitei | _has_yaku)
    )
    can_kyuushu = (
        config.enable_special_abortive_draw
        & Hand.can_kyuushu(hand[c_p])
        & _is_first_turn(state.round_state.next_deck_ix - 1)
        & (state.players.meld_counts.sum() == 0)
    )
    mask = mask.at[Action.KYUUSHU].set(can_kyuushu)
    return mask


def _mask_after_draw_in_riichi(
    state: State, hand: Array, c_p: Array, new_tile: Array
) -> Array:
    """
    Legal action mask for the player who drew a tile and declared RIICHI
    - Set if the player can play CLOSED_KAN
    - Set if the player can win by TSUMO
    """
    new_tile_type = Tile.to_tile_type(new_tile)
    mask = ZERO_MASK_1D.at[Action.TSUMOGIRI].set(TRUE)
    can_closed_kan = (
        Hand.can_closed_kan_after_riichi(hand[c_p], new_tile_type, state.players.can_win[c_p])
        & ~state.round_state.is_haitei
        & ~(state.players.n_kan.sum() >= 4)  # riichi does not lift the four-kan limit
    )
    mask = mask.at[Tile.NUM_TILE_TYPE_WITH_RED + new_tile_type].set(can_closed_kan)
    mask = mask.at[Action.TSUMO].set(state.players.can_win[c_p, new_tile_type])
    return mask


def _discard(state: State, tile: Array, game_config: Optional[GameConfig] = None) -> State:
    """
    Discard a tile from the hand and update the state
    - Move the discarded tile to the river
    - Calculate YAKU for the discarded tile and the next drawn tile
    - Calculate the legal action for OTHER players (melds and RON)
    - Update furiten by discard
    - Disable AfterKan and Ippatsu
    - If the player can meld, set the next player (RON > KAN, PON > CHI)
    - Check if the game is ended (abortive_draw_normal (流局))
    """
    # An open or added kan's new indicator turns over with the kanner's discard, so a ron on it counts.
    state = _reveal_pending_kan_dora(state)
    c_p = state.current_player
    config = _resolve_game_config(game_config)
    had_after_kan = state.round_state.can_after_kan
    is_tsumogiri = tile == Action.TSUMOGIRI
    tile = jnp.where(tile == Action.TSUMOGIRI, state.round_state.last_draw, tile)
    is_riichi = state.players.riichi_declared[c_p]
    river = River.add_discard(
        state.players.river, tile, c_p, state.players.discard_counts[c_p], is_tsumogiri, is_riichi
    )
    n_river = state.players.discard_counts.at[c_p].add(1)
    hand_with_red_row = Hand.sub(state.players.hand_with_red[c_p], tile)
    state = _replace_state(
        state,
        last_draw=jnp.int8(-1),
        river=river,
        discard_counts=n_river,
        has_nagashi_mangan=state.players.has_nagashi_mangan.at[c_p].set(
            state.players.has_nagashi_mangan[c_p] & Tile.is_yaochu(tile)
        ),
    )
    state = _set_player_hand(state, c_p, hand_with_red_row)
    next_tile = state.round_state.deck[state.round_state.next_deck_ix]
    has_yaku, fan, fu = yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile(
        state, tile, next_tile, use_red_fives=config.use_red_fives
    )
    can_win = jax.vmap(Hand.can_ron, in_axes=(None, 0))(state.players.hand[c_p], TILE_RANGE)
    is_furiten_by_river = jax.vmap(_is_waiting_tile, in_axes=(None, 0))(
        can_win, River.decode_tile(river[c_p])
    ).any()
    state = _replace_state(
        state,
        has_yaku=has_yaku,
        fan=fan,
        fu=fu,
        can_win=state.players.can_win.at[c_p].set(can_win),
        furiten_by_discard=state.players.furiten_by_discard.at[c_p].set(is_furiten_by_river),
        ippatsu=state.players.ippatsu.at[c_p].set(FALSE),
    )
    legal_action_mask_4p = jax.vmap(
        _mask_after_discard, in_axes=(None, 0, 0, None)
    )(
        state, state.players.hand_with_red, jnp.arange(4), tile
    )
    legal_action_mask_4p = legal_action_mask_4p.at[c_p, :].set(
        FALSE
    )  # Set the legal action for the player who discarded the tile to False
    # A hand this tile completes without a yaku is not offered RON, but letting the win go by still makes it furiten.
    missed_win = (
        (jnp.arange(4) != c_p)
        & state.players.can_win[:, Tile.to_tile_type(tile)]
        & ~legal_action_mask_4p[:, Action.RON]
    )
    state = _replace_state(state, furiten_by_pass=state.players.furiten_by_pass | missed_win)
    # 三家和 (triple ron) is decided in ``_ron`` itself when the third player
    # actually declares RON. Counting RON candidates pre-emptively here would
    # mis-fire when one of the three candidates ends up passing.
    is_four_kan_draw = (
        config.enable_special_abortive_draw
        & had_after_kan
        & (state.players.n_kan.sum() >= 4)
        & ((state.players.n_kan > 0).sum() >= 2)
    )
    # After the fourth kan only a ron can take the discard; unclaimed, the round is drawn.
    legal_action_mask_4p = jnp.where(
        is_four_kan_draw,
        ZERO_MASK_2D.at[:, Action.RON].set(legal_action_mask_4p[:, Action.RON]),
        legal_action_mask_4p,
    )

    next_meld_player, can_any = _next_meld_player(
        legal_action_mask_4p, c_p
    )  # Set the next player
    no_ron_player = jnp.logical_not(legal_action_mask_4p[:, Action.RON].any())
    no_meld_player = jnp.logical_not(can_any)
    # Check if the game is ended (abortive_draw_normal)
    is_abortive_draw_normal = state.round_state.next_deck_ix < _live_wall_end_ix(state)
    state = _replace_state(
        state,
        can_after_kan=FALSE,
        is_haitei=state.round_state.is_haitei | is_abortive_draw_normal,
    )
    state = jax.lax.cond(
        is_four_kan_draw & no_ron_player,
        lambda: _trigger_special_abortive_draw(
            # A riichi declared on this discard stands, and its stick stays on the table.
            _accept_riichi(_replace_state(state, last_player=jnp.int8(c_p), target=jnp.int8(tile)))
        ),
        lambda: jax.lax.cond(
            no_meld_player | (is_abortive_draw_normal & no_ron_player),
            lambda: _replace_state(state,   # type:ignore
                current_player=jnp.int8((c_p + 1) % 4),
                last_player=jnp.int8(c_p),
                target=jnp.int8(-1),
                draw_next=TRUE,
                is_abortive_draw_normal=is_abortive_draw_normal,
            ),
            lambda: _replace_state(state,   # type:ignore
                current_player=jnp.int8(next_meld_player),
                last_player=jnp.int8(c_p),
                target=jnp.int8(tile),
                legal_action_mask=legal_action_mask_4p.at[
                    next_meld_player, Action.PASS
                ].set(
                    TRUE
                ),  # Add the pass action to the legal action
                draw_next=FALSE,
            ),
        ),
    )
    return state


def _mask_after_discard(
    state: State, hand: Array, c_p: Array, tile: Array
) -> Array:
    """
    Legal action mask for the player who discarded a tile
    - For melds (CHI, PON, OPEN_KAN)
    - For RON
    """
    # ``Houtei`` is the final discard before exhaustive draw. Normal last-tile
    # draws propagate via ``is_haitei``; after-kan discard chains can also be
    # the final discard once the live wall is exhausted, so we detect that here.
    haitei = state.round_state.is_haitei | (
        state.round_state.next_deck_ix < _live_wall_end_ix(state)
    )
    riichi = state.players.riichi[c_p]
    discarder = state.current_player
    src = (discarder - c_p) % 4
    cannot_meld = riichi | haitei
    cannot_kan = (
        state.players.n_kan.sum() >= 4
    )  # If the number of kan is 4 or more, cannot play OPEN_KAN
    chi_mask = (
        _mask_for_chi(hand, tile) & ~cannot_meld & (src == 3)
    )  # Cannot play CHI from the player who is not the upper player
    pm_mask = _mask_for_pon_open_kan(hand, tile, cannot_kan) & ~cannot_meld
    can_ron = state.players.can_win[c_p, Tile.to_tile_type(tile)]
    has_yaku = state.players.has_yaku[
        c_p, 0
    ]  # Reference the information about the discarded tile and the next drawn tile
    is_furiten = state.players.furiten_by_discard[c_p] | state.players.furiten_by_pass[c_p]
    ron_ok = ((has_yaku | haitei) & can_ron) & ~is_furiten
    # Combine the 1D mask and expand it to 4×NUM_ACTION
    mask = chi_mask | pm_mask
    mask = mask.at[Action.RON].set(ron_ok)
    return mask


def _discard_tile_mask_after_chi(hand_after_chi: Array, target: Array, action: Array) -> Array:
    """
    Tile-type discard mask after CHI with kuikae (swap-calling) prohibited.
    - ``hand_after_chi`` is the caller's hand (with red) after the CHI meld is removed.
    - Prohibit the swap-calling tile type (if any) and the called tile type.
    """
    prohibitive_tile_type = Meld.prohibitive_tile_type_after_chi(action, target)
    tile_mask = hand_after_chi > 0
    tile_mask = jax.lax.cond(
        prohibitive_tile_type >= 0,
        lambda: _set_tile_type_action(tile_mask, prohibitive_tile_type, FALSE),
        lambda: tile_mask,
    )
    tile_mask = _set_tile_type_action(tile_mask, Tile.to_tile_type(target), FALSE)
    return tile_mask


def _mask_for_chi(hand: Array, tile: Array) -> Array:
    """
    - Check if the player can play CHI with the target tile
    - A CHI is legal only when at least one discard remains after the kuikae
      restrictions; otherwise it would produce an active state with an all-false
      legal action mask (see issue #61).
    """

    def _can_chi_and_discard(action: Array) -> Array:
        can_chi = Hand.can_chi(hand, tile, action)
        has_discard = jax.lax.cond(
            can_chi,
            lambda: _discard_tile_mask_after_chi(Hand.chi(hand, tile, action), tile, action).any(),
            lambda: FALSE,
        )
        return can_chi & has_discard

    chi_results = jax.vmap(_can_chi_and_discard)(CHI_ACTIONS)
    legal_action_mask = ZERO_MASK_1D.at[CHI_ACTIONS].set(chi_results)
    return legal_action_mask


def _mask_for_pon_open_kan(hand: Array, tile: Array, cannot_kan: Array) -> Array:
    """
    - Check if the player can play PON or OPEN_KAN with the target tile
    """
    legal_action_mask = ZERO_MASK_1D
    legal_action_mask = legal_action_mask.at[Action.PON].set(Hand.can_no_red_pon(hand, tile))
    legal_action_mask = legal_action_mask.at[Action.PON_RED].set(Hand.can_red_pon(hand, tile))
    legal_action_mask = legal_action_mask.at[Action.OPEN_KAN].set(Hand.can_open_kan(hand, tile) & ~cannot_kan)
    return legal_action_mask


def _next_ron_player(legal_action_mask_4p: Array, discarded_player: Array) -> Array:
    """
    - Check if the player can play RON with the target tile
    - If multiple players can play RON, prioritize the player who is closest to the discarded player
    Example:
        discarded_player = 1, legal_action_mask_4p = [True, False, True, False]
        return 2

    """
    can_ron = (
        legal_action_mask_4p[:, Action.RON] > 0
    )  # Whether each player can play RON
    can_any_ron = can_ron.any()
    # Calculate the distance from the discarded player and prioritize the player who is closest to the discarded player
    distance = (jnp.arange(4) - discarded_player) % 4
    distance = jnp.where(can_ron, distance, jnp.inf)
    idx = jnp.argmin(distance)
    return idx, can_any_ron


def _next_meld_player(legal_action_mask_4p: Array, discarded_player: Array) -> Array:
    """
    Set the next player from the legal action for melding.
    - Set the next player from the legal action (RON > OPEN_KAN, PON > CHI)
    - Prioritize the player who is the closest to the discarded player if multiple players can play RON
    - Used in _discard() to set the next player for melding
    """
    can_chi = (
        legal_action_mask_4p[:, Action.CHI_L : Action.CHI_R_RED + 1].sum(axis=1) > 0
    )  # (4,)
    can_pon = (legal_action_mask_4p[:, Action.PON] | legal_action_mask_4p[:, Action.PON_RED]) > 0  # (4,)
    can_open_kan = legal_action_mask_4p[:, Action.OPEN_KAN] > 0  # (4,)
    can_ron = legal_action_mask_4p[:, Action.RON] > 0  # (4,)

    can_any = jnp.any(
        jnp.stack([can_chi, can_pon, can_open_kan, can_ron], axis=1), axis=1
    )
    # Priority: RON > OPEN_KAN > PON > CHI > NONE
    priority = jnp.where(
        can_ron,
        3,
        jnp.where(can_open_kan, 2, jnp.where(can_pon, 1, jnp.where(can_chi, 0, -1))),
    )
    idx = jnp.argmax(priority)
    can_multiple_ron = can_ron.sum() > 1

    # If multiple players can play RON, prioritize the player who is the closest to the discarded player
    def ron_case():
        distance = (jnp.arange(4) - discarded_player) % 4
        # Set the distance of the players who cannot play RON to infinity
        distance = jnp.where(can_ron, distance, jnp.inf)
        return jnp.argmin(distance)

    idx = jnp.where(can_multiple_ron, ron_case(), idx)
    return idx, can_any.any()


def _claims_open_to(legal_action_mask_4p: Array, player: Array) -> Array:
    """
    The part of the player's row they may answer now.
    - Every call on a discard is heard before one is carried out, so RON > PON, OPEN_KAN > CHI holds across players
    - A PON or OPEN_KAN waits while another player can still RON; a CHI also waits while another player can still PON or OPEN_KAN
    - A player who can RON answers RON or PASS before any call of their own, whoever else can claim the tile
    - A waiting call stays in ``legal_action_mask_4p`` and is offered once the claims above it are passed
    """
    others = jnp.arange(4) != player
    other_ron = (legal_action_mask_4p[:, Action.RON] & others).any()
    other_pon = (legal_action_mask_4p[:, Action.PON : Action.OPEN_KAN + 1].any(axis=1) & others).any()
    mask = legal_action_mask_4p[player]
    mask = mask.at[Action.PON : Action.OPEN_KAN + 1].set(mask[Action.PON : Action.OPEN_KAN + 1] & ~other_ron)
    mask = mask.at[Action.CHI_L : Action.CHI_R_RED + 1].set(
        mask[Action.CHI_L : Action.CHI_R_RED + 1] & ~(other_ron | other_pon)
    )
    # Otherwise the prompt would say whether anyone else can ron or pon the tile,
    # and in a double ron whether the player before this one declared RON.
    ron_or_pass = ZERO_MASK_1D.at[Action.RON].set(TRUE).at[Action.PASS].set(TRUE)
    return jnp.where(mask[Action.RON], mask & ron_or_pass, mask)


def _append_meld(state: State, meld: Array, player: Array) -> State:
    """
    Append the meld to the state
    """
    melds = state.players.melds.at[(player, state.players.meld_counts[player])].set(meld)
    n_meld = state.players.meld_counts.at[player].add(1)
    return _replace_state(state, melds=melds, meld_counts=n_meld)  # type:ignore


def _accept_riichi(state: State) -> State:
    """
    Accept the RIICHI
    - Set the RIICHI flag
    - Subtract the score of the player who accepted the RIICHI
    - Provide rewards
    - Update the kyotaku
    - Set the Ippatsu flag
    - Check if the player has Double Riichi
    """
    l_p = state.round_state.last_player
    already_riichi = state.players.riichi[l_p]  # Whether the player has already RIICHI
    has_l_p_riichi_declared = jnp.logical_and(
        jnp.logical_not(already_riichi), state.players.riichi_declared[l_p]
    )
    _score = state.round_state.score.at[l_p].add(
        has_l_p_riichi_declared * -10
    )  # Subtract the score of the player who accepted the RIICHI
    rewards = (
        jnp.zeros(4, dtype=jnp.float32).at[l_p].set(has_l_p_riichi_declared * -10)
    )  # Rewards for the player who accepted the RIICHI
    _kyotaku = state.round_state.kyotaku + jnp.int8(has_l_p_riichi_declared)
    riichi = state.players.riichi.at[l_p].set(has_l_p_riichi_declared)
    is_ippatsu = jnp.where(has_l_p_riichi_declared, TRUE, state.players.ippatsu[l_p])

    is_double_riichi = _is_first_turn(state.round_state.next_deck_ix) & (
        state.players.meld_counts.sum() == 0
    )  # If the player has no meld, the player has Double Riichi
    is_double_riichi = jnp.where(
        has_l_p_riichi_declared, is_double_riichi, state.players.double_riichi[l_p]
    )
    state = jax.lax.cond(
        already_riichi,
        lambda: state,
        lambda: _replace_state(state, 
            riichi=riichi,
            riichi_declared=state.players.riichi_declared.at[l_p].set(FALSE),
            score=jnp.int32(_score),
            rewards=rewards,
            kyotaku=_kyotaku,
            double_riichi=state.players.double_riichi.at[l_p].set(is_double_riichi),
            ippatsu=state.players.ippatsu.at[l_p].set(
                is_ippatsu
            ),  # Enable Ippatsu for the player who accepted the RIICHI
        ),
    )
    return state


def _is_waiting_tile(can_ron: Array, tile: int) -> bool:
    """
    Check if the tile is a waiting tile
    """
    tile_type = Tile.to_tile_type(tile)
    return (tile != -1) & can_ron[tile_type]


def _reveal_pending_kan_dora(state: State) -> State:
    """
    Turn over the new indicator of an open or added kan if it is still face down
    - A closed kan turns its indicator over at once in ``_kan``
    - An open or added kan's waits for the kanner's discard or next kan, and while it waits ``n_kan_doras`` is one behind the completed kans
    """
    n_kan_doras = state.round_state.n_kan_doras
    return jax.lax.cond(
        n_kan_doras.astype(jnp.int32) < state.players.n_kan.sum().astype(jnp.int32),
        lambda s: _replace_state(
            s,
            n_kan_doras=n_kan_doras + 1,
            dora_indicators=s.round_state.dora_indicators.at[n_kan_doras + 1].set(
                s.round_state.deck[9 - 2 * (n_kan_doras + 1)]
            ),
            ura_dora_indicators=s.round_state.ura_dora_indicators.at[n_kan_doras + 1].set(
                s.round_state.deck[8 - 2 * (n_kan_doras + 1)]
            ),
        ),
        lambda s: s,
        state,
    )


def _draw_rinshan(state: State, game_config: Optional[GameConfig] = None):
    """
    Process when a KAN is Accepted
    - Disable Ippatsu
    - Disable Double Riichi
    - Update the last deck index (王牌の繰り / 海底位置の調整; 本手牌の next は進めない)
    - Disable the kan flag
    - Draw the rinshan tile
    - Calculate legal_action_mask for the player who drew the tile
    - Set the AfterKan flag (嶺上開花)

    大明槓・加槓の槓ドラはここでもめくらず、その打牌か続く槓でめくる（``_reveal_pending_kan_dora``）。
    暗槓は槍槓がないため ``_kan`` で槓ドラを先にめくる（``n_kan_doras > n_kan.sum()`` の間だけ一時的に不整合）。
    """
    # A previous open or added kan's indicator turns over just before this rinshan draw,
    # once no one has robbed the added kan that led here (see ``_kan``).
    state = _reveal_pending_kan_dora(state)
    c_p = state.current_player
    config = _resolve_game_config(game_config)
    n_kan = state.players.n_kan.sum()
    rinshan_tile = state.round_state.deck[jnp.int32(10 + n_kan)]
    kan_dora_pre_flipped = state.round_state.n_kan_doras.astype(jnp.int32) > n_kan.astype(
        jnp.int32
    )

    def _after_open_or_added_kan(s: State) -> State:
        return _replace_state(
            s,
            ippatsu=jnp.zeros(4, dtype=jnp.bool_),
            can_after_kan=TRUE,
            n_kan=s.players.n_kan.at[c_p].add(1),
            kan_declared=FALSE,
            last_deck_ix=s.round_state.last_deck_ix + 1,
        )

    def _after_kan_dora_already_done(s: State) -> State:
        return _replace_state(
            s,
            ippatsu=jnp.zeros(4, dtype=jnp.bool_),
            can_after_kan=TRUE,
            n_kan=s.players.n_kan.at[c_p].add(1),
            kan_declared=FALSE,
        )

    state = jax.lax.cond(
        kan_dora_pre_flipped, _after_kan_dora_already_done, _after_open_or_added_kan, state
    )
    # Rinshan draws come from the dead wall and never qualify as Haitei.
    is_haitei = FALSE
    hand_with_red = state.players.hand_with_red.at[c_p].set(
        Hand.add(state.players.hand_with_red[c_p], rinshan_tile)
    )
    # ``can_win`` caches waits for a 13-tile hand before the winning tile is added.
    # Normal draws reuse values computed on the previous discard boundary; rinshan
    # draws need the same convention using the post-kan 13-tile hand.
    can_ron = jax.vmap(Hand.can_ron, in_axes=(None, 0))(state.players.hand[c_p], TILE_RANGE)
    state = _replace_state(state, can_win=state.players.can_win.at[c_p].set(can_ron))
    is_riichi = state.players.riichi[c_p]
    # For the masks only: a kan moves a live-wall tile to the dead wall, so once the
    # live wall is empty this rinshan tile cannot be kanned, just like a haitei tile.
    draw_eval_state = _replace_state(
        state, is_haitei=state.round_state.next_deck_ix < _live_wall_end_ix(state)
    )
    legal_action_mask_c_p = jax.lax.cond(
        is_riichi,
        lambda: _mask_after_draw_in_riichi(draw_eval_state, hand_with_red, c_p, rinshan_tile),
        # Rinshan comes off the dead wall: next_deck_ix is untouched and
        # last_deck_ix has already advanced (王牌繰り above), so the live
        # draws remaining after this draw is the between-turns count.
        lambda: _mask_after_draw(
            draw_eval_state, hand_with_red, c_p, rinshan_tile, game_config,
            live_draws_left=state.round_state.next_deck_ix
            - _live_wall_end_ix(state) + 1),
    )
    legal_action_mask_4p = state.players.legal_action_mask.at[c_p, :].set(
        legal_action_mask_c_p
    )  # Update the legal action for the player who drew the tile
    normal_state = _replace_state(state,   # type:ignore
        last_draw=rinshan_tile,
        legal_action_mask=legal_action_mask_4p,
        has_yaku=state.players.has_yaku.at[c_p, 0].set(state.players.has_yaku[c_p, 1]),
        fan=state.players.fan.at[c_p, 0].set(state.players.fan[c_p, 1]),
        fu=state.players.fu.at[c_p, 0].set(state.players.fu[c_p, 1]),
        is_haitei=is_haitei,
    )
    return _set_player_hand(normal_state, c_p, hand_with_red[c_p])


def _kan(state: State, action, game_config: Optional[GameConfig] = None):
    """
    Process when a KAN is Declared
    - Process the KAN
    - Calculate YAKU for the Robbing KAN and the rinshan tile
    - Apply KAN action
    - Disable Ippatsu
    """
    # A kan made before discarding turns over the previous open or added kan's indicator
    # before its own rinshan draw. An added kan can be robbed first, and that ron is
    # scored, and seen, with the indicator still face down (tenhou: "明槓/加槓は後めくり
    # （打牌または続く嶺上の直前）"); ``_draw_rinshan`` turns it over if nobody robs.
    face_down = state.round_state
    state = _reveal_pending_kan_dora(state)
    c_p = state.current_player
    config = _resolve_game_config(game_config)
    self_kan_tile_type = action - Tile.NUM_TILE_TYPE_WITH_RED
    # An added kan of a five puts the red copy in the meld once the black ones are
    # gone, and a chankan must score and report that red identity.
    self_kan_tile = jnp.where(
        Hand.has_red_of(state.players.hand_with_red[c_p], self_kan_tile_type)
        & (state.players.hand_with_red[c_p, self_kan_tile_type] == 0),
        Tile.to_red(self_kan_tile_type),
        self_kan_tile_type,
    )
    tile = jnp.where(
        action == Action.OPEN_KAN,
        state.round_state.target,
        self_kan_tile,
    )
    rinshan_tile = state.round_state.deck[
        jnp.int32(10 + state.players.n_kan.sum())
    ]  # Reference the deck in _state.py
    # Apply KAN action to hand, meld, river
    is_open_kan = action == Action.OPEN_KAN
    pon = state.players.pon[(c_p, Tile.to_tile_type(tile))]
    is_added_kan = pon != 0  # TODO: Is it correct?
    open_kan_state = _open_kan(state)
    selfkan_state = _selfkan(state, action, is_added_kan)
    state = jax.lax.cond(
        is_open_kan,
        lambda: open_kan_state,
        lambda: selfkan_state,
    )
    # 暗槓のみ実際の卓と同様に槓ドラが即開示される。嶺上牌はまだ引かない（``_draw_rinshan``）。
    is_closed_kan = (~is_open_kan) & (~is_added_kan)
    state = jax.lax.cond(
        is_closed_kan,
        lambda s: _replace_state(
            s,
            n_kan_doras=s.round_state.n_kan_doras + 1,
            dora_indicators=s.round_state.dora_indicators.at[s.round_state.n_kan_doras + 1].set(
                s.round_state.deck[9 - 2 * (s.round_state.n_kan_doras + 1)]
            ),
            ura_dora_indicators=s.round_state.ura_dora_indicators.at[
                s.round_state.n_kan_doras + 1
            ].set(s.round_state.deck[8 - 2 * (s.round_state.n_kan_doras + 1)]),
            last_deck_ix=s.round_state.last_deck_ix + 1,
        ),
        lambda s: s,
        state,
    )
    has_yaku, fan, fu = yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile(
        state,
        tile,
        rinshan_tile,
        face_down.dora_indicators,
        face_down.ura_dora_indicators,
        use_red_fives=config.use_red_fives,
    )  # (4, 2); the RON half only matters for a robbed added kan
    state = _replace_state(state, 
        has_yaku=has_yaku,
        fan=fan,
        fu=fu,
    )
    # Check if the player can win by RON
    is_furiten = state.players.furiten_by_discard | state.players.furiten_by_pass  # (4,)
    legal_action_mask_4p = state.players.legal_action_mask.at[:, Action.RON].set(
        state.players.can_win[:, Tile.to_tile_type(tile)] & ~is_furiten
    )
    legal_action_mask_4p = legal_action_mask_4p.at[c_p, Action.RON].set(
        FALSE
    )  # Disable the legal action for the player who declared the KAN
    state = _replace_state(state, legal_action_mask=legal_action_mask_4p)
    # Triple ron is settled inside ``_ron`` itself; only added-kan opens up the
    # robbing-kan window (closed/open kan are not ronnable in default rules).
    next_ron_player, can_any_ron = _next_ron_player(legal_action_mask_4p, c_p)
    chankan_mask = ZERO_MASK_2D.at[:, Action.RON].set(legal_action_mask_4p[:, Action.RON])
    return jax.lax.cond(
        is_added_kan & can_any_ron,
        lambda: _replace_state(state,   # type:ignore
            target=jnp.int8(tile),
            last_player=c_p,
            current_player=jnp.int8(next_ron_player),
            legal_action_mask=chankan_mask.at[next_ron_player, Action.PASS].set(TRUE),
            kan_declared=TRUE,  # KAN is declared
            draw_next=FALSE,
            dora_indicators=face_down.dora_indicators,
            ura_dora_indicators=face_down.ura_dora_indicators,
            n_kan_doras=face_down.n_kan_doras,
        ),
        lambda: _replace_state(state,   # type:ignore
            target=jnp.int8(-1),
            # Same as the branch above, so that whether anyone could rob the kan
            # does not show in ``last_player``.
            last_player=jnp.where(is_added_kan, c_p, state.round_state.last_player),
            legal_action_mask=ZERO_MASK_2D,
            kan_declared=TRUE,  # KAN is declared
            draw_next=FALSE,
        ),
    )


def _selfkan(state: State, action, is_added_kan):
    """
    Apply SelfKan
    - Branch between ADDED_KAN and CLOSED_KAN
    - Draw the rinshan tile
    - Set the legal action after drawing the rinshan tile
    """
    target = action - Tile.NUM_TILE_TYPE_WITH_RED  # Convert to 0-33
    added_kan_state = _added_kan(state, target)
    closed_kan_state = _closed_kan(state, target)
    return jax.lax.cond(
        is_added_kan,
        lambda: added_kan_state,
        lambda: closed_kan_state,
    )


def _closed_kan(state: State, target):
    """
    Apply CLOSED_KAN
    - Generate a meld from the target
    - Update the hand and meld
    """
    c_p = state.current_player
    meld = Meld.init(target + Tile.NUM_TILE_TYPE_WITH_RED, target, src=0)
    state = _append_meld(state, meld, c_p)
    return _set_player_hand(state, c_p, Hand.closed_kan(state.players.hand_with_red[c_p], target))


def _added_kan(state: State, target):
    """
    Apply ADDED_KAN
    - Generate a meld from the target
    - Update the hand and meld
    """
    c_p = state.current_player
    pon = state.players.pon[(c_p, target)]
    pon_src = pon >> 2
    pon_idx = pon & 0b11
    melds = state.players.melds.at[(c_p, pon_idx)].set(
        Meld.init(target + Tile.NUM_TILE_TYPE_WITH_RED, target, pon_src)
    )
    hand_with_red_row = Hand.added_kan(state.players.hand_with_red[c_p], target)
    pon = state.players.pon.at[(c_p, target)].set(jnp.int8(0))
    return _set_player_hand(_replace_state(state, melds=melds, pon=pon), c_p, hand_with_red_row)


def _clear_furiten_by_pass(state: State, c_p: Array) -> State:
    """Furiten by pass lasts only until the player's own turn comes round again.

    ``_draw`` does this for a normal turn; a turn taken by pon / chi / open kan
    has no draw, so those call it directly. Riichi keeps the player furiten.
    """
    return _replace_state(
        state,
        furiten_by_pass=state.players.furiten_by_pass.at[c_p].set(
            state.players.furiten_by_pass[c_p] & state.players.riichi[c_p]
        ),
    )


def _open_kan(state: State):
    """
    Apply OPEN_KAN
    - Generate a meld from the target
    - Update the hand and meld
    """
    c_p = state.current_player
    l_p = state.round_state.last_player
    state = _accept_riichi(state)
    state = _clear_furiten_by_pass(state, c_p)
    src = (l_p - c_p) % 4
    meld = Meld.init(Action.OPEN_KAN, state.round_state.target, src)
    state = _append_meld(state, meld, c_p)
    hand_with_red_row = Hand.open_kan(state.players.hand_with_red[c_p], state.round_state.target)
    is_hand_concealed = state.players.is_hand_concealed.at[c_p].set(FALSE)
    # Add the meld to the river
    river = River.add_meld(
        state.players.river, Action.OPEN_KAN, l_p, state.players.discard_counts[l_p] - 1, src
    )
    state = _replace_state(state,   # type:ignore
        target=jnp.int8(-1),
        is_hand_concealed=is_hand_concealed,
        river=river,
        has_nagashi_mangan=state.players.has_nagashi_mangan.at[l_p].set(FALSE),
    )
    return _set_player_hand(state, c_p, hand_with_red_row)


def _pon(state: State, action: Array):
    """
    Apply PON
    - Generate a meld from the target
    - Update the hand and meld
    """
    c_p = state.current_player
    l_p = state.round_state.last_player
    tar = state.round_state.target
    state = _accept_riichi(state)
    state = _clear_furiten_by_pass(state, c_p)
    src = (l_p - c_p) % 4
    meld = Meld.init(action, tar, src)
    state = _append_meld(state, meld, c_p)
    pon_hand = Hand.pon(state.players.hand_with_red[c_p], tar, action)
    hand_with_red = state.players.hand_with_red.at[c_p].set(pon_hand)
    is_hand_concealed = state.players.is_hand_concealed.at[c_p].set(FALSE)
    tar_type = Tile.to_tile_type(tar)
    pon = state.players.pon.at[(c_p, tar_type)].set(jnp.int8(src << 2 | state.players.meld_counts[c_p] - 1))
    river = River.add_meld(state.players.river, action, l_p, state.players.discard_counts[l_p] - 1, src)
    # Kuikae is by tile type, so a pon of a five also forbids the other color of five.
    legal_action_mask_4p = (
        ZERO_MASK_2D.at[c_p, : Tile.NUM_TILE_TYPE_WITH_RED]
        .set(_set_tile_type_action((hand_with_red[c_p] > 0).astype(jnp.bool_), tar, FALSE))
        .at[c_p, Action.PASS]
        .set(FALSE)
    )
    state = _replace_state(state,   # type:ignore
        target=jnp.int8(-1),
        is_hand_concealed=is_hand_concealed,
        pon=pon,
        legal_action_mask=legal_action_mask_4p,
        river=river,
        has_nagashi_mangan=state.players.has_nagashi_mangan.at[l_p].set(FALSE),
        ippatsu=jnp.zeros(4, dtype=jnp.bool_),  # Disable Ippatsu
        draw_next=FALSE,
    )
    return _set_player_hand(state, c_p, pon_hand)


def _chi(state: State, action: Array):
    """
    Apply CHI
    - Generate a meld from the target
    - Update the hand and meld
    """
    c_p = state.current_player
    tar_p = state.round_state.last_player  # Absolute position
    tar = state.round_state.target
    state = _accept_riichi(state)
    state = _clear_furiten_by_pass(state, c_p)
    meld = Meld.init(action, tar, src=jnp.int32(3))
    state = _append_meld(state, meld, c_p)
    chi_hand = Hand.chi(state.players.hand_with_red[c_p], tar, action)
    hand_with_red = state.players.hand_with_red.at[c_p].set(chi_hand)
    is_hand_concealed = state.players.is_hand_concealed.at[c_p].set(FALSE)
    legal_action_mask_4p = (
        _mask_after_chi(state, hand_with_red, c_p, tar, action)
        .at[c_p, Action.PASS]
        .set(FALSE)
    )
    # Add the meld to the river
    river = River.add_meld(
        state.players.river, action, tar_p, state.players.discard_counts[tar_p] - 1, jnp.int32(3)
    )
    state = _replace_state(state,   # type:ignore
        target=jnp.int8(-1),
        is_hand_concealed=is_hand_concealed,
        legal_action_mask=legal_action_mask_4p,
        river=river,
        has_nagashi_mangan=state.players.has_nagashi_mangan.at[tar_p].set(FALSE),
        ippatsu=jnp.zeros(4, dtype=jnp.bool_),  # Disable Ippatsu
        draw_next=FALSE,
    )
    return _set_player_hand(state, c_p, chi_hand)


def _mask_after_chi(
    state: State, hand: Array, c_p: Array, target: Array, action: Array
) -> Array:
    """
    Generate legal action after CHI
    - Prohibit eating changes (喰いかえ)
    - If the prohibited tile is 5, also prohibit red tiles
    """
    tile_mask = _discard_tile_mask_after_chi(hand[c_p], target, action)
    player_mask = ZERO_MASK_1D.at[: Tile.NUM_TILE_TYPE_WITH_RED].set(tile_mask)
    # Build the legal action mask for the player
    legal_action_mask_4p = ZERO_MASK_2D.at[c_p].set(player_mask)
    return legal_action_mask_4p


def _pass(state: State, game_config: Optional[GameConfig] = None):
    """
    Apply PASS
    - For discarded tile
    - If the player who can RON, KAN, PON, CHI passes, set the next player from the legal action
    - If no next meld player, proceed to the next player's draw
    """
    config = _resolve_game_config(game_config)
    c_p = state.current_player
    # If the player who declared the KAN passes, set the next player from the legal action
    can_robbing_kan = state.round_state.kan_declared
    is_ron_player = jnp.bool_(state.players.legal_action_mask[c_p, Action.RON])
    # Only what was offered is declined; a call still waiting on a higher claim stays.
    legal_action_mask_4p = state.players.legal_action_mask.at[c_p, :].set(
        state.players.legal_action_mask[c_p] & ~_claims_open_to(state.players.legal_action_mask, c_p)
    )
    post_ron_mask = ZERO_MASK_2D.at[:, Action.RON].set(legal_action_mask_4p[:, Action.RON])
    next_ron_player, can_any_ron = _next_ron_player(
        post_ron_mask, state.round_state.last_player
    )
    is_post_ron = state.pending_winners.any()
    # Set the next player from the legal action
    next_meld_player, can_any = _next_meld_player(
        legal_action_mask_4p, state.round_state.last_player
    )  # Reference the current wind for the next meld player
    no_meld_player = jnp.logical_not(can_any)
    # Check if the game is ended (abortive_draw_normal (流局))
    is_abortive_draw_normal = state.round_state.next_deck_ix < _live_wall_end_ix(state)
    post_ron_state = jax.lax.cond(
        config.allow_double_ron & can_any_ron,
        lambda: _replace_state(
            state,
            current_player=jnp.int8(next_ron_player),
            # Keep every row as a PASS before any RON leaves it, lower calls included:
            # a stored row must not show whether an earlier candidate declared RON.
            legal_action_mask=legal_action_mask_4p.at[next_ron_player, Action.PASS].set(TRUE),
            furiten_by_pass=state.players.furiten_by_pass.at[c_p].set(
                state.players.furiten_by_pass[c_p] | is_ron_player
            ),
            draw_next=FALSE,
        ),
        lambda: _settle_ron(
            _replace_state(
                state,
                furiten_by_pass=state.players.furiten_by_pass.at[c_p].set(
                    state.players.furiten_by_pass[c_p] | is_ron_player
                ),
            )
        ),
    )
    passed = jax.lax.cond(
        is_post_ron,
        lambda: post_ron_state,
        lambda: jax.lax.cond(
            no_meld_player,
            lambda: _replace_state(state,   # type:ignore
                current_player=jnp.where(
                    can_robbing_kan,
                    jnp.int8(state.round_state.last_player),
                    jnp.int8((state.round_state.last_player + 1) % 4),
                ),
                target=jnp.int8(-1),
                furiten_by_pass=state.players.furiten_by_pass.at[c_p].set(
                    state.players.furiten_by_pass[c_p] | is_ron_player
                ),
                draw_next=TRUE & ~can_robbing_kan,
                is_abortive_draw_normal=is_abortive_draw_normal,
                legal_action_mask=legal_action_mask_4p,
            ),
            lambda: _replace_state(state,   # type:ignore
                current_player=jnp.int8(next_meld_player),
                target=jnp.int8(state.round_state.target),
                legal_action_mask=legal_action_mask_4p.at[
                    next_meld_player, Action.PASS
                ].set(TRUE),
                furiten_by_pass=state.players.furiten_by_pass.at[c_p].set(
                    state.players.furiten_by_pass[c_p] | is_ron_player
                ),
            ),
        ),
    )
    # Nobody took the discard after the fourth kan: the four-kan abortive draw. Four kans by
    # two or more players can only mean that discard, since it ends the round either way.
    is_four_kan_draw = (
        config.enable_special_abortive_draw
        & ~can_robbing_kan
        & (state.players.n_kan.sum() >= 4)
        & ((state.players.n_kan > 0).sum() >= 2)
    )
    return jax.lax.cond(
        is_four_kan_draw & ~is_post_ron & no_meld_player,
        # A riichi declared on that discard stands, and its stick stays on the table.
        lambda: _trigger_special_abortive_draw(_accept_riichi(passed)),
        lambda: passed,
    )


def _riichi(state: State):
    """
    Apply RIICHI
    - Set the RIICHI declared flag
    - Generate the legal action for the player after RIICHI
    """
    c_p = state.current_player
    discard_candidates = jax.vmap(Hand.is_tenpai)(
        jax.vmap(Hand.sub, in_axes=(None, 0))(
            state.players.hand_with_red[c_p], jnp.arange(Tile.NUM_TILE_TYPE_WITH_RED)
        )
    )
    legal_action_mask_for_discard = discard_candidates & (state.players.hand_with_red[c_p] > 0)
    player_mask = ZERO_MASK_1D.at[: Tile.NUM_TILE_TYPE_WITH_RED].set(legal_action_mask_for_discard)
    player_mask = player_mask.at[state.round_state.last_draw].set(
        (state.players.hand_with_red[c_p, state.round_state.last_draw] >= 2)
        & legal_action_mask_for_discard[state.round_state.last_draw]
    )
    player_mask = player_mask.at[Action.TSUMOGIRI].set(legal_action_mask_for_discard[state.round_state.last_draw])
    return _replace_state(state,   # type:ignore
        legal_action_mask=state.players.legal_action_mask.at[c_p].set(player_mask),
        riichi_declared=state.players.riichi_declared.at[c_p].set(TRUE),
        draw_next=FALSE,
    )


def _ron(state: State, game_config: Optional[GameConfig] = None) -> State:
    """
    Apply RON
    - Calculate the score of the winner (consider only the remainder when divided by 100)
    - Hold it back until everyone who can ron the tile has answered (``_settle_ron``)
    """
    c_p = state.current_player
    config = _resolve_game_config(game_config)
    is_ippatsu = state.players.ippatsu[c_p] & state.players.riichi[c_p]  # Ippatsu (一発)
    is_double_riichi = state.players.double_riichi[c_p]  # Double Riichi (ダブル立直)
    can_robbing_kan = state.round_state.kan_declared  # RobbingKan (槍槓)
    is_houtei = state.round_state.is_haitei & ~can_robbing_kan  # 河底撈魚
    is_yakuman = state.players.fu[c_p, 0] == 0  # When Yakuman, fu is 0
    basic_score = Yaku.score(
        state.players.fan[c_p, 0]
        + (
            jnp.int32(is_ippatsu)
            + jnp.int32(is_double_riichi)
            + jnp.int32(can_robbing_kan)
            + jnp.int32(is_houtei)
        )
        * (
            1 - is_yakuman
        ),  # When Yakuman, do not add ippatsu, double riichi, robbing_kan, houtei
        state.players.fu[c_p, 0],
    )
    score = jnp.where(state.round_state.dealer == c_p, basic_score * 6, basic_score * 4)
    # Round up the score to the nearest multiple of 100
    score = jnp.ceil(score / 100)
    # In double-ron, honba and kyotaku are paid only once on the first ron.
    is_first_ron = ~state.pending_winners.any()
    honba = jnp.where(is_first_ron, state.round_state.honba * 3, 0)
    # Build reward array more efficiently
    normal_reward = jnp.zeros(4, dtype=jnp.float32)
    normal_reward = normal_reward.at[c_p].set(score + honba)
    normal_reward = normal_reward.at[state.round_state.last_player].set(-score - honba)
    is_pao, pao_player = _pao(state, c_p)
    pao_reward = jnp.zeros(4, dtype=jnp.float32)
    pao_reward = pao_reward.at[c_p].set(score + honba)
    # The points are split with the discarder, but the honba is the liable player's alone.
    pao_reward = pao_reward.at[pao_player].add(-score / 2 - honba)
    pao_reward = pao_reward.at[state.round_state.last_player].add(-score / 2)
    reward = jnp.where(config.enable_pao & is_pao, pao_reward, normal_reward)
    # The Kyotaku is already paid when the RIICHI is declared, so we only need to add the Kyotaku to the winner
    kyotaku_bonus = 10 * state.round_state.kyotaku.astype(jnp.int32) * is_first_ron  # int8 wraps at 13 sticks
    reward = reward.at[c_p].add(kyotaku_bonus)
    # Nothing is paid, recorded or cleared off the table until everyone who can
    # ron this tile has answered: the ones still to answer must not learn how the
    # ones before them did. ``_settle_ron`` pays the whole chain at once.
    pending = jnp.where(is_first_ron, jnp.zeros_like(state.rewards), state.pending_rewards) + reward
    pending_winners = state.pending_winners.at[c_p].set(TRUE)
    remaining_ron_mask = ZERO_MASK_2D.at[:, Action.RON].set(
        state.players.legal_action_mask[:, Action.RON]
    )
    remaining_ron_mask = remaining_ron_mask.at[c_p, Action.RON].set(FALSE)
    next_ron_player, can_any_ron = _next_ron_player(
        remaining_ron_mask, state.round_state.last_player
    )
    # The rows a RON leaves are the rows a PASS leaves (see ``_pass``): only what was
    # offered is answered. Calls left there never run, as the window has a winner.
    answered_mask = state.players.legal_action_mask.at[c_p, :].set(
        state.players.legal_action_mask[c_p] & ~_claims_open_to(state.players.legal_action_mask, c_p)
    )
    # 三家和: this is the 3rd RON declared on the same discard. Nobody wins, and as
    # nothing has been paid yet the round is abandoned as it stands; the window is
    # closed, so the three RONs go into the history.
    is_triple_ron = config.enable_special_abortive_draw & (pending_winners.sum() >= 3)
    triple_ron_state = _trigger_special_abortive_draw(
        _replace_state(
            _record_rons(state, pending_winners),
            pending_rewards=jnp.zeros_like(state.rewards),
            pending_winners=jnp.zeros_like(pending_winners),
        )
    )
    # ``continue_ron`` covers both 2nd-RON (double) and 3rd-RON (triple) cases:
    # we hand the turn to the next eligible RON candidate so they can choose
    # RON / PASS. ``allow_double_ron`` config flag gates the whole multi-RON
    # behaviour (legacy name).
    continue_ron = config.allow_double_ron & can_any_ron
    continue_state = _replace_state(
        state,
        current_player=jnp.int8(next_ron_player),
        pending_rewards=jnp.float32(pending),
        pending_winners=pending_winners,
        legal_action_mask=answered_mask.at[next_ron_player, Action.PASS].set(TRUE),
        draw_next=FALSE,
    )
    final_state = _settle_ron(
        _replace_state(state, pending_rewards=jnp.float32(pending), pending_winners=pending_winners)
    )
    return jax.lax.cond(
        is_triple_ron,
        lambda: triple_ron_state,
        lambda: jax.lax.cond(
            continue_ron,
            lambda: continue_state,
            lambda: final_state,
        ),
    )


def _settle_ron(state: State) -> State:
    """
    Close a ron window once its last player has answered, RON or PASS
    - Pay every ron declared on the tile at once, and record them
    - Leave the turn with the last winner, so the closed round does not show whether anyone declined
    """
    winners = state.pending_winners
    # The winner ``_next_ron_player`` asked last, i.e. the farthest from the discarder.
    distance = (jnp.arange(4) - state.round_state.last_player) % 4
    last_winner = jnp.argmax(jnp.where(winners, distance, -1))
    return _replace_state(
        _record_rons(state, winners),
        current_player=jnp.int8(last_winner),
        terminated_round=TRUE,
        score=jnp.int32(state.round_state.score + state.pending_rewards),
        rewards=jnp.float32(state.pending_rewards),
        pending_rewards=jnp.zeros_like(state.rewards),
        pending_winners=jnp.zeros_like(winners),
        kyotaku=jnp.int8(0),
        has_won=state.players.has_won | winners,
        legal_action_mask=ZERO_MASK_2D.at[:, Action.DUMMY].set(TRUE),
        kan_declared=FALSE,
        draw_next=FALSE,
    )


def _record_rons(state: State, winners: Array) -> State:
    """Record a RON for each of ``winners``, in the order ``_next_ron_player`` asked them."""
    for distance in range(4):
        seat = (state.round_state.last_player + distance) % 4
        state = _write_action_history(state, seat, jnp.int32(Action.RON), winners[seat])
    return state


def _tsumo(state: State, game_config: Optional[GameConfig] = None) -> State:
    """
    Apply TSUUMO
    - Calculate the score of the winner
    - Update the score
    - terminated=true
    - Clear the Kyotaku
    - Check if the game is ended (abortive_draw_normal (流局))
    """
    c_p = state.current_player
    config = _resolve_game_config(game_config)
    _dealer = state.round_state.dealer
    can_after_kan = (
        state.round_state.can_after_kan
    )  # Check if the game is ended (AfterKan (嶺上開花))
    is_ippatsu = state.players.ippatsu[c_p] & state.players.riichi[c_p]  # Ippatsu (一発)
    is_double_riichi = state.players.double_riichi[c_p]  # Double Riichi (ダブル立直)
    is_haitei = state.round_state.is_haitei & ~can_after_kan  # 海底摸月 (last live-wall tsumo)
    # Check for Blessing of the Heaven/Earth (天和/地和)
    is_pure_first_turn = _is_first_turn(state.round_state.next_deck_ix) & (
        state.players.meld_counts.sum() == 0
    )
    is_hand_yakuman = state.players.fu[c_p, 0] == 0
    is_yakuman = is_hand_yakuman | is_pure_first_turn
    fan = state.players.fan[c_p, 0]
    fan = jnp.where(
        is_hand_yakuman,
        fan + is_pure_first_turn,
        jnp.where(
            is_pure_first_turn,  # Blessing of the Heaven(天和) and Earth(地和)
            1,
            fan,
        ),
    )
    fan = jnp.where(
        is_yakuman,
        fan,
        fan
        + (
            jnp.int8(can_after_kan)
            + jnp.int8(is_ippatsu)
            + jnp.int8(is_double_riichi)
            + jnp.int8(is_haitei)
        ),
    )  # When Yakuman, do not add AfterKan, Ippatsu, Double Riichi, Bottom of the Sea (海底摸月)
    fu = jnp.where(is_yakuman, 0, state.players.fu[c_p, 0])
    basic_score = Yaku.score(
        jnp.int32(fan),  # Calculate the score when the tile is discarded
        jnp.int32(fu),
    )
    honba = state.round_state.honba * 1  # 1 Honba is 100 points (per player)
    s1 = jnp.ceil(basic_score / 100)
    s2 = jnp.ceil(basic_score * 2 / 100)
    score = jnp.where(_dealer == c_p, basic_score * 6, basic_score * 4)
    score = jnp.ceil(score / 100)
    # Build reward array more efficiently
    normal_reward = jnp.where(
        _dealer == c_p,
        # If c_p is dealer
        jnp.full(4, -s2 - honba, dtype=jnp.float32),
        # If c_p is not dealer
        jnp.full(4, -s1 - honba, dtype=jnp.float32),
    )
    # Update specific positions based on dealer condition
    normal_reward = jnp.where(
        _dealer == c_p,
        normal_reward.at[c_p].set(s2 * 3 + 3 * honba),  # The dealer pays the score
        normal_reward.at[_dealer]
        .set(-s2 - honba)
        .at[c_p]
        .set(s1 * 2 + s2 + 3 * honba),  # The non-dealer pays the score
    )
    is_pao, pao_player = _pao(state, c_p)
    pao_reward = jnp.zeros(4, dtype=jnp.float32)
    pao_reward = pao_reward.at[c_p].set(score + 3 * honba)
    pao_reward = pao_reward.at[pao_player].set(-score - 3 * honba)
    reward = jnp.where(config.enable_pao & is_pao, pao_reward, normal_reward)
    # The Kyotaku is already paid when the RIICHI is declared, so we only need to add the Kyotaku to the winner
    kyotaku_bonus = 10 * state.round_state.kyotaku.astype(jnp.int32)  # int8 wraps at 13 sticks
    reward = reward.at[c_p].add(kyotaku_bonus)
    score = state.round_state.score + reward
    reward = reward
    return _replace_state(state,   # type:ignore
        terminated_round=TRUE,
        rewards=jnp.float32(reward),
        score=jnp.int32(score),
        kyotaku=jnp.int8(0),  # Clear the Kyotaku
        has_won=state.players.has_won.at[c_p].set(TRUE),
        legal_action_mask=ZERO_MASK_2D.at[:, Action.DUMMY].set(
            TRUE
        ),  # Set the dummy action
    )


def _abortive_draw_normal(state: State) -> State:
    """
    Apply ABORTIVE_DRAW_NORMAL
    - Calculate the score of the winner
    - Update the score
    - terminated=true
    - Clear the Kyotaku
    - Check if the game is ended (abortive_draw_normal (流局))
    """
    # Normal Draw (流局)
    tenpai = state.players.can_win.any(axis=-1)  # (4,)
    n_tenpai = tenpai.sum()
    n_not_tenpai = 4 - n_tenpai
    rewards = jnp.zeros(4, dtype=jnp.int32)
    total_rewards = 30
    normal_rewards = jnp.where(
        tenpai, total_rewards // n_tenpai, -total_rewards // n_not_tenpai
    )
    normal_rewards = jnp.where(
        jnp.logical_or(n_tenpai == 0, n_tenpai == 4),
        jnp.zeros(4, dtype=jnp.int32),
        normal_rewards,
    )
    is_nagashi = state.players.has_nagashi_mangan
    has_nagashi = is_nagashi.any()
    nagashi_rewards = (
        jax.vmap(_mangan_tsumo, in_axes=(0, None, None))(
            jnp.arange(4, dtype=jnp.int32),
            state.round_state.dealer,
            jnp.int8(0),
        )
        * is_nagashi[:, None]
    ).sum(axis=0)
    rewards = jnp.where(has_nagashi, nagashi_rewards.astype(jnp.int32), normal_rewards)
    return _replace_state(state,   # type:ignore
        rewards=rewards.astype(jnp.float32),
        score=jnp.int32(state.round_state.score + jnp.float32(rewards)),  # Update the score
        legal_action_mask=ZERO_MASK_2D.at[:, Action.DUMMY].set(
            TRUE
        ),  # Set the dummy action
        terminated_round=TRUE,
        draw_next=FALSE,
    )


def _special_next_round(
    state: State,
    key: PRNGKey,
    game_config: Optional[GameConfig] = None,
    *,
    prepared_next_round: Optional[Tuple[Array, Array, Array, Array, Array]] = None,
) -> State:
    """After a special abortive draw the dealer deals again, with one more honba."""
    if prepared_next_round is None:
        prepared_next_round = _prepare_next_round_assets(key, game_config)
    return _init_for_next_round_from_prepared(
        _carry_over(state, will_dealer_continue=TRUE), prepared_next_round, game_config
    )


def _pao(state: State, winner: Array) -> Tuple[Array, Array]:
    melds = state.players.melds[winner]
    valid = jnp.arange(melds.shape[0]) < state.players.meld_counts[winner]
    targets = jax.vmap(Meld.target)(melds)
    actions = jax.vmap(Meld.action)(melds)
    srcs = jax.vmap(Meld.src)(melds)
    # An added kan keeps the pon's slot and ``src`` (see ``_added_kan``), so it still
    # counts as the set that was called.
    is_open_set = valid & (
        (actions == Action.PON)
        | (actions == Action.PON_RED)
        | (actions == Action.OPEN_KAN)
        | (Action.is_selfkan(actions) & (srcs != 0))
    )

    def _has_open_set(tile_type: int) -> Array:
        return jnp.any(is_open_set & (targets == tile_type))

    big_three_dragons = _has_open_set(31) & _has_open_set(32) & _has_open_set(33)
    big_four_winds = _has_open_set(27) & _has_open_set(28) & _has_open_set(29) & _has_open_set(30)
    relevant = is_open_set & (
        ((targets >= 31) & (targets <= 33) & big_three_dragons)
        | ((targets >= 27) & (targets <= 30) & big_four_winds)
    )
    relevant_idx = jnp.where(relevant, jnp.arange(melds.shape[0], dtype=jnp.int32), -1)
    last_idx = jnp.max(relevant_idx)
    is_pao = last_idx >= 0
    pao_player = jnp.where(is_pao, (winner + srcs[last_idx]) % 4, jnp.int32(0))
    return is_pao, pao_player


def _mangan_tsumo(winner: Array, dealer: Array, honba: Array) -> Array:
    basic_score = Yaku.score(jnp.int32(5), jnp.int32(30))
    honba_payment = honba * 1
    s1 = jnp.ceil(basic_score / 100) + honba_payment
    s2 = jnp.ceil(basic_score * 2 / 100) + honba_payment
    return jnp.where(
        dealer == winner,
        jnp.full(4, -s2, dtype=jnp.float32).at[winner].set(s2 * 3),
        jnp.full(4, -s1, dtype=jnp.float32)
        .at[dealer]
        .set(-s2)
        .at[winner]
        .set(s1 * 2 + s2),
    )


def _rank_order(round_state) -> Array:
    """Players from first place to last: by score, and on equal score by the seat wind they started the game with."""
    return jnp.lexsort((round_state.init_wind, -round_state.score))


def _final_score(round_state) -> Array:
    """Score at game end: raw score plus uma, with the riichi sticks to the top."""
    order = _rank_order(round_state)
    rank_points = jnp.zeros_like(round_state.score).at[order].set(round_state.order_points)
    score = round_state.score + rank_points
    return score.at[order[0]].add(10 * round_state.kyotaku.astype(jnp.int32))


def _is_game_end(round_state, will_dealer_continue: Array) -> Array:
    """Whether the hanchan/tonpuusen is over now that a kyoku has ended.

    ``round`` is 0-based and ``round_limit`` is the index of the last regular
    kyoku, so a half game ends after South-4 (``round_limit == 7``). If nobody
    has reached ``TARGET_SCORE`` there, the deal runs on into the extra rounds
    (西入) and ends as soon as someone does, or unconditionally once the last
    extra round has been played.
    """
    score = round_state.score
    is_final_round = round_state.round >= round_state.round_limit
    is_last_extra_round = round_state.round >= round_state.round_limit + SUDDEN_DEATH_ROUNDS
    # The deal leaves the current dealer, or it stays but they are top and stop.
    deal_passes = ~will_dealer_continue | (_rank_order(round_state)[0] == round_state.dealer)
    return (
        (score < 0).any()
        # The last extra round still gives way to the dealer's renchan.
        | (is_last_extra_round & ~will_dealer_continue)
        | (is_final_round & deal_passes & (score.max() >= TARGET_SCORE))
    )


def _will_dealer_continue(state: State) -> Array:
    """Whether the dealer deals the next round too: they won, or nobody won and they are tenpai."""
    hora = state.players.has_won  # (4,)
    is_tempai = state.players.can_win.any(axis=-1)  # (4,)
    dealer = state.round_state.dealer
    has_other_than_dealer_won = hora.any() & ~hora[dealer]
    return jnp.logical_or(is_tempai[dealer] & ~has_other_than_dealer_won, hora[dealer])


def _carry_over(state: State, will_dealer_continue: Array) -> State:
    """The state the next round is dealt into.

    If ``will_dealer_continue``, the dealer and the round stay; otherwise the deal
    passes to the next seat. The honba goes up when nobody won or the dealer
    continues, and resets otherwise. The round limit, the starting winds, the uma,
    the scores and the riichi sticks carry over.
    """
    dealer = state.round_state.dealer
    next_dealer = jnp.where(will_dealer_continue, dealer, (dealer + 1) % 4)
    has_winner = state.players.has_won.any()
    return _make_state(
        current_player=next_dealer,  # Start from the dealer
        dealer=next_dealer,
        seat_wind=_calc_wind(next_dealer),
        round=jnp.where(will_dealer_continue, state.round_state.round, state.round_state.round + 1),
        # _make_state starts from default_state(), so ANY field not passed here
        # silently reverts to the dataclass default -- round_limit would drop to 7
        # regardless of the 4 ('east') or 8 the env configured, so the game would
        # run past its intended end and the observation would report the wrong
        # round_limit from the second kyoku onward.
        round_limit=state.round_state.round_limit,
        init_wind=state.round_state.init_wind,
        order_points=state.round_state.order_points,
        honba=jnp.where(~has_winner | will_dealer_continue, state.round_state.honba + 1, 0),
        kyotaku=state.round_state.kyotaku,
        score=state.round_state.score,
    )


def _next_round(
    state: State,
    key: PRNGKey,
    game_config: Optional[GameConfig] = None,
    *,
    prepared_next_round: Optional[Tuple[Array, Array, Array, Array, Array]] = None,
) -> State:
    """
    Move to the next round (``key`` deals the new wall)
    - Process the next round
    - Move the dealer
    - Process the round
    - If the round is ended, calculate the rank points. The Kyotaku is the top (same point wind order)
    - DUMMY sharing: 3 times of rotation (cp to +1 mod 4) after the 4th time, the result is determined
    """
    dc = state.round_state.dummy_count  # int8
    if prepared_next_round is None:
        prepared_next_round = _prepare_next_round_assets(key, game_config)

    # ---- During the DUMMY sharing phase, only rotate ----
    def _rotate_once(s: State):
        will_dealer_continue = _will_dealer_continue(s)
        final_score = _final_score(s.round_state)

        game_end = _is_game_end(s.round_state, will_dealer_continue)
        return _replace_state(s, 
            current_player=jnp.int8((s.current_player + 1) % 4),
            terminated=(s.round_state.dummy_count == 0) & game_end,
            dummy_count=jnp.int8(
                s.round_state.dummy_count + jnp.int8(1)
            ),  # Strictly set the dtype
            score=jnp.where(
                (s.round_state.dummy_count == 0) & game_end, final_score, s.round_state.score
            ),  # Reflect the final score in the first DUMMY sharing phase
            rewards=jnp.where(
                (s.round_state.dummy_count == 0) & game_end,
                jnp.float32(final_score - s.round_state.score),
                s.rewards,
            ),
        )

    # ---- After the DUMMY sharing phase (=3), determine the next round or end the game ----
    def _finalize_and_start_next(s: State):
        will_dealer_continue = _will_dealer_continue(s)
        next_round_state = _init_for_next_round_from_prepared(
            _carry_over(s, will_dealer_continue),
            prepared_next_round,
            game_config,
        )

        terminated_state = _replace_state(
            s,
            score=s.round_state.score,
            terminated=TRUE,
        )

        game_end = _is_game_end(s.round_state, will_dealer_continue)
        # Determine the next round or end the game
        return jax.lax.cond(
            game_end,
            lambda: _replace_state(terminated_state, 
                current_player=jnp.int8(terminated_state.round_state.dealer),
                dummy_count=jnp.int8(0),
            ),
            lambda: _replace_state(next_round_state, 
                current_player=jnp.int8(next_round_state.round_state.dealer),
                dummy_count=jnp.int8(0),
            ),
        )

    # Branch at the entrance
    final_start_state = _finalize_and_start_next(state)
    rotate_state = _rotate_once(state)
    return jax.lax.cond(
        dc == jnp.int8(3),
        lambda: final_start_state,
        lambda: rotate_state,  # Early return here
    )


def _start_next_round_or_end_game(
    state: State,
    key: PRNGKey,
    game_config: Optional[GameConfig] = None,
) -> State:
    """``auto`` round_end round transition (no DUMMY sharing).

    Called when
    ``terminated_round`` becomes True (RON / TSUMO / 流局) and the env is not in
    ``single`` mode. Branches into game-end (terminated=True with final score) or
    next-round init state (rewards from the round-end step are preserved).
    """
    final_score = _final_score(state.round_state)
    will_dealer_continue = _will_dealer_continue(state)
    game_end = _is_game_end(state.round_state, will_dealer_continue)

    prepared = _prepare_next_round_assets(key, game_config)
    next_round_state = _init_for_next_round_from_prepared(
        _carry_over(state, will_dealer_continue), prepared, game_config
    )
    next_round_state = _replace_state(next_round_state,
        current_player=jnp.int8(next_round_state.round_state.dealer),
        rewards=state.rewards,
        step_count=state.step_count,
    )

    terminated_state = _replace_state(state,
        score=jnp.int32(final_score),
        rewards=jnp.float32(state.rewards + (final_score - state.round_state.score)),
        terminated=TRUE,
    )

    return jax.lax.cond(
        game_end,
        lambda: terminated_state,
        lambda: next_round_state,
    )


def _dora_array(state: State) -> Array:
    """
    - Create an array of length 34, where the number of tiles is stored in the index of the dora tile
    """

    def update_dora_counts(dora_counts: Array, dora_indicator: Array) -> Array:
        is_dora_valid = dora_indicator != -1
        dora_tile_type = Tile.to_tile_type(dora_indicator)
        return dora_counts.at[DORA_ARRAY[dora_tile_type]].add(is_dora_valid)

    # Count occurrences of each dora type more efficiently using bincount-like approach
    # For normal dora
    dora_counts = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.int8)
    dora_counts = jax.vmap(update_dora_counts, in_axes=(None, 0))(
        dora_counts, state.round_state.dora_indicators
    ).sum(axis=0)
    # For ura dora
    ura_dora_counts = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.int8)
    ura_dora_counts = jax.vmap(update_dora_counts, in_axes=(None, 0))(
        ura_dora_counts, state.round_state.ura_dora_indicators
    ).sum(axis=0)
    return jnp.array([dora_counts, ura_dora_counts])
