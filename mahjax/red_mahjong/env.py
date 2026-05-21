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


from typing import Dict, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp

from mahjax.core import Env

from .action import Action
from .constants import DORA_ARRAY, FALSE, FIRST_DRAW_IDX, TILE_RANGE, TRUE, ZERO_MASK_1D, ZERO_MASK_2D
from .hand import Hand
from .meld import Meld
from .shanten import Shanten
from .state import GameConfig, State, default_game_config, default_state
from .tile import River, Tile
from .types import Array, PRNGKey
from .yaku import Yaku
from .observation import _observe_dict, _observe_2D

v_can_win = jax.vmap(
    jax.vmap(Hand.can_ron, in_axes=(None, 0)), in_axes=(0, None)
)  # For each player and tile, check if the player can win by RON


ACTION_FUN_MAP = jnp.zeros(Action.NUM_ACTION, dtype=jnp.int32)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[: Tile.NUM_TILE_TYPE_WITH_RED].set(0)  # discard
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.TSUMOGIRI].set(0)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Tile.NUM_TILE_TYPE_WITH_RED : Action.TSUMOGIRI].set(
    1
)  # closed_kan/added_kan
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.RIICHI].set(2)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.RON].set(3)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.TSUMO].set(4)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.PON].set(5)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.PON_RED].set(5)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.OPEN_KAN].set(1)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.CHI_L : Action.CHI_R_RED + 1].set(6)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.PASS].set(7)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.KYUUSHU].set(8)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.DUMMY].set(9)

_PLAYER_FIELDS = {
    "hand",
    "hand_with_red",
    "hand_ids",
    "hand_counts",
    "drawn_tile",
    "legal_action_mask",
    "can_win",
    "has_yaku",
    "fan",
    "fu",
    "melds",
    "meld_tiles",
    "meld_info",
    "meld_counts",
    "river",
    "discards",
    "discard_info",
    "discard_counts",
    "riichi",
    "riichi_declared",
    "riichi_step",
    "double_riichi",
    "ippatsu",
    "furiten_by_discard",
    "furiten_by_pass",
    "is_hand_concealed",
    "pon",
    "has_won",
    "n_kan",
    "has_nagashi_mangan",
}

_ROUND_FIELDS = {
    "rng_key",
    "action_history",
    "shanten_current_player",
    "round",
    "round_limit",
    "terminated_round",
    "honba",
    "kyotaku",
    "init_wind",
    "seat_wind",
    "dealer",
    "order_points",
    "score",
    "deck",
    "next_deck_ix",
    "last_deck_ix",
    "draw_next",
    "last_draw",
    "last_player",
    "dora_indicators",
    "ura_dora_indicators",
    "is_abortive_draw_normal",
    "dummy_count",
    "is_haitei",
    "target",
    "n_kan_doras",
    "kan_declared",
    "can_after_kan",
    "can_robbing_kan",
}


def _replace_state(state: State, **updates) -> State:
    env_updates = {}
    player_updates = {}
    round_updates = {}
    for key, value in updates.items():
        if key == "legal_action_mask":
            ndim = getattr(value, "ndim", None)
            if ndim == 1:
                env_updates[key] = value
            else:
                player_updates[key] = value
        elif key in _PLAYER_FIELDS:
            player_updates[key] = value
        elif key in _ROUND_FIELDS:
            round_updates[key] = value
        else:
            env_updates[key] = value

    players = state.players if not player_updates else state.players.replace(**player_updates)
    round_state = state.round_state if not round_updates else state.round_state.replace(**round_updates)
    current_player = env_updates.get("current_player", state.current_player)

    if "legal_action_mask" in env_updates and "legal_action_mask" not in player_updates:
        players = players.replace(
            legal_action_mask=players.legal_action_mask.at[current_player].set(env_updates["legal_action_mask"])
        )
    elif "legal_action_mask" not in env_updates and ("legal_action_mask" in player_updates or "current_player" in env_updates):
        env_updates["legal_action_mask"] = players.legal_action_mask[current_player]

    env_updates.setdefault("players", players)
    env_updates.setdefault("round_state", round_state)
    return state.replace(**env_updates)


def _make_state(**updates) -> State:
    return _replace_state(default_state(), **updates)


def _resolve_game_config(game_config: Optional[GameConfig]) -> GameConfig:
    return default_game_config() if game_config is None else game_config


def _apply_red_five_config(deck: Array, game_config: Optional[GameConfig] = None) -> Array:
    config = _resolve_game_config(game_config)
    return jax.lax.cond(
        config.use_red_fives,
        lambda: deck,
        lambda: Tile.to_tile_type(deck).astype(jnp.int8),
    )


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


def _has_red_discard_action(mask: Array) -> Array:
    return mask[Action.PON] | mask[Action.PON_RED]


def _set_player_hand(state: State, player: Array, hand_with_red_row: Array) -> State:
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
    state: State, tile: Array, next_tile: Array
) -> Tuple[Array, Array, Array]:
    """
    Calculate YAKU for the discarded tile and the next drawn tile.
    Returns per-player cached values for RON on ``tile`` and TSUMO on ``next_tile``.
    """
    ron_state = _replace_state(state, target=jnp.int8(tile))
    tsumo_state = _replace_state(state, last_draw=jnp.int8(next_tile))
    is_rons2 = jnp.array([True, False], dtype=jnp.bool_)
    idx = jnp.arange(8)
    player_idx = idx // 2
    is_ron_idx = idx % 2

    hand_b = state.players.hand_with_red[player_idx]
    is_ron_b = is_rons2[is_ron_idx]

    def f(hand, player, is_ron):
        rs = jax.lax.cond(is_ron, lambda: ron_state, lambda: tsumo_state)
        return Yaku.judge(hand, is_ron, player, rs)

    yaku8, fan8, fu8 = jax.vmap(f)(hand_b, player_idx, is_ron_b)
    yaku42 = yaku8.reshape(4, 2, -1)
    fan42 = fan8.reshape(4, 2)
    fu42 = fu8.reshape(4, 2)
    has_yaku = yaku42.any(axis=-1)
    return has_yaku, fan42.astype(jnp.int32), fu42.astype(jnp.int32)


class RedMahjong(Env):
    def __init__(
        self,
        round_mode: Literal["single", "east", "half"] = "half",
        observe_type: str = "dict",
        order_points: List[int] = [
            30,
            10,
            -10,
            -30,
        ],  # No oka, 10-30, SAIKOUISEN rule https://saikouisen.com/about/rules/
        game_config: Optional[GameConfig] = None,
        next_round_style: Literal["auto", "dummy_share"] = "auto",
    ):
        if round_mode not in ("single", "east", "half"):
            raise ValueError(f"round_mode must be one of ('single', 'east', 'half'), got: {round_mode}")
        if observe_type == "2D":
            raise ValueError(f"observe type 2D is not developed yet")
        if next_round_style not in ("auto", "dummy_share"):
            raise ValueError(
                f"next_round_style must be one of ('auto', 'dummy_share'), got: {next_round_style}"
            )
        self.round_mode = round_mode
        self.one_round = round_mode == "single"
        self.round_limit = jnp.int8(4 if round_mode == "east" else 8)
        self.observe_func = _observe_dict if observe_type == "dict" else _observe_2D
        self.order_points = order_points
        self.game_config = _resolve_game_config(game_config)
        self.next_round_style = next_round_style
        self._step_fn = _step_auto if next_round_style == "auto" else _step_dummy_share

    def init(self, key: PRNGKey) -> State:
        """Return the initial state. Note that no internal state of
        environment changes.
        Args:
            key: pseudo-random generator key in JAX. Consumed in this function.
        Returns:
            State: initial state of environment
        """
        state = _init(key, self.game_config)
        state = _replace_state(
            state,
            order_points=jnp.array(self.order_points, dtype=jnp.int32),
            round_limit=self.round_limit,
        )
        shanten_val = Shanten.number(state.players.hand[state.current_player]).astype(jnp.int8)
        return _replace_state(state, shanten_current_player=shanten_val)

    def step(
        self,
        state: State,
        action: Array,
        key: Optional[Array] = None,
    ) -> State:
        del key
        """Step function."""
        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player
        state = _replace_state(
            state,
            order_points=jnp.array(self.order_points, dtype=jnp.int32),
        )


        stepped_state = _replace_state(
            self._step_fn(state, action, self.game_config),
            step_count=state.step_count + 1,
        )
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: _replace_state(state, rewards=jnp.zeros_like(state.rewards)),
            lambda: stepped_state,
        )
        state = jax.lax.cond(
            state.round_state.terminated_round & self.one_round,
            lambda: _replace_state(state, terminated=TRUE),
            lambda: state,
        )
        # ``auto`` next_round_style: see the comment in no_red_mahjong/env.py.
        if self.next_round_style == "auto":
            state = jax.lax.cond(
                state.round_state.terminated_round & ~state.terminated & ~jnp.bool_(self.one_round),
                lambda: _advance_to_next_round_auto(state, self.game_config),
                lambda: state,
            )
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )
        state = jax.lax.cond(
            state.terminated,
            lambda: _replace_state(state, legal_action_mask=jnp.ones_like(state.legal_action_mask)),
            lambda: state,
        )
        return state

    def verify_step(
        self,
        state: State,
        action: Array,
        key: Optional[Array] = None,
    ) -> tuple[State, Array]:
        del key
        return verify_step(state, action, self.game_config)

    def observe(self, state: State) -> Array:
        assert isinstance(state, State)
        return self.observe_func(state)

    @property
    def id(self) -> str:
        return "red_mahjong"  # type:ignore

    @property
    def version(self) -> str:
        return "beta"

    @property
    def num_players(self) -> int:
        return 4

    @property
    def num_actions(self) -> int:
        """Return the size of action space (e.g., 9 in Tic-tac-toe)"""
        state = default_state()
        return int(state.legal_action_mask.shape[0])

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Return the matrix shape of observation"""
        state = default_state()
        obs = self.observe(state)
        return obs.shape

    @property
    def _illegal_action_penalty(self) -> float:
        """Negative reward given when illegal action is selected."""
        return -1.0

    def _step_with_illegal_action(self, state: State, loser: Array) -> State:
        penalty = self._illegal_action_penalty
        reward = jnp.ones_like(state.rewards) * (-1 * penalty) * (self.num_players - 1)
        reward = reward.at[loser].set(penalty)
        return _replace_state(state, rewards=reward, terminated=TRUE)


def _init(rng: PRNGKey, game_config: Optional[GameConfig] = None) -> State:
    """
    Initialize the state.
    """
    rng, subkey = jax.random.split(rng)
    current_player = jnp.int8(jax.random.randint(rng, (), 0, 4))
    last_player = jnp.int8(-1)
    deck = Tile.from_tile_id_to_tile(jax.random.permutation(rng, jnp.arange(136))).astype(jnp.int8)
    deck = _apply_red_five_config(deck, game_config)
    init_hand_with_red = Hand.make_init_hand(deck)
    init_hand = jax.vmap(Hand.to_34)(init_hand_with_red)
    dora_indicators = jnp.array([deck[9], -1, -1, -1, -1], dtype=jnp.int8)
    ura_dora_indicators = jnp.array([deck[8], -1, -1, -1, -1], dtype=jnp.int8)
    state = _make_state(
        current_player=current_player,
        dealer=current_player,
        init_wind=_calc_wind(current_player),
        seat_wind=_calc_wind(current_player),
        last_player=last_player,
        deck=deck,
        dora_indicators=dora_indicators,
        ura_dora_indicators=ura_dora_indicators,
        hand=init_hand,
        hand_with_red=init_hand_with_red,
    )
    can_ron = v_can_win(state.players.hand, TILE_RANGE)
    c_p = state.current_player
    new_tile = state.round_state.deck[state.round_state.next_deck_ix]
    new_tile_type = Tile.to_tile_type(new_tile)
    next_deck_ix = state.round_state.next_deck_ix - 1
    eval_state = _replace_state(state, last_draw=new_tile)
    _, yakuman_num, _ = Yaku.judge_yakuman(
        state.players.hand_with_red[c_p],
        FALSE,
        c_p,
        eval_state,
    )
    hand = state.players.hand.at[c_p].set(Hand.add(state.players.hand[c_p], new_tile))
    hand_with_red = state.players.hand_with_red.at[c_p].set(Hand.add(state.players.hand_with_red[c_p], new_tile))
    legal_action_mask_c_p = _make_legal_action_mask_after_draw(
        state, hand_with_red, c_p, new_tile, game_config
    )
    legal_action_mask_4p = ZERO_MASK_2D.at[c_p, :].set(legal_action_mask_c_p)
    state = _replace_state(
        state,
        has_yaku=state.players.has_yaku.at[c_p, 0].set(can_ron[c_p, new_tile_type]),
        fan=state.players.fan.at[c_p, 0].set(jnp.int32(yakuman_num)),
        fu=state.players.fu.at[c_p, 0].set(jnp.int32(0)),
        can_win=can_ron,
        legal_action_mask=legal_action_mask_4p,
        next_deck_ix=next_deck_ix,
        hand=hand,
        hand_with_red=hand_with_red,
        last_draw=new_tile,
        target=jnp.int8(-1),
    )
    return state


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
) -> Tuple[PRNGKey, Array, Array, Array, Array, Array, Array]:
    rng, subkey = jax.random.split(rng)
    deck = Tile.from_tile_id_to_tile(jax.random.permutation(rng, jnp.arange(136))).astype(jnp.int8)
    deck = _apply_red_five_config(deck, game_config)
    init_hand_with_red = Hand.make_init_hand(deck)
    init_hand = jax.vmap(Hand.to_34)(init_hand_with_red)
    dora_indicators = jnp.array([deck[9], -1, -1, -1, -1], dtype=jnp.int8)
    ura_dora_indicators = jnp.array([deck[8], -1, -1, -1, -1], dtype=jnp.int8)
    can_ron = v_can_win(init_hand, TILE_RANGE)
    return subkey, deck, dora_indicators, ura_dora_indicators, init_hand, init_hand_with_red, can_ron


def _init_for_next_round_from_prepared(
    state: State,
    prepared: Tuple[PRNGKey, Array, Array, Array, Array, Array, Array],
    game_config: Optional[GameConfig] = None,
) -> State:
    last_player = jnp.int8(-1)
    subkey, deck, dora_indicators, ura_dora_indicators, init_hand, init_hand_with_red, can_ron = prepared
    state = _replace_state(
        state,
        last_player=last_player,
        deck=deck,
        dora_indicators=dora_indicators,
        ura_dora_indicators=ura_dora_indicators,
        hand=init_hand,
        hand_with_red=init_hand_with_red,
        rng_key=subkey,
    )
    c_p = state.current_player
    new_tile = state.round_state.deck[state.round_state.next_deck_ix]
    new_tile_type = Tile.to_tile_type(new_tile)
    next_deck_ix = state.round_state.next_deck_ix - 1
    eval_state = _replace_state(state, last_draw=new_tile)
    _, yakuman_num, _ = Yaku.judge_yakuman(
        state.players.hand_with_red[c_p],
        FALSE,
        c_p,
        eval_state,
    )
    hand = state.players.hand.at[c_p].set(Hand.add(state.players.hand[c_p], new_tile))
    hand_with_red = state.players.hand_with_red.at[c_p].set(Hand.add(state.players.hand_with_red[c_p], new_tile))
    legal_action_mask_c_p = _make_legal_action_mask_after_draw(
        state, hand_with_red, c_p, new_tile, game_config
    )
    legal_action_mask_4p = ZERO_MASK_2D.at[c_p, :].set(legal_action_mask_c_p)
    state = _replace_state(
        state,
        has_yaku=state.players.has_yaku.at[c_p, 0].set(can_ron[c_p, new_tile_type]),
        fan=state.players.fan.at[c_p, 0].set(jnp.int32(yakuman_num)),
        fu=state.players.fu.at[c_p, 0].set(jnp.int32(0)),
        can_win=can_ron,
        legal_action_mask=legal_action_mask_4p,
        next_deck_ix=next_deck_ix,
        hand=hand,
        hand_with_red=hand_with_red,
        last_draw=new_tile,
        target=jnp.int8(-1),
    )
    return state


def _calc_wind(east_player: Array) -> Array:
    east_player = jnp.asarray(east_player, dtype=jnp.int32)
    players = jnp.arange(4, dtype=jnp.int32)
    return ((players - east_player) % 4).astype(jnp.int8)


def _is_first_turn(next_deck_ix: Array) -> Array:
    return next_deck_ix >= FIRST_DRAW_IDX - 4


def _append_action_history(state: State, action: Array) -> Array:
    action_i32 = jnp.asarray(action, dtype=jnp.int32)
    is_tsumogiri = action_i32 == Action.TSUMOGIRI
    is_discard = ((0 <= action_i32) & (action_i32 < Tile.NUM_TILE_TYPE_WITH_RED)) | is_tsumogiri
    history_action = jnp.where(is_tsumogiri, state.round_state.last_draw, action_i32)
    history_action = jnp.where(is_discard, history_action, action_i32).astype(jnp.int8)
    history_tsumogiri = jnp.where(
        is_discard,
        is_tsumogiri.astype(jnp.int8),
        jnp.int8(-1),
    )

    action_history = state.round_state.action_history.at[0, state.step_count].set(
        state.current_player
    )
    action_history = action_history.at[1, state.step_count].set(history_action)
    return action_history.at[2, state.step_count].set(history_tsumogiri)


def _step_dummy_share(
    state: State, action: Array, game_config: Optional[GameConfig] = None
) -> State:
    """Step used by ``next_round_style='dummy_share'``.

    Dispatch table includes the dummy-rotation ``_next_round`` branch (selected
    when ``action == DUMMY``) so that callers can drive the four-DUMMY share
    phase explicitly. Heavier than ``_step_auto`` because both
    ``_special_next_round`` and ``_next_round`` are pre-computed every step.
    """
    action_history = _append_action_history(state, action)
    state = _replace_state(state, action_history=action_history)  # type:ignore
    state = _dispatch_action_dummy_share(state, action, game_config)
    return _finalize_step_state(state, game_config, update_shanten=TRUE)


def _step_auto(
    state: State, action: Array, game_config: Optional[GameConfig] = None
) -> State:
    """Step used by ``next_round_style='auto'`` (RL default).

    Skips the dummy-rotation ``_next_round`` branch entirely — under ``auto``
    the auto-advance in ``RedMahjong.step`` is responsible for round
    transitions, and ``DUMMY`` is never a legal action. The dispatch table
    still keeps the ``_special_next_round`` branch for ``KYUUSHU``.
    """
    action_history = _append_action_history(state, action)
    state = _replace_state(state, action_history=action_history)  # type:ignore
    state = _dispatch_action_auto(state, action, game_config)
    return _finalize_step_state(state, game_config, update_shanten=TRUE)


# Back-compat alias for internal tests that import ``_step`` directly. They
# exercise the dummy-rotation flow, which is exactly ``_step_dummy_share``.
_step = _step_dummy_share


def _dispatch_action_dummy_share(
    state: State, action: Array, game_config: Optional[GameConfig] = None
) -> State:
    discard_state = _discard(state, action, game_config)
    kan_state = _kan(state, action, game_config)
    riichi_state = _riichi(state)
    ron_state = _ron(state, game_config)
    tsumo_state = _tsumo(state, game_config)
    pon_state = _pon(state, action)
    chi_state = _chi(state, action)
    pass_state = _pass(state, game_config)
    _, next_round_rng = jax.random.split(state.round_state.rng_key)
    prepared_next_round = _prepare_next_round_assets(next_round_rng)
    special_next_round_state = _special_next_round(
        state,
        game_config,
        next_round_rng=next_round_rng,
        prepared_next_round=prepared_next_round,
    )
    next_round_state = _next_round(
        state,
        game_config,
        next_round_rng=next_round_rng,
        prepared_next_round=prepared_next_round,
    )
    fn_idx = ACTION_FUN_MAP[action]
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


def _dispatch_action_auto(
    state: State, action: Array, game_config: Optional[GameConfig] = None
) -> State:
    """Dispatch table for ``auto`` next_round_style. Drops the ``_next_round``
    (dummy rotation) computation — its slot is a no-op since ``DUMMY`` is
    never a legal action under ``auto`` (any erroneous DUMMY is caught by
    ``env.step``'s ``is_illegal`` check).
    """
    discard_state = _discard(state, action, game_config)
    kan_state = _kan(state, action, game_config)
    riichi_state = _riichi(state)
    ron_state = _ron(state, game_config)
    tsumo_state = _tsumo(state, game_config)
    pon_state = _pon(state, action)
    chi_state = _chi(state, action)
    pass_state = _pass(state, game_config)
    _, next_round_rng = jax.random.split(state.round_state.rng_key)
    prepared_next_round = _prepare_next_round_assets(next_round_rng)
    special_next_round_state = _special_next_round(
        state,
        game_config,
        next_round_rng=next_round_rng,
        prepared_next_round=prepared_next_round,
    )
    fn_idx = ACTION_FUN_MAP[action]
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
            lambda: state,  # DUMMY: no-op; illegal under ``auto``, env.step catches it.
        ],
    )


def _dispatch_action_lazy(
    state: State, action: Array, game_config: Optional[GameConfig] = None
) -> State:
    """Lazy dispatcher used by :func:`verify_step` only. Avoids the eager
    pre-computation of every branch — branches are built as ``lambda s: ...``
    so ``jax.lax.switch`` only traces the selected branch. Slower compile but
    smaller working set; reserved for replay verification.
    """
    fn_idx = ACTION_FUN_MAP[action]
    return jax.lax.switch(
        fn_idx,
        [
            lambda s: _discard(s, action, game_config),
            lambda s: _kan(s, action, game_config),
            lambda s: _riichi(s),
            lambda s: _ron(s, game_config),
            lambda s: _tsumo(s, game_config),
            lambda s: _pon(s, action),
            lambda s: _chi(s, action),
            lambda s: _pass(s, game_config),
            lambda s: _special_next_round(s, game_config),
            lambda s: _next_round(s, game_config),
        ],
        state,
    )


def _step_verify_lazy(
    state: State, action: Array, game_config: Optional[GameConfig] = None
) -> State:
    action_history = _append_action_history(state, action)
    state = _replace_state(state, action_history=action_history)
    state = _dispatch_action_lazy(state, action, game_config)
    return _finalize_step_state(state, game_config, update_shanten=FALSE)


def verify_step(
    state: State,
    action: Array,
    game_config: Optional[GameConfig] = None,
) -> tuple[State, Array]:
    """Step variant for replay verification.

    Returns ``(state_after, is_illegal)``. Unlike :class:`RedMahjong`'s public
    ``step``, this does **not** apply the illegal-action penalty: when ``action``
    is illegal, the input state is returned unchanged together with
    ``is_illegal=True``. Used by ``mahjax_tenhou_test`` to compare env behavior
    against tenhou mjlogs without triggering the env's penalty path.
    """
    is_illegal = ~state.legal_action_mask[action]
    stepped_state = _replace_state(
        _step_verify_lazy(state, action, game_config),
        step_count=state.step_count + 1,
    )
    state = jax.lax.cond(
        (state.terminated | state.truncated) | is_illegal,
        lambda: state,
        lambda: stepped_state,
    )
    return state, is_illegal


def _finalize_step_state(
    state: State,
    game_config: Optional[GameConfig] = None,
    *,
    update_shanten: Array = TRUE,
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
        lambda: _draw_after_kan(state, game_config),
        lambda: state,
    )
    state = jax.lax.cond(
        state.round_state.is_abortive_draw_normal & (state.round_state.dummy_count == 0) & ~state.terminated,
        lambda: _abortive_draw_normal(state),
        lambda: state,
    )
    state = _replace_state(
        state,
        legal_action_mask=state.players.legal_action_mask[state.current_player],
    )
    return jax.lax.cond(
        update_shanten,
        lambda: _replace_state(
            state,
            shanten_current_player=Shanten.number(state.players.hand[state.current_player]).astype(jnp.int8),
        ),
        lambda: state,
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
        _make_legal_action_mask_after_draw_w_riichi(draw_eval_state, hand_with_red, c_p, new_tile),
        _make_legal_action_mask_after_draw(draw_eval_state, hand_with_red, c_p, new_tile, game_config),
    )
    legal_action_mask_4p = state.players.legal_action_mask.at[c_p, :].set(legal_action_mask_c_p)
    normal_state = _replace_state(
        state,
        target=jnp.int8(-1),
        has_yaku=state.players.has_yaku.at[c_p, 0].set(state.players.has_yaku[c_p, 1]),
        fan=state.players.fan.at[c_p, 0].set(state.players.fan[c_p, 1]),
        fu=state.players.fu.at[c_p, 0].set(state.players.fu[c_p, 1]),
        next_deck_ix=next_deck_ix,
        hand=state.players.hand.at[c_p].set(Hand.to_34(hand_with_red[c_p])),
        hand_with_red=hand_with_red,
        last_draw=new_tile,
        legal_action_mask=legal_action_mask_4p,
        furiten_by_pass=state.players.furiten_by_pass.at[c_p].set(
            state.players.furiten_by_pass[c_p] & state.players.riichi[c_p]
        ),
        is_haitei=is_haitei,
        draw_next=FALSE,
    )
    return jax.lax.cond(
        is_special_abortive_draw,
        lambda: special_state,
        lambda: normal_state,
    )


def _make_legal_action_mask_after_draw(
    state: State,
    hand: Array,
    c_p: Array,
    new_tile: Array,
    game_config: Optional[GameConfig] = None,
) -> Array:
    """
    Legal action mask for the player who drew a tile
    - Set discardable tiles
    - Set if the player can play CLOSED_KAN or ADDED_KAN
    - Set if the player can declare RIICHI
    - Set if the player can win by TSUMO
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
    live_wall_end = _live_wall_end_ix(state)
    # 残りツモが 4 回**未満**のときは立直不可。``visualization.remaining_tiles`` と同じ
    # ``next - last + 1`` を使うと、禁止は ``next < last + 3``（従来の ``+ 4`` は残り4でも禁止になる）。
    no_next_draw = state.round_state.next_deck_ix < live_wall_end + 3
    can_riichi = jnp.where(
        state.players.riichi[c_p] | ~state.players.is_hand_concealed[c_p] | no_next_draw,
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
        & _is_first_turn(state.round_state.next_deck_ix)
        & (state.players.meld_counts.sum() == 0)
    )
    mask = mask.at[Action.KYUUSHU].set(can_kyuushu)
    return mask


def _make_legal_action_mask_after_draw_w_riichi(
    state: State, hand: Array, c_p: Array, new_tile: Array
) -> Array:
    """
    Legal action mask for the player who drew a tile and declared RIICHI
    - Set if the player can play CLOSED_KAN
    - Set if the player can win by TSUMO
    """
    new_tile_type = Tile.to_tile_type(new_tile)
    mask = ZERO_MASK_1D.at[Action.TSUMOGIRI].set(TRUE)
    tile_types = jnp.arange(Tile.NUM_TILE_TYPE, dtype=jnp.int32)
    can_closed_kan = jax.vmap(
        lambda tile_type: (
            Hand.can_closed_kan_after_riichi(hand[c_p], tile_type, state.players.can_win[c_p])
            & ~state.round_state.is_haitei
        )
    )(tile_types)
    mask = mask.at[Tile.NUM_TILE_TYPE_WITH_RED : Action.TSUMOGIRI].set(can_closed_kan)
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
    hand_with_red = state.players.hand_with_red.at[c_p].set(
        Hand.sub(state.players.hand_with_red[c_p], tile)
    )
    hand = state.players.hand.at[c_p].set(Hand.to_34(hand_with_red[c_p]))
    state = _replace_state(
        state,
        last_draw=jnp.int8(-1),
        hand=hand,
        hand_with_red=hand_with_red,
        river=river,
        discard_counts=n_river,
        has_nagashi_mangan=state.players.has_nagashi_mangan.at[c_p].set(
            state.players.has_nagashi_mangan[c_p] & Tile.is_yaochu(tile)
        ),
    )
    next_tile = state.round_state.deck[state.round_state.next_deck_ix]
    has_yaku, fan, fu = yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile(
        state, tile, next_tile
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
        _make_legal_action_mask_after_discard, in_axes=(None, 0, 0, None)
    )(
        state, state.players.hand_with_red, jnp.arange(4), tile
    )
    legal_action_mask_4p = legal_action_mask_4p.at[c_p, :].set(
        FALSE
    )  # Set the legal action for the player who discarded the tile to False
    # 三家和 (triple ron) is decided in ``_ron`` itself when the third player
    # actually declares RON. Counting RON candidates pre-emptively here would
    # mis-fire when one of the three candidates ends up passing.
    is_four_kan_draw = (
        config.enable_special_abortive_draw
        & had_after_kan
        & (state.players.n_kan.sum() >= 4)
        & ((state.players.n_kan > 0).sum() >= 2)
        & ~legal_action_mask_4p[:, Action.RON].any()
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
        is_four_kan_draw,
        lambda: _trigger_special_abortive_draw(
            _replace_state(state, last_player=jnp.int8(c_p), target=jnp.int8(tile))
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


def _make_legal_action_mask_after_discard(
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


def _mask_for_chi(hand: Array, tile: Array) -> Array:
    """
    - Check if the player can play CHI with the target tile
    """
    chi_results = jax.vmap(Hand.can_chi, in_axes=(None, None, 0))(hand, tile, CHI_ACTIONS)
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


def _draw_after_kan(state: State, game_config: Optional[GameConfig] = None):
    """
    Process when a KAN is Accepted
    - Disable Ippatsu
    - Disable Double Riichi
    - Update the KAN dora (except 暗槓は ``_kan`` で既にめくり済み)
    - Update the last deck index (王牌の繰り / 海底位置の調整; 本手牌の next は進めない)
    - Disable the kan flag
    - Draw the rinshan tile
    - Calculate legal_action_mask for the player who drew the tile
    - Set the AfterKan flag (嶺上開花)

    大明槓・加槓は槍槓のため ``_kan`` では槓ドラをめくらず、不成立後にここでめくる。
    暗槓は槍槓がないため ``_kan`` で槓ドラを先にめくる（``n_kan_doras > n_kan.sum()`` の間だけ一時的に不整合）。
    """
    c_p = state.current_player
    config = _resolve_game_config(game_config)
    n_kan = state.players.n_kan.sum()
    rinshan_tile = state.round_state.deck[jnp.int32(10 + n_kan)]
    kan_dora_pre_flipped = state.round_state.n_kan_doras.astype(jnp.int32) > n_kan.astype(
        jnp.int32
    )
    n_kan_doras = state.round_state.n_kan_doras
    next_kan_dora = state.round_state.deck[9 - 2 * (n_kan_doras + 1)]
    next_kan_ura = state.round_state.deck[8 - 2 * (n_kan_doras + 1)]

    def _after_kan_flip_dora(s: State) -> State:
        return _replace_state(
            s,
            ippatsu=jnp.zeros(4, dtype=jnp.bool_),
            can_after_kan=TRUE,
            n_kan=s.players.n_kan.at[c_p].add(1),
            kan_declared=FALSE,
            n_kan_doras=s.round_state.n_kan_doras + 1,
            dora_indicators=s.round_state.dora_indicators.at[s.round_state.n_kan_doras + 1].set(
                next_kan_dora
            ),
            ura_dora_indicators=s.round_state.ura_dora_indicators.at[
                s.round_state.n_kan_doras + 1
            ].set(next_kan_ura),
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
        kan_dora_pre_flipped, _after_kan_dora_already_done, _after_kan_flip_dora, state
    )
    # Rinshan draws come from the dead wall and never qualify as Haitei.
    is_haitei = FALSE
    hand_with_red = state.players.hand_with_red.at[c_p].set(
        Hand.add(state.players.hand_with_red[c_p], rinshan_tile)
    )
    hand = state.players.hand.at[c_p].set(Hand.to_34(hand_with_red[c_p]))
    # ``can_win`` caches waits for a 13-tile hand before the winning tile is added.
    # Normal draws reuse values computed on the previous discard boundary; rinshan
    # draws need the same convention using the post-kan 13-tile hand.
    can_ron = jax.vmap(Hand.can_ron, in_axes=(None, 0))(state.players.hand[c_p], TILE_RANGE)
    state = _replace_state(state, can_win=state.players.can_win.at[c_p].set(can_ron))
    is_riichi = state.players.riichi[c_p]
    draw_eval_state = _replace_state(state, is_haitei=is_haitei)
    legal_action_mask_c_p = jax.lax.cond(
        is_riichi,
        lambda: _make_legal_action_mask_after_draw_w_riichi(draw_eval_state, hand_with_red, c_p, rinshan_tile),
        lambda: _make_legal_action_mask_after_draw(draw_eval_state, hand_with_red, c_p, rinshan_tile, game_config),
    )
    legal_action_mask_4p = state.players.legal_action_mask.at[c_p, :].set(
        legal_action_mask_c_p
    )  # Update the legal action for the player who drew the tile
    normal_state = _replace_state(state,   # type:ignore
        last_draw=rinshan_tile,
        hand=hand,
        hand_with_red=hand_with_red,
        legal_action_mask=legal_action_mask_4p,
        has_yaku=state.players.has_yaku.at[c_p, 0].set(state.players.has_yaku[c_p, 1]),
        fan=state.players.fan.at[c_p, 0].set(state.players.fan[c_p, 1]),
        fu=state.players.fu.at[c_p, 0].set(state.players.fu[c_p, 1]),
        is_haitei=is_haitei,
    )
    return normal_state


def _kan(state: State, action, game_config: Optional[GameConfig] = None):
    """
    Process when a KAN is Declared
    - Process the KAN
    - Calculate YAKU for the Robbing KAN and the rinshan tile
    - Apply KAN action
    - Disable Ippatsu
    """
    c_p = state.current_player
    config = _resolve_game_config(game_config)
    tile = jnp.where(
        action == Action.OPEN_KAN,
        state.round_state.target,
        action - Tile.NUM_TILE_TYPE_WITH_RED,
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
    # 暗槓のみ実際の卓と同様に槓ドラが即開示される。嶺上牌はまだ引かない（``_draw_after_kan``）。
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
        state, tile, rinshan_tile
    )  # (4, 2)
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
    return jax.lax.cond(
        is_added_kan & can_any_ron,
        lambda: _replace_state(state,   # type:ignore
            target=jnp.int8(tile),
            last_player=c_p,
            current_player=jnp.int8(next_ron_player),
            legal_action_mask=state.players.legal_action_mask.at[
                next_ron_player, Action.PASS
            ].set(
                TRUE
            ),  # Robbing KAN player can PASS
            kan_declared=TRUE,  # KAN is declared
            draw_next=FALSE,
        ),
        lambda: _replace_state(state,   # type:ignore
            target=jnp.int8(-1),
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
    hand_with_red = state.players.hand_with_red.at[c_p].set(Hand.closed_kan(state.players.hand_with_red[c_p], target))
    return _replace_state(state, hand=state.players.hand.at[c_p].set(Hand.to_34(hand_with_red[c_p])), hand_with_red=hand_with_red)


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
    hand_with_red = state.players.hand_with_red.at[c_p].set(Hand.added_kan(state.players.hand_with_red[c_p], target))
    pon = state.players.pon.at[(c_p, target)].set(jnp.int8(0))
    return _replace_state(
        state,
        melds=melds,
        hand=state.players.hand.at[c_p].set(Hand.to_34(hand_with_red[c_p])),
        hand_with_red=hand_with_red,
        pon=pon,
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
    src = (l_p - c_p) % 4
    meld = Meld.init(Action.OPEN_KAN, state.round_state.target, src)
    state = _append_meld(state, meld, c_p)
    hand_with_red = state.players.hand_with_red.at[c_p].set(
        Hand.open_kan(state.players.hand_with_red[c_p], state.round_state.target)
    )
    hand = state.players.hand.at[c_p].set(Hand.to_34(hand_with_red[c_p]))
    is_hand_concealed = state.players.is_hand_concealed.at[c_p].set(FALSE)
    # Add the meld to the river
    river = River.add_meld(
        state.players.river, Action.OPEN_KAN, l_p, state.players.discard_counts[l_p] - 1, src
    )
    return _replace_state(state,   # type:ignore
        hand=hand,
        hand_with_red=hand_with_red,
        target=jnp.int8(-1),
        is_hand_concealed=is_hand_concealed,
        river=river,
        has_nagashi_mangan=state.players.has_nagashi_mangan.at[l_p].set(FALSE),
    )


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
    src = (l_p - c_p) % 4
    meld = Meld.init(action, tar, src)
    state = _append_meld(state, meld, c_p)
    pon_hand = Hand.pon(state.players.hand_with_red[c_p], tar, action)
    hand_with_red = state.players.hand_with_red.at[c_p].set(pon_hand)
    hand = state.players.hand.at[c_p].set(Hand.to_34(pon_hand))
    is_hand_concealed = state.players.is_hand_concealed.at[c_p].set(FALSE)
    tar_type = Tile.to_tile_type(tar)
    pon = state.players.pon.at[(c_p, tar_type)].set(jnp.int8(src << 2 | state.players.meld_counts[c_p] - 1))
    river = River.add_meld(state.players.river, action, l_p, state.players.discard_counts[l_p] - 1, src)
    legal_action_mask_4p = (
        ZERO_MASK_2D.at[c_p, : Tile.NUM_TILE_TYPE_WITH_RED]
        .set((hand_with_red[c_p] > 0).astype(jnp.bool_))
        .at[c_p, tar]
        .set(FALSE)
        .at[c_p, Action.PASS]
        .set(FALSE)
    )
    return _replace_state(state,   # type:ignore
        target=jnp.int8(-1),
        is_hand_concealed=is_hand_concealed,
        pon=pon,
        hand=hand,
        hand_with_red=hand_with_red,
        legal_action_mask=legal_action_mask_4p,
        river=river,
        has_nagashi_mangan=state.players.has_nagashi_mangan.at[l_p].set(FALSE),
        ippatsu=jnp.zeros(4, dtype=jnp.bool_),  # Disable Ippatsu
        draw_next=FALSE,
    )


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
    meld = Meld.init(action, tar, src=jnp.int32(3))
    state = _append_meld(state, meld, c_p)
    chi_hand = Hand.chi(state.players.hand_with_red[c_p], tar, action)
    hand_with_red = state.players.hand_with_red.at[c_p].set(chi_hand)
    hand = state.players.hand.at[c_p].set(Hand.to_34(chi_hand))
    is_hand_concealed = state.players.is_hand_concealed.at[c_p].set(FALSE)
    legal_action_mask_4p = (
        _make_legal_action_mask_after_chi(state, hand_with_red, c_p, tar, action)
        .at[c_p, Action.PASS]
        .set(FALSE)
    )
    # Add the meld to the river
    river = River.add_meld(
        state.players.river, action, tar_p, state.players.discard_counts[tar_p] - 1, jnp.int32(3)
    )
    return _replace_state(state,   # type:ignore
        target=jnp.int8(-1),
        is_hand_concealed=is_hand_concealed,
        hand=hand,
        hand_with_red=hand_with_red,
        legal_action_mask=legal_action_mask_4p,
        river=river,
        has_nagashi_mangan=state.players.has_nagashi_mangan.at[tar_p].set(FALSE),
        ippatsu=jnp.zeros(4, dtype=jnp.bool_),  # Disable Ippatsu
        draw_next=FALSE,
    )


def _make_legal_action_mask_after_chi(
    state: State, hand: Array, c_p: Array, target: Array, action: Array
) -> Array:
    """
    Generate legal action after CHI
    - Prohibit eating changes (喰いかえ)
    - If the prohibited tile is 5, also prohibit red tiles
    """
    prohibitive_tile_type = Meld.prohibitive_tile_type_after_chi(action, target)
    tile_mask = hand[c_p] > 0
    tile_mask = jax.lax.cond(
        prohibitive_tile_type >= 0,
        lambda: _set_tile_type_action(tile_mask, prohibitive_tile_type, FALSE),
        lambda: tile_mask,
    )
    tile_mask = _set_tile_type_action(tile_mask, Tile.to_tile_type(target), FALSE)
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
    legal_action_mask_4p = state.players.legal_action_mask.at[c_p, :].set(FALSE)
    post_ron_mask = ZERO_MASK_2D.at[:, Action.RON].set(legal_action_mask_4p[:, Action.RON])
    next_ron_player, can_any_ron = _next_ron_player(
        post_ron_mask, state.round_state.last_player
    )
    is_post_ron = state.players.has_won.any() & ~can_robbing_kan
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
            legal_action_mask=post_ron_mask.at[next_ron_player, Action.PASS].set(TRUE),
            furiten_by_pass=state.players.furiten_by_pass.at[c_p].set(is_ron_player),
            draw_next=FALSE,
        ),
        lambda: _replace_state(
            state,
            target=jnp.int8(-1),
            furiten_by_pass=state.players.furiten_by_pass.at[c_p].set(is_ron_player),
            legal_action_mask=ZERO_MASK_2D.at[:, Action.DUMMY].set(TRUE),
            terminated_round=TRUE,
            draw_next=FALSE,
        ),
    )
    return jax.lax.cond(
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
                    is_ron_player & ~can_robbing_kan
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
                    is_ron_player & ~can_robbing_kan
                ),
            ),
        ),
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
    - Clear the Kyotaku
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
    # In double-ron, honba is paid only once on the first ron.
    honba = jnp.where(state.players.has_won.any(), 0, state.round_state.honba * 3)
    # Build reward array more efficiently
    normal_reward = jnp.zeros(4, dtype=jnp.float32)
    normal_reward = normal_reward.at[c_p].set(score + honba)
    normal_reward = normal_reward.at[state.round_state.last_player].set(-score - honba)
    is_pao, pao_player = _pao(state, c_p)
    pao_reward = jnp.zeros(4, dtype=jnp.float32)
    pao_reward = pao_reward.at[c_p].set(score + honba)
    pao_reward = pao_reward.at[pao_player].add(-score / 2)
    pao_reward = pao_reward.at[state.round_state.last_player].add(-score / 2 - honba)
    reward = jnp.where(config.enable_pao & is_pao, pao_reward, normal_reward)
    # The Kyotaku is already paid when the RIICHI is declared, so we only need to add the Kyotaku to the winner
    kyotaku_bonus = 10 * (state.round_state.kyotaku)
    reward = reward.at[c_p].add(kyotaku_bonus)
    score = state.round_state.score + jnp.float32(reward)
    remaining_ron_mask = ZERO_MASK_2D.at[:, Action.RON].set(
        state.players.legal_action_mask[:, Action.RON]
    )
    remaining_ron_mask = remaining_ron_mask.at[c_p, Action.RON].set(FALSE)
    next_ron_player, can_any_ron = _next_ron_player(
        remaining_ron_mask, state.round_state.last_player
    )
    # 三家和: this is the 3rd RON declared on the same discard. Trigger
    # ``_trigger_special_abortive_draw`` instead of applying this RON's
    # score / has_won — abortive draw means no one wins.
    # (The prior two RONs' scores and has_won bits have already been written
    # to ``state`` by previous ``_ron`` calls; we leave those as-is, a known
    # small inaccuracy vs tenhou's "no payment, no winners" rule.)
    is_triple_ron = (
        config.enable_special_abortive_draw
        & ((state.players.has_won.sum() + 1) >= 3)
    )
    triple_ron_state = _trigger_special_abortive_draw(state)
    # ``continue_ron`` covers both 2nd-RON (double) and 3rd-RON (triple) cases:
    # we hand the turn to the next eligible RON candidate so they can choose
    # RON / PASS. ``allow_double_ron`` config flag gates the whole multi-RON
    # behaviour (legacy name).
    continue_ron = config.allow_double_ron & can_any_ron
    continue_state = _replace_state(
        state,
        current_player=jnp.int8(next_ron_player),
        score=jnp.int32(score),
        rewards=jnp.float32(reward),
        kyotaku=jnp.int8(0),
        has_won=state.players.has_won.at[c_p].set(TRUE),
        legal_action_mask=remaining_ron_mask.at[next_ron_player, Action.PASS].set(TRUE),
        draw_next=FALSE,
    )
    final_state = _replace_state(
        state,
        terminated_round=TRUE,
        score=jnp.int32(score),
        rewards=jnp.float32(reward),
        kyotaku=jnp.int8(0),
        has_won=state.players.has_won.at[c_p].set(TRUE),
        legal_action_mask=ZERO_MASK_2D.at[:, Action.DUMMY].set(TRUE),
        draw_next=FALSE,
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
    kyotaku_bonus = 10 * state.round_state.kyotaku
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
    game_config: Optional[GameConfig] = None,
    *,
    next_round_rng: Optional[PRNGKey] = None,
    prepared_next_round: Optional[Tuple[PRNGKey, Array, Array, Array, Array, Array, Array]] = None,
) -> State:
    if next_round_rng is None:
        _, next_round_rng = jax.random.split(state.round_state.rng_key)
    if prepared_next_round is None:
        prepared_next_round = _prepare_next_round_assets(next_round_rng)
    dealer = state.round_state.dealer
    base_next = _make_state(
        rng_key=next_round_rng,
        current_player=dealer,
        dealer=dealer,
        init_wind=state.round_state.init_wind,
        seat_wind=state.round_state.seat_wind,
        round=state.round_state.round,
        round_limit=state.round_state.round_limit,
        honba=state.round_state.honba + 1,
        kyotaku=state.round_state.kyotaku,
        score=state.round_state.score,
        order_points=state.round_state.order_points,
    )
    return _init_for_next_round_from_prepared(base_next, prepared_next_round, game_config)


def _pao(state: State, winner: Array) -> Tuple[Array, Array]:
    melds = state.players.melds[winner]
    valid = jnp.arange(melds.shape[0]) < state.players.meld_counts[winner]
    targets = jax.vmap(Meld.target)(melds)
    actions = jax.vmap(Meld.action)(melds)
    srcs = jax.vmap(Meld.src)(melds)
    is_open_set = valid & (
        (actions == Action.PON) | (actions == Action.PON_RED) | (actions == Action.OPEN_KAN)
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


def _next_round(
    state: State,
    game_config: Optional[GameConfig] = None,
    *,
    next_round_rng: Optional[PRNGKey] = None,
    prepared_next_round: Optional[Tuple[PRNGKey, Array, Array, Array, Array, Array, Array]] = None,
) -> State:
    """
    Move to the next round
    - Process the next round
    - Move the dealer
    - Process the round
    - If the round is ended, calculate the rank points. The Kyotaku is the top (same point wind order)
    - DUMMY sharing: 3 times of rotation (cp to +1 mod 4) after the 4th time, the result is determined
    """
    dc = state.round_state.dummy_count  # int8
    if next_round_rng is None:
        _, next_round_rng = jax.random.split(state.round_state.rng_key)
    if prepared_next_round is None:
        prepared_next_round = _prepare_next_round_assets(next_round_rng)

    # ---- During the DUMMY sharing phase, only rotate ----
    def _rotate_once(s: State):
        is_tempai = s.players.can_win.any(axis=-1)  # (4,)
        dealer = s.round_state.dealer
        hora = s.players.has_won  # (4,)
        will_dealer_continue = jnp.logical_or(
            is_tempai[dealer], hora[dealer]
        )  # Check if the dealer continues (win or temporary win)
        order = jnp.argsort(
            -s.round_state.score
        )  # Example: score=[10,20,30,40] -> order=[3,2,1,0]
        rank_points = (
            jnp.zeros_like(s.round_state.score).at[order].set(s.round_state.order_points)
        )  # Assign the rank points
        score = s.round_state.score + rank_points  # Add the rank points
        top = jnp.argmax(score)
        final_score = score.at[top].add(
            10 * s.round_state.kyotaku
        )  # Add the Kyotaku (10 points per Riichi stick × number of Honba) to the top

        # Check if the round is ended
        is_final_round = s.round_state.round == s.round_state.round_limit
        has_dealer_end = jnp.logical_not(will_dealer_continue)
        is_dealer_top = jnp.arange(4)[top] == s.round_state.dealer
        has_minus_score = (s.round_state.score < 0).any()
        _is_game_end = (
            (is_final_round & has_dealer_end)
            | (has_minus_score)
            | (is_final_round & is_dealer_top)
        )
        return _replace_state(s, 
            current_player=jnp.int8((s.current_player + 1) % 4),
            terminated=(s.round_state.dummy_count == 0) & _is_game_end,
            dummy_count=jnp.int8(
                s.round_state.dummy_count + jnp.int8(1)
            ),  # Strictly set the dtype
            score=jnp.where(
                (s.round_state.dummy_count == 0) & _is_game_end, final_score, s.round_state.score
            ),  # Reflect the final score in the first DUMMY sharing phase
        )

    # ---- After the DUMMY sharing phase (=3), determine the next round or end the game ----
    def _finalize_and_start_next(s: State):
        hora = s.players.has_won  # (4,)
        is_tempai = s.players.can_win.any(axis=-1)  # (4,)
        dealer = s.round_state.dealer
        is_eight_consecutive_deals = (
            s.round_state.honba >= 8
        )  # 8 consecutive deals means the honba moves to the next round
        has_other_than_dealer_won = hora.any() & ~hora[dealer]
        will_dealer_continue = jnp.logical_or(
            is_tempai[dealer] & ~has_other_than_dealer_won, hora[dealer]
        )
        will_dealer_continue = will_dealer_continue & ~is_eight_consecutive_deals
        next_round = jnp.where(will_dealer_continue, s.round_state.round, s.round_state.round + 1)
        has_winner = hora.any()
        next_honba = jnp.where(
            ~has_winner | will_dealer_continue, s.round_state.honba + 1, 0
        )  # if there is no winner or the dealer continues, the honba is incremented
        next_dealer = jnp.where(
            will_dealer_continue, dealer, (dealer + 1) % 4
        )  # if the dealer continues, the dealer is kept, otherwise the dealer is incremented

        # ★ Initialize only at this timing
        base_next = _make_state(  # type: ignore
            rng_key=next_round_rng,
            current_player=next_dealer,  # Start from the dealer
            dealer=next_dealer,
            seat_wind=_calc_wind(next_dealer),
            round=next_round,
            honba=next_honba,
            kyotaku=s.round_state.kyotaku,
            score=s.round_state.score,
        )
        next_round_state = _init_for_next_round_from_prepared(
            base_next,
            prepared_next_round,
            game_config,
        )

        terminated_state = _replace_state(
            s,
            score=s.round_state.score,
            terminated=TRUE,
        )

        # Check if the round is ended
        top = jnp.argmax(s.round_state.score)
        is_final_round = s.round_state.round == s.round_state.round_limit
        has_dealer_end = jnp.logical_not(
            will_dealer_continue
        )  # Check if the dealer continues (win or temporary win)
        is_dealer_top = (
            jnp.arange(4)[top] == s.round_state.dealer
        )  # Check if the dealer is the top
        has_minus_score = (
            s.round_state.score < 0
        ).any()  # Check if there is a player with negative score
        _is_game_end = (
            (is_final_round & has_dealer_end)
            | (has_minus_score)
            | (is_final_round & is_dealer_top)
        )
        # Determine the next round or end the game
        return jax.lax.cond(
            _is_game_end,
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


def _advance_to_next_round_auto(
    state: State,
    game_config: Optional[GameConfig] = None,
) -> State:
    """``auto`` next_round_style round transition (no DUMMY sharing) for red_mahjong.

    Mirrors ``no_red_mahjong._advance_to_next_round_auto``: called when
    ``terminated_round`` becomes True (RON / TSUMO / 流局) and the env is not in
    ``single`` mode. Branches into game-end (terminated=True with final score) or
    next-round init state (rewards from the round-end step are preserved).
    """
    hora = state.players.has_won  # (4,)
    is_tempai = state.players.can_win.any(axis=-1)  # (4,)
    dealer = state.round_state.dealer

    order = jnp.argsort(-state.round_state.score)
    rank_points = (
        jnp.zeros_like(state.round_state.score).at[order].set(state.round_state.order_points)
    )
    score_with_rank = state.round_state.score + rank_points
    top_after_rank = jnp.argmax(score_with_rank)
    final_score = score_with_rank.at[top_after_rank].add(10 * state.round_state.kyotaku)

    is_eight_consecutive_deals = state.round_state.honba >= 8
    has_other_than_dealer_won = hora.any() & ~hora[dealer]
    will_dealer_continue = jnp.logical_or(
        is_tempai[dealer] & ~has_other_than_dealer_won, hora[dealer]
    ) & ~is_eight_consecutive_deals

    top_pre_rank = jnp.argmax(state.round_state.score)
    is_final_round = state.round_state.round == state.round_state.round_limit
    has_dealer_end = jnp.logical_not(will_dealer_continue)
    is_dealer_top = jnp.arange(4)[top_pre_rank] == state.round_state.dealer
    has_minus_score = (state.round_state.score < 0).any()
    _is_game_end = (
        (is_final_round & has_dealer_end)
        | has_minus_score
        | (is_final_round & is_dealer_top)
    )

    next_round = jnp.where(will_dealer_continue, state.round_state.round, state.round_state.round + 1)
    has_winner = hora.any()
    next_honba = jnp.where(
        ~has_winner | will_dealer_continue, state.round_state.honba + 1, 0
    )
    next_dealer = jnp.where(will_dealer_continue, dealer, (dealer + 1) % 4)

    _, next_round_rng = jax.random.split(state.round_state.rng_key)
    prepared = _prepare_next_round_assets(next_round_rng, game_config)
    base_next = _make_state(
        rng_key=next_round_rng,
        current_player=next_dealer,
        dealer=next_dealer,
        seat_wind=_calc_wind(next_dealer),
        round=next_round,
        round_limit=state.round_state.round_limit,
        order_points=state.round_state.order_points,
        honba=next_honba,
        kyotaku=state.round_state.kyotaku,
        score=state.round_state.score,
    )
    next_round_state = _init_for_next_round_from_prepared(base_next, prepared, game_config)
    next_round_state = _replace_state(next_round_state,
        current_player=jnp.int8(next_round_state.round_state.dealer),
        rewards=state.rewards,
        step_count=state.step_count,
    )

    terminated_state = _replace_state(state,
        score=jnp.int32(final_score),
        terminated=TRUE,
    )

    return jax.lax.cond(
        _is_game_end,
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
