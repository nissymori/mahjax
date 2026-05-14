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

from mahjax._src.types import Array, PRNGKey
from mahjax.core import Env
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.hand import Hand
from mahjax.no_red_mahjong.meld import Meld
from mahjax.no_red_mahjong.shanten import Shanten
from mahjax.no_red_mahjong.state import DORA_ARRAY, FIRST_DRAW_IDX, State, default_state
from mahjax.no_red_mahjong.tile import River, Tile
from mahjax.no_red_mahjong.yaku import Yaku
from mahjax.no_red_mahjong.observation import _observe_dict, _observe_2D

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

TILE_RANGE = jnp.arange(Tile.NUM_TILE_TYPE)
ZERO_MASK_1D = jnp.zeros(Action.NUM_ACTION, dtype=jnp.bool_)
ZERO_MASK_2D = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)


_PLAYER_FIELDS = {
    "hand",
    "legal_action_mask",
    "can_win",
    "has_yaku",
    "fan",
    "fu",
    "melds",
    "meld_counts",
    "river",
    "discard_counts",
    "riichi",
    "riichi_declared",
    "double_riichi",
    "ippatsu",
    "furiten_by_discard",
    "furiten_by_pass",
    "is_hand_concealed",
    "pon",
    "has_won",
    "n_kan",
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

v_can_win = jax.vmap(
    jax.vmap(Hand.can_ron, in_axes=(None, 0)), in_axes=(0, None)
)  # For each player and tile, check if the player can win by RON


ACTION_FUN_MAP = jnp.zeros(Action.NUM_ACTION, dtype=jnp.int32)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[: Tile.NUM_TILE_TYPE].set(0)  # discard
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.TSUMOGIRI].set(0)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Tile.NUM_TILE_TYPE : Action.TSUMOGIRI].set(
    1
)  # closed_kan/added_kan
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.RIICHI].set(2)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.RON].set(3)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.TSUMO].set(4)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.PON].set(5)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.OPEN_KAN].set(1)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.CHI_L : Action.CHI_R + 1].set(6)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.PASS].set(7)
ACTION_FUN_MAP = ACTION_FUN_MAP.at[Action.DUMMY].set(8)


@jax.jit
def yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile(
    state: State, tile: Array, next_tile: Array, prevalent_wind: Array
) -> Tuple[Array, Array, Array]:
    """
    Calculate YAKU for the discarded tile and the next drawn tile
    Args:
        state: State
        tile: Discarded tile or ADDED_KAN tile
        next_tile: Next drawn tile or RINSHAN_KAN tile
        prevalent_wind: Prevalent wind
    Returns:
        has_yaku: Whether each player has Yaku for the discarded tile and the next drawn tile (4,2) : 4 players, 2 cases (RON/TSUMO)
        fan: Fan number of the Yaku for the discarded tile and the next drawn tile (4,2) : 4 players, 2 cases (RON/TSUMO)
        fu: Fu number of the Yaku for the discarded tile and the next drawn tile (4,2) : 4 players, 2 cases (RON/TSUMO)
    """
    dora = _dora_array(state)
    tiles2 = jnp.array([tile, next_tile])  # (2,)
    is_rons2 = jnp.array([True, False], dtype=jnp.bool_)  # (2,)
    # Create 8 batches
    idx = jnp.arange(8)
    i_idx = idx // 2  # 0,0,1,1,2,2,3,3  → player
    j_idx = idx % 2  # 0,1,0,1,...      → case (RON/TSUMO)

    hand_b = state.players.hand[i_idx]  # (8, ...)
    melds_b = state.players.melds[i_idx]  # (8, ...)
    n_meld_b = state.players.meld_counts[i_idx]  # (8,)
    riichi_b = state.players.riichi[i_idx]  # (8,)
    cur_wind_b = state.round_state.seat_wind[i_idx]  # (8,)
    tile_b = tiles2[j_idx]  # (8,)
    is_ron_b = is_rons2[j_idx]  # (8,)

    def f(hand, melds, n_meld, riichi, cur_wind, t, is_ron):
        return Yaku.judge(
            hand, melds, n_meld, t, riichi, is_ron, prevalent_wind, cur_wind, dora
        )

    yaku8, fan8, fu8 = jax.vmap(f)(
        hand_b, melds_b, n_meld_b, riichi_b, cur_wind_b, tile_b, is_ron_b
    )
    # yaku8: (8, n_yaku) → (4,2,n_yaku) Reshape to (4,2,n_yaku)
    yaku42 = yaku8.reshape(4, 2, -1)
    fan42 = fan8.reshape(4, 2)
    fu42 = fu8.reshape(4, 2)
    has_yaku = yaku42.any(axis=-1)  # (4,2)
    return has_yaku, fan42.astype(jnp.int32), fu42.astype(jnp.int32)


class NoRedMahjong(Env):
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
    ):
        if round_mode not in ("single", "east", "half"):
            raise ValueError(f"round_mode must be one of ('single', 'east', 'half'), got: {round_mode}")
        self.round_mode = round_mode
        self.one_round = round_mode == "single"
        self.round_limit = jnp.int8(4 if round_mode == "east" else 8)
        self.observe_func = _observe_dict if observe_type == "dict" else _observe_2D
        self.order_points = order_points

    def init(self, key: PRNGKey) -> State:
        """Return the initial state. Note that no internal state of
        environment changes.
        Args:
            key: pseudo-random generator key in JAX. Consumed in this function.
        Returns:
            State: initial state of environment
        """
        state = _init(key)
        state = _replace_state(state,   # type:ignore
            order_points=jnp.array(self.order_points, dtype=jnp.int32),
            round_limit=self.round_limit,
        )  # type: ignore
        shanten_val = Shanten.number(state.players.hand[state.current_player]).astype(jnp.int8)
        state = _replace_state(state,   # type:ignore
            shanten_current_player=shanten_val
        )
        return state

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
        state = _replace_state(state,   # type:ignore
            order_points=jnp.array(self.order_points, dtype=jnp.int32),
        )  # type: ignore reflect the order points

        # If the state is already terminated or truncated, environment does not take usual step,
        # but return the same state with zero-rewards for all players
        stepped_state = _replace_state(_step(state, action), step_count=state.step_count + 1)
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: _replace_state(state, rewards=jnp.zeros_like(state.rewards)),  # type: ignore
            lambda: stepped_state,  # type: ignore
        )
        state = jax.lax.cond(
            state.round_state.terminated_round & self.one_round,
            lambda: _replace_state(state, terminated=TRUE),
            lambda: state,
        )
        # Taking illegal action leads to immediate game terminal with negative reward
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )
        # All legal_action_mask elements are **TRUE** at terminal state
        # This is to avoid zero-division error when normalizing action probability
        # Taking any action at terminal state does not give any effect to the state
        state = jax.lax.cond(
            state.terminated,
            lambda: _replace_state(state, legal_action_mask=jnp.ones_like(state.legal_action_mask)),  # type: ignore
            lambda: state,
        )
        return state

    def observe(self, state: State) -> Array:
        assert isinstance(state, State)
        return self.observe_func(state)

    @property
    def id(self) -> str:
        return "no_red_mahjong"  # type:ignore

    @property
    def version(self) -> str:
        return "beta"

    @property
    def num_players(self) -> int:
        return 4

    @property
    def num_actions(self) -> int:
        """Return the size of action space (e.g., 9 in Tic-tac-toe)"""
        state = State()
        return int(state.legal_action_mask.shape[0])

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Return the matrix shape of observation"""
        state = State()
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
        return _replace_state(state, rewards=reward, terminated=TRUE)  # type: ignore


def _init(rng: PRNGKey) -> State:
    """
    Initialize the state
    - Generate the initial hand
    - Set decks
    - Set game-related variables (dealer, seat wind, last player, deck, dora indicators, ura dora indicators, hand, rng key)
    - Calculate the can_win
    - Calculate the YAKU for the initial hand
    - Generate the legal action mask

    Args:
        rng (PRNGKey): Random number generator key

    Returns:
        State: Initial state of the game
    """
    rng, subkey = jax.random.split(rng)
    current_player = jnp.int8(jax.random.randint(rng, (), 0, 4))
    last_player = jnp.int8(-1)
    deck = Tile.from_tile_id_to_tile(
        jax.random.permutation(rng, jnp.arange(136))
    ).astype(
        jnp.int8
    )  # (0-34)
    init_hand = Hand.make_init_hand(deck)  # (4, 34)
    dora_indicators = jnp.array(
        [deck[9], -1, -1, -1, -1], dtype=jnp.int8
    )  # Refer to state.py's _dora_indicators
    ura_dora_indicators = jnp.array(
        [deck[8], -1, -1, -1, -1], dtype=jnp.int8
    )  # Refer to state.py's _ura_dora_indicators
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
    )
    can_ron = v_can_win(state.players.hand, TILE_RANGE)  # (4, 34)
    c_p = (
        state.current_player
    )  # To avoid recurrence by drawing, explicitly write the first draw.
    new_tile = state.round_state.deck[state.round_state.next_deck_ix]
    next_deck_ix = state.round_state.next_deck_ix - 1
    # Only judge the Yakuman.
    prevalent_wind = state.round_state.round % 4
    dora = _dora_array(state)
    _, yakuman_num, _ = Yaku.judge_yakuman(
        state.players.hand[c_p],
        state.players.melds[c_p],
        state.players.meld_counts[c_p],
        new_tile,
        state.players.riichi[c_p],
        FALSE,
        prevalent_wind,
        state.round_state.seat_wind[c_p],
        dora,
    )
    hand = state.players.hand.at[c_p].set(Hand.add(state.players.hand[c_p], new_tile))
    # Generate the legal action for the player who drew the tile after the draw
    legal_action_mask_c_p = _make_legal_action_mask_after_draw(
        state, hand, c_p, new_tile
    )
    legal_action_mask_4p = ZERO_MASK_2D.at[c_p, :].set(legal_action_mask_c_p)
    state = _replace_state(state,   # type:ignore
        has_yaku=state.players.has_yaku.at[c_p, 0].set(
            can_ron[c_p, new_tile]
        ),  # If the combination is horable, the yaku is always attached (Blessing of Heaven).
        fan=state.players.fan.at[c_p, 0].set(
            jnp.int32(yakuman_num)
        ),  # Only judge the Yakuman.
        fu=state.players.fu.at[c_p, 0].set(jnp.int32(0)),  # If the player wins, the fu is 0.
        can_win=can_ron,
        legal_action_mask=legal_action_mask_4p,
        next_deck_ix=next_deck_ix,
        hand=hand,
        last_draw=new_tile,
        target=jnp.int8(-1),
    )
    return state


def _init_for_next_round(rng: PRNGKey, state: State) -> State:
    """
    Initialize the state for the next round
    - Generate the new deck
    - Set game-related variables (last player, deck, dora indicators, ura dora indicators, hand, rng key)
    - Succeed the process of _next_round (dealer, seat wind, round, honba, kyotaku, score, etc.)
    """
    rng, subkey = jax.random.split(rng)
    last_player = jnp.int8(-1)
    deck = Tile.from_tile_id_to_tile(
        jax.random.permutation(rng, jnp.arange(136))
    ).astype(
        jnp.int8
    )  # (0-34)
    init_hand = Hand.make_init_hand(deck)  # (4, 34)
    dora_indicators = jnp.array(
        [deck[9], -1, -1, -1, -1], dtype=jnp.int8
    )  # Refer to state.py's _dora_indicators
    ura_dora_indicators = jnp.array(
        [deck[8], -1, -1, -1, -1], dtype=jnp.int8
    )  # Refer to state.py's _ura_dora_indicators
    state = _replace_state(state,   # type:ignore
        last_player=last_player,
        deck=deck,
        dora_indicators=dora_indicators,
        ura_dora_indicators=ura_dora_indicators,
        hand=init_hand,
        rng_key=subkey,
    )
    can_ron = v_can_win(state.players.hand, TILE_RANGE)  # (4, 34)
    c_p = (
        state.current_player
    )  # To avoid recurrence by drawing, explicitly write the first draw.
    new_tile = state.round_state.deck[state.round_state.next_deck_ix]
    next_deck_ix = state.round_state.next_deck_ix - 1
    # Only judge the Yakuman.
    prevalent_wind = state.round_state.round % 4
    dora = _dora_array(state)
    _, yakuman_num, _ = Yaku.judge_yakuman(
        state.players.hand[c_p],
        state.players.melds[c_p],
        state.players.meld_counts[c_p],
        new_tile,
        state.players.riichi[c_p],
        FALSE,
        prevalent_wind,
        state.round_state.seat_wind[c_p],
        dora,
    )

    hand = state.players.hand.at[c_p].set(Hand.add(state.players.hand[c_p], new_tile))
    # Generate the legal action for the player who drew the tile after the draw
    legal_action_mask_c_p = _make_legal_action_mask_after_draw(
        state, hand, c_p, new_tile
    )
    legal_action_mask_4p = ZERO_MASK_2D.at[c_p, :].set(legal_action_mask_c_p)

    state = _replace_state(state,   # type:ignore
        has_yaku=state.players.has_yaku.at[c_p, 0].set(
            can_ron[c_p, new_tile]
        ),  # If the player wins, the yaku is always attached.
        fan=state.players.fan.at[c_p, 0].set(
            jnp.int32(yakuman_num)
        ),  # Only judge the Yakuman.
        fu=state.players.fu.at[c_p, 0].set(jnp.int32(0)),  # If the player wins, the fu is 0.
        can_win=can_ron,
        legal_action_mask=legal_action_mask_4p,
        next_deck_ix=next_deck_ix,
        hand=hand,
        last_draw=new_tile,
        target=jnp.int8(-1),
    )
    return state


def _calc_wind(east_player: Array) -> Array:
    return jnp.array(
        [
            east_player,
            (east_player + 1) % 4,
            (east_player + 2) % 4,
            (east_player + 3) % 4,
        ],
        dtype=jnp.int8,
    )


def _is_first_turn(next_deck_ix: Array) -> Array:
    return next_deck_ix >= FIRST_DRAW_IDX - 4


def _step(state: State, action: Array) -> State:
    """
    Branch the process according to the action
    The type of action is referred to mahjong/_action.py
    """
    action_i8 = jnp.int8(action)
    # add current player
    action_history = state.round_state.action_history.at[0, state.step_count].set(
        state.current_player
    )
    # For discards, row 1 stores the discarded tile and row 2 stores the
    # tsumogiri flag. Non-discard actions keep the raw action id and use -1.
    is_tsumogiri = action_i8 == Action.TSUMOGIRI
    is_discard = ((0 <= action_i8) & (action_i8 < Tile.NUM_TILE_TYPE)) | is_tsumogiri
    recorded_action = jnp.where(is_tsumogiri, state.round_state.last_draw, action_i8).astype(
        jnp.int8
    )
    tsumogiri_flag = jnp.where(
        is_discard,
        is_tsumogiri.astype(jnp.int8),
        jnp.int8(-1),
    )
    action_history = action_history.at[1, state.step_count].set(recorded_action)
    action_history = action_history.at[2, state.step_count].set(tsumogiri_flag)
    state = _replace_state(state,   # type:ignore
        action_history=action_history
    )
    # execute actions
    discard_state = _discard(state, action)
    kan_state = _kan(state, action)
    riichi_state = _riichi(state)
    ron_state = _ron(state)
    tsumo_state = _tsumo(state)
    pon_state = _pon(state, action)
    chi_state = _chi(state, action)
    pass_state = _pass(state)
    next_round_state = _next_round(state)
    fn_idx = ACTION_FUN_MAP[action]
    state = jax.lax.switch(
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
            lambda: next_round_state,
        ],
    )
    state = jax.lax.cond(
        state.round_state.draw_next & ~state.round_state.is_abortive_draw_normal,
        lambda: _draw(state),
        lambda: state,
    )  # If the player draws a tile, call _draw.
    state = jax.lax.cond(
        state.round_state.kan_declared
        & ~state.round_state.is_abortive_draw_normal
        & ~state.players.legal_action_mask[
            :, Action.RON
        ].any(),  # If the player cannot declare a RobbingKan, call _draw_after_kan.
        lambda: _draw_after_kan(state),
        lambda: state,
    )  # If the player declares a Kan, call _draw_after_kan.
    state = jax.lax.cond(
        state.round_state.is_abortive_draw_normal & (state.round_state.dummy_count == 0) & ~state.terminated,
        lambda: _abortive_draw_normal(state),
        lambda: state,
    )  # If the game is ended (abortive_draw_normal (流局)), call _abortive_draw_normal.
    state = _replace_state(state,   # type:ignore
        legal_action_mask=state.players.legal_action_mask[state.current_player]
    )  # Set the legal action mask for the current player.
    shanten_val = Shanten.number(state.players.hand[state.current_player]).astype(jnp.int8)
    state = _replace_state(state,   # type:ignore
        shanten_current_player=shanten_val
    )
    return state


def _draw(state: State) -> State:
    """
    Draw a tile from the deck
    - Update the next drawn tile
    - Generate the legal action for the player who drew the tile
    - Accept the riichi
    - Update the furiten by pass
    - Update the is haitei flag
    """
    state = _accept_riichi(
        state
    )  # Cancel the riichi flag and subtract the score when the riichi is accepted
    c_p = state.current_player
    is_haitei = state.round_state.next_deck_ix == state.round_state.last_deck_ix
    new_tile = state.round_state.deck[state.round_state.next_deck_ix]
    next_deck_ix = state.round_state.next_deck_ix - 1
    hand = state.players.hand.at[c_p].set(Hand.add(state.players.hand[c_p], new_tile))
    # Generate the legal action for the player who drew the tile
    legal_action_mask_c_p = jax.lax.select(
        state.players.riichi[c_p],
        _make_legal_action_mask_after_draw_w_riichi(state, hand, c_p, new_tile),
        _make_legal_action_mask_after_draw(state, hand, c_p, new_tile),
    )
    legal_action_mask_4p = state.players.legal_action_mask.at[c_p, :].set(
        legal_action_mask_c_p
    )
    return _replace_state(state,   # type:ignore
        target=jnp.int8(-1),
        has_yaku=state.players.has_yaku.at[c_p, 0].set(
            state.players.has_yaku[c_p, 1]
        ),  # Update the information about the current drawn tile
        fan=state.players.fan.at[c_p, 0].set(
            state.players.fan[c_p, 1]
        ),  # Update the information about the current drawn tile
        fu=state.players.fu.at[c_p, 0].set(
            state.players.fu[c_p, 1]
        ),  # Update the information about the current drawn tile
        next_deck_ix=next_deck_ix,
        hand=hand,
        last_draw=new_tile,
        legal_action_mask=legal_action_mask_4p,
        furiten_by_pass=state.players.furiten_by_pass.at[c_p].set(
            state.players.furiten_by_pass[c_p] & state.players.riichi[c_p]
        ),  # Once the player with riichi is passed, the furiten by pass is not released.
        is_haitei=is_haitei,
        draw_next=FALSE,
    )


def _make_legal_action_mask_after_draw(
    state: State, hand: Array, c_p: Array, new_tile: Array
) -> Array:
    """
    Legal action mask for the player who drew a tile
    - Set discardable tiles
    - Set if the player can play CLOSED_KAN or ADDED_KAN
    - Set if the player can declare RIICHI
    - Set if the player can win by TSUMO
    """
    tiles_ok = (hand[c_p] > 0).astype(jnp.bool_)
    tiles_ok = tiles_ok.at[new_tile].set(
        hand[c_p, new_tile] >= 2
    )  # Drawn tile cannot be discarded by normal discard action if it is less than 2 (otherwise done by TSUMOGIRI)
    mask = ZERO_MASK_1D.at[: Tile.NUM_TILE_TYPE].set(tiles_ok)
    mask = mask.at[Action.TSUMOGIRI].set(TRUE)
    # Check if the player can declare CLOSED_KAN or ADDED_KAN
    cannot_kan = (
        state.players.n_kan.sum() >= 4
    )  # If the number of kan is 4 or more, the player cannot declare kan
    can_kan = (
        (
            Hand.can_closed_kan(hand[c_p], new_tile)
            | (
                Hand.can_added_kan(hand[c_p], new_tile)
                & (state.players.pon[(c_p, new_tile)] > 0)
            )
        )
        & ~state.round_state.is_haitei
        & ~cannot_kan
    )
    mask = mask.at[new_tile + Tile.NUM_TILE_TYPE].set(can_kan)
    # Check if the player can declare RIICHI
    no_next_draw = state.round_state.next_deck_ix < state.round_state.last_deck_ix + 4
    can_riichi = jnp.where(
        state.players.riichi[c_p] | ~state.players.is_hand_concealed[c_p] | no_next_draw,
        FALSE,
        Hand.can_riichi(hand[c_p]),
    )
    mask = mask.at[Action.RIICHI].set(can_riichi)
    can_tsumo = state.players.can_win[c_p, new_tile]
    _can_after_kan = state.round_state.can_after_kan
    _is_haitei = state.round_state.is_haitei
    _has_yaku = state.players.has_yaku[
        c_p, 1
    ]  # Whether each player has Yaku for the drawn tile is pre-calculated in previous discard action. Therefore, we can refer to it here.
    mask = mask.at[Action.TSUMO].set(
        can_tsumo
        & (state.players.is_hand_concealed[c_p] | _can_after_kan | _is_haitei | _has_yaku)
    )  # Even if the player does not have Yaku for their hand, they can win by TSUMO if it is AfterKan, Haitei.
    return mask


def _make_legal_action_mask_after_draw_w_riichi(
    state: State, hand: Array, c_p: Array, new_tile: Array
) -> Array:
    """
    Legal action mask for the player who drew a tile and declared RIICHI
    - Set if the player can play CLOSED_KAN
    - Set if the player can win by TSUMO
    """
    mask = ZERO_MASK_1D.at[Action.TSUMOGIRI].set(TRUE)
    can_closed_kan = (
        Hand.can_closed_kan_after_riichi(hand[c_p], new_tile, state.players.can_win[c_p])
        & ~state.round_state.is_haitei
    )
    mask = mask.at[new_tile + Tile.NUM_TILE_TYPE].set(can_closed_kan)
    mask = mask.at[Action.TSUMO].set(state.players.can_win[c_p, new_tile])
    return mask


def _discard(state: State, tile: Array) -> State:
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
    is_tsumogiri = tile == Action.TSUMOGIRI
    tile = jnp.where(
        tile == Action.TSUMOGIRI, state.round_state.last_draw, tile
    )  # If the tile is TSUUMOGIRI, use the last drawn tile
    is_riichi = state.players.riichi_declared[c_p]
    river = River.add_discard(
        state.players.river, tile, c_p, state.players.discard_counts[c_p], is_tsumogiri, is_riichi
    )  # Add the discarded tile to the river
    n_river = state.players.discard_counts.at[c_p].add(1)
    hand = state.players.hand.at[c_p].set(Hand.sub(state.players.hand[c_p], tile))
    state = _replace_state(state,   # type:ignore
        last_draw=jnp.int8(-1),
        hand=hand,
        river=river,
        discard_counts=n_river,
    )
    # Calculate YAKU for the discarded tile and the next drawn tile
    prevalent_wind = state.round_state.round % 4
    next_tile = state.round_state.deck[state.round_state.next_deck_ix]
    has_yaku, fan, fu = yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile(
        state, tile, next_tile, prevalent_wind
    )
    # Generate the legal action for the player who discarded the tile
    can_win = jax.vmap(Hand.can_ron, in_axes=(None, 0))(
        state.players.hand[c_p], TILE_RANGE
    )  # (34,)
    # Check if the player is furiten by the river
    is_furiten_by_river = jax.vmap(_is_waiting_tile, in_axes=(None, 0))(
        can_win, River.decode_tile(river[c_p])
    ).any()
    state = _replace_state(state,   # type:ignore
        has_yaku=has_yaku,
        fan=fan,
        fu=fu,
        can_win=state.players.can_win.at[c_p].set(can_win),
        furiten_by_discard=state.players.furiten_by_discard.at[c_p].set(is_furiten_by_river),
        can_after_kan=FALSE,  # AfterKan is disabled when the player discards a tile
        ippatsu=state.players.ippatsu.at[c_p].set(
            FALSE
        ),  # Ippatsu is disabled when the player discards a tile
    )
    # Generate the legal action for OTHER players (melds and ron)
    legal_action_mask_4p = jax.vmap(
        _make_legal_action_mask_after_discard, in_axes=(None, 0, 0, None)
    )(
        state, state.players.hand, jnp.arange(4), tile
    )  # (4, 87)
    legal_action_mask_4p = legal_action_mask_4p.at[c_p, :].set(
        FALSE
    )  # Set the legal action for the player who discarded the tile to False

    next_meld_player, can_any = _next_meld_player(
        legal_action_mask_4p, c_p
    )  # Set the next player
    no_ron_player = jnp.logical_not(legal_action_mask_4p[:, Action.RON].any())
    no_meld_player = jnp.logical_not(can_any)
    # Check if the game is ended (abortive_draw_normal)
    is_abortive_draw_normal = (
        state.round_state.next_deck_ix < state.round_state.last_deck_ix
    )  # If the next drawn tile is not left, the game is ended
    state = jax.lax.cond(
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
    haitei = state.round_state.is_haitei
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
    can_ron = state.players.can_win[c_p, tile]
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
    chi_results = jax.vmap(Hand.can_chi, in_axes=(None, None, 0))(
        hand, tile, jnp.arange(Action.CHI_L, Action.CHI_R + 1)
    )
    legal_action_mask = ZERO_MASK_1D.at[Action.CHI_L : Action.CHI_R + 1].set(
        chi_results
    )
    return legal_action_mask


def _mask_for_pon_open_kan(hand: Array, tile: Array, cannot_kan: Array) -> Array:
    """
    - Check if the player can play PON or OPEN_KAN with the target tile
    """
    pon_result = Hand.can_pon(hand, tile)
    open_kan_result = Hand.can_open_kan(hand, tile) & ~cannot_kan
    legal_action_mask = ZERO_MASK_1D.at[Action.PON].set(pon_result)
    legal_action_mask = legal_action_mask.at[Action.OPEN_KAN].set(open_kan_result)
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
        legal_action_mask_4p[:, Action.CHI_L : Action.CHI_R + 1].sum(axis=1) > 0
    )  # (4,)
    can_pon = legal_action_mask_4p[:, Action.PON] > 0  # (4,)
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
    return (tile != -1) & can_ron[tile]


def _draw_after_kan(state: State):
    """
    Process when a KAN is Accepted
    - Disable Ippatsu
    - Disable Double Riichi
    - Update the KAN dora
    - Update the Haitei tile
    - Disable the kan flag
    - Draw the rinshan tile
    - Calculate legal_action_mask for the player who drew the tile
    - Set the AfterKan flag (嶺上開花)
    """
    c_p = state.current_player
    n_kan = state.players.n_kan.sum()  # The number of kan
    rinshan_tile = state.round_state.deck[
        10 + n_kan
    ]  # Reference the deck in _state.py TODO: Is it correct?

    # Process the KAN dora
    n_kan_doras = state.round_state.n_kan_doras  # The number of kan dora before updating
    next_kan_dora = state.round_state.deck[
        9 - 2 * (n_kan_doras + 1)
    ]  # Reference the deck in _state.py
    next_kan_ura = state.round_state.deck[
        8 - 2 * (n_kan_doras + 1)
    ]  # Reference the deck in _state.py
    state = _replace_state(state, 
        ippatsu=jnp.zeros(4, dtype=jnp.bool_),  # Disable Ippatsu
        can_after_kan=TRUE,
        n_kan=state.players.n_kan + 1,
        kan_declared=FALSE,
        n_kan_doras=state.round_state.n_kan_doras + 1,
        dora_indicators=state.round_state.dora_indicators.at[state.round_state.n_kan_doras + 1].set(
            next_kan_dora
        ),  # Reveal the KAN dora
        ura_dora_indicators=state.round_state.ura_dora_indicators.at[state.round_state.n_kan_doras + 1].set(
            next_kan_ura
        ),  # Reveal the KAN dora
        last_deck_ix=state.round_state.last_deck_ix
        + 1,  # Update the last deck index after drawing the rinshan tile
    )

    hand = state.players.hand.at[c_p].set(Hand.add(state.players.hand[c_p], rinshan_tile))
    can_ron = jax.vmap(Hand.can_ron, in_axes=(None, 0))(
        state.players.hand[c_p], TILE_RANGE
    )  # (34,) Update the legal action for the player who drew the tile
    state = _replace_state(state, 
        can_win=state.players.can_win.at[c_p].set(can_ron),
    )
    is_riichi = state.players.riichi[c_p]
    legal_action_mask_c_p = jax.lax.cond(
        is_riichi,
        lambda: _make_legal_action_mask_after_draw_w_riichi(
            state, hand, c_p, rinshan_tile
        ),
        lambda: _make_legal_action_mask_after_draw(state, hand, c_p, rinshan_tile),
    )
    legal_action_mask_4p = state.players.legal_action_mask.at[c_p, :].set(
        legal_action_mask_c_p
    )  # Update the legal action for the player who drew the tile
    return _replace_state(state,   # type:ignore
        last_draw=rinshan_tile,
        hand=hand,
        legal_action_mask=legal_action_mask_4p,
        has_yaku=state.players.has_yaku.at[c_p, 0].set(state.players.has_yaku[c_p, 1]),
        fan=state.players.fan.at[c_p, 0].set(state.players.fan[c_p, 1]),
        fu=state.players.fu.at[c_p, 0].set(state.players.fu[c_p, 1]),
    )


def _kan(state: State, action):
    """
    Process when a KAN is Declared
    - Process the KAN
    - Calculate YAKU for the Robbing KAN and the rinshan tile
    - Apply KAN action
    - Disable Ippatsu
    """
    c_p = state.current_player
    tile = action - Tile.NUM_TILE_TYPE
    prevalent_wind = state.round_state.round % 4
    rinshan_tile = state.round_state.deck[
        jnp.int32(10 + state.players.n_kan.sum())
    ]  # Reference the deck in _state.py
    # Apply KAN action to hand, meld, river
    is_open_kan = action == Action.OPEN_KAN
    pon = state.players.pon[(c_p, tile)]
    is_added_kan = pon != 0  # TODO: Is it correct?
    state = jax.lax.cond(
        is_open_kan,
        lambda: _open_kan(state),
        lambda: _selfkan(state, action, is_added_kan),
    )
    # Calculate YAKU for the RobbingKan and the rinshan tile
    has_yaku, fan, fu = yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile(
        state, tile, rinshan_tile, prevalent_wind
    )  # (4, 2)
    state = _replace_state(state, 
        has_yaku=has_yaku,
        fan=fan,
        fu=fu,
    )
    # Check if the player can win by RON
    is_furiten = state.players.furiten_by_discard | state.players.furiten_by_pass  # (4,)
    legal_action_mask_4p = state.players.legal_action_mask.at[:, Action.RON].set(
        state.players.can_win[:, tile] & ~is_furiten
    )
    legal_action_mask_4p = legal_action_mask_4p.at[c_p, Action.RON].set(
        FALSE
    )  # Disable the legal action for the player who declared the KAN
    state = _replace_state(state, 
        legal_action_mask=legal_action_mask_4p,
    )
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
    target = action - Tile.NUM_TILE_TYPE  # Convert to 0-34
    return jax.lax.cond(
        is_added_kan,
        lambda: _added_kan(state, target),
        lambda: _closed_kan(state, target),
    )


def _closed_kan(state: State, target):
    """
    Apply CLOSED_KAN
    - Generate a meld from the target
    - Update the hand and meld
    """
    c_p = state.current_player
    meld = Meld.init(target + Tile.NUM_TILE_TYPE, target, src=0)
    state = _append_meld(state, meld, c_p)
    hand = state.players.hand.at[c_p].set(Hand.closed_kan(state.players.hand[c_p], target))
    return _replace_state(state,   # type:ignore
        hand=hand,
    )


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
        Meld.init(target + Tile.NUM_TILE_TYPE, target, pon_src)
    )
    hand = state.players.hand.at[c_p].set(Hand.added_kan(state.players.hand[c_p], target))
    # Since the ADDED_KAN consumes the pon, set it to 0
    pon = state.players.pon.at[(c_p, target)].set(jnp.int8(0))
    return _replace_state(state,   # type:ignore
        melds=melds,
        hand=hand,
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
    hand = state.players.hand.at[c_p].set(Hand.open_kan(state.players.hand[c_p], state.round_state.target))
    is_hand_concealed = state.players.is_hand_concealed.at[c_p].set(FALSE)
    # Add the meld to the river
    river = River.add_meld(
        state.players.river, Action.OPEN_KAN, l_p, state.players.discard_counts[l_p] - 1, src
    )
    return _replace_state(state,   # type:ignore
        hand=hand,
        target=jnp.int8(-1),
        is_hand_concealed=is_hand_concealed,
        river=river,
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
    meld = Meld.init(Action.PON, tar, src)
    state = _append_meld(state, meld, c_p)
    pon_hand = Hand.pon(state.players.hand[c_p], tar)
    hand = state.players.hand.at[c_p].set(pon_hand)
    is_hand_concealed = state.players.is_hand_concealed.at[c_p].set(FALSE)
    # Add the pon information for the ADDED_KAN
    pon = state.players.pon.at[(c_p, tar)].set(jnp.int8(src << 2 | state.players.meld_counts[c_p] - 1))
    # Add the meld to the river
    river = River.add_meld(state.players.river, Action.PON, l_p, state.players.discard_counts[l_p] - 1, src)
    legal_action_mask_4p = (
        ZERO_MASK_2D.at[c_p, : Tile.NUM_TILE_TYPE]
        .set((hand[c_p] > 0).astype(jnp.bool_))
        .at[c_p, tar]
        .set(FALSE)  # The target tile is prohibited
        .at[c_p, Action.PASS]
        .set(FALSE)
    )  # Update the legal action for the player who declared the PON
    return _replace_state(state,   # type:ignore
        target=jnp.int8(-1),
        is_hand_concealed=is_hand_concealed,
        pon=pon,
        hand=hand,
        legal_action_mask=legal_action_mask_4p,
        river=river,
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
    chi_hand = Hand.chi(state.players.hand[c_p], tar, action)
    hand = state.players.hand.at[c_p].set(chi_hand)
    is_hand_concealed = state.players.is_hand_concealed.at[c_p].set(FALSE)
    legal_action_mask_4p = (
        _make_legal_action_mask_after_chi(state, hand, c_p, tar, action)
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
        legal_action_mask=legal_action_mask_4p,
        river=river,
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
    prohibitive_tile_type = Meld.prohibitive_tile_type_after_chi(
        action, target
    )  # Prohibit Swap-Calling: [1]23 -> 4 is prohibited
    # Create player's mask efficiently
    tile_mask = hand[c_p] > 0
    # Apply prohibitive tile restriction
    tile_mask = tile_mask.at[prohibitive_tile_type].set(
        jnp.logical_and(tile_mask[prohibitive_tile_type], prohibitive_tile_type == -1)
    )
    # Prohibit the target tile to be discarded
    tile_mask = tile_mask.at[target].set(FALSE)
    player_mask = ZERO_MASK_1D.at[: Tile.NUM_TILE_TYPE].set(tile_mask)
    # Build the legal action mask for the player
    legal_action_mask_4p = ZERO_MASK_2D.at[c_p].set(player_mask)
    return legal_action_mask_4p


def _pass(state: State):
    """
    Apply PASS
    - For discarded tile
    - If the player who can RON, KAN, PON, CHI passes, set the next player from the legal action
    - If no next meld player, proceed to the next player's draw
    """
    c_p = state.current_player
    # If the player who declared the KAN passes, set the next player from the legal action
    can_robbing_kan = state.round_state.kan_declared
    is_ron_player = jnp.bool_(state.players.legal_action_mask[c_p, Action.RON])
    legal_action_mask_4p = state.players.legal_action_mask.at[c_p, :].set(FALSE)
    # Set the next player from the legal action
    next_meld_player, can_any = _next_meld_player(
        legal_action_mask_4p, state.round_state.last_player
    )  # Reference the current wind for the next meld player
    no_meld_player = jnp.logical_not(can_any)
    # Check if the game is ended (abortive_draw_normal (流局))
    is_abortive_draw_normal = (
        state.round_state.next_deck_ix < state.round_state.last_deck_ix
    )  # If the next deck index is less than the last deck index, the game is ended (abortive_draw_normal (流局))
    return jax.lax.cond(
        no_meld_player,
        lambda: _replace_state(state,   # type:ignore
            current_player=jnp.where(
                can_robbing_kan,
                jnp.int8(state.round_state.last_player),
                jnp.int8((state.round_state.last_player + 1) % 4),
            ),  # If the player who declared the KAN passes, set the last player
            target=jnp.int8(-1),
            furiten_by_pass=state.players.furiten_by_pass.at[c_p].set(
                is_ron_player & ~can_robbing_kan
            ),  # If the player who RON passes, set the furiten
            draw_next=TRUE
            & ~can_robbing_kan,  # If no next player for robbing KAN, draw the rinshan tile
            is_abortive_draw_normal=is_abortive_draw_normal,
            legal_action_mask=legal_action_mask_4p,
        ),
        lambda: _replace_state(state,   # type:ignore
            current_player=jnp.int8(next_meld_player),
            target=jnp.int8(state.round_state.target),  # Do not change the target
            legal_action_mask=legal_action_mask_4p.at[
                next_meld_player, Action.PASS
            ].set(
                TRUE
            ),  # Add the pass action to the legal action
            furiten_by_pass=state.players.furiten_by_pass.at[c_p].set(
                is_ron_player & ~can_robbing_kan
            ),  # If the player who RON passes, set the furiten
        ),
    )


def _riichi(state: State):
    """
    Apply RIICHI
    - Set the RIICHI declared flag
    - Generate the legal action for the player after RIICHI
    """
    c_p = state.current_player
    legal_action_mask_for_discard = jax.vmap(Hand.is_tenpai)(
        jax.vmap(Hand.sub, in_axes=(None, 0))(state.players.hand[c_p], TILE_RANGE)
    )  # Only tiles that can maintain tenpai can be discarded
    legal_action_mask_for_discard = jnp.logical_and(
        legal_action_mask_for_discard, state.players.hand[c_p]
    )
    # Set tile actions
    player_mask = ZERO_MASK_1D.at[: Tile.NUM_TILE_TYPE].set(
        legal_action_mask_for_discard
    )
    # The last drawn tile must have at least 2 tiles to be discarded
    player_mask = player_mask.at[state.round_state.last_draw].set(
        (state.players.hand[c_p, state.round_state.last_draw] >= 2)
        & legal_action_mask_for_discard[state.round_state.last_draw]
    )
    # Set TSUUMOGIRI action
    player_mask = player_mask.at[Action.TSUMOGIRI].set(
        legal_action_mask_for_discard[state.round_state.last_draw]
    )
    return _replace_state(state,   # type:ignore
        legal_action_mask=state.players.legal_action_mask.at[c_p].set(player_mask),
        riichi_declared=state.players.riichi_declared.at[c_p].set(TRUE),
        draw_next=FALSE,
    )


def _ron(state: State) -> State:
    """
    Apply RON
    - Calculate the score of the winner (consider only the remainder when divided by 100)
    - Clear the Kyotaku
    """
    c_p = state.current_player
    is_ippatsu = state.players.ippatsu[c_p] & state.players.riichi[c_p]  # Ippatsu (一発)
    is_double_riichi = state.players.double_riichi[c_p]  # Double Riichi (ダブル立直)
    can_robbing_kan = state.round_state.kan_declared  # RobbingKan (槍槓)
    is_houtei = state.round_state.is_haitei  # Bottom of the River (河底摸月)
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
    honba = state.round_state.honba * 3  # 1 Honba is 300 points (per player)
    # Build reward array more efficiently
    reward = jnp.zeros(4, dtype=jnp.float32)
    reward = reward.at[c_p].set(score + honba)
    reward = reward.at[state.round_state.last_player].set(-score - honba)
    # The Kyotaku is already paid when the RIICHI is declared, so we only need to add the Kyotaku to the winner
    kyotaku_bonus = 10 * (state.round_state.kyotaku)
    reward = reward.at[c_p].add(kyotaku_bonus)
    score = state.round_state.score + jnp.float32(reward)
    return _replace_state(state,   # type:ignore
        terminated_round=TRUE,
        score=jnp.int32(score),  # Update the score
        rewards=jnp.float32(reward),
        kyotaku=jnp.int8(0),  # Clear the Kyotaku
        has_won=state.players.has_won.at[c_p].set(TRUE),
        legal_action_mask=ZERO_MASK_2D.at[:, Action.DUMMY].set(
            TRUE
        ),  # Set the dummy action
        draw_next=FALSE,
    )


def _tsumo(state: State) -> State:
    """
    Apply TSUUMO
    - Calculate the score of the winner
    - Update the score
    - terminated=true
    - Clear the Kyotaku
    - Check if the game is ended (abortive_draw_normal (流局))
    """
    c_p = state.current_player
    _dealer = state.round_state.dealer
    can_after_kan = (
        state.round_state.can_after_kan
    )  # Check if the game is ended (AfterKan (嶺上開花))
    is_ippatsu = state.players.ippatsu[c_p] & state.players.riichi[c_p]  # Ippatsu (一発)
    is_double_riichi = state.players.double_riichi[c_p]  # Double Riichi (ダブル立直)
    is_haitei = state.round_state.is_haitei  # Bottom of the River (河底摸月)
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
    fu = jnp.where(is_yakuman, 0, state.players.fu[c_p, 0] + (2 * can_after_kan))
    basic_score = Yaku.score(
        jnp.int32(fan),  # Calculate the score when the tile is discarded
        jnp.int32(fu),  # When AfterKan, add 2 fu
    )
    honba = state.round_state.honba * 1  # 1 Honba is 100 points (per player)
    s1 = jnp.ceil(basic_score / 100)
    s2 = jnp.ceil(basic_score * 2 / 100)
    score = jnp.where(_dealer == c_p, basic_score * 6, basic_score * 4)
    score = jnp.ceil(score / 100)
    # Build reward array more efficiently
    reward = jnp.where(
        _dealer == c_p,
        # If c_p is dealer
        jnp.full(4, -s2 - honba, dtype=jnp.float32),
        # If c_p is not dealer
        jnp.full(4, -s1 - honba, dtype=jnp.float32),
    )
    # Update specific positions based on dealer condition
    reward = jnp.where(
        _dealer == c_p,
        reward.at[c_p].set(s2 * 3 + 3 * honba),  # The dealer pays the score
        reward.at[_dealer]
        .set(-s2 - honba)
        .at[c_p]
        .set(s1 * 2 + s2 + 3 * honba),  # The non-dealer pays the score
    )
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
    rewards = jnp.where(
        tenpai, total_rewards // n_tenpai, -total_rewards // n_not_tenpai
    )
    rewards = jnp.where(
        jnp.logical_or(n_tenpai == 0, n_tenpai == 4),
        jnp.zeros(4, dtype=jnp.int32),
        rewards,
    )
    return _replace_state(state,   # type:ignore
        rewards=rewards.astype(jnp.float32),
        score=jnp.int32(state.round_state.score + jnp.float32(rewards)),  # Update the score
        legal_action_mask=ZERO_MASK_2D.at[:, Action.DUMMY].set(
            TRUE
        ),  # Set the dummy action
        terminated_round=TRUE,
        draw_next=FALSE,
    )


def _next_round(state: State) -> State:
    """
    Move to the next round
    - Process the next round
    - Move the dealer
    - Process the round
    - If the round is ended, calculate the rank points. The Kyotaku is the top (same point wind order)
    - DUMMY sharing: 3 times of rotation (cp to +1 mod 4) after the 4th time, the result is determined
    """
    dc = state.round_state.dummy_count  # int8

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

        rng, subkey = jax.random.split(s.round_state.rng_key)

        # ★ Initialize only at this timing
        base_next = _make_state(
            rng_key=subkey,
            current_player=next_dealer,  # Start from the dealer
            dealer=next_dealer,
            seat_wind=_calc_wind(next_dealer),
            round=next_round,
            honba=next_honba,
            kyotaku=s.round_state.kyotaku,
            score=s.round_state.score,
        )
        next_round_state = _init_for_next_round(subkey, base_next)

        terminated_state = _make_state(
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


def _dora_array(state: State) -> Array:
    """
    - Create an array of length 34, where the number of tiles is stored in the index of the dora tile
    """

    def update_dora_counts(dora_counts: Array, dora_indicator: Array) -> Array:
        is_dora_valid = dora_indicator != -1
        return dora_counts.at[DORA_ARRAY[dora_indicator]].add(is_dora_valid)

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
