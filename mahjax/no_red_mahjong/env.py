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


from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from mahjax._src.types import Array, PRNGKey
from mahjax.core import Env
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.hand import Hand
from mahjax.no_red_mahjong.meld import Meld
from mahjax.no_red_mahjong.shanten import Shanten
from mahjax.no_red_mahjong.state import DORA_ARRAY, FIRST_DRAW_IDX, State
from mahjax.no_red_mahjong.tile import River, Tile
from mahjax.no_red_mahjong.yaku import Yaku

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)

TILE_RANGE = jnp.arange(Tile.NUM_TILE_TYPE)
ZERO_MASK_1D = jnp.zeros(Action.NUM_ACTION, dtype=jnp.bool_)
ZERO_MASK_2D = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)

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

    hand_b = state._hand[i_idx]  # (8, ...)
    melds_b = state._melds[i_idx]  # (8, ...)
    n_meld_b = state._n_meld[i_idx]  # (8,)
    riichi_b = state._riichi[i_idx]  # (8,)
    cur_wind_b = state._seat_wind[i_idx]  # (8,)
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
        one_round: bool = False,
        observe_type: str = "dict",
        order_points: List[int] = [
            30,
            10,
            -10,
            -30,
        ],  # No oka, 10-30, SAIKOUISEN rule https://saikouisen.com/about/rules/
    ):
        self.one_round = one_round
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
        state = state.replace(  # type:ignore
            _order_points=jnp.array(self.order_points, dtype=jnp.int32),
        )  # type: ignore
        shanten_val = Shanten.number(state._hand[state.current_player]).astype(jnp.int8)
        state = state.replace(  # type:ignore
            _shanten_c_p=shanten_val
        )
        # observation = self.observe(state)
        # return state.replace(observation=observation)  # type: ignore
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
        state = state.replace(  # type:ignore
            _order_points=jnp.array(self.order_points, dtype=jnp.int32),
        )  # type: ignore reflect the order points

        # If the state is already terminated or truncated, environment does not take usual step,
        # but return the same state with zero-rewards for all players
        stepped_state = _step(state, action).replace(_step_count=state._step_count + 1)
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(rewards=jnp.zeros_like(state.rewards)),  # type: ignore
            lambda: stepped_state,  # type: ignore
        )
        state = jax.lax.cond(
            state._terminated_round & self.one_round,
            lambda: state.replace(terminated=TRUE),
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
            lambda: state.replace(legal_action_mask=jnp.ones_like(state.legal_action_mask)),  # type: ignore
            lambda: state,
        )
        # observation = self.observe(state)
        # state = state.replace(observation=observation)  # type: ignore
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
        return state.replace(rewards=reward, terminated=TRUE)  # type: ignore


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
    state = State(  # type:ignore
        current_player=current_player,
        _dealer=current_player,
        _init_wind=_calc_wind(current_player),
        _seat_wind=_calc_wind(current_player),
        _last_player=last_player,
        _deck=deck,
        _dora_indicators=dora_indicators,
        _ura_dora_indicators=ura_dora_indicators,
        _hand=init_hand,
    )
    can_ron = v_can_win(state._hand, TILE_RANGE)  # (4, 34)
    c_p = (
        state.current_player
    )  # To avoid recurrence by drawing, explicitly write the first draw.
    new_tile = state._deck[state._next_deck_ix]
    next_deck_ix = state._next_deck_ix - 1
    # Only judge the Yakuman.
    prevalent_wind = state._round % 4
    dora = _dora_array(state)
    _, yakuman_num, _ = Yaku.judge_yakuman(
        state._hand[c_p],
        state._melds[c_p],
        state._n_meld[c_p],
        new_tile,
        state._riichi[c_p],
        FALSE,
        prevalent_wind,
        state._seat_wind[c_p],
        dora,
    )
    hand = state._hand.at[c_p].set(Hand.add(state._hand[c_p], new_tile))
    # Generate the legal action for the player who drew the tile after the draw
    legal_action_mask_c_p = _make_legal_action_mask_after_draw(
        state, hand, c_p, new_tile
    )
    legal_action_mask_4p = ZERO_MASK_2D.at[c_p, :].set(legal_action_mask_c_p)
    state = state.replace(  # type:ignore
        legal_action_mask=legal_action_mask_4p[c_p],
        _has_yaku=state._has_yaku.at[c_p, 0].set(
            can_ron[c_p, new_tile]
        ),  # If the combination is horable, the yaku is always attached (Blessing of Heaven).
        _fan=state._fan.at[c_p, 0].set(
            jnp.int32(yakuman_num)
        ),  # Only judge the Yakuman.
        _fu=state._fu.at[c_p, 0].set(jnp.int32(0)),  # If the player wins, the fu is 0.
        _can_win=can_ron,
        _legal_action_mask_4p=legal_action_mask_4p,
        _next_deck_ix=next_deck_ix,
        _hand=hand,
        _last_draw=new_tile,
        _target=jnp.int8(-1),
    )
    return state


def _init_for_next_round(rng: PRNGKey, state: State) -> State:
    """
    Initialize the state for the next round
    - Generate the new deck
    - Set game-related variables (last player, deck, dora indicators, ura dora indicators, hand, rng key)
    - Succeed the process of _next_round (dealer, seat wind, round, honba, kyotaku, score, etc.)

    Args:
        rng (PRNGKey): Random number generator key
        state (State): Current state of the game

    Returns:
        State: State after processing when the next round is determined
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
    state = state.replace(  # type:ignore
        _last_player=last_player,
        _deck=deck,
        _dora_indicators=dora_indicators,
        _ura_dora_indicators=ura_dora_indicators,
        _hand=init_hand,
        _rng_key=subkey,
    )
    can_ron = v_can_win(state._hand, TILE_RANGE)  # (4, 34)
    c_p = (
        state.current_player
    )  # To avoid recurrence by drawing, explicitly write the first draw.
    new_tile = state._deck[state._next_deck_ix]
    next_deck_ix = state._next_deck_ix - 1
    # Only judge the Yakuman.
    prevalent_wind = state._round % 4
    dora = _dora_array(state)
    _, yakuman_num, _ = Yaku.judge_yakuman(
        state._hand[c_p],
        state._melds[c_p],
        state._n_meld[c_p],
        new_tile,
        state._riichi[c_p],
        FALSE,
        prevalent_wind,
        state._seat_wind[c_p],
        dora,
    )

    hand = state._hand.at[c_p].set(Hand.add(state._hand[c_p], new_tile))
    # Generate the legal action for the player who drew the tile after the draw
    legal_action_mask_c_p = _make_legal_action_mask_after_draw(
        state, hand, c_p, new_tile
    )
    legal_action_mask_4p = ZERO_MASK_2D.at[c_p, :].set(legal_action_mask_c_p)

    state = state.replace(  # type:ignore
        _has_yaku=state._has_yaku.at[c_p, 0].set(
            can_ron[c_p, new_tile]
        ),  # If the player wins, the yaku is always attached.
        _fan=state._fan.at[c_p, 0].set(
            jnp.int32(yakuman_num)
        ),  # Only judge the Yakuman.
        _fu=state._fu.at[c_p, 0].set(jnp.int32(0)),  # If the player wins, the fu is 0.
        _can_win=can_ron,
        _legal_action_mask_4p=legal_action_mask_4p,
        _next_deck_ix=next_deck_ix,
        _hand=hand,
        _last_draw=new_tile,
        _target=jnp.int8(-1),
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
    Args:
        state: State: Current state of the game
        action: Array: Action to be performed

    Returns:
        State: State after processing the action
    """
    # add action history
    action_history = state._action_history.at[0, state._step_count].set(
        state.current_player
    )
    action_history = action_history.at[1, state._step_count].set(action)
    state = state.replace(  # type:ignore
        _action_history=action_history
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
        state._draw_next & ~state._is_abortive_draw_normal,
        lambda: _draw(state),
        lambda: state,
    )  # If the player draws a tile, call _draw.
    state = jax.lax.cond(
        state._kan_declared
        & ~state._is_abortive_draw_normal
        & ~state._legal_action_mask_4p[
            :, Action.RON
        ].any(),  # If the player cannot declare a RobbingKan, call _draw_after_kan.
        lambda: _draw_after_kan(state),
        lambda: state,
    )  # If the player declares a Kan, call _draw_after_kan.
    state = jax.lax.cond(
        state._is_abortive_draw_normal & (state._dummy_count == 0) & ~state.terminated,
        lambda: _abortive_draw_normal(state),
        lambda: state,
    )  # If the game is ended (abortive_draw_normal (流局)), call _abortive_draw_normal.
    state = state.replace(  # type:ignore
        legal_action_mask=state._legal_action_mask_4p[state.current_player]
    )  # Set the legal action mask for the current player.
    shanten_val = Shanten.number(state._hand[state.current_player]).astype(jnp.int8)
    state = state.replace(  # type:ignore
        _shanten_c_p=shanten_val
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

    Args:
        state: State: Current state of the game

    Returns:
        State: State after drawing a tile
    """
    state = _accept_riichi(
        state
    )  # Cancel the riichi flag and subtract the score when the riichi is accepted
    c_p = state.current_player
    is_haitei = state._next_deck_ix == state._last_deck_ix
    new_tile = state._deck[state._next_deck_ix]
    next_deck_ix = state._next_deck_ix - 1
    hand = state._hand.at[c_p].set(Hand.add(state._hand[c_p], new_tile))
    # Generate the legal action for the player who drew the tile
    legal_action_mask_c_p = jax.lax.select(
        state._riichi[c_p],
        _make_legal_action_mask_after_draw_w_riichi(state, hand, c_p, new_tile),
        _make_legal_action_mask_after_draw(state, hand, c_p, new_tile),
    )
    legal_action_mask_4p = state._legal_action_mask_4p.at[c_p, :].set(
        legal_action_mask_c_p
    )
    return state.replace(  # type:ignore
        _target=jnp.int8(-1),
        _has_yaku=state._has_yaku.at[c_p, 0].set(
            state._has_yaku[c_p, 1]
        ),  # Update the information about the current drawn tile
        _fan=state._fan.at[c_p, 0].set(
            state._fan[c_p, 1]
        ),  # Update the information about the current drawn tile
        _fu=state._fu.at[c_p, 0].set(
            state._fu[c_p, 1]
        ),  # Update the information about the current drawn tile
        _next_deck_ix=next_deck_ix,
        _hand=hand,
        _last_draw=new_tile,
        _legal_action_mask_4p=legal_action_mask_4p,
        _furiten_by_pass=state._furiten_by_pass.at[c_p].set(
            state._furiten_by_pass[c_p] & state._riichi[c_p]
        ),  # Once the player with riichi is passed, the furiten by pass is not released.
        _is_haitei=is_haitei,
        _draw_next=FALSE,
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

    Args:
        state: State: Current state of the game
        hand: Array: Hand of the player
        c_p: Array: Current player
        new_tile: Array: New tile drawn

    Returns:
        Array: legal_action_mask (NUM_ACTIONS,)
    """
    tiles_ok = (hand[c_p] > 0).astype(jnp.bool_)
    tiles_ok = tiles_ok.at[new_tile].set(
        hand[c_p, new_tile] >= 2
    )  # Drawn tile cannot be discarded by normal discard action if it is less than 2 (otherwise done by TSUMOGIRI)
    mask = ZERO_MASK_1D.at[: Tile.NUM_TILE_TYPE].set(tiles_ok)
    mask = mask.at[Action.TSUMOGIRI].set(TRUE)
    # Check if the player can declare CLOSED_KAN or ADDED_KAN
    cannot_kan = (
        state._n_kan.sum() >= 4
    )  # If the number of kan is 4 or more, the player cannot declare kan
    can_kan = (
        (
            Hand.can_closed_kan(hand[c_p], new_tile)
            | (
                Hand.can_added_kan(hand[c_p], new_tile)
                & (state._pon[(c_p, new_tile)] > 0)
            )
        )
        & ~state._is_haitei
        & ~cannot_kan
    )
    mask = mask.at[new_tile + Tile.NUM_TILE_TYPE].set(can_kan)
    # Check if the player can declare RIICHI
    no_next_draw = state._next_deck_ix < state._last_deck_ix + 4
    can_riichi = jnp.where(
        state._riichi[c_p] | ~state._is_hand_concealed[c_p] | no_next_draw,
        FALSE,
        Hand.can_riichi(hand[c_p]),
    )
    mask = mask.at[Action.RIICHI].set(can_riichi)
    can_tsumo = state._can_win[c_p, new_tile]
    _can_after_kan = state._can_after_kan
    _is_haitei = state._is_haitei
    _has_yaku = state._has_yaku[
        c_p, 1
    ]  # Whether each player has Yaku for the drawn tile is pre-calculated in previous discard action. Therefore, we can refer to it here.
    mask = mask.at[Action.TSUMO].set(
        can_tsumo
        & (state._is_hand_concealed[c_p] | _can_after_kan | _is_haitei | _has_yaku)
    )  # Even if the player does not have Yaku for their hand, they can win by TSUMO if it is AfterKan, Haitei.
    return mask


def _make_legal_action_mask_after_draw_w_riichi(
    state: State, hand: Array, c_p: Array, new_tile: Array
) -> Array:
    """
    Legal action mask for the player who drew a tile and declared RIICHI
    - Set if the player can play CLOSED_KAN
    - Set if the player can win by TSUMO

    Args:
        state: State: Current state of the game
        hand: Array: Hand of the player
        c_p: Array: Current player
        new_tile: Array: New tile drawn

    Returns:
        Array: legal_action_mask (NUM_ACTIONS,)
    """
    mask = ZERO_MASK_1D.at[Action.TSUMOGIRI].set(TRUE)
    can_closed_kan = (
        Hand.can_closed_kan_after_riichi(hand[c_p], new_tile, state._can_win[c_p])
        & ~state._is_haitei
    )
    mask = mask.at[new_tile + Tile.NUM_TILE_TYPE].set(can_closed_kan)
    mask = mask.at[Action.TSUMO].set(state._can_win[c_p, new_tile])
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

    Args:
        state: State: Current state of the game
        tile: Array: Discarded tile

    Returns:
        State: State after discarding a tile
    """
    c_p = state.current_player
    is_tsumogiri = tile == Action.TSUMOGIRI
    tile = jnp.where(
        tile == Action.TSUMOGIRI, state._last_draw, tile
    )  # If the tile is TSUUMOGIRI, use the last drawn tile
    is_riichi = state._riichi_declared
    river = River.add_discard(
        state._river, tile, c_p, state._n_river[c_p], is_tsumogiri, is_riichi
    )  # Add the discarded tile to the river
    n_river = state._n_river.at[c_p].add(1)
    hand = state._hand.at[c_p].set(Hand.sub(state._hand[c_p], tile))
    state = state.replace(  # type:ignore
        _last_draw=jnp.int8(-1),
        _hand=hand,
        _river=river,
        _n_river=n_river,
    )
    # Calculate YAKU for the discarded tile and the next drawn tile
    prevalent_wind = state._round % 4
    next_tile = state._deck[state._next_deck_ix]
    has_yaku, fan, fu = yaku_judge_for_discarded_or_kanned_tile_and_next_draw_tile(
        state, tile, next_tile, prevalent_wind
    )
    # Generate the legal action for the player who discarded the tile
    can_win = jax.vmap(Hand.can_ron, in_axes=(None, 0))(
        state._hand[c_p], TILE_RANGE
    )  # (34,)
    # Check if the player is furiten by the river
    is_furiten_by_river = jax.vmap(_is_waiting_tile, in_axes=(None, 0))(
        can_win, River.decode_tile(river[c_p])
    ).any()
    state = state.replace(  # type:ignore
        _has_yaku=has_yaku,
        _fan=fan,
        _fu=fu,
        _can_win=state._can_win.at[c_p].set(can_win),
        _furiten_by_discard=state._furiten_by_discard.at[c_p].set(is_furiten_by_river),
        _can_after_kan=FALSE,  # AfterKan is disabled when the player discards a tile
        _ippatsu=state._ippatsu.at[c_p].set(
            FALSE
        ),  # Ippatsu is disabled when the player discards a tile
    )
    # Generate the legal action for OTHER players (melds and ron)
    legal_action_mask_4p = jax.vmap(
        _make_legal_action_mask_after_discard, in_axes=(None, 0, 0, None)
    )(
        state, state._hand, jnp.arange(4), tile
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
        state._next_deck_ix < state._last_deck_ix
    )  # If the next drawn tile is not left, the game is ended
    state = jax.lax.cond(
        no_meld_player | (is_abortive_draw_normal & no_ron_player),
        lambda: state.replace(  # type:ignore
            current_player=jnp.int8((c_p + 1) % 4),
            _last_player=jnp.int8(c_p),
            _target=jnp.int8(-1),
            _draw_next=TRUE,
            _is_abortive_draw_normal=is_abortive_draw_normal,
        ),
        lambda: state.replace(  # type:ignore
            current_player=jnp.int8(next_meld_player),
            _last_player=jnp.int8(c_p),
            _target=jnp.int8(tile),
            _legal_action_mask_4p=legal_action_mask_4p.at[
                next_meld_player, Action.PASS
            ].set(
                TRUE
            ),  # Add the pass action to the legal action
            _draw_next=FALSE,
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

    Args:
        state: State: Current state of the game
        hand: Array: Hand of the player
        c_p: Array: Current player
        tile: Array: Discarded tile

    Returns:
        Array: legal_action_mask (NUM_ACTIONS,)
    """
    haitei = state._is_haitei
    riichi = state._riichi[c_p]
    discarder = state.current_player
    src = (discarder - c_p) % 4
    cannot_meld = riichi | haitei
    cannot_kan = (
        state._n_kan.sum() >= 4
    )  # If the number of kan is 4 or more, cannot play OPEN_KAN
    chi_mask = (
        _mask_for_chi(hand, tile) & ~cannot_meld & (src == 3)
    )  # Cannot play CHI from the player who is not the upper player
    pm_mask = _mask_for_pon_open_kan(hand, tile, cannot_kan) & ~cannot_meld
    can_ron = state._can_win[c_p, tile]
    has_yaku = state._has_yaku[
        c_p, 0
    ]  # Reference the information about the discarded tile and the next drawn tile
    is_furiten = state._furiten_by_discard[c_p] | state._furiten_by_pass[c_p]
    ron_ok = ((has_yaku | haitei) & can_ron) & ~is_furiten
    # Combine the 1D mask and expand it to 4×NUM_ACTION
    mask = chi_mask | pm_mask
    mask = mask.at[Action.RON].set(ron_ok)
    return mask


def _mask_for_chi(hand: Array, tile: Array) -> Array:
    """
    - Check if the player can play CHI

    Args:
        hand: Array: Hand of the player to be checked
        tile: Array: Discarded tile

    Returns:
        Array: legal_action_mask (NUM_ACTIONS,)
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
    - Check if the player can play PON or OPEN_KAN

    Args:
        hand: Array: Hand of the player to be checked
        tile: Array: Discarded tile
        cannot_kan: Array: Whether the player cannot play OPEN_KAN

    Returns:
        Array: legal_action_mask (NUM_ACTIONS,)
    """
    pon_result = Hand.can_pon(hand, tile)
    open_kan_result = Hand.can_open_kan(hand, tile) & ~cannot_kan
    legal_action_mask = ZERO_MASK_1D.at[Action.PON].set(pon_result)
    legal_action_mask = legal_action_mask.at[Action.OPEN_KAN].set(open_kan_result)
    return legal_action_mask


def _next_ron_player(legal_action_mask_4p: Array, discarded_player: Array) -> Array:
    """
    - Check if the player can play RON
    - If multiple players can play RON, prioritize the player who is closest to the discarded player
    Example:
        discarded_player = 1, legal_action_mask_4p = [True, False, True, False]
        return 2

    Args:
        legal_action_mask_4p: Array: Legal action mask for each player
        discarded_player: Array: Discarded player

    Returns:
        Array: next_ron_player: Index of the next player to play RON
        can_any_ron: Whether any player can play RON
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
    Set the next player from the legal action.
    - Set the next player from the legal action (RON > OPEN_KAN > PON > CHI)
    - Prioritize the player who is the closest to the discarded player if multiple players can play RON
    - Used in _discard()

    Args:
        legal_action_mask_4p: Array: Legal action mask for each player
        discarded_player: Array: Discarded player

    Returns:
        Array: next_meld_player: Index of the next player to play MELD
        can_any: Whether any player can play MELD
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

    Args:
        state: State: Current state of the game
        meld: Array: Meld to be appended
        player: Array: Player who played the meld

    Returns:
        State: State after appending the meld
    """
    melds = state._melds.at[(player, state._n_meld[player])].set(meld)
    n_meld = state._n_meld.at[player].add(1)
    return state.replace(_melds=melds, _n_meld=n_meld)  # type:ignore


def _accept_riichi(state: State) -> State:
    """
    Accept the RIICHI
    - Set the RIICHI flag
    - Subtract the score of the player who accepted the RIICHI
    - Provide rewards
    - Update the kyotaku
    - Set the Ippatsu flag
    - Check if the player has Double Riichi

    Args:
        state: State: Current state of the game

    Returns:
        State: State after accepting the RIICHI
    """
    l_p = state._last_player
    already_riichi = state._riichi[l_p]  # Whether the player has already RIICHI
    has_l_p_riichi_declared = jnp.logical_and(
        jnp.logical_not(already_riichi), state._riichi_declared
    )
    _score = state._score.at[l_p].add(
        has_l_p_riichi_declared * -10
    )  # Subtract the score of the player who accepted the RIICHI
    rewards = (
        jnp.zeros(4, dtype=jnp.float32).at[l_p].set(has_l_p_riichi_declared * -10)
    )  # Rewards for the player who accepted the RIICHI
    _kyotaku = state._kyotaku + jnp.int8(has_l_p_riichi_declared)
    riichi = state._riichi.at[l_p].set(has_l_p_riichi_declared)
    is_ippatsu = jnp.where(has_l_p_riichi_declared, TRUE, state._ippatsu[l_p])

    is_double_riichi = _is_first_turn(state._next_deck_ix) & (
        state._n_meld.sum() == 0
    )  # If the player has no meld, the player has Double Riichi
    is_double_riichi = jnp.where(
        has_l_p_riichi_declared, is_double_riichi, state._double_riichi[l_p]
    )
    state = jax.lax.cond(
        already_riichi,
        lambda: state,
        lambda: state.replace(
            _riichi=riichi,
            _riichi_declared=FALSE,
            _score=jnp.int32(_score),
            rewards=rewards,
            _kyotaku=_kyotaku,
            _double_riichi=state._double_riichi.at[l_p].set(is_double_riichi),
            _ippatsu=state._ippatsu.at[l_p].set(
                is_ippatsu
            ),  # Enable Ippatsu for the player who accepted the RIICHI
        ),
    )
    return state


def _is_waiting_tile(can_ron: Array, tile: int) -> bool:
    """
    Check if the tile is a waiting tile

    Args:
        can_ron: Array: Whether the player can play RON
        tile: Array: Tile to be checked

    Returns:
        Array: Whether the tile is a waiting tile
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

    Args:
        state: State: Current state of the game

    Returns:
        State: State after processing when a KAN is Accepted
    """
    c_p = state.current_player
    n_kan = state._n_kan.sum()  # The number of kan
    rinshan_tile = state._deck[
        10 + n_kan
    ]  # Reference the deck in _state.py TODO: Is it correct?

    # Process the KAN dora
    n_kan_doras = state._n_kan_doras  # The number of kan dora before updating
    next_kan_dora = state._deck[
        9 - 2 * (n_kan_doras + 1)
    ]  # Reference the deck in _state.py
    next_kan_ura = state._deck[
        8 - 2 * (n_kan_doras + 1)
    ]  # Reference the deck in _state.py
    state = state.replace(
        _ippatsu=jnp.zeros(4, dtype=jnp.bool_),  # Disable Ippatsu
        _can_after_kan=TRUE,
        _n_kan=state._n_kan + 1,
        _kan_declared=FALSE,
        _n_kan_doras=state._n_kan_doras + 1,
        _dora_indicators=state._dora_indicators.at[state._n_kan_doras + 1].set(
            next_kan_dora
        ),  # Reveal the KAN dora
        _ura_dora_indicators=state._ura_dora_indicators.at[state._n_kan_doras + 1].set(
            next_kan_ura
        ),  # Reveal the KAN dora
        _last_deck_ix=state._last_deck_ix
        + 1,  # Update the last deck index after drawing the rinshan tile
    )

    hand = state._hand.at[c_p].set(Hand.add(state._hand[c_p], rinshan_tile))
    can_ron = jax.vmap(Hand.can_ron, in_axes=(None, 0))(
        state._hand[c_p], TILE_RANGE
    )  # (34,) Update the legal action for the player who drew the tile
    state = state.replace(
        _can_win=state._can_win.at[c_p].set(can_ron),
    )
    is_riichi = state._riichi[c_p]
    legal_action_mask_c_p = jax.lax.cond(
        is_riichi,
        lambda: _make_legal_action_mask_after_draw_w_riichi(
            state, hand, c_p, rinshan_tile
        ),
        lambda: _make_legal_action_mask_after_draw(state, hand, c_p, rinshan_tile),
    )
    legal_action_mask_4p = state._legal_action_mask_4p.at[c_p, :].set(
        legal_action_mask_c_p
    )  # Update the legal action for the player who drew the tile
    return state.replace(  # type:ignore
        _last_draw=rinshan_tile,
        _hand=hand,
        _legal_action_mask_4p=legal_action_mask_4p,
        _has_yaku=state._has_yaku.at[c_p, 0].set(state._has_yaku[c_p, 1]),
        _fan=state._fan.at[c_p, 0].set(state._fan[c_p, 1]),
        _fu=state._fu.at[c_p, 0].set(state._fu[c_p, 1]),
    )


def _kan(state: State, action):
    """
    Process when a KAN is Declared
    - Process the KAN
    - Calculate YAKU for the Robbing KAN and the rinshan tile
    - Apply KAN action
    - Disable Ippatsu

    Args:
        state: State: Current state of the game
        action: Array: Action to be processed

    Returns:
        State: State after processing when a KAN is Declared
    """
    c_p = state.current_player
    tile = action - Tile.NUM_TILE_TYPE
    prevalent_wind = state._round % 4
    rinshan_tile = state._deck[
        jnp.int32(10 + state._n_kan.sum())
    ]  # Reference the deck in _state.py
    # Apply KAN action to hand, meld, river
    is_open_kan = action == Action.OPEN_KAN
    pon = state._pon[(c_p, tile)]
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
    state = state.replace(
        _has_yaku=has_yaku,
        _fan=fan,
        _fu=fu,
    )
    # Check if the player can win by RON
    is_furiten = state._furiten_by_discard | state._furiten_by_pass  # (4,)
    legal_action_mask_4p = state._legal_action_mask_4p.at[:, Action.RON].set(
        state._can_win[:, tile] & ~is_furiten
    )
    legal_action_mask_4p = legal_action_mask_4p.at[c_p, Action.RON].set(
        FALSE
    )  # Disable the legal action for the player who declared the KAN
    state = state.replace(
        _legal_action_mask_4p=legal_action_mask_4p,
    )
    next_ron_player, can_any_ron = _next_ron_player(legal_action_mask_4p, c_p)
    return jax.lax.cond(
        is_added_kan & can_any_ron,
        lambda: state.replace(  # type:ignore
            _target=jnp.int8(tile),
            _last_player=c_p,
            current_player=jnp.int8(next_ron_player),
            _legal_action_mask_4p=state._legal_action_mask_4p.at[
                next_ron_player, Action.PASS
            ].set(
                TRUE
            ),  # Robbing KAN player can PASS
            _kan_declared=TRUE,  # KAN is declared
            _draw_next=FALSE,
        ),
        lambda: state.replace(  # type:ignore
            _target=jnp.int8(-1),
            _kan_declared=TRUE,  # KAN is declared
            _draw_next=FALSE,
        ),
    )


def _selfkan(state: State, action, is_added_kan):
    """
    Apply SelfKan
    - Branch between ADDED_KAN and CLOSED_KAN
    - Draw the rinshan tile
    - Set the legal action after drawing the rinshan tile

    Args:
        state: State: Current state of the game
        action: Array: Action to be processed
        is_added_kan: Array: Whether the player can play ADDED_KAN

    Returns:
        State: State after processing when a KAN is Declared
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

    Args:
        state: State: Current state of the game
        target: Array: Target tile for CLOSED_KAN

    Returns:
        State: State after processing when a CLOSED_KAN is Declared
    """
    c_p = state.current_player
    meld = Meld.init(target + Tile.NUM_TILE_TYPE, target, src=0)
    state = _append_meld(state, meld, c_p)
    hand = state._hand.at[c_p].set(Hand.closed_kan(state._hand[c_p], target))
    return state.replace(  # type:ignore
        _hand=hand,
    )


def _added_kan(state: State, target):
    """
    Apply ADDED_KAN
    - Generate a meld from the target
    - Update the hand and meld

    Args:
        state: State: Current state of the game
        target: Array: Target tile for ADDED_KAN

    Returns:
        State: State after processing when a ADDED_KAN is Declared
    """
    c_p = state.current_player
    pon = state._pon[(c_p, target)]
    pon_src = pon >> 2
    pon_idx = pon & 0b11
    melds = state._melds.at[(c_p, pon_idx)].set(
        Meld.init(target + Tile.NUM_TILE_TYPE, target, pon_src)
    )
    hand = state._hand.at[c_p].set(Hand.added_kan(state._hand[c_p], target))
    # Since the ADDED_KAN consumes the pon, set it to 0
    pon = state._pon.at[(c_p, target)].set(jnp.int8(0))
    return state.replace(  # type:ignore
        _melds=melds,
        _hand=hand,
        _pon=pon,
    )


def _open_kan(state: State):
    """
    Apply OPEN_KAN
    - Generate a meld from the target
    - Update the hand and meld

    Args:
        state: State: Current state of the game
        target: Array: Target tile for OPEN_KAN

    Returns:
        State: State after processing when a OPEN_KAN is Declared
    """
    c_p = state.current_player
    l_p = state._last_player
    state = _accept_riichi(state)
    src = (l_p - c_p) % 4
    meld = Meld.init(Action.OPEN_KAN, state._target, src)
    state = _append_meld(state, meld, c_p)
    hand = state._hand.at[c_p].set(Hand.open_kan(state._hand[c_p], state._target))
    is_hand_concealed = state._is_hand_concealed.at[c_p].set(FALSE)
    # Add the meld to the river
    river = River.add_meld(
        state._river, Action.OPEN_KAN, l_p, state._n_river[l_p] - 1, src
    )
    return state.replace(  # type:ignore
        _hand=hand,
        _target=jnp.int8(-1),
        _is_hand_concealed=is_hand_concealed,
        _river=river,
    )


def _pon(state: State, action):
    """
    Apply PON
    - Generate a meld from the target
    - Update the hand and meld

    Args:
        state: State: Current state of the game
        action: Array: Action to be processed

    Returns:
        State: State after processing when a PON is Declared
    """
    c_p = state.current_player
    l_p = state._last_player
    tar = state._target
    state = _accept_riichi(state)
    src = (l_p - c_p) % 4
    meld = Meld.init(Action.PON, tar, src)
    state = _append_meld(state, meld, c_p)
    pon_hand = Hand.pon(state._hand[c_p], tar)
    hand = state._hand.at[c_p].set(pon_hand)
    is_hand_concealed = state._is_hand_concealed.at[c_p].set(FALSE)
    # Add the pon information for the ADDED_KAN
    pon = state._pon.at[(c_p, tar)].set(jnp.int8(src << 2 | state._n_meld[c_p] - 1))
    # Add the meld to the river
    river = River.add_meld(state._river, Action.PON, l_p, state._n_river[l_p] - 1, src)
    legal_action_mask_4p = (
        ZERO_MASK_2D.at[c_p, : Tile.NUM_TILE_TYPE]
        .set((hand[c_p] > 0).astype(jnp.bool_))
        .at[c_p, tar]
        .set(FALSE)  # The target tile is prohibited
        .at[c_p, Action.PASS]
        .set(FALSE)
    )  # Update the legal action for the player who declared the PON
    return state.replace(  # type:ignore
        _target=jnp.int8(-1),
        _is_hand_concealed=is_hand_concealed,
        _pon=pon,
        _hand=hand,
        _legal_action_mask_4p=legal_action_mask_4p,
        _river=river,
        _ippatsu=jnp.zeros(4, dtype=jnp.bool_),  # Disable Ippatsu
        _draw_next=FALSE,
    )


def _chi(state: State, action):
    """
    Apply CHI
    - Generate a meld from the target
    - Update the hand and meld

    Args:
        state: State: Current state of the game
        action: Array: Action to be processed

    Returns:
        State: State after processing when a CHI is Declared
    """
    c_p = state.current_player
    tar_p = state._last_player  # Absolute position
    tar = state._target
    state = _accept_riichi(state)
    meld = Meld.init(action, tar, src=jnp.int32(3))
    state = _append_meld(state, meld, c_p)
    chi_hand = Hand.chi(state._hand[c_p], tar, action)
    hand = state._hand.at[c_p].set(chi_hand)
    is_hand_concealed = state._is_hand_concealed.at[c_p].set(FALSE)
    legal_action_mask_4p = (
        _make_legal_action_mask_after_chi(state, hand, c_p, tar, action)
        .at[c_p, Action.PASS]
        .set(FALSE)
    )
    # Add the meld to the river
    river = River.add_meld(
        state._river, action, tar_p, state._n_river[tar_p] - 1, jnp.int32(3)
    )
    return state.replace(  # type:ignore
        _target=jnp.int8(-1),
        _is_hand_concealed=is_hand_concealed,
        _hand=hand,
        _legal_action_mask_4p=legal_action_mask_4p,
        _river=river,
        _ippatsu=jnp.zeros(4, dtype=jnp.bool_),  # Disable Ippatsu
        _draw_next=FALSE,
    )


def _make_legal_action_mask_after_chi(
    state: State, hand: Array, c_p: Array, target: Array, action: Array
) -> Array:
    """
    Generate legal action for CHI
    - Prohibit eating changes
    - If the prohibited tile is 5, also prohibit red tiles

    Args:
        state: State: Current state of the game
        hand: Array: Hand of the player
        c_p: Array: Current player
        target: Array: Target tile for CHI
        action: Array: Action to be processed

    Returns:
        Array: Legal action for CHI
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

    Args:
        state: State: Current state of the game

    Returns:
        State: State after processing when a PASS is Declared
    """
    c_p = state.current_player
    # If the player who declared the KAN passes, set the next player from the legal action
    can_robbing_kan = state._kan_declared
    is_ron_player = jnp.bool_(state._legal_action_mask_4p[c_p, Action.RON])
    legal_action_mask_4p = state._legal_action_mask_4p.at[c_p, :].set(FALSE)
    # Set the next player from the legal action
    next_meld_player, can_any = _next_meld_player(
        legal_action_mask_4p, state._last_player
    )  # Reference the current wind for the next meld player
    no_meld_player = jnp.logical_not(can_any)
    # Check if the game is ended (abortive_draw_normal (流局))
    is_abortive_draw_normal = (
        state._next_deck_ix < state._last_deck_ix
    )  # If the next deck index is less than the last deck index, the game is ended (abortive_draw_normal (流局))
    return jax.lax.cond(
        no_meld_player,
        lambda: state.replace(  # type:ignore
            current_player=jnp.where(
                can_robbing_kan,
                jnp.int8(state._last_player),
                jnp.int8((state._last_player + 1) % 4),
            ),  # If the player who declared the KAN passes, set the last player
            _target=jnp.int8(-1),
            _furiten_by_pass=state._furiten_by_pass.at[c_p].set(
                is_ron_player & ~can_robbing_kan
            ),  # If the player who RON passes, set the furiten
            _draw_next=TRUE
            & ~can_robbing_kan,  # If no next player for robbing KAN, draw the rinshan tile
            _is_abortive_draw_normal=is_abortive_draw_normal,
            _legal_action_mask_4p=legal_action_mask_4p,
        ),
        lambda: state.replace(  # type:ignore
            current_player=jnp.int8(next_meld_player),
            _target=jnp.int8(state._target),  # Do not change the target
            _legal_action_mask_4p=legal_action_mask_4p.at[
                next_meld_player, Action.PASS
            ].set(
                TRUE
            ),  # Add the pass action to the legal action
            _furiten_by_pass=state._furiten_by_pass.at[c_p].set(
                is_ron_player & ~can_robbing_kan
            ),  # If the player who RON passes, set the furiten
        ),
    )


def _riichi(state: State):
    """
    Apply RIICHI
    - Set the RIICHI declared flag
    - Generate the legal action for the player after RIICHI

    Args:
        state: State: Current state of the game

    Returns:
        State: State after processing when a RIICHI is Declared
    """
    c_p = state.current_player
    legal_action_mask_for_discard = jax.vmap(Hand.is_tenpai)(
        jax.vmap(Hand.sub, in_axes=(None, 0))(state._hand[c_p], TILE_RANGE)
    )  # Only tiles that can maintain tenpai can be discarded
    legal_action_mask_for_discard = jnp.logical_and(
        legal_action_mask_for_discard, state._hand[c_p]
    )
    # Set tile actions
    player_mask = ZERO_MASK_1D.at[: Tile.NUM_TILE_TYPE].set(
        legal_action_mask_for_discard
    )
    # The last drawn tile must have at least 2 tiles to be discarded
    player_mask = player_mask.at[state._last_draw].set(
        (state._hand[c_p, state._last_draw] >= 2)
        & legal_action_mask_for_discard[state._last_draw]
    )
    # Set TSUUMOGIRI action
    player_mask = player_mask.at[Action.TSUMOGIRI].set(
        legal_action_mask_for_discard[state._last_draw]
    )
    return state.replace(  # type:ignore
        _legal_action_mask_4p=state._legal_action_mask_4p.at[c_p].set(player_mask),
        _riichi_declared=TRUE,
        _draw_next=FALSE,
    )


def _ron(state: State) -> State:
    """
    Apply RON
    - Calculate the score of the winner (consider only the remainder when divided by 100)
    - Clear the Kyotaku

    Args:
        state: State: Current state of the game

    Returns:
        State: State after processing when a RON is Declared
    """
    c_p = state.current_player
    is_ippatsu = state._ippatsu[c_p] & state._riichi[c_p]  # Ippatsu (一発)
    is_double_riichi = state._double_riichi[c_p]  # Double Riichi (ダブル立直)
    can_robbing_kan = state._kan_declared  # RobbingKan (槍槓)
    is_houtei = state._is_haitei  # Bottom of the River (河底摸月)
    is_yakuman = state._fu[c_p, 0] == 0  # When Yakuman, fu is 0
    basic_score = Yaku.score(
        state._fan[c_p, 0]
        + (
            jnp.int32(is_ippatsu)
            + jnp.int32(is_double_riichi)
            + jnp.int32(can_robbing_kan)
            + jnp.int32(is_houtei)
        )
        * (
            1 - is_yakuman
        ),  # When Yakuman, do not add ippatsu, double riichi, robbing_kan, houtei
        state._fu[c_p, 0],
    )
    score = jnp.where(state._dealer == c_p, basic_score * 6, basic_score * 4)
    # Round up the score to the nearest multiple of 100
    score = jnp.ceil(score / 100)
    honba = state._honba * 3  # 1 Honba is 300 points (per player)
    # Build reward array more efficiently
    reward = jnp.zeros(4, dtype=jnp.float32)
    reward = reward.at[c_p].set(score + honba)
    reward = reward.at[state._last_player].set(-score - honba)
    # The Kyotaku is already paid when the RIICHI is declared, so we only need to add the Kyotaku to the winner
    kyotaku_bonus = 10 * (state._kyotaku - state._riichi[c_p])
    reward = reward.at[c_p].add(kyotaku_bonus)
    score = state._score + jnp.float32(reward)
    return state.replace(  # type:ignore
        _terminated_round=TRUE,
        _score=jnp.int32(score),  # Update the score
        rewards=jnp.float32(reward),
        _kyotaku=jnp.int8(0),  # Clear the Kyotaku
        _has_won=state._has_won.at[c_p].set(TRUE),
        _legal_action_mask_4p=ZERO_MASK_2D.at[:, Action.DUMMY].set(
            TRUE
        ),  # Set the dummy action
        _draw_next=FALSE,
    )


def _tsumo(state: State) -> State:
    """
    Apply TSUUMO
    - Calculate the score of the winner
    - Update the score
    - terminated=true
    - Clear the Kyotaku
    - Check if the game is ended (abortive_draw_normal (流局))
    Args:
        state: State: Current state of the game

    Returns:
        State: State after processing when a TSUUMO is Declared
    """
    c_p = state.current_player
    _dealer = state._dealer
    can_after_kan = (
        state._can_after_kan
    )  # Check if the game is ended (AfterKan (嶺上開花))
    is_ippatsu = state._ippatsu[c_p] & state._riichi[c_p]  # Ippatsu (一発)
    is_double_riichi = state._double_riichi[c_p]  # Double Riichi (ダブル立直)
    is_haitei = state._is_haitei  # Bottom of the River (河底摸月)
    # Check for Blessing of the Heaven/Earth (天和/地和)
    is_pure_first_turn = _is_first_turn(state._next_deck_ix) & (
        state._n_meld.sum() == 0
    )
    is_hand_yakuman = state._fu[c_p, 0] == 0
    is_yakuman = is_hand_yakuman | is_pure_first_turn
    fan = state._fan[c_p, 0]
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
    fu = jnp.where(is_yakuman, 0, state._fu[c_p, 0] + (2 * can_after_kan))
    basic_score = Yaku.score(
        jnp.int32(fan),  # Calculate the score when the tile is discarded
        jnp.int32(fu),  # When AfterKan, add 2 fu
    )
    honba = state._honba * 1  # 1 Honba is 100 points (per player)
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
    kyotaku_bonus = 10 * (state._kyotaku - state._riichi[c_p])
    reward = reward.at[c_p].add(kyotaku_bonus)
    score = state._score + reward
    reward = reward
    return state.replace(  # type:ignore
        _terminated_round=TRUE,
        rewards=jnp.float32(reward),
        _score=jnp.int32(score),
        _kyotaku=jnp.int8(0),  # Clear the Kyotaku
        _has_won=state._has_won.at[c_p].set(TRUE),
        _legal_action_mask_4p=ZERO_MASK_2D.at[:, Action.DUMMY].set(
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
    Args:
        state: State: Current state of the game

    Returns:
        State: State after processing when a ABORTIVE_DRAW_NORMAL (流局) happens
    """
    # Normal Draw (流局)
    tenpai = state._can_win.any(axis=-1)  # (4,)
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
    return state.replace(  # type:ignore
        rewards=rewards.astype(jnp.float32),
        _score=jnp.int32(state._score + jnp.float32(rewards)),  # Update the score
        _legal_action_mask_4p=ZERO_MASK_2D.at[:, Action.DUMMY].set(
            TRUE
        ),  # Set the dummy action
        _terminated_round=TRUE,
        _draw_next=FALSE,
    )


def _next_round(state: State) -> State:
    """
    Move to the next round
    - Process the next round
    - Move the dealer
    - Process the round
    - If the round is ended, calculate the rank points. The Kyotaku is the top (same point wind order)
    - DUMMY sharing: 3 times of rotation (cp to +1 mod 4) after the 4th time, the result is determined
    Args:
        state: State: Current state of the game

    Returns:
        State: State after processing when the next round is determined
    """
    dc = state._dummy_count  # int8

    # ---- During the DUMMY sharing phase, only rotate ----
    def _rotate_once(s: State):
        is_tempai = s._can_win.any(axis=-1)  # (4,)
        dealer = s._dealer
        hora = s._has_won  # (4,)
        will_dealer_continue = jnp.logical_or(
            is_tempai[dealer], hora[dealer]
        )  # Check if the dealer continues (win or temporary win)
        order = jnp.argsort(
            -s._score
        )  # Example: score=[10,20,30,40] -> order=[3,2,1,0]
        rank_points = (
            jnp.zeros_like(s._score).at[order].set(s._order_points)
        )  # Assign the rank points
        score = s._score + rank_points  # Add the rank points
        top = jnp.argmax(score)
        final_score = score.at[top].add(
            10 * s._kyotaku
        )  # Add the Kyotaku (10 points per Riichi stick × number of Honba) to the top

        # Check if the round is ended
        is_final_round = s._round == s._round_limit
        has_dealer_end = jnp.logical_not(will_dealer_continue)
        is_dealer_top = jnp.arange(4)[top] == s._dealer
        has_minus_score = (s._score < 0).any()
        _is_game_end = (
            (is_final_round & has_dealer_end)
            | (has_minus_score)
            | (is_final_round & is_dealer_top)
        )
        return s.replace(
            current_player=jnp.int8((s.current_player + 1) % 4),
            terminated=(s._dummy_count == 0) & _is_game_end,
            _dummy_count=jnp.int8(
                s._dummy_count + jnp.int8(1)
            ),  # Strictly set the dtype
            _score=jnp.where(
                (s._dummy_count == 0) & _is_game_end, final_score, s._score
            ),  # Reflect the final score in the first DUMMY sharing phase
        )

    # ---- After the DUMMY sharing phase (=3), determine the next round or end the game ----
    def _finalize_and_start_next(s: State):
        hora = s._has_won  # (4,)
        is_tempai = s._can_win.any(axis=-1)  # (4,)
        dealer = s._dealer
        is_eight_consecutive_deals = (
            s._honba >= 8
        )  # 8 consecutive deals means the honba moves to the next round
        has_other_than_dealer_won = hora.any() & ~hora[dealer]
        will_dealer_continue = jnp.logical_or(
            is_tempai[dealer] & ~has_other_than_dealer_won, hora[dealer]
        )
        will_dealer_continue = will_dealer_continue & ~is_eight_consecutive_deals
        next_round = jnp.where(will_dealer_continue, s._round, s._round + 1)
        has_winner = hora.any()
        next_honba = jnp.where(
            ~has_winner | will_dealer_continue, s._honba + 1, 0
        )  # if there is no winner or the dealer continues, the honba is incremented
        next_dealer = jnp.where(
            will_dealer_continue, dealer, (dealer + 1) % 4
        )  # if the dealer continues, the dealer is kept, otherwise the dealer is incremented

        rng, subkey = jax.random.split(s._rng_key)

        # ★ Initialize only at this timing
        base_next = State(  # type: ignore
            _rng_key=subkey,
            current_player=next_dealer,  # Start from the dealer
            _dealer=next_dealer,
            _seat_wind=_calc_wind(next_dealer),
            _round=next_round,
            _honba=next_honba,
            _kyotaku=s._kyotaku,
            _score=s._score,
        )
        next_round_state = _init_for_next_round(subkey, base_next)

        terminated_state = State(
            _score=s._score,
            terminated=TRUE,
        )

        # Check if the round is ended
        top = jnp.argmax(s._score)
        is_final_round = s._round == s._round_limit
        has_dealer_end = jnp.logical_not(
            will_dealer_continue
        )  # Check if the dealer continues (win or temporary win)
        is_dealer_top = (
            jnp.arange(4)[top] == s._dealer
        )  # Check if the dealer is the top
        has_minus_score = (
            s._score < 0
        ).any()  # Check if there is a player with negative score
        _is_game_end = (
            (is_final_round & has_dealer_end)
            | (has_minus_score)
            | (is_final_round & is_dealer_top)
        )
        # Determine the next round or end the game
        return jax.lax.cond(
            _is_game_end,
            lambda: terminated_state.replace(
                current_player=jnp.int8(terminated_state._dealer),
                _dummy_count=jnp.int8(0),
            ),
            lambda: next_round_state.replace(
                current_player=jnp.int8(next_round_state._dealer),
                _dummy_count=jnp.int8(0),
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
    Args:
        state: State: Current state of the game

    Returns:
        jnp.ndarray: Array of length 34, where the number of tiles is stored in the index of the dora tile
            - dora counts: (34,) The number of dora tiles
            - ura dora counts: (34,) The number of ura dora tiles
    """

    def update_dora_counts(dora_counts: Array, dora_indicator: Array) -> Array:
        is_dora_valid = dora_indicator != -1
        return dora_counts.at[DORA_ARRAY[dora_indicator]].add(is_dora_valid)

    # Count occurrences of each dora type more efficiently using bincount-like approach
    # For normal dora
    dora_counts = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.int8)
    dora_counts = jax.vmap(update_dora_counts, in_axes=(None, 0))(
        dora_counts, state._dora_indicators
    ).sum(axis=0)
    # For ura dora
    ura_dora_counts = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.int8)
    ura_dora_counts = jax.vmap(update_dora_counts, in_axes=(None, 0))(
        ura_dora_counts, state._ura_dora_indicators
    ).sum(axis=0)
    return jnp.array([dora_counts, ura_dora_counts])


@jax.jit
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
    - action history: (2, 200) action history [player, action], player index is relative to the current player in [0, 3], action is in [0, 78] (-1 means no action)
    - shanten count: (1,) The number of shanten (0-6)
    - furiten: (1,) Whether the player is in furiten [True/False]
    - scores: (4,) The scores of the players ordered from the current player's perspective (c_p, right, across, left)
    - round: (1,) The round number (0-12)
    - honba: (1,) The honba number
    - kyotaku: (1,) The kyotaku number
    - round wind: (1,) The round wind [0-3]
    - seat wind: (1,) The seat wind [0-3]
    - dora indicators: (4,) The dora indicators [0-33], -1 means no dora
    Args:
        state: State: Current state of the game

    Returns:
        Dict: Dictionary of the observed features
    """
    c_p = state.current_player
    c_p_based_order = (jnp.arange(4) + c_p) % 4
    # hand features
    hand_c_p_34 = state._hand[c_p]
    hand_c_p_14 = hand_counts_to_idx(hand_c_p_34)
    # action histories
    player_history = state._action_history[0, :].astype(jnp.int32)  # (200,)
    valid_history = player_history >= 0  # default value is -1, so we need to mask it
    relative_player_history = jnp.mod(player_history - jnp.int32(c_p), 4).astype(
        state._action_history.dtype
    )  # translate the player index to the relative index. e.g. if the original player index is 1, and the current player index is 3, then the relative player index is 2.
    relative_player_history = jnp.where(
        valid_history, relative_player_history, state._action_history[0, :]
    )
    action_history = state._action_history.at[0, :].set(relative_player_history)

    # game features
    shanten_c_p = state._shanten_c_p
    furiten = state._furiten_by_discard[c_p] | state._furiten_by_pass[c_p]
    scores = state._score[c_p_based_order]
    _round = state._round
    honba = state._honba
    kyotaku = state._kyotaku
    prevalent_wind = state._seat_wind[c_p]
    seat_wind = state._init_wind[c_p]
    dora_indicators = state._dora_indicators[:4]  # (4,)
    return {
        "hand": hand_c_p_14,
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
    hand_c = state._hand[c_p]  # (34,)
    # Number of tiles: >=1, >=2, >=3, ==4
    thresholds = jnp.array([1, 2, 3, 4], dtype=jnp.int32)[:, None]  # (4,1)
    hand_bins = (hand_c[None, :] >= thresholds).astype(jnp.float32)  # (4,34)
    # Waiting tile (can ron)
    wait_feat = state._can_win[c_p][None, :].astype(jnp.float32)  # (1,34)
    # Furiten (broadcast scalar)
    is_furiten = state._furiten_by_discard[c_p] | state._furiten_by_pass[c_p]
    furiten_feat = jnp.full((1, 34), is_furiten, dtype=jnp.float32)
    # Shanten (0..6 to 0..1 normalized)
    shanten_val = state._shanten_c_p  # Shanten count is pre-calculated
    shanten_feat = jnp.full((1, 34), (shanten_val / 6.0), dtype=jnp.float32)
    hand_block = jnp.concatenate(
        [hand_bins, wait_feat, furiten_feat, shanten_feat], axis=0
    )  # (7,34)

    # ---------- Game Features (15 ch) ----------
    # The highest score is 100000, the lowest score is -250, so normalize to 0-1 by adding 250
    score_norm = ((state._score + 250) / 1250.0).astype(jnp.float32)[
        c_p_based_order, None
    ]  # (4,1)
    score_feat = jnp.repeat(score_norm, 34, axis=1)  # (4,34)
    # Rank (higher score is higher rank): 0..3 to 0..1
    # rank_idx = 0: highest rank, 3: lowest rank
    order = jnp.argsort(-state._score)
    inv = jnp.argsort(order)
    rank_idx = inv[c_p].astype(jnp.float32)
    rank_feat = jnp.full((1, 34), rank_idx / 3.0, dtype=jnp.float32)
    # Round/Honba/Kyotaku normalization
    round_feat = jnp.full(
        (1, 34),
        (
            state._round.astype(jnp.float32)
            / jnp.maximum(1.0, state._round_limit.astype(jnp.float32))
        ),
        dtype=jnp.float32,
    )
    honba_feat = jnp.full(
        (1, 34), (state._honba.astype(jnp.float32) / 10.0), dtype=jnp.float32
    )
    kyotaku_feat = jnp.full(
        (1, 34), (state._kyotaku.astype(jnp.float32) / 10.0), dtype=jnp.float32
    )
    # Wind (seat wind and prevalent wind) 0..3 to 0..1
    seat_wind = state._seat_wind[c_p].astype(jnp.float32) / 3.0
    prevalent_wind = (state._round % 4).astype(jnp.float32) / 3.0
    wind_feat = jnp.stack(
        [
            jnp.full((34,), seat_wind, dtype=jnp.float32),
            jnp.full((34,), prevalent_wind, dtype=jnp.float32),
        ],
        axis=0,
    )  # (2,34)
    # Remaining tsumo (approximately): (_next_deck_ix - _last_deck_ix + 1) / 70
    tiles_rem = (
        state._next_deck_ix.astype(jnp.int32)
        - state._last_deck_ix.astype(jnp.int32)
        + 1
    )
    tiles_rem = jnp.clip(tiles_rem, 0, 70).astype(jnp.float32) / 70.0
    tiles_rem_feat = jnp.full((1, 34), tiles_rem, dtype=jnp.float32)

    # Dora display (maximum 4 tiles to 4ch one-hot)
    # -1 is ignored (zero)
    dora_inds = state._dora_indicators[:4].astype(jnp.int32)  # (4,)
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
    river = state._river[c_p_based_order]
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
    melds = state._melds[c_p_based_order].reshape(-1)  # (16,)
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
        state._riichi[c_p_based_order].astype(jnp.float32)[:, None], 34, axis=1
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
    lam = state._legal_action_mask_4p[c_p]  # (NUM_ACTION,)
    # discard: 0..33 any
    feat_discard = lam[: Tile.NUM_TILE_TYPE].any()
    # selfkan: 34..67 any (from here, closed_kan/added_kan is estimated)
    selfkan_mask = lam[Tile.NUM_TILE_TYPE : Action.TSUMOGIRI]
    has_selfkan = selfkan_mask.any()
    # added_kan estimation: If there is a tile in the selfkan candidate that has _pon information, then "added kan"
    # Extract one tile idx weighted by the sum (multiple can be set, use the maximum idx)
    idx34 = jnp.arange(Tile.NUM_TILE_TYPE)
    cand_idx = jnp.argmax(selfkan_mask * idx34)  # 0..33 any
    has_added_kan = has_selfkan & (state._pon[(c_p, cand_idx)] > 0)
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
