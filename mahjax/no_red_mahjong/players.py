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


import jax
import jax.numpy as jnp

from mahjax._src.types import Array, PRNGKey
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.hand import Hand
from mahjax.no_red_mahjong.meld import Meld
from mahjax.no_red_mahjong.shanten import Shanten
from mahjax.no_red_mahjong.state import State
from mahjax.no_red_mahjong.tile import River, Tile
from mahjax.no_red_mahjong.yaku import Yaku

PRIORITY_MASK = jnp.array(
    [
        5,
        4,
        3,
        2,
        1,
        2,
        3,
        4,
        5,
        5,
        4,
        3,
        2,
        1,
        2,
        3,
        4,
        5,
        5,
        4,
        3,
        2,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        7,
        7,
        6,
        6,
        6,
    ]
)

OUTSIDE_MASK = jnp.array(
    [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
)

YAKU_TILES = jnp.array(
    [
        27,
        31,
        32,
        33,
    ]
)

# Hyperparameters for Pon
BASIC_PON_PROB = 0.04
YAKU_PON_PROB = 0.7
YAKU_MELD_PON_PROB = 0.6
HAS_PUNG_PON_PROB = 0.05
# Hyperparameters for Chi
BASIC_CHI_PROB = 0.02
YAKU_MELD_CHI_PROB = 0.5
HAS_PUNG_CHI_PROB = 0.05
# Hyperparameters for Open Kan
open_kan_PROB = 0.05
# Hyperparameters for Riichi
RIICHI_PROB = 0.9


def has_river_tile(hand: Array, tile: Array) -> Array:
    return jnp.where(
        tile == -1,
        jnp.zeros(34, dtype=jnp.int32),
        jnp.zeros(34, dtype=jnp.int32).at[tile].set(hand[tile]),
    )


def _discard_logic(state: State, unflatten_hand: Array, flatten_hand: Array) -> Array:
    """
    - Discard the tile that minimizes the shanten number.
    - Prioritize the outside tiles when discarding.
    - When listening, discard the tile that makes the wait wider.
    - If the player is tan-yao, discard the tan-yao tile.
    - If other players are riichi, and the player has a shanten less than 2, discard the tile that the other players are waiting for.
    """
    c_p = state.current_player
    n_meld = state._n_meld[c_p]
    # Shanten
    detailed_shantens = Shanten.detailed_discard(flatten_hand)
    best_shanten_normal = jnp.min(detailed_shantens[:, 0])
    best_shanten_7_pairs = jnp.min(detailed_shantens[:, 1])
    best_shanten_13_orphan = jnp.min(detailed_shantens[:, 2])

    best_shanten = jnp.where(
        jnp.logical_or(
            best_shanten_normal < best_shanten_7_pairs, best_shanten_7_pairs <= 3
        ),
        best_shanten_normal,
        best_shanten_7_pairs,
    )
    shantens = jnp.where(
        jnp.logical_or(
            best_shanten_normal < best_shanten_7_pairs, best_shanten_7_pairs <= 3
        ),
        detailed_shantens[:, 0],
        detailed_shantens[:, 1],
    )
    best_shanten = jnp.where(
        best_shanten < best_shanten_13_orphan + 2, best_shanten, best_shanten_13_orphan
    )
    shantens = jnp.where(
        best_shanten < best_shanten_13_orphan + 2, shantens, detailed_shantens[:, 2]
    )
    best_shanten = jnp.where(n_meld > 0, best_shanten_normal, best_shanten)
    shantens = jnp.where(n_meld > 0, detailed_shantens[:, 0], shantens)
    best_shanten_mask = shantens == best_shanten
    priority_mask = (
        best_shanten_mask * PRIORITY_MASK * (unflatten_hand > 0)
    )  # Select the tile from the tiles the player has.
    best_shanten_action = jnp.argmax(priority_mask)
    # Waiting
    is_tempai = best_shanten == 0

    def f(i):
        i = jnp.int32(i)
        return jax.lax.cond(
            jnp.logical_or(flatten_hand[i] == 0, shantens[i] != 0),
            lambda: jnp.int32(-1),
            lambda: jax.vmap(Hand.can_ron, in_axes=(None, 0))(
                flatten_hand.at[i].set(flatten_hand[i] - 1),
                jnp.arange(Tile.NUM_TILE_TYPE),
            )
            .astype(jnp.int32)
            .sum(),
        )

    can_rons = jax.vmap(f)(jnp.arange(Tile.NUM_TILE_TYPE))  # (34,)
    best_waiting_action = jnp.argmax(can_rons)  # (,)
    action = jnp.where(is_tempai, best_waiting_action, best_shanten_action)
    # Defense
    other_riichi = jnp.asarray(state._riichi)
    other_riichi = other_riichi.at[c_p].set(False)
    is_other_riichi = other_riichi.any()  # (4,)
    riichi_player = jnp.argmax(
        other_riichi
    )  # The player who is riichi, considering only one riichi.
    riich_player_river = River.decode_tile(state._river[riichi_player])
    hand_within_river = jax.vmap(has_river_tile, in_axes=(None, 0))(
        flatten_hand, riich_player_river
    ).sum(
        axis=-1
    )  # (34,)
    no_genbutsu = hand_within_river.sum() == 0
    defense_action = jnp.argmax(hand_within_river)
    defense_action = jnp.where(
        no_genbutsu, action, defense_action
    )  # If the player has no genbutsu, discard the tile that makes the player's hand advance.
    action = jnp.where(
        is_other_riichi & best_shanten >= 2, defense_action, action
    )  # If the player has a shanten less than 2 and other players are riichi, discard the tile that the other players are waiting for.
    return action


def _is_equal(a: Array, arr: Array) -> Array:
    return (a == arr).any()


def _pon_logic(
    state: State, unflatten_hand: Array, flatten_hand: Array, rng: PRNGKey
) -> Array:
    """
    - Pon the yaku tile with 70% probability.
    - Other tiles,
        - If the yaku tile is melded, pon with 60% probability
    - If the tile is a pung, pon with 5% probability
    """
    # Whether the target is a yaku tile
    target_tile = state._target
    is_global_yaku = (target_tile == YAKU_TILES).any()
    is_wind_yaku = target_tile == 27 + state._seat_wind[state.current_player]
    is_yaku = is_global_yaku | is_wind_yaku

    # Whether the yaku tile is melded
    meld = state._melds[state.current_player]
    meld_tile = Meld.target(meld)
    is_yaku_meld = (jax.vmap(_is_equal, in_axes=(0, None))(meld_tile, YAKU_TILES)).any()
    is_wind_meld = (meld_tile == 27 + state._seat_wind[state.current_player]).any()
    is_yaku_meld = is_yaku_meld | is_wind_meld

    # Whether the tile is a pung
    has_pung = unflatten_hand[target_tile] >= 3

    # Probability of pon
    basic_prob = (flatten_hand * ~OUTSIDE_MASK).sum() * BASIC_PON_PROB  # 最大52%
    prob = jnp.where(is_yaku, YAKU_PON_PROB, basic_prob)
    prob = jnp.where(is_yaku_meld, YAKU_MELD_PON_PROB, prob)
    prob = jnp.where(has_pung, HAS_PUNG_PON_PROB, prob)
    do_pon = jax.random.bernoulli(rng, prob)
    return jnp.where(do_pon, Action.PON, Action.PASS)


def _chi_logic(
    state: State, unflatten_hand: Array, flatten_hand: Array, rng: PRNGKey
) -> Array:
    """
    - If the yaku tile is melded, chi with 50% probability
    - Otherwise, chi with the probability proportional to the number of tan-yao tiles and the wind
    - If the tile is a pung, chi with 5% probability
    """
    # Whether the yaku tile is melded
    target_tile = state._target

    meld = state._melds[state.current_player]
    meld_tile = Meld.target(meld)
    is_yaku_meld = (jax.vmap(_is_equal, in_axes=(0, None))(meld_tile, YAKU_TILES)).any()
    is_wind_meld = (meld_tile == 27 + state._seat_wind[state.current_player]).any()
    is_yaku_meld = is_yaku_meld | is_wind_meld

    # Whether the tile is a pung
    has_pung = unflatten_hand[target_tile] >= 3

    # Probability of chi
    basic_prob = (flatten_hand * ~OUTSIDE_MASK).sum() * BASIC_CHI_PROB  # 最大26%
    prob = jnp.where(is_yaku_meld, YAKU_MELD_CHI_PROB, basic_prob)
    prob = jnp.where(has_pung, HAS_PUNG_CHI_PROB, prob)
    do_chi = jax.random.bernoulli(rng, prob)
    chi_mask = state.legal_action_mask * jnp.zeros(
        Action.NUM_ACTION, dtype=jnp.int32
    ).at[Action.CHI_L : Action.CHI_R + 1].set(1)
    chi_logits = jnp.log(chi_mask.astype(jnp.float32))
    chi_action = jax.random.categorical(rng, logits=chi_logits)
    return jnp.where(do_chi, chi_action, Action.PASS)


def _open_kan_logic(
    state: State, unflatten_hand: Array, flatten_hand: Array, rng: PRNGKey
) -> Array:
    """
    - Open kan with 5% probability
    """
    # Probability of open kan
    do_open_kan = jax.random.bernoulli(rng, 0.05)
    return jnp.where(do_open_kan, Action.OPEN_KAN, Action.PASS)


def _riichi_logic(state: State, current_action: Array, rng: PRNGKey) -> Array:
    """
    - If the player has a yaku, riichi with 80% probability
    """
    do_riichi = jax.random.bernoulli(rng, RIICHI_PROB)
    return jnp.where(do_riichi, Action.RIICHI, current_action)


def rule_based_player(state: State, rng: PRNGKey) -> Array:
    unflatten_hand = state._hand[state.current_player]
    melds = state._melds[state.current_player]
    n_meld = state._n_meld[state.current_player]
    legal_action_mask = state.legal_action_mask
    flatten_hand = Yaku.flatten(unflatten_hand, melds, n_meld)
    # discard logic
    discard_action = _discard_logic(state, unflatten_hand, flatten_hand)
    discard_action = jnp.where(
        discard_action == state._last_draw, Action.TSUMOGIRI, discard_action
    )

    is_legal = legal_action_mask[discard_action]
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    random_action = jax.random.categorical(rng, logits=logits)
    action = jnp.where(is_legal, discard_action, random_action)

    # Melded tiles
    is_chi = legal_action_mask[Action.CHI_L : Action.CHI_R + 1].any()
    is_pon = legal_action_mask[Action.PON]
    is_open_kan = legal_action_mask[Action.OPEN_KAN]
    rng, chi_rng, pon_rng, open_kan_rng = jax.random.split(rng, 4)
    action = jnp.where(
        is_chi, _chi_logic(state, unflatten_hand, flatten_hand, chi_rng), action
    )
    action = jnp.where(
        is_pon, _pon_logic(state, unflatten_hand, flatten_hand, pon_rng), action
    )
    action = jnp.where(
        is_open_kan,
        _open_kan_logic(state, unflatten_hand, flatten_hand, open_kan_rng),
        action,
    )

    # Riichi
    is_riichi = legal_action_mask[Action.RIICHI]
    rng, riichi_rng = jax.random.split(rng)
    action = jnp.where(is_riichi, _riichi_logic(state, action, riichi_rng), action)

    # If the player can riichi, riichi
    can_riichi = legal_action_mask[Action.RIICHI]
    action = jnp.where(can_riichi, Action.RIICHI, action)
    # If the player can tsumo, tsumo
    can_tsumo = legal_action_mask[Action.TSUMO]
    action = jnp.where(can_tsumo, Action.TSUMO, action)
    # If the player can ron, ron
    can_ron = legal_action_mask[Action.RON]
    action = jnp.where(can_ron, Action.RON, action)
    return action


def random_player(state: State, rng: PRNGKey) -> Array:
    legal_action_mask = state.legal_action_mask
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits=logits)
