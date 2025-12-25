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

import mahjax.core as core
from mahjax._src.struct import dataclass
from mahjax._src.types import Array, PRNGKey
from mahjax.no_red_mahjong.meld import EMPTY_MELD
from mahjax.no_red_mahjong.tile import EMPTY_RIVER, Tile

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
FIRST_DRAW_IDX = (
    135 - 13 * 4
)  # The index of the first drawn tile after drawing 13*4 tiles from the deck
DORA_ARRAY = jnp.array(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        0,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        9,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        19,
        28,
        29,
        30,
        27,
        32,
        33,
        31,
    ]
)


@dataclass
class State(core.State):
    current_player: Array = jnp.int8(0)  # The player to take an action
    rewards: Array = jnp.zeros(4, dtype=jnp.float32)  # The rewards (hundreds place)
    legal_action_mask: Array = jnp.zeros(79, dtype=jnp.bool_)
    terminated: Array = FALSE
    truncated: Array = FALSE
    _rng_key: PRNGKey = jax.random.PRNGKey(0)
    _step_count: Array = jnp.int32(0)
    # action history (player, action), 70 (discard) + 70 (every pass) + 16 (four players meld 4 times) + 16 (discard for the melds) + 4 (dummy actions) + 20 (buffer)
    _action_history: Array = jnp.full(
        (2, 200), -1, dtype=jnp.int8
    )  # the default value is -1 which means no-action is performed.
    # --- Mahjong specific ---
    # --- Related to the hand ---
    _hand: Array = jnp.zeros(
        (4, Tile.NUM_TILE_TYPE), dtype=jnp.int8
    )  # The hand of the players (player, tile)
    _legal_action_mask_4p: Array = jnp.zeros(
        (4, 79), dtype=jnp.bool_
    )  # Legal action mask for each player
    _can_win: Array = jnp.zeros(
        (4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_
    )  # Whether the players can win with the tile (player, tile)
    _has_yaku: Array = jnp.zeros(
        (4, 2), dtype=jnp.bool_
    )  # Whether the players have yaku (player, discarded tile/next drawn tile) Updated at the beginning of the game and when discarding
    _fan: Array = jnp.zeros(
        (4, 2), dtype=jnp.int32
    )  # The number of fans (player, discarded tile/next drawn tile)
    _fu: Array = jnp.zeros(
        (4, 2), dtype=jnp.int32
    )  # The number of fu (player, discarded tile/next drawn tile)
    _shanten_c_p: Array = jnp.int8(0)  # The number of shanten for the current player
    # --- Related to the game ---
    _round: Array = jnp.int8(0)
    _round_limit: Array = jnp.int8(7)  # The maximum number of rounds (East-South game)
    _terminated_round: Array = FALSE  # Whether the round is terminated
    _honba: Array = jnp.int8(0)  # The number of honba
    _kyotaku: Array = jnp.int8(0)  # The number of kyotaku
    _init_wind: Array = jnp.array(
        [0, 1, 2, 3], dtype=jnp.int8
    )  # The wind determined by the initial seat order
    _seat_wind: Array = jnp.array(
        [0, 1, 2, 3], dtype=jnp.int8
    )  # The wind determined by the relative seat order from the dealer
    _dealer: Array = jnp.int8(0)  # The dealer of the round is (state._dealer+round) % 4
    _order_points: Array = jnp.array(
        [30, 10, -10, -30], dtype=jnp.int32
    )  # The points for each rank
    _score: Array = jnp.full(4, 250, dtype=jnp.int32)  # The score (hundreds place)
    _deck: Array = jnp.zeros(136, dtype=jnp.int8)  # The deck of tiles
    _next_deck_ix: Array = jnp.int32(FIRST_DRAW_IDX)  # The index of the next drawn tile
    _last_deck_ix: Array = jnp.int8(14)  # The index of the last drawn tile
    _draw_next: Array = FALSE  # Whether the next player should draw a tile
    _last_draw: Array = jnp.int8(
        0
    )  # The last drawn tile for the player with 3n+2 tiles. If no last drawn tile is set, it is 0.
    _last_player: Array = jnp.int8(0)  # The last player who discarded a tile.
    # The river of the players (player, tile)
    # int8
    # 0b  0     0    0 0 0 0 0 0
    #    gray|riichi|    tile(0-33)
    _river: Array = jnp.full((4, 24), EMPTY_RIVER, dtype=jnp.uint16)
    _n_river: Array = jnp.zeros(4, dtype=jnp.int8)  # The number of river (player)
    _dora_indicators: Array = (
        jnp.ones(5, dtype=jnp.int8) * -1
    )  # The dora indicators (dora)
    _ura_dora_indicators: Array = (
        jnp.ones(5, dtype=jnp.int8) * -1
    )  # The ura dora indicators (ura dora)
    _has_won: Array = jnp.zeros(
        4, dtype=jnp.bool_
    )  # Whether each player has won (player)
    _is_abortive_draw_normal: Array = (
        FALSE  # Whether the game is a abortive_draw_normal
    )
    _dummy_count: Array = jnp.int8(0)  # The number of dummy actions
    _is_haitei: Array = FALSE  # Whether the last drawn tile is the haitei tile
    _furiten_by_discard: Array = jnp.zeros(
        4, dtype=jnp.bool_
    )  # Friten flag by having discarded a waiting tile
    _furiten_by_pass: Array = jnp.zeros(
        4, dtype=jnp.bool_
    )  # Friten flag by passing to ron a waiting tile
    # --- Related to the meld ---
    _melds: Array = jnp.full(
        (4, 4), EMPTY_MELD, dtype=jnp.uint16
    )  # melds[i][j]: player i's j-th meld (j=1,2,3,4). If no meld is set, it is EMPTY_MELD.
    _target: Array = jnp.int8(
        -1
    )  # Target tile for melding and winning. If no target tile is set, it is -1.
    _is_hand_concealed: Array = jnp.ones(
        4, dtype=jnp.bool_
    )  # Whether each player is in menzen (player)
    _pon: Array = jnp.zeros(
        (4, 34), dtype=jnp.int32
    )  # pon[i][j]: player i has j-th pon of tile j. If no pon is set, it is 0. TODO: optimize this
    _n_meld: Array = jnp.zeros(4, dtype=jnp.int8)  # The number of melds (player)
    _n_kan: Array = jnp.zeros(4, dtype=jnp.int8)  # The number of kan (player)
    _n_kan_doras: Array = jnp.int8(0)  # The number of kan doras (kan dora)
    _kan_declared: Array = FALSE  # Whether the kan is declared (For RobbingKan)
    _can_after_kan: Array = FALSE  # Whether the kan is declared (For AfterKan)
    _can_robbing_kan: Array = (
        FALSE  # For RobbingKan (槍槓). It is True after the kan is declared and until the next player draws a tile.
    )
    # --- Related to Riichi ---
    _riichi_declared: Array = (
        FALSE  # Whether the riichi is declared (For Riichi). It is True after the riichi is declared and until the player discards a tile.
    )
    _riichi: Array = jnp.zeros(
        4, dtype=jnp.bool_
    )  # Whether each player has riichi (player)
    _double_riichi: Array = jnp.zeros(
        4, dtype=jnp.bool_
    )  # Whether each player has double riichi (player)
    _ippatsu: Array = jnp.zeros(4, dtype=jnp.bool_)  # Ippatsu flag (player)

    @property
    def env_id(self) -> core.EnvId:
        # TODO add envid
        return "mahjong"  # type:ignore

    @property
    def json(self):
        import json

        class NumpyEncoder(json.JSONEncoder):
            """Special json encoder for numpy types"""

            def default(self, obj):
                if isinstance(obj, jnp.integer):
                    return int(obj)
                elif isinstance(obj, jnp.floating):
                    return float(obj)
                elif isinstance(obj, jnp.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        return json.dumps(self.__dict__, cls=NumpyEncoder)

    @classmethod
    def from_json(cls, path):
        import json

        def decode_state(data: dict):
            return cls(  # type:ignore
                current_player=jnp.int8(data["current_player"]),
                rewards=jnp.array(data["rewards"], dtype=jnp.float32),
                terminated=jnp.bool_(data["terminated"]),
                truncated=jnp.bool_(data["truncated"]),
                legal_action_mask=jnp.array(data["legal_action_mask"], dtype=jnp.bool_),
                _rng_key=jnp.array(data["_rng_key"]),
                _step_count=jnp.int32(data["_step_count"]),
                _round=jnp.int8(data["_round"]),
                _honba=jnp.int8(data["_honba"]),
                _dealer=jnp.int8(data["_dealer"]),
                _score=jnp.array(data["_score"], dtype=jnp.float32),
                _deck=jnp.array(data["_deck"], dtype=jnp.int8),
                _next_deck_ix=jnp.int8(data["_next_deck_ix"]),
                _hand=jnp.array(data["_hand"], dtype=jnp.int8),
                _river=jnp.array(data["_river"], dtype=jnp.uint8),
                _n_river=jnp.array(data["_n_river"], dtype=jnp.int8),
                _dora_indicators=jnp.array(data["_dora_indicators"], dtype=jnp.int8),
                _ura_dora_indicators=jnp.array(
                    data["_ura_dora_indicators"], dtype=jnp.int8
                ),
                _n_kan=jnp.int8(data["_n_kan"]),
                _target=jnp.int8(data["_target"]),
                _last_draw=jnp.int8(data["_last_draw"]),
                _last_player=jnp.int8(data["_last_player"]),
                _furo_check_num=jnp.uint8(data["_furo_check_num"]),
                _riichi_declared=jnp.bool_(data["_riichi_declared"]),
                _riichi=jnp.array(data["_riichi"], dtype=jnp.bool_),
                _n_meld=jnp.array(data["_n_meld"], dtype=jnp.int8),
                _melds=jnp.array(data["_melds"], dtype=jnp.int32),
                _is_hand_concealed=jnp.array(
                    data["_is_hand_concealed"], dtype=jnp.bool_
                ),
                _pon=jnp.array(data["_pon"], dtype=jnp.int32),
            )

        with open(path, "r") as f:
            state = json.load(f, object_hook=decode_state)

        return state

    def __eq__(self, b):
        a = self
        return (
            a.current_player == b.current_player
            and (a.rewards == b.rewards).all()
            and a.terminated == b.terminated
            and a.truncated == b.truncated
            and (a.legal_action_mask == b.legal_action_mask).all()
            and (a._rng_key == b._rng_key).all()
            and a._step_count == b._step_count
            and a._round == b._round
            and a._honba == b._honba
            and a._dealer == b._dealer
            and (a._score == b._score).all()
            and (a._deck == b._deck).all()
            and a._next_deck_ix == b._next_deck_ix
            and (a._hand == b._hand).all()
            and (a._river == b._river).all()
            and (a._doras == b._doras).all()
            and a._n_kan == b._n_kan
            and a._target == b._target
            and a._last_draw == b._last_draw
            and a._last_player == b._last_player
            and a._furo_check_num == b._furo_check_num
            and a._riichi_declared == b._riichi_declared
            and (a._riichi == b._riichi).all()
            and (a._n_meld == b._n_meld).all()
            and (a._melds == b._melds).all()
            and (a._is_hand_concealed == b._is_hand_concealed).all()
            and (a._pon == b._pon).all()
        )
