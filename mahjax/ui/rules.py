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

"""Action-space and state-decoding facade for the two mahjong envs.

The UI never re-derives mahjong rules -- it only mirrors ``legal_action_mask``.
What it does need is a way to say "which action id is Riichi here" and "what
tiles does this meld word stand for", and both differ between the envs:

===================  ==================  ==================
                     ``red_mahjong``     ``no_red_mahjong``
===================  ==================  ==================
tile ids             0-36 (34-36 red)    0-33
discard actions      0-36                0-33
kan actions          37-70               34-67
action space         87                  79
red pon/chi variants yes                 no
nine terminals       action 85           absent
abortive draws       yes                 absent
===================  ==================  ==================

Everything above this module speaks a single vocabulary: tile ids are always
red-aware (0-33 types, 34/35/36 red fives), which for ``no_red_mahjong`` simply
means 0-33 is all that ever appears.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

NUM_TILE_TYPES = 34
NUM_PLAYERS = 4
RED_FIVE_TYPES = (4, 13, 22)  # 5m, 5p, 5s
RED_TILE_BASE = NUM_TILE_TYPES  # 34/35/36 are the red 5m/5p/5s

#: ``dora_indicator tile type -> dora tile type`` (the next tile in the cycle).
DORA_NEXT = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 0]
    + [10, 11, 12, 13, 14, 15, 16, 17, 9]
    + [19, 20, 21, 22, 23, 24, 25, 26, 18]
    + [28, 29, 30, 27]
    + [32, 33, 31],
    dtype=np.int32,
)


def tile_type(tile: int) -> int:
    """Red-aware tile id -> 0-33 tile type."""
    tile = int(tile)
    if tile >= RED_TILE_BASE:
        return RED_FIVE_TYPES[tile - RED_TILE_BASE]
    return tile


def is_red(tile: int) -> bool:
    return int(tile) >= RED_TILE_BASE


def red_of(tile_type_: int) -> int:
    """0-33 tile type -> its red-five id, or the type itself when it has none."""
    tile_type_ = int(tile_type_)
    if tile_type_ in RED_FIVE_TYPES:
        return RED_TILE_BASE + RED_FIVE_TYPES.index(tile_type_)
    return tile_type_


def dora_of(indicator: int) -> int:
    """Dora indicator (red-aware id) -> the 0-33 tile type it points at."""
    return int(DORA_NEXT[tile_type(indicator)])


class Rules:
    """Per-env view of the action space plus state decoding helpers.

    Construct one per env id and keep it for the life of the match; it holds no
    mutable state.
    """

    def __init__(self, env_id: str) -> None:
        if env_id not in ("red_mahjong", "no_red_mahjong"):
            raise ValueError(f"Unknown env_id: {env_id}")
        self.env_id = env_id
        self.has_red = env_id == "red_mahjong"

        if self.has_red:
            from mahjax.red_mahjong.action import Action
            from mahjax.red_mahjong.meld import Meld
            from mahjax.red_mahjong.tile import River
        else:
            from mahjax.no_red_mahjong.action import Action  # type: ignore[assignment]
            from mahjax.no_red_mahjong.meld import Meld  # type: ignore[assignment]
            from mahjax.no_red_mahjong.tile import River  # type: ignore[assignment]

        self._Action = Action
        self._Meld = Meld
        self._River = River

        # Discard action ids are exactly the tile ids, so this doubles as the
        # number of distinct tiles the env can name.
        self.n_tiles: int = 37 if self.has_red else 34
        self.kan_base: int = self.n_tiles

        self.TSUMOGIRI: int = int(Action.TSUMOGIRI)
        self.RIICHI: int = int(Action.RIICHI)
        self.TSUMO: int = int(Action.TSUMO)
        self.RON: int = int(Action.RON)
        self.PON: int = int(Action.PON)
        self.OPEN_KAN: int = int(Action.OPEN_KAN)
        self.PASS: int = int(Action.PASS)
        self.DUMMY: int = int(Action.DUMMY)
        self.NUM_ACTION: int = int(Action.NUM_ACTION)
        self.PON_RED: Optional[int] = int(Action.PON_RED) if self.has_red else None
        self.KYUUSHU: Optional[int] = int(Action.KYUUSHU) if self.has_red else None

        # (action id, index of the called tile inside the run, uses a red five)
        if self.has_red:
            self.chi_actions = [
                (int(Action.CHI_L), 0, False),
                (int(Action.CHI_L_RED), 0, True),
                (int(Action.CHI_M), 1, False),
                (int(Action.CHI_M_RED), 1, True),
                (int(Action.CHI_R), 2, False),
                (int(Action.CHI_R_RED), 2, True),
            ]
        else:
            self.chi_actions = [
                (int(Action.CHI_L), 0, False),
                (int(Action.CHI_M), 1, False),
                (int(Action.CHI_R), 2, False),
            ]
        self._chi_offset = {a: off for a, off, _ in self.chi_actions}
        self._chi_is_red = {a: red for a, _, red in self.chi_actions}
        self.pon_actions = [self.PON] + ([self.PON_RED] if self.PON_RED is not None else [])

    # ------------------------------------------------------------------ env

    def make_env(self, round_mode: str) -> Any:
        """The env this UI always wants: stop at round end, share via DUMMY."""
        import mahjax

        return mahjax.make(
            self.env_id, round_mode=round_mode, next_round_style="dummy_share"
        )

    # --------------------------------------------------------------- actions

    def is_discard(self, action: int) -> bool:
        return 0 <= int(action) < self.n_tiles

    def is_kan(self, action: int) -> bool:
        return self.kan_base <= int(action) < self.TSUMOGIRI

    def kan_action(self, tile_type_: int) -> int:
        return self.kan_base + int(tile_type_)

    def kan_tile_type(self, action: int) -> int:
        return int(action) - self.kan_base

    def is_chi(self, action: int) -> bool:
        return int(action) in self._chi_offset

    def is_pon(self, action: int) -> bool:
        return int(action) in self.pon_actions

    def chi_tiles(self, action: int, target: int) -> List[int]:
        """The three tiles a chi action forms, with red fives where they apply."""
        offset = self._chi_offset[int(action)]
        base = tile_type(target)
        tiles = [base - offset, base - offset + 1, base - offset + 2]
        if self._chi_is_red.get(int(action)):
            tiles = [red_of(t) if t in RED_FIVE_TYPES else t for t in tiles]
        tiles[offset] = int(target)  # the called tile keeps its own identity
        return tiles

    def pon_tiles(self, action: int, target: int) -> List[int]:
        """The three tiles a pon action forms; index 2 is the called tile."""
        base = tile_type(target)
        tiles = [base, base, int(target)]
        if action == self.PON_RED and not is_red(target):
            tiles[0] = red_of(base)
        return tiles

    def open_kan_tiles(self, target: int) -> List[int]:
        base = tile_type(target)
        tiles = [base, base, base, int(target)]
        if base in RED_FIVE_TYPES and not is_red(target):
            tiles[0] = red_of(base)
        return tiles

    def closed_kan_tiles(self, tile_type_: int) -> List[int]:
        """A concealed/added kan consumes all four copies, red one included."""
        base = int(tile_type_)
        tiles = [base] * 4
        if base in RED_FIVE_TYPES:
            tiles[0] = red_of(base)
        return tiles

    def legal_actions(self, state: Any) -> List[int]:
        mask = np.asarray(state.legal_action_mask, dtype=bool)
        return [int(a) for a in np.flatnonzero(mask)]

    # ----------------------------------------------------------------- state

    def hand(self, state: Any, seat: int) -> np.ndarray:
        """Per-tile-id counts for one seat (length :attr:`n_tiles`)."""
        arr = state.players.hand_with_red if self.has_red else state.players.hand
        return np.asarray(arr[int(seat)], dtype=np.int32)

    def hand_tiles(self, state: Any, seat: int) -> List[int]:
        counts = self.hand(state, seat)
        tiles: List[int] = []
        for tile, n in enumerate(counts):
            tiles.extend([tile] * int(n))
        tiles.sort(key=lambda t: (tile_type(t), is_red(t)))
        return tiles

    def hand_count(self, state: Any, seat: int) -> int:
        return int(self.hand(state, seat).sum())

    def melds(self, state: Any, seat: int) -> List[Dict[str, Any]]:
        """Decode one seat's melds into ``{kind, tiles, called, from}`` dicts.

        ``called`` indexes into ``tiles`` (``None`` for a concealed kan) and
        ``from`` is the absolute seat the tile came from.
        """
        Meld = self._Meld
        seat = int(seat)
        n = int(state.players.meld_counts[seat])
        out: List[Dict[str, Any]] = []
        for i in range(n):
            word = state.players.melds[seat, i]
            if bool(Meld.is_empty(word)):
                continue
            action = int(Meld.action(word))
            target_type = int(Meld.target(word))
            src = int(Meld.src(word))
            if self.has_red and bool(Meld.is_target_red(word)):
                target = red_of(target_type)
            else:
                target = target_type
            from_seat = None if src == 0 else (seat + src) % NUM_PLAYERS

            if bool(Meld.is_chi(word)):
                kind = "chi"
                tiles = self.chi_tiles(action, target)
                called = self._chi_offset[action]
            elif bool(Meld.is_pon(word)):
                kind = "pon"
                tiles = self.pon_tiles(action, target)
                called = 2
            elif bool(Meld.is_closed_kan(word)):
                kind = "kan_closed"
                tiles = self.closed_kan_tiles(target_type)
                called = None
                from_seat = None
            elif action == self.OPEN_KAN:
                kind = "kan_open"
                tiles = self.open_kan_tiles(target)
                called = 3
            else:  # added kan: the meld word replaced the pon it upgraded
                kind = "kan_added"
                tiles = self.closed_kan_tiles(target_type)
                if is_red(target):
                    tiles[0] = target
                called = 3
            out.append(
                {"kind": kind, "tiles": tiles, "called": called, "from": from_seat}
            )
        return out

    def river(self, state: Any, seat: int) -> List[Dict[str, Any]]:
        """Decode one seat's discard pond in discard order."""
        seat = int(seat)
        decoded = np.asarray(self._River.decode_river(state.players.river[seat]))
        n = int(state.players.discard_counts[seat])
        out: List[Dict[str, Any]] = []
        for i in range(n):
            out.append(
                {
                    "tile": int(decoded[0, i]),
                    "riichi": bool(decoded[1, i]),
                    "called": bool(decoded[2, i]),
                    "tsumogiri": bool(decoded[3, i]),
                }
            )
        return out

    def last_discard(self, state: Any) -> Optional[Dict[str, int]]:
        """The most recent discard as ``{seat, index}``, or ``None``.

        ``round_state.target`` is not usable for this: it is set only while a
        claimant is being asked, and is cleared once the discard passes.
        """
        seat = int(state.round_state.last_player)
        if seat < 0:
            return None
        count = int(state.players.discard_counts[seat])
        if count <= 0:
            return None
        return {"seat": seat, "index": count - 1}

    def riichi_state(self, state: Any, seat: int) -> str:
        seat = int(seat)
        if bool(state.players.riichi[seat]):
            return "accepted"
        if bool(state.players.riichi_declared[seat]):
            return "declared"
        return "none"

    def scores(self, state: Any) -> List[int]:
        """Points in the usual 25000-style units (the env stores hundreds)."""
        return [int(s) * 100 for s in np.asarray(state.round_state.score)]

    def rewards(self, state: Any) -> List[int]:
        return [int(round(float(r) * 100)) for r in np.asarray(state.rewards)]

    def dora_indicators(self, state: Any) -> List[int]:
        return [int(t) for t in np.asarray(state.round_state.dora_indicators) if int(t) >= 0]

    def ura_dora_indicators(self, state: Any) -> List[int]:
        return [
            int(t) for t in np.asarray(state.round_state.ura_dora_indicators) if int(t) >= 0
        ]

    def remaining_draws(self, state: Any) -> int:
        rs = state.round_state
        return max(int(rs.next_deck_ix) - int(rs.last_deck_ix) + 1, 0)

    def drawn_tile(self, state: Any, seat: int) -> Optional[int]:
        """The tile this seat just drew, if it is theirs and still in hand.

        ``last_draw`` is cleared on discard and untouched by calls, so it only
        means something for the seat on turn.
        """
        if int(state.current_player) != int(seat):
            return None
        tile = int(state.round_state.last_draw)
        if tile < 0:
            return None
        if int(self.hand(state, seat)[tile]) <= 0:
            return None
        return tile

    def is_round_over(self, state: Any) -> bool:
        """True once the round is settled and only DUMMY sharing remains.

        Checked on the mask rather than on ``terminated_round`` because a double
        ron keeps ``terminated_round`` set while the second claimant is still
        being asked.
        """
        mask = np.asarray(state.legal_action_mask, dtype=bool)
        if bool(state.terminated):
            return True
        return bool(mask[self.DUMMY]) and int(mask.sum()) == 1

    def tenpai(self, state: Any) -> List[bool]:
        return [bool(x) for x in np.asarray(state.players.can_win).any(axis=-1)]

    def has_won(self, state: Any) -> List[bool]:
        return [bool(x) for x in np.asarray(state.players.has_won)]

    def nagashi_mangan(self, state: Any) -> List[bool]:
        arr = getattr(state.players, "has_nagashi_mangan", None)
        if arr is None:
            return [False] * NUM_PLAYERS
        return [bool(x) for x in np.asarray(arr)]


_CACHE: Dict[str, Rules] = {}


def rules_for(env_id: str) -> Rules:
    """Shared :class:`Rules` instance for an env id (they are immutable)."""
    if env_id not in _CACHE:
        _CACHE[env_id] = Rules(env_id)
    return _CACHE[env_id]


__all__ = [
    "Rules",
    "rules_for",
    "tile_type",
    "is_red",
    "red_of",
    "dora_of",
    "NUM_PLAYERS",
    "NUM_TILE_TYPES",
    "RED_TILE_BASE",
    "RED_FIVE_TYPES",
]
