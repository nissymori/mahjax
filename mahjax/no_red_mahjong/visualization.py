from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import jax.numpy as jnp

from mahjax.red_mahjong.state import State as RedState
from mahjax.red_mahjong.state import default_state as default_red_state
from mahjax.red_mahjong.visualization import (
    render_round_svg as render_red_round_svg,
)
from mahjax.red_mahjong.visualization import (
    render_svg_animation as render_red_svg_animation,
)

from .state import State

Language = Literal["ja", "en"]


_SHARED_PLAYER_FIELDS = (
    "hand",
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
)

_SHARED_ROUND_FIELDS = (
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
)


def to_red_visual_state(state: State) -> RedState:
    rs = default_red_state()
    hand34 = state.players.hand.astype(jnp.int8)
    hand37 = jnp.zeros((4, 37), dtype=jnp.int8).at[:, :34].set(hand34)
    # Pad no_red's narrower legal_action_mask to red's wider action space.
    legal_4p = state.players.legal_action_mask
    pad_2d = rs.players.legal_action_mask.shape[1] - legal_4p.shape[1]
    if pad_2d > 0:
        legal_4p = jnp.pad(legal_4p, ((0, 0), (0, pad_2d)), constant_values=False)
    legal_1p = state.legal_action_mask
    pad_1d = rs.legal_action_mask.shape[0] - legal_1p.shape[0]
    if pad_1d > 0:
        legal_1p = jnp.pad(legal_1p, (0, pad_1d), constant_values=False)

    player_updates = {f: getattr(state.players, f) for f in _SHARED_PLAYER_FIELDS}
    player_updates["hand_with_red"] = hand37
    player_updates["legal_action_mask"] = legal_4p
    round_updates = {f: getattr(state.round_state, f) for f in _SHARED_ROUND_FIELDS}
    return rs.replace(
        current_player=state.current_player,
        legal_action_mask=legal_1p,
        players=rs.players.replace(**player_updates),
        round_state=rs.round_state.replace(**round_updates),
        rewards=state.rewards,
        terminated=state.terminated,
        truncated=state.truncated,
    )


def render_round_svg(
    state: State,
    show_all_hands: bool = True,
    visible_player: int = 0,
    language: Language = "ja",
) -> str:
    return render_red_round_svg(
        to_red_visual_state(state),
        show_all_hands=show_all_hands,
        visible_player=visible_player,
        language=language,
    )


def save_svg(
    state: State,
    filename: str | Path,
    show_all_hands: bool = True,
    language: Language = "ja",
) -> None:
    Path(filename).write_text(
        render_round_svg(
            state,
            show_all_hands=show_all_hands,
            language=language,
        ),
        encoding="utf-8",
    )


def render_svg_animation(
    states: Sequence[State],
    frame_duration_seconds: float = 0.2,
    show_all_hands: bool = True,
    language: Language = "ja",
) -> str:
    return render_red_svg_animation(
        [to_red_visual_state(state) for state in states],
        frame_duration_seconds=frame_duration_seconds,
        show_all_hands=show_all_hands,
        language=language,
    )


def save_svg_animation(
    states: Sequence[State],
    filename: str | Path,
    frame_duration_seconds: float = 0.2,
    show_all_hands: bool = True,
    language: Language = "ja",
) -> None:
    Path(filename).write_text(
        render_svg_animation(
            states,
            frame_duration_seconds=frame_duration_seconds,
            show_all_hands=show_all_hands,
            language=language,
        ),
        encoding="utf-8",
    )
