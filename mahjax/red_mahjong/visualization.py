from __future__ import annotations

import base64
import importlib.resources as resources
from functools import lru_cache
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp

from .action import Action
from .constants import (
    LEGAL_ACTION_SIZE,
    MAX_DISCARDS_PER_PLAYER,
    MAX_MELDS_PER_PLAYER,
    NUM_PLAYERS,
    NUM_TILE_TYPES_WITH_RED,
)
from .env import RedMahjong
from .meld import Meld
from .state import EnvState
from .tile import River, Tile

_W = 32.0
_H = 48.0
_BOARD = 700.0
_CENTER = _BOARD / 2.0
_CENTER_BOX_SIZE = 6.0 * _W
_HAND_X = 120.0
_HAND_Y = 640.0
_RIVER_X0 = _CENTER - 3.0 * _W
_RIVER_Y0 = 450.0
_DORA_SCALE = 0.70
_DORA_GAP = 4.0
_DORA_Y = _CENTER - 8.0
_REMAINING_TILES_Y = _CENTER + 47.0
_SCORE_Y = _CENTER + _CENTER_BOX_SIZE / 2.0 - 10.0
_RIICHI_BAR_Y = _CENTER + _CENTER_BOX_SIZE / 2.0 - 4.0
_RIICHI_DOT_Y = _CENTER + _CENTER_BOX_SIZE / 2.0 + 1.0
Language = Literal["ja", "en"]
_PLAYER_WIND_LABELS = {
    "ja": ("東", "南", "西", "北"),
    "en": ("E", "S", "W", "N"),
}
_ROUND_WIND_LABELS = {
    "ja": ("東", "南"),
    "en": ("East", "South"),
}
_JA_FONT_FAMILY = "'Noto Sans CJK JP', 'Noto Sans CJK', sans-serif"


def _normalize_language(language: Language | str) -> Language:
    if language not in ("ja", "en"):
        raise ValueError(f"Unsupported language: {language}")
    return language


def _is_red_tile(tile: int) -> bool:
    return tile >= Tile.NUM_TILE_TYPE


def _tile_sort_key(tile: int) -> tuple[int, int]:
    return (int(Tile.to_tile_type(tile)), int(_is_red_tile(tile)))


def _tile_asset_name(tile: int) -> str:
    if tile < 0:
        return "back.svg"
    tile_type = int(Tile.to_tile_type(tile))
    is_red = _is_red_tile(tile)
    if 0 <= tile_type <= 8:
        n = tile_type + 1
        return f"{n}mr.svg" if (tile_type == 4 and is_red) else f"{n}m.svg"
    if 9 <= tile_type <= 17:
        n = tile_type - 8
        return f"{n}pr.svg" if (tile_type == 13 and is_red) else f"{n}p.svg"
    if 18 <= tile_type <= 26:
        n = tile_type - 17
        return f"{n}sr.svg" if (tile_type == 22 and is_red) else f"{n}s.svg"
    if tile_type == 27:
        return "east.svg"
    if tile_type == 28:
        return "south.svg"
    if tile_type == 29:
        return "west.svg"
    if tile_type == 30:
        return "north.svg"
    if tile_type == 31:
        return "white.svg"
    if tile_type == 32:
        return "gd.svg"
    if tile_type == 33:
        return "rd.svg"
    return "back.svg"


@lru_cache(maxsize=512)
def _tile_data_uri(language: Language, name: str) -> str:
    data = resources.files("mahjax._src.assets.tiles").joinpath(language, name).read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode("ascii")


def _image_tag(
    tile: int,
    x: float,
    y: float,
    language: Language,
    rotate: bool = False,
    opacity: float = 1.0,
    rotate_anchor: str = "center",
    rotate_offset: tuple[float, float] | None = None,
) -> str:
    href = _tile_data_uri(language, _tile_asset_name(tile))
    transform = ""
    if rotate:
        if rotate_anchor == "topleft":
            cx = x
            cy = y
        elif rotate_anchor == "slot":
            x += (_H - _W) / 2.0
            y += (_H - _W) / 2.0
            cx = x + _W / 2.0
            cy = y + _H / 2.0
        else:
            cx = x + _W / 2.0
            cy = y + _H / 2.0
        if rotate_offset is not None:
            dx, dy = rotate_offset
            transform = f' transform="translate({dx:.2f} {dy:.2f}) rotate(-90 {cx:.2f} {cy:.2f})"'
        else:
            transform = f' transform="rotate(-90 {cx:.2f} {cy:.2f})"'
    return (
        f'<image href="{href}" x="{x:.2f}" y="{y:.2f}" width="{_W:.2f}" height="{_H:.2f}"'
        f'{transform} opacity="{opacity:.3f}" />'
    )


def _image_tag_scaled(
    tile: int,
    x: float,
    y: float,
    scale: float,
    language: Language,
    opacity: float = 1.0,
) -> str:
    href = _tile_data_uri(language, _tile_asset_name(tile))
    w = _W * scale
    h = _H * scale
    return (
        f'<image href="{href}" x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}"'
        f' opacity="{opacity:.3f}" />'
    )


def _river_tsumogiri_overlay(x: float, y: float, rotate: bool = False) -> str:
    if rotate:
        x += (_H - _W) / 2.0
        y += (_H - _W) / 2.0
        cx = x + _W / 2.0
        cy = y + _H / 2.0
        transform = (
            f' transform="rotate(-90 {cx:.2f} {cy:.2f})"'
        )
    else:
        transform = ""
    parts = [
        f'<g class="tsumogiri-overlay"{transform}>',
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{_W:.2f}" height="{_H:.2f}" fill="#c2c2c2" opacity="0.28" />',
    ]
    line_x = x + 3.0
    while line_x < x + _W:
        parts.append(
            f'<line x1="{line_x:.2f}" y1="{y+2:.2f}" x2="{line_x:.2f}" y2="{y+_H-2:.2f}" '
            'stroke="#777" stroke-width="0.8" opacity="0.45" />'
        )
        line_x += 3.5
    parts.append("</g>")
    return "".join(parts)


def _counts_to_tiles(counts: jnp.ndarray) -> list[int]:
    tiles: list[int] = []
    for tile in range(NUM_TILE_TYPES_WITH_RED):
        tiles.extend([tile] * int(counts[tile]))
    tiles.sort(key=_tile_sort_key)
    return tiles


def _draw_tile_for_player(state: EnvState, player: int, tiles: list[int]) -> int:
    if not tiles:
        return -1
    if int(state.current_player) != player:
        return -1
    # After calls, concealed hand size is not always 14.
    # Detect draw phase by legal TSUMOGIRI availability instead.
    if not bool(state.legal_action_mask[Action.TSUMOGIRI]):
        return -1
    draw_tile = int(state.round_state.last_draw)
    if draw_tile >= 0 and draw_tile in tiles:
        return draw_tile
    # Keep natural hand sorting when draw provenance is unavailable.
    return -1


def _chi_index(action: int) -> int:
    if action in (Action.CHI_L, Action.CHI_L_RED):
        return 0
    if action in (Action.CHI_M, Action.CHI_M_RED):
        return 1
    if action in (Action.CHI_R, Action.CHI_R_RED):
        return 2
    return -1


def _meld_tile_list(meld: jnp.ndarray) -> tuple[list[int], int]:
    action = int(Meld.action(meld))
    target = int(Meld.target(meld))
    target_tile = int(Meld.target_tile(meld))
    is_target_red = bool(Meld.is_target_red(meld))
    if Meld.is_chi(meld):
        chi_index = _chi_index(action)
        start = target - chi_index
        tiles = [start, start + 1, start + 2]
        if action in (Action.CHI_L_RED, Action.CHI_M_RED, Action.CHI_R_RED):
            for i, tile in enumerate(tiles):
                if int(Tile.is_tile_type_five(tile)):
                    tiles[i] = int(Tile.to_red(tile))
        if is_target_red:
            tiles[chi_index] = target_tile
        return tiles, chi_index

    if Meld.is_pon(meld):
        tiles = [target, target, target]
        called_index = 2
        tiles[called_index] = target_tile
        if action == Action.PON_RED and int(Tile.is_tile_type_five(target)) and not is_target_red:
            for i in range(len(tiles)):
                if i != called_index:
                    tiles[i] = int(Tile.to_red(target))
                    break
        return tiles, called_index

    if Meld.is_kan(meld):
        tiles = [target, target, target, target]
        if int(Tile.is_tile_type_five(target)) and not is_target_red:
            tiles[0] = int(Tile.to_red(target))
        if action == Action.OPEN_KAN:
            tiles[-1] = target_tile
            return tiles, 3
        if is_target_red:
            tiles[0] = target_tile
        return tiles, -1

    return [], -1


def _meld_layout(meld: jnp.ndarray) -> list[tuple[int, bool]]:
    tiles, called_index = _meld_tile_list(meld)
    if not tiles:
        return []
    src = int(Meld.src(meld))
    if Meld.is_chi(meld):
        return [(tiles[called_index], True)] + [
            (tile, False) for i, tile in enumerate(tiles) if i != called_index
        ]
    if Meld.is_pon(meld) or int(Meld.action(meld)) == Action.OPEN_KAN:
        rotate_pos = 2
        if src == 3:
            rotate_pos = 0
        elif src == 2:
            rotate_pos = 1
        called_tile = tiles[called_index]
        remaining = [tile for i, tile in enumerate(tiles) if i != called_index]
        ordered = remaining[:]
        ordered.insert(min(rotate_pos, len(ordered)), called_tile)
        rot_idx = min(rotate_pos, len(ordered) - 1)
        return [(tile, i == rot_idx) for i, tile in enumerate(ordered)]
    return [(tile, False) for tile in tiles]


def _player_group(
    state: EnvState,
    player: int,
    show_all_hands: bool,
    visible_player: int,
    language: Language,
) -> str:
    parts: list[str] = []
    seat_wind = _PLAYER_WIND_LABELS[language][(player - int(state.round_state.dealer)) % 4]
    score = int(state.round_state.score[player]) * 100
    seat_wind_font = f' font-family="{_JA_FONT_FAMILY}"' if language == "ja" else ""
    parts.append(
        f'<text x="265" y="435" font-size="22" fill="#000"{seat_wind_font}>{seat_wind}</text>'
        f'<text x="{_CENTER:.1f}" y="{_SCORE_Y:.1f}" text-anchor="middle" font-size="20" fill="#000">{score}</text>'
    )
    if bool(state.players.riichi[player]):
        parts.append(
            f'<rect x="{_CENTER-50:.1f}" y="{_RIICHI_BAR_Y:.1f}" width="100" height="10" fill="#fff" stroke="#000" />'
            f'<circle cx="{_CENTER:.1f}" cy="{_RIICHI_DOT_Y:.1f}" r="3" fill="red" />'
        )

    tiles = _counts_to_tiles(state.players.hand_with_red[player])
    draw_tile = _draw_tile_for_player(state, player, tiles)
    offset = 0.0
    if not show_all_hands and player != int(visible_player):
        for _ in range(len(tiles)):
            parts.append(_image_tag(-1, _HAND_X + offset, _HAND_Y, language))
            offset += _W
    else:
        shown = tiles[:]
        if draw_tile >= 0 and draw_tile in shown:
            shown.remove(draw_tile)
        for tile in shown:
            parts.append(_image_tag(tile, _HAND_X + offset, _HAND_Y, language))
            offset += _W
        if draw_tile >= 0:
            offset += _W * 0.5
            parts.append(_image_tag(draw_tile, _HAND_X + offset, _HAND_Y, language))
            offset += _W

    offset += _W

    meld_count = int(state.players.meld_counts[player])
    for m in reversed(range(min(meld_count, MAX_MELDS_PER_PLAYER))):
        meld = state.players.melds[player, m]
        if bool(Meld.is_empty(meld)):
            continue
        for tile, rotate in _meld_layout(meld):
            if rotate:
                parts.append(
                    _image_tag(
                        tile,
                        _HAND_X + offset,
                        _HAND_Y,
                        language,
                        rotate=True,
                        rotate_anchor="slot",
                    )
                )
                offset += _H
            else:
                parts.append(_image_tag(tile, _HAND_X + offset, _HAND_Y, language))
                offset += _W
        offset += _W

    river = River.decode_river(state.players.river[player])
    x = _RIVER_X0
    y = _RIVER_Y0
    dcount = int(state.players.discard_counts[player])
    for di in range(min(dcount, MAX_DISCARDS_PER_PLAYER)):
        tile = int(river[0, di])
        riichi_decl_tile = int(river[1, di]) > 0
        called = int(river[2, di]) > 0
        tsumogiri = int(river[3, di]) > 0
        opacity = 0.5 if called else 1.0
        if riichi_decl_tile:
            parts.append(
                _image_tag(
                    tile,
                    x,
                    y,
                    language,
                    rotate=True,
                    opacity=opacity,
                    rotate_anchor="slot",
                )
            )
            if tsumogiri:
                parts.append(_river_tsumogiri_overlay(x, y, rotate=True))
        else:
            parts.append(_image_tag(tile, x, y, language, opacity=opacity))
            if tsumogiri:
                parts.append(_river_tsumogiri_overlay(x, y, rotate=False))
        # Riichi declaration tile is placed sideways and consumes one extra slot.
        x += _H if riichi_decl_tile else _W
        if di % 6 == 5:
            x = _RIVER_X0
            y += _H
    return "".join(parts)


def render_round_svg(
    state: EnvState,
    show_all_hands: bool = True,
    visible_player: int = 0,
    language: Language = "ja",
) -> str:
    language = _normalize_language(language)
    dora = [int(tile) for tile in jnp.asarray(state.round_state.dora_indicators) if int(tile) >= 0][:4]
    round_index = int(state.round_state.round)
    if language == "ja":
        round_label = f"{_ROUND_WIND_LABELS[language][round_index // 4]}{round_index % 4 + 1}局"
        if int(state.round_state.honba) > 0:
            round_label += f" {int(state.round_state.honba)}本場"
    else:
        round_label = f"{_ROUND_WIND_LABELS[language][round_index // 4]} {round_index % 4 + 1}"
        if int(state.round_state.honba) > 0:
            round_label += f" {int(state.round_state.honba)} Honba"
    remaining_tiles = max(int(state.round_state.next_deck_ix) - int(state.round_state.last_deck_ix) + 1, 0)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_BOARD:.0f}" height="{_BOARD:.0f}" viewBox="0 0 {_BOARD:.0f} {_BOARD:.0f}">',
        f'<rect x="0" y="0" width="{_BOARD:.0f}" height="{_BOARD:.0f}" fill="#ffffff" />',
        (
            f'<rect x="{_CENTER-_CENTER_BOX_SIZE/2.0:.1f}" y="{_CENTER-_CENTER_BOX_SIZE/2.0:.1f}" '
            f'width="{_CENTER_BOX_SIZE:.1f}" height="{_CENTER_BOX_SIZE:.1f}" '
            'fill="#fff" stroke="#000" stroke-width="2" rx="3" ry="3" />'
        ),
        (
            f'<text x="{_CENTER:.1f}" y="{_CENTER-25:.1f}" text-anchor="middle" font-size="22" fill="#000"'
            f' font-family="{_JA_FONT_FAMILY}">{round_label}</text>'
            if language == "ja"
            else f'<text x="{_CENTER:.1f}" y="{_CENTER-25:.1f}" text-anchor="middle" font-size="22" fill="#000">{round_label}</text>'
        ),
    ]
    if dora:
        dora_w = _W * _DORA_SCALE
        total_w = len(dora) * dora_w + max(0, len(dora) - 1) * _DORA_GAP
        dx = _CENTER - total_w / 2.0
        dy = _DORA_Y
        for i, tile in enumerate(dora):
            parts.append(
                _image_tag_scaled(
                    tile,
                    dx + i * (dora_w + _DORA_GAP),
                    dy,
                    _DORA_SCALE,
                    language,
                )
            )
    parts.append(
        f'<text x="{_CENTER-15:.1f}" y="{_REMAINING_TILES_Y:.1f}" font-size="20" fill="#000">x {remaining_tiles}</text>'
    )
    for player in range(NUM_PLAYERS):
        group = _player_group(
            state,
            player,
            show_all_hands=show_all_hands,
            visible_player=visible_player,
            language=language,
        )
        parts.append(f'<g transform="rotate({-90*player} {_CENTER:.1f} {_CENTER:.1f})">{group}</g>')
    parts.append("</svg>")
    return "".join(parts)


def save_svg(
    state: EnvState,
    filename: str | Path,
    show_all_hands: bool = True,
    visible_player: int = 0,
    language: Language = "ja",
) -> None:
    Path(filename).write_text(
        render_round_svg(
            state,
            show_all_hands=show_all_hands,
            visible_player=visible_player,
            language=language,
        ),
        encoding="utf-8",
    )


def render_play_history_svg(
    states: list[EnvState],
    columns: int = 3,
    padding: float = 20.0,
    show_all_hands: bool = True,
    language: Language = "ja",
) -> str:
    language = _normalize_language(language)
    if columns <= 0:
        raise ValueError("columns must be >= 1")
    if not states:
        raise ValueError("states must not be empty")
    rows = (len(states) + columns - 1) // columns
    width = columns * _BOARD + (columns - 1) * padding
    height = rows * _BOARD + (rows - 1) * padding
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#ffffff" />',
    ]
    for idx, state in enumerate(states):
        svg = render_round_svg(state, show_all_hands=show_all_hands, language=language)
        uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
        col = idx % columns
        row = idx // columns
        x = col * (_BOARD + padding)
        y = row * (_BOARD + padding)
        parts.append(
            f'<image href="{uri}" x="{x:.2f}" y="{y:.2f}" width="{_BOARD:.2f}" height="{_BOARD:.2f}" />'
        )
    parts.append("</svg>")
    return "".join(parts)


def save_play_history_svg(
    states: list[EnvState],
    filename: str | Path,
    columns: int = 3,
    padding: float = 20.0,
    show_all_hands: bool = True,
    language: Language = "ja",
) -> None:
    Path(filename).write_text(
        render_play_history_svg(
            states,
            columns=columns,
            padding=padding,
            show_all_hands=show_all_hands,
            language=language,
        ),
        encoding="utf-8",
    )


def render_svg_animation(
    states: list[EnvState],
    frame_duration_seconds: float = 0.2,
    show_all_hands: bool = True,
    visible_player: int = 0,
    language: Language = "ja",
) -> str:
    language = _normalize_language(language)
    if not states:
        raise ValueError("states must not be empty")
    total_seconds = frame_duration_seconds * len(states)
    step = 100.0 / len(states)
    style = f".frame{{visibility:hidden; animation:{total_seconds}s linear _k infinite;}}"
    style += f"@keyframes _k{{0%,{step}%{{visibility:visible}}{step * 1.000001}%,100%{{visibility:hidden}}}}"
    images: list[str] = []
    for i, state in enumerate(states):
        svg = render_round_svg(
            state,
            show_all_hands=show_all_hands,
            visible_player=visible_player,
            language=language,
        )
        uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
        frame_id = f"_fr{i:x}"
        images.append(
            f'<image href="{uri}" x="0" y="0" width="{_BOARD:.2f}" height="{_BOARD:.2f}" id="{frame_id}" class="frame" />'
        )
        style += f"#{frame_id}{{animation-delay:{i * frame_duration_seconds}s}}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_BOARD:.0f}" height="{_BOARD:.0f}" viewBox="0 0 {_BOARD:.0f} {_BOARD:.0f}">'
        f'<rect x="0" y="0" width="{_BOARD:.0f}" height="{_BOARD:.0f}" fill="#ffffff" />'
        f"<defs><style>{style}</style></defs>"
        f"{''.join(images)}</svg>"
    )


def save_svg_animation(
    states: list[EnvState],
    filename: str | Path,
    frame_duration_seconds: float = 0.2,
    show_all_hands: bool = True,
    visible_player: int = 0,
    language: Language = "ja",
) -> None:
    Path(filename).write_text(
        render_svg_animation(
            states,
            frame_duration_seconds=frame_duration_seconds,
            show_all_hands=show_all_hands,
            visible_player=visible_player,
            language=language,
        ),
        encoding="utf-8",
    )


def generate_play_history_states(
    seed: int = 0,
    max_steps: int = 80,
    policy: str = "first_legal",
) -> list[EnvState]:
    env = RedMahjong()
    key = jax.random.PRNGKey(seed)
    state = env.init(key)
    history: list[EnvState] = [state]
    rng = key

    def _sample_action(mask: jnp.ndarray, rng_key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        count = jnp.sum(mask.astype(jnp.int32))
        legal_idx = jnp.nonzero(mask, size=LEGAL_ACTION_SIZE, fill_value=-1)[0]
        rng_key, sub = jax.random.split(rng_key)
        idx = jax.random.randint(sub, (), 0, jnp.maximum(count, 1))
        action = jax.lax.cond(count > 0, lambda: legal_idx[idx], lambda: jnp.int32(-1))
        return action.astype(jnp.int32), rng_key, count

    for _ in range(max_steps):
        if policy == "random":
            action, rng, count = _sample_action(state.legal_action_mask, rng)
            if int(count) == 0:
                break
        else:
            if int(jnp.sum(state.legal_action_mask)) == 0:
                break
            action = jnp.argmax(state.legal_action_mask).astype(jnp.int32)
        state = env.step(state, action)
        history.append(state)
        if bool(state.terminated | state.truncated):
            break
    return history


def random_play_and_save_svg(
    filename: str | Path,
    seed: int = 0,
    max_steps: int = 80,
    language: Language = "ja",
) -> EnvState:
    history = generate_play_history_states(seed=seed, max_steps=max_steps, policy="random")
    state = history[-1]
    save_svg(state, filename, language=language)
    return state


def random_play_history_and_save_svg(
    filename: str | Path,
    seed: int = 0,
    max_steps: int = 80,
    columns: int = 3,
    padding: float = 20.0,
    show_all_hands: bool = True,
    language: Language = "ja",
) -> EnvState:
    history = generate_play_history_states(seed=seed, max_steps=max_steps, policy="random")
    state = history[-1]
    save_play_history_svg(
        history,
        filename,
        columns=columns,
        padding=padding,
        show_all_hands=show_all_hands,
        language=language,
    )
    return state


def play_history_and_save_svg_animation(
    filename: str | Path,
    seed: int = 0,
    max_steps: int = 80,
    frame_duration_seconds: float = 0.2,
    show_all_hands: bool = True,
    policy: str = "first_legal",
    language: Language = "ja",
) -> EnvState:
    history = generate_play_history_states(seed=seed, max_steps=max_steps, policy=policy)
    save_svg_animation(
        history,
        filename,
        frame_duration_seconds=frame_duration_seconds,
        show_all_hands=show_all_hands,
        language=language,
    )
    return history[-1]
