import numpy as np

from mahjax._src.dwg.mahjong_tile import TilePath
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.env import State as MahjongState
from mahjax.no_red_mahjong.meld import Meld

# ==== 追加: riverの16bitレイアウトに合わせたマスク類 ====
TILE_MASK = 0b0000000000111111  # bits 0..5
BIT_RIICHI = 1 << 6  # bit 6
BIT_GRAY = 1 << 7  # bit 7
EMPTY_RIVER = 0xFFFF  # 空スロット

path_list = TilePath.str_list
tile_w = 30
tile_h = 45
hand_x = 120
hand_y = 640
wind = ["東", "南", "西", "北"]


def _make_mahjong_dwg_jp(dwg, state: MahjongState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]
    board_g = dwg.g(
        style="stroke:#000000;stroke-width:0.01mm;fill:#000000",
        fill_rule="evenodd",
    )

    # background
    board_g.add(
        dwg.rect(
            (0, 0),
            (
                BOARD_WIDTH * GRID_SIZE,
                BOARD_HEIGHT * GRID_SIZE,
            ),
            fill=color_set.background_color,
        )
    )
    # central info
    width = 180
    board_g.add(
        dwg.rect(
            (
                (BOARD_WIDTH * GRID_SIZE - width) / 2,
                (BOARD_HEIGHT * GRID_SIZE - width) / 2,
            ),
            (width, width),
            fill=color_set.background_color,
            stroke=color_set.grid_color,
            stroke_width="2px",
            rx="3px",
            ry="3px",
        )
    )
    kanji = ["", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十"]
    ro = state._round
    round = f"{wind[ro//4]}{kanji[ro%4+1]}局"
    if state._honba > 0:
        round += f"{kanji[state._honba]}本場"

    fontsize = 20
    y = -25
    board_g.add(
        dwg.text(
            text=round,
            insert=(
                (BOARD_WIDTH * GRID_SIZE) / 2 - len(round) * fontsize / 2,
                (BOARD_HEIGHT * GRID_SIZE) / 2 + y,
            ),
            fill=color_set.text_color,
            font_size=f"{fontsize}px",
            font_family="serif",
        )
    )

    # dora
    dora_scale = 0.6
    x = (BOARD_WIDTH * GRID_SIZE) / 2 - tile_w * dora_scale * 2.5
    y = (BOARD_WIDTH * GRID_SIZE) / 2 - 15
    for _x, dora in enumerate(state._dora_indicators):
        if dora == -1:
            dora = 34
        p = dwg.path(d=path_list[dora])
        p.translate(x + _x * tile_w * dora_scale, y)
        p.scale(dora_scale)
        board_g.add(p)

    # yama
    yama_scale = 0.6
    x = (BOARD_WIDTH * GRID_SIZE) / 2 - 25
    y = (BOARD_WIDTH * GRID_SIZE) / 2 + 22
    fontsize = 20
    p = dwg.path(d=path_list[34])
    p.translate(x, y)
    p.scale(yama_scale)
    board_g.add(p)
    board_g.add(
        dwg.text(
            text=f"x {state._next_deck_ix-14+1}",
            insert=(x + tile_w * yama_scale + 5, y + tile_h * yama_scale - 5),
            fill=color_set.text_color,
            font_size=f"{fontsize}px",
            font_family="serif",
        )
    )

    # board
    for i in range(4):
        players_g = _make_players_dwg_jp(
            dwg, state, i, color_set, BOARD_WIDTH, BOARD_HEIGHT, GRID_SIZE
        )
        players_g.rotate(
            angle=-90 * i,
            center=(BOARD_WIDTH * GRID_SIZE / 2, BOARD_WIDTH * GRID_SIZE / 2),
        )
        board_g.add(players_g)

    return board_g


def _make_players_dwg_jp(
    dwg,
    state: MahjongState,
    i,
    color_set,
    BOARD_WIDTH,
    BOARD_HEIGHT,
    GRID_SIZE,
):
    players_g = dwg.g(
        style="stroke:#000000;stroke-width:0.01mm;fill:#000000",
        fill_rule="evenodd",
    )

    # wind
    x = 265
    y = 435
    fontsize = 22
    players_g.add(
        dwg.text(
            text=wind[(i - state._dealer) % 4],
            insert=(x, y),
            fill=color_set.text_color,
            font_size=f"{fontsize}px",
            font_family="serif",
        )
    )

    # score
    fontsize = 20
    score = str(int(state._score[i]) * 100)
    y = 70
    players_g.add(
        dwg.text(
            text=score,
            insert=(
                (BOARD_WIDTH * GRID_SIZE) / 2 - len(score) * fontsize / 4,
                BOARD_HEIGHT * GRID_SIZE / 2 + y,
            ),
            fill=color_set.text_color,
            font_size=f"{fontsize}px",
            font_family="serif",
        )
    )

    # riichi bou
    width = 100
    height = 10
    y = 75
    if state._riichi[i]:
        players_g.add(
            dwg.rect(
                (
                    (BOARD_WIDTH * GRID_SIZE - width) / 2,
                    BOARD_HEIGHT * GRID_SIZE / 2 + y,
                ),
                (width, height),
                fill=color_set.background_color,
                stroke=color_set.grid_color,
                stroke_width="1px",
                rx="3px",
                ry="3px",
            )
        )
        players_g.add(
            dwg.circle(
                center=(
                    BOARD_HEIGHT * GRID_SIZE / 2,
                    BOARD_HEIGHT * GRID_SIZE / 2 + y + height / 2,
                ),
                r="3px",
                fill="red",
            )
        )

    # hand
    offset = 0
    hand_raw = np.array(state._hand[i])
    is_hidden_hand = bool((hand_raw < 0).any())
    hand = np.abs(hand_raw)
    current_player = int(state.current_player)
    last_draw = int(getattr(state, "_last_draw", -1))
    draw_tile = None
    separated = False
    if not is_hidden_hand and current_player == i and last_draw >= 0:
        draw_count = int(hand[int(last_draw)])
        if draw_count > 0:
            draw_tile = last_draw
    if is_hidden_hand:
        total_tiles = int(hand.sum())
        separate_draw = current_player == i and total_tiles % 3 == 2
        visible_tiles = total_tiles - (1 if separate_draw and total_tiles > 0 else 0)
        offset = _draw_hidden_hand_tiles(
            dwg,
            players_g,
            visible_tiles,
            offset,
            separate_draw,
        )
    else:
        for tile, num in enumerate(hand):
            count = int(num)
            if draw_tile is not None and not separated and tile == draw_tile:
                count -= 1
                separated = True
            for _ in range(count):
                p = dwg.path(d=path_list[tile])
                p.translate(hand_x + offset, hand_y)
                players_g.add(p)
                offset += tile_w

        if draw_tile is not None and separated:
            offset += tile_w * 0.5
            p = dwg.path(d=path_list[draw_tile])
            p.translate(hand_x + offset, hand_y)
            players_g.add(p)
            offset += tile_w

    offset += tile_w

    # meld
    for meld in state._melds[i]:
        if meld == 0:
            continue
        if Meld.action(meld) == Action.PON:
            players_g, offset = _apply_pon(dwg, players_g, meld, offset)
        elif (
            (Meld.action(meld) == Action.CHI_L)
            or (Meld.action(meld) == Action.CHI_M)
            or (Meld.action(meld) == Action.CHI_R)
        ):
            players_g, offset = _apply_chi(dwg, players_g, meld, offset)
        elif (34 <= Meld.action(meld) <= 67) and Meld.src(meld) == 0:
            players_g, offset = _apply_closed_kan(dwg, players_g, meld, offset)
        elif 34 <= Meld.action(meld) <= 67:
            players_g, offset = _apply_added_kan(dwg, players_g, meld, offset)
        elif Meld.action(meld) == Action.OPEN_KAN:
            players_g, offset = _apply_open_kan(dwg, players_g, meld, offset)

    # river
    x = BOARD_WIDTH * GRID_SIZE / 2 - 3 * tile_w
    y = 450

    river = state._river[i]
    for river_ix, raw in enumerate(river):
        raw = int(raw)

        # === 修正: 空スロット判定（16bitの 0xFFFF） ===
        if raw == EMPTY_RIVER:
            # 改行処理だけ進める
            if river_ix % 6 == 5:
                x = BOARD_WIDTH * GRID_SIZE / 2 - 3 * tile_w
                y += tile_h
            continue

        # === 修正: ビット抽出は16bitレイアウト準拠 ===
        fill = "black"
        if (raw & BIT_GRAY) != 0:
            fill = "gray"

        is_riichi = (raw & BIT_RIICHI) != 0
        tile_id = raw & TILE_MASK  # 0..33

        if is_riichi:
            # リーチ宣言牌は90度回転表示
            p = dwg.path(d=path_list[tile_id], fill=fill)
            p.rotate(angle=-90, center=(x, y))
            p.translate(x - tile_h + 4, y + 2)
            players_g.add(p)
            x += tile_h
        elif tile_id < 34:
            p = dwg.path(d=path_list[tile_id], fill=fill)
            p.translate(x, y)
            players_g.add(p)
            x += tile_w

        if river_ix % 6 == 5:
            x = BOARD_WIDTH * GRID_SIZE / 2 - 3 * tile_w
            y += tile_h

    return players_g


def _draw_hidden_hand_tiles(dwg, group, tile_count, offset, separate_draw):
    """Draw a sequence of facedown tiles for hidden opponents."""
    for _ in range(tile_count):
        p = dwg.path(d=path_list[34])
        p.translate(hand_x + offset, hand_y)
        group.add(p)
        offset += tile_w
    if separate_draw:
        offset += tile_w * 0.5
        p = dwg.path(d=path_list[34])
        p.translate(hand_x + offset, hand_y)
        group.add(p)
        offset += tile_w
    return offset


def _apply_pon(dwg, g, meld, offset):
    tile = Meld.target(meld)
    if Meld.src(meld) == 3:
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
    elif Meld.src(meld) == 2:
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
    elif Meld.src(meld) == 1:
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
    return g, offset + tile_w


def _apply_chi(dwg, g, meld, offset):
    tile = Meld.target(meld)
    if Meld.action(meld) == Action.CHI_L:
        tile1 = tile
        tile2 = tile + 1
        tile3 = tile + 2
    elif Meld.action(meld) == Action.CHI_M:
        tile1 = tile
        tile2 = tile - 1
        tile3 = tile + 1
    else:
        tile1 = tile
        tile2 = tile - 1
        tile3 = tile - 2

    if Meld.src(meld) == 3:
        p = dwg.path(d=path_list[tile1])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
        p = dwg.path(d=path_list[tile2])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile3])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
    elif Meld.src(meld) == 2:
        p = dwg.path(d=path_list[tile2])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile1])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
        p = dwg.path(d=path_list[tile3])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
    elif Meld.src(meld) == 1:
        p = dwg.path(d=path_list[tile3])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile2])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile1])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
    return g, offset + tile_w


def _apply_closed_kan(dwg, g, meld, offset):
    tile = Meld.target(meld)
    p = dwg.path(d=path_list[tile])
    p.translate(hand_x + offset, hand_y)
    g.add(p)
    offset += tile_w
    p = dwg.path(d=TilePath.back)
    p.translate(hand_x + offset, hand_y)
    g.add(p)
    offset += tile_w
    p = dwg.path(d=TilePath.back)
    p.translate(hand_x + offset, hand_y)
    g.add(p)
    offset += tile_w
    p = dwg.path(d=path_list[tile])
    p.translate(hand_x + offset, hand_y)
    g.add(p)
    offset += tile_w

    return g, offset + tile_w


def _apply_added_kan(dwg, g, meld, offset):
    tile = Meld.target(meld)
    if Meld.src(meld) == 3:
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, 0)
        g.add(p)
        offset += tile_h
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
    elif Meld.src(meld) == 2:
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, 0)
        g.add(p)
        offset += tile_h
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
    elif Meld.src(meld) == 1:
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, 0)
        g.add(p)
        offset += tile_h
    return g, offset + tile_w


def _apply_open_kan(dwg, g, meld, offset):
    tile = Meld.target(meld)
    if Meld.src(meld) == 3:
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
    elif Meld.src(meld) == 2:
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
    elif Meld.src(meld) == 1:
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.translate(hand_x + offset, hand_y)
        g.add(p)
        offset += tile_w
        p = dwg.path(d=path_list[tile])
        p.rotate(-90, center=(hand_x + offset, hand_y))
        p.translate(hand_x + offset - tile_h + 4, hand_y + 1)
        g.add(p)
        offset += tile_h
    return g, offset + tile_w


def _make_mahjong_dwg_en(dwg, state: MahjongState, config):
    GRID_SIZE = config["GRID_SIZE"]
    BOARD_WIDTH = config["BOARD_WIDTH"]
    BOARD_HEIGHT = config["BOARD_HEIGHT"]
    color_set = config["COLOR_SET"]
    board_g = dwg.g(
        style="stroke:#000000;stroke-width:0.01mm;fill:#000000",
        fill_rule="evenodd",
    )

    # background
    board_g.add(
        dwg.rect(
            (0, 0),
            (
                BOARD_WIDTH * GRID_SIZE,
                BOARD_HEIGHT * GRID_SIZE,
            ),
            fill=color_set.background_color,
        )
    )
    # central info
    width = 180
    center_x = (BOARD_WIDTH * GRID_SIZE - width) / 2
    center_y = (BOARD_HEIGHT * GRID_SIZE - width) / 2
    board_g.add(
        dwg.rect(
            (center_x, center_y),
            (width, width),
            fill=color_set.background_color,
            stroke=color_set.grid_color,
            stroke_width="2px",
            rx="3px",
            ry="3px",
        )
    )

    # Round Info (English)
    winds_en = ["E", "S", "W", "N"]
    ro = state._round
    round_str = f"{winds_en[ro//4]} {ro%4+1}"  # e.g., "E 1"
    if state._honba > 0:
        round_str += f"-{state._honba}"  # e.g., "E 1-1"

    fontsize = 24
    y = -25
    board_g.add(
        dwg.text(
            text=round_str,
            insert=(
                (BOARD_WIDTH * GRID_SIZE) / 2,
                (BOARD_HEIGHT * GRID_SIZE) / 2 + y,
            ),
            fill=color_set.text_color,
            font_size=f"{fontsize}px",
            font_family="Arial, sans-serif",
            font_weight="bold",
            text_anchor="middle",  # Center align
        )
    )

    # Helper to draw a single English tile (Simplified for Dora/Center)
    def _draw_simple_en_tile(group, tx, ty, t_idx, scale=1.0):
        w, h = tile_w * scale, tile_h * scale

        # Tile Body
        group.add(
            dwg.rect(
                insert=(tx, ty),
                size=(w, h),
                fill="#fff",
                stroke="#333",
                stroke_width="1",
                rx="2",
                ry="2",
            )
        )

        # Text
        txt, color = _get_tile_text_color(t_idx)
        group.add(
            dwg.text(
                txt,
                insert=(tx + w / 2, ty + h / 2 + 5 * scale),
                fill=color,
                font_size=f"{18*scale}px",
                font_family="Arial",
                font_weight="bold",
                text_anchor="middle",
            )
        )

    # dora
    dora_scale = 0.6
    dora_w = tile_w * dora_scale
    start_x = (BOARD_WIDTH * GRID_SIZE) / 2 - dora_w * 2.5
    dora_y = (BOARD_WIDTH * GRID_SIZE) / 2 - 15

    for _x, dora in enumerate(state._dora_indicators):
        if dora == -1:
            # Draw back of tile
            w, h = tile_w * dora_scale, tile_h * dora_scale
            board_g.add(
                dwg.rect(
                    insert=(start_x + _x * dora_w, dora_y),
                    size=(w, h),
                    fill="#444",
                    stroke="#333",
                    stroke_width="1",
                    rx="2",
                    ry="2",
                )
            )
        else:
            _draw_simple_en_tile(
                board_g, start_x + _x * dora_w, dora_y, dora, dora_scale
            )

    # yama (Remaining Tiles)
    x = (BOARD_WIDTH * GRID_SIZE) / 2
    y = (BOARD_WIDTH * GRID_SIZE) / 2 + 35
    fontsize = 18
    # Icon-like rect for deck
    board_g.add(
        dwg.rect(
            insert=(x - 40, y - 15),
            size=(20, 28),
            fill="#444",
            stroke="#333",
            stroke_width="1",
            rx="2",
            ry="2",
        )
    )
    board_g.add(
        dwg.text(
            text=f"x {state._next_deck_ix-14+1}",
            insert=(x - 15, y + 5),
            fill=color_set.text_color,
            font_size=f"{fontsize}px",
            font_family="Arial, sans-serif",
        )
    )

    # board (Players)
    for i in range(4):
        players_g = _make_players_dwg_en(
            dwg, state, i, color_set, BOARD_WIDTH, BOARD_HEIGHT, GRID_SIZE
        )
        players_g.rotate(
            angle=-90 * i,
            center=(BOARD_WIDTH * GRID_SIZE / 2, BOARD_WIDTH * GRID_SIZE / 2),
        )
        board_g.add(players_g)

    return board_g


def _get_tile_text_color(tile_idx):
    """Helper to return text and color for a tile index (0-33)."""
    if 0 <= tile_idx < 9:
        return f"{tile_idx+1}m", "#b71c1c"  # Manzu (Red)
    if 9 <= tile_idx < 18:
        return f"{tile_idx-9+1}p", "#0d47a1"  # Pinzu (Blue)
    if 18 <= tile_idx < 27:
        return f"{tile_idx-18+1}s", "#1b5e20"  # Souzu (Green)

    winds = ["E", "S", "W", "N"]
    if 27 <= tile_idx < 31:
        return winds[tile_idx - 27], "#000"

    dragons = ["wd", "gd", "rd"]
    if tile_idx == 31:
        return "wd", "#555"  # White Dragon (Gray text)
    if tile_idx == 32:
        return "gd", "#1b5e20"  # Green Dragon
    if tile_idx == 33:
        return "rd", "#b71c1c"  # Red Dragon
    return "?", "#000"


def _make_players_dwg_en(
    dwg,
    state: MahjongState,
    i,
    color_set,
    BOARD_WIDTH,
    BOARD_HEIGHT,
    GRID_SIZE,
):
    players_g = dwg.g(
        style="stroke:#000000;stroke-width:0.01mm;fill:#000000",
        fill_rule="evenodd",
    )

    # --- Helper: Draw Tile ---
    def draw_en_tile(
        group, x, y, tile_idx, rotate=False, is_back=False, transparent=False
    ):
        """Draws a tile at x,y. Uses tile_w/tile_h global vars."""
        w, h = tile_w, tile_h
        opacity = 0.6 if transparent else 1.0
        bg_color = "#fff" if not is_back else "#444"
        stroke_c = "#999"

        def _place(element):
            if rotate:
                element.rotate(-90, center=(x, y))
                element.translate(x - tile_h + 4, y + 1)
            else:
                element.translate(x, y)
            element.update({"opacity": opacity})
            group.add(element)

        body = dwg.rect(
            insert=(0, 0),
            size=(w, h),
            fill=bg_color,
            stroke=stroke_c,
            stroke_width="1",
            rx="3",
            ry="3",
        )
        _place(body)

        if not is_back and tile_idx is not None:
            txt, color = _get_tile_text_color(tile_idx)
            text = dwg.text(
                txt,
                insert=(w / 2, h / 2 + 6),
                fill=color,
                font_size="20px",
                font_family="Arial, sans-serif",
                font_weight="bold",
                text_anchor="middle",
            )
            _place(text)

    # --- Wind ---
    wind_en = ["E", "S", "W", "N"]
    x_pos = 265
    y_pos = 435
    fontsize = 24
    players_g.add(
        dwg.text(
            text=wind_en[(i - state._dealer) % 4],
            insert=(x_pos, y_pos),
            fill=color_set.text_color,
            font_size=f"{fontsize}px",
            font_family="Arial, sans-serif",
            font_weight="bold",
        )
    )

    # --- Score ---
    fontsize = 20
    score = str(int(state._score[i]) * 100)
    y_pos = 70
    players_g.add(
        dwg.text(
            text=score,
            insert=(
                (BOARD_WIDTH * GRID_SIZE) / 2 - len(score) * fontsize / 4,
                BOARD_HEIGHT * GRID_SIZE / 2 + y_pos,
            ),
            fill=color_set.text_color,
            font_size=f"{fontsize}px",
            font_family="Arial, sans-serif",
        )
    )

    # --- Riichi Stick ---
    width = 100
    height = 10
    y_pos = 75
    if state._riichi[i]:
        players_g.add(
            dwg.rect(
                (
                    (BOARD_WIDTH * GRID_SIZE - width) / 2,
                    BOARD_HEIGHT * GRID_SIZE / 2 + y_pos,
                ),
                (width, height),
                fill="#eee",
                stroke="#666",
                stroke_width="1px",
                rx="3px",
                ry="3px",
            )
        )
        players_g.add(
            dwg.circle(
                center=(
                    BOARD_HEIGHT * GRID_SIZE / 2,
                    BOARD_HEIGHT * GRID_SIZE / 2 + y_pos + height / 2,
                ),
                r="3px",
                fill="red",
            )
        )

    # --- Hand ---
    offset = 0
    hand_raw = np.array(state._hand[i])
    is_hidden_hand = bool((hand_raw < 0).any())
    hand = np.abs(hand_raw)
    current_player = int(state.current_player)
    last_draw = int(getattr(state, "_last_draw", -1))
    draw_tile = None
    separated = False

    if not is_hidden_hand and current_player == i and last_draw >= 0:
        draw_count = int(hand[int(last_draw)])
        if draw_count > 0:
            draw_tile = last_draw

    if is_hidden_hand:
        total_tiles = int(hand.sum())
        separate_draw = current_player == i and total_tiles % 3 == 2
        visible_tiles = total_tiles - (1 if separate_draw and total_tiles > 0 else 0)

        for _ in range(visible_tiles):
            draw_en_tile(players_g, hand_x + offset, hand_y, None, is_back=True)
            offset += tile_w

        if separate_draw:
            offset += tile_w * 0.5
            draw_en_tile(players_g, hand_x + offset, hand_y, None, is_back=True)
            offset += tile_w
    else:
        for tile, num in enumerate(hand):
            count = int(num)
            if draw_tile is not None and not separated and tile == draw_tile:
                count -= 1
                separated = True
            for _ in range(count):
                draw_en_tile(players_g, hand_x + offset, hand_y, tile)
                offset += tile_w

        if draw_tile is not None and separated:
            offset += tile_w * 0.5
            draw_en_tile(players_g, hand_x + offset, hand_y, draw_tile)
            offset += tile_w

    offset += tile_w

    # --- Meld (Implemented locally for English tiles) ---
    for meld in state._melds[i]:
        if meld == 0:
            continue

        action = Meld.action(meld)
        target = Meld.target(meld)
        src = Meld.src(meld)  # 1:Right, 2:Center, 3:Left

        # Define layout logic
        layout = []  # List of (tile_id, rotate_bool)

        if action == Action.PON:
            if src == 3:
                layout = [(target, True), (target, False), (target, False)]
            elif src == 2:
                layout = [(target, False), (target, True), (target, False)]
            else:
                layout = [(target, False), (target, False), (target, True)]

        elif action in (Action.CHI_L, Action.CHI_M, Action.CHI_R):
            if action == Action.CHI_L:
                t = [target, target + 1, target + 2]
            elif action == Action.CHI_M:
                t = [target - 1, target, target + 1]
            else:
                t = [target - 2, target - 1, target]

            if src == 3:
                layout = [(t[0], True), (t[1], False), (t[2], False)]
            elif src == 2:
                layout = [(t[1], False), (t[0], True), (t[2], False)]
            else:
                layout = [(t[2], False), (t[1], False), (t[0], True)]

        elif 34 <= action <= 67:  # KAN (Closed/Added)
            if src == 0:  # Closed Kan
                # Draw Back, Back, Tile, Tile (simplified representation for closed kan)
                # Or standard: Tile, Back, Back, Tile
                draw_en_tile(players_g, hand_x + offset, hand_y, target)
                offset += tile_w
                draw_en_tile(players_g, hand_x + offset, hand_y, None, is_back=True)
                offset += tile_w
                draw_en_tile(players_g, hand_x + offset, hand_y, None, is_back=True)
                offset += tile_w
                draw_en_tile(players_g, hand_x + offset, hand_y, target)
                offset += tile_w
                continue
            else:  # Added Kan
                # Similar layout to Pon but the rotated one gets another tile
                # For simplified SVG, we just draw 4 tiles, rotating the source one
                if src == 3:
                    layout = [
                        (target, True),
                        (target, False),
                        (target, False),
                        (target, True),
                    ]
                elif src == 2:
                    layout = [
                        (target, False),
                        (target, True),
                        (target, False),
                        (target, True),
                    ]
                else:
                    layout = [
                        (target, False),
                        (target, False),
                        (target, True),
                        (target, True),
                    ]

        elif action == Action.OPEN_KAN:
            if src == 3:
                layout = [
                    (target, True),
                    (target, False),
                    (target, False),
                    (target, False),
                ]
            elif src == 2:
                layout = [
                    (target, False),
                    (target, True),
                    (target, False),
                    (target, False),
                ]
            else:
                layout = [
                    (target, False),
                    (target, False),
                    (target, False),
                    (target, True),
                ]

        # Render Meld Layout
        for t_idx, rot in layout:
            draw_en_tile(players_g, hand_x + offset, hand_y, t_idx, rotate=rot)
            if rot:
                offset += tile_h  # Rotated tiles take height as width
            else:
                offset += tile_w
        offset += 5  # Small gap

    # --- River ---
    # Centering logic from JP version
    river_x_base = BOARD_WIDTH * GRID_SIZE / 2 - 3 * tile_w
    x = river_x_base
    y = 450

    river = state._river[i]
    for river_ix, raw in enumerate(river):
        raw = int(raw)

        if raw == EMPTY_RIVER:
            if river_ix % 6 == 5:
                x = river_x_base
                y += tile_h
            continue

        is_gray = (raw & BIT_GRAY) != 0
        is_riichi = (raw & BIT_RIICHI) != 0
        tile_id = raw & TILE_MASK

        if is_riichi:
            draw_en_tile(
                players_g, x, y + 10, tile_id, rotate=True, transparent=is_gray
            )
            x += tile_h  # Advance by height since rotated
        elif tile_id < 34:
            draw_en_tile(players_g, x, y, tile_id, rotate=False, transparent=is_gray)
            x += tile_w

        # Wrap every 6 tiles
        if river_ix % 6 == 5:
            x = river_x_base
            y += tile_h

    return players_g
