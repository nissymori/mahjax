from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
JA = ROOT / "ja"
EN = ROOT / "en"

SUIT_COLORS = {"m": "#c62828", "p": "#1f3f99", "s": "#16703c"}
LABEL_COLORS = {
    "east": "#111111",
    "south": "#111111",
    "west": "#111111",
    "north": "#111111",
    "rd": "#c62828",
    "gd": "#16703c",
}
LABELS = {
    "east": "E",
    "south": "S",
    "west": "W",
    "north": "N",
    "rd": "R",
    "gd": "G",
}

TEXT_STYLE = (
    "font-family=\"'Noto Serif', 'Noto Serif JP', 'Yu Mincho', "
    "'Hiragino Mincho ProN', Georgia, serif\" "
    "font-size=\"14.6\" font-weight=\"700\""
)
TEXT_X = 3.5
TEXT_Y = 15.1
STANDARD_BOX = (10.8, 15.9, 15.9, 20.9)
LARGE_BOX = (10.3, 15.9, 16.3, 20.9)
MOTIF_PAD_RATIO = 0.045
MOTIF_MIN_PAD = 0.16


def read_svg(name: str):
    text = (JA / name).read_text()
    view_box = re.search(r'viewBox="([^"]+)"', text).group(1)
    group_open = re.search(r"(<g[^>]*>)", text).group(1)
    path_d = re.search(r'<path d="([^"]+)"', text).group(1)
    return text, view_box, group_open, path_d


def split_subpaths(path_d: str) -> list[str]:
    parts = path_d.split(" M ")
    return [part if index == 0 else "M " + part for index, part in enumerate(parts)]


def bbox_for_path(path_d: str) -> tuple[float, float, float, float]:
    token_re = re.compile(r"[A-Z]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
    tokens = token_re.findall(path_d)
    cur = (0.0, 0.0)
    start = None
    xs: list[float] = []
    ys: list[float] = []
    i = 0

    def add_point(x: float, y: float) -> None:
        xs.append(float(x))
        ys.append(float(y))

    while i < len(tokens):
        cmd = tokens[i]
        i += 1
        if cmd == "M":
            x = float(tokens[i])
            y = float(tokens[i + 1])
            i += 2
            cur = (x, y)
            start = (x, y)
            add_point(x, y)
        elif cmd == "L":
            x = float(tokens[i])
            y = float(tokens[i + 1])
            i += 2
            add_point(*cur)
            add_point(x, y)
            cur = (x, y)
        elif cmd == "Q":
            x1 = float(tokens[i])
            y1 = float(tokens[i + 1])
            x = float(tokens[i + 2])
            y = float(tokens[i + 3])
            i += 4
            add_point(*cur)
            add_point(x1, y1)
            add_point(x, y)
            cur = (x, y)
        elif cmd == "C":
            x1 = float(tokens[i])
            y1 = float(tokens[i + 1])
            x2 = float(tokens[i + 2])
            y2 = float(tokens[i + 3])
            x = float(tokens[i + 4])
            y = float(tokens[i + 5])
            i += 6
            add_point(*cur)
            add_point(x1, y1)
            add_point(x2, y2)
            add_point(x, y)
            cur = (x, y)
        elif cmd == "A":
            x = float(tokens[i + 5])
            y = float(tokens[i + 6])
            i += 7
            add_point(*cur)
            add_point(x, y)
            cur = (x, y)
        elif cmd == "Z":
            if start is not None:
                add_point(*cur)
                add_point(*start)
                cur = start
        else:
            raise ValueError(f"Unexpected command: {cmd}")

    return min(xs), min(ys), max(xs), max(ys)


def label_for(name: str) -> tuple[str, str]:
    stem = Path(name).stem
    if stem in LABELS:
        return LABELS[stem], LABEL_COLORS[stem]
    if stem == "white":
        return "", "#111111"
    number, suit = re.fullmatch(r"([1-9])(m|p|s)r?", stem).groups()
    return number, SUIT_COLORS[suit]


def build_motif_markup(name: str, border_parts: set[str]) -> str:
    text = (JA / name).read_text()
    groups = re.findall(r"(<g[^>]*>)(.*?)</g>", text, flags=re.DOTALL)
    chunks: list[str] = []
    for group_open, group_body in groups:
        paths = re.findall(r'<path d="([^"]+)"', group_body)
        kept_paths = []
        for path_d in paths:
            kept = [part for part in split_subpaths(path_d) if part not in border_parts]
            if kept:
                kept_paths.append(" ".join(kept))
        if kept_paths:
            body = "".join(
                f'<path d="{path_d}" vector-effect="non-scaling-stroke"/>'
                for path_d in kept_paths
            )
            chunks.append(f"{group_open}{body}</g>")
    return "".join(chunks)


def main() -> None:
    _, _, frame_group_open, white_path = read_svg("white.svg")
    standard_border_parts = set(split_subpaths(white_path))
    white_frame = (JA / "white.svg").read_text()
    frame_path = re.search(r'<path d="([^"]+)"', white_frame).group(1)

    _, _, _, red_path = read_svg("rd.svg")
    large_border_parts = set(split_subpaths(red_path)[:2])

    EN.mkdir(exist_ok=True)

    for src in sorted(JA.glob("*.svg")):
        name = src.name
        if re.fullmatch(r"[1-46-9][mps]r\.svg", name):
            continue
        if name in {"back.svg", "b.svg"}:
            (EN / name).write_text(src.read_text())
            continue
        if name == "white.svg":
            (EN / name).write_text(white_frame)
            continue
        if name == "oya.svg":
            out = (
                '<svg width="58" height="83" viewBox="0 0 29 41.5" '
                'xmlns="http://www.w3.org/2000/svg">'
                f'{frame_group_open}<path d="{frame_path}" '
                'vector-effect="non-scaling-stroke"/></g>'
                '<text x="14.5" y="23.7" text-anchor="middle" '
                f'{TEXT_STYLE} fill="#111111">D</text></svg>'
            )
            (EN / name).write_text(out)
            continue

        _, view_box, _, _ = read_svg(name)
        border_parts = (
            standard_border_parts
            if view_box == "0 0 29 41.5"
            else large_border_parts
        )
        motif_markup = build_motif_markup(name, border_parts)
        label, color = label_for(name)
        text_svg = (
            f'<text x="{TEXT_X}" y="{TEXT_Y}" {TEXT_STYLE} fill="{color}">{label}</text>'
            if label
            else ""
        )
        motif_svg = ""
        if motif_markup:
            path_list = re.findall(r'<path d="([^"]+)"', motif_markup)
            min_x = min(bbox_for_path(path_d)[0] for path_d in path_list)
            min_y = min(bbox_for_path(path_d)[1] for path_d in path_list)
            max_x = max(bbox_for_path(path_d)[2] for path_d in path_list)
            max_y = max(bbox_for_path(path_d)[3] for path_d in path_list)
            width = max_x - min_x
            height = max_y - min_y
            pad_x = max(MOTIF_MIN_PAD, width * MOTIF_PAD_RATIO)
            pad_y = max(MOTIF_MIN_PAD, height * MOTIF_PAD_RATIO)
            motif_view_box = (
                f"{min_x - pad_x:.3f} {min_y - pad_y:.3f} "
                f"{width + pad_x * 2:.3f} {height + pad_y * 2:.3f}"
            )
            x, y, width_box, height_box = (
                LARGE_BOX if name in {"rd.svg", "gd.svg"} else STANDARD_BOX
            )
            motif_svg = (
                f'<svg x="{x}" y="{y}" width="{width_box}" height="{height_box}" '
                f'viewBox="{motif_view_box}" overflow="visible" '
                'preserveAspectRatio="xMidYMid meet">'
                f"{motif_markup}</svg>"
            )

        out = (
            '<svg width="58" height="83" viewBox="0 0 29 41.5" '
            'xmlns="http://www.w3.org/2000/svg">'
            f'{frame_group_open}<path d="{frame_path}" '
            'vector-effect="non-scaling-stroke"/></g>'
            f"{text_svg}{motif_svg}</svg>"
        )
        (EN / name).write_text(out)


if __name__ == "__main__":
    main()
