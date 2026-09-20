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

"""Paint the tile faces the way a real set is painted.

    python -m mahjax._src.assets.tiles.colorize            # the ja set
    python mahjax/_src/assets/tiles/generate_en_tiles.py   # then rebuild en from it

The faces are drawn as one ``<path>`` holding every subpath: the rounded frame
(an outer rectangle and an inset one, filled evenodd so only the ring shows)
followed by the character. One path means one colour, which is why the whole
set reads as black on white. Splitting the frame from the character lets the
character take the suit's colour while the frame stays ink, and a manzu tile
can keep a black numeral above its red 萬 the way a real tile does.

A pin or sou face is not one colour either. It takes its suit's colour, and the
parts a real set paints otherwise are laid over it: the red centre of 1p, 3p and
5p, the lower block of 6p and 7p, the middle row of 9p, the red sticks of 5s, 7s
and 9s, and the peacock's ink body and red legs on 1s. Which part takes which
colour follows the FluffyStuff riichi-mahjong-tiles set (public domain), which
these faces are drawn after. An accent is the whole character again, masked to
its region, because a region can cut across a subpath -- 7p's pips share one
outline -- and splitting the subpaths would break their evenodd holes.

The English set copies these faces, colours and masks included, so
``generate_en_tiles.py`` has to run after this.

The split is derived from the geometry every time, so running this twice is the
same as running it once.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

HERE = Path(__file__).resolve().parent

#: The frame is drawn in ink whatever the face says, so a red five reads as a
#: red character and not as a red tile.
FRAME = "#000"
RED = "#c62828"  # 萬, 中, the red fives, and the red pips and sticks
GREEN = "#16703c"  # bamboo, and 發
BLUE = "#1f3f99"  # the circles of a pin tile
SUIT = {"m": FRAME, "p": BLUE, "s": GREEN}

#: Backs and the dealer marker are not faces and keep the colour they have.
SKIP = {"b.svg", "back.svg", "oya.svg"}

#: A manzu numeral sits in the top of the tile and 萬 fills the rest. Measured:
#: the lowest numeral stroke ends at 0.44 of the height, the highest 萬 stroke
#: starts at 0.43, so the midpoint of a subpath separates them cleanly.
MAN_SPLIT = 0.42


@dataclass(frozen=True)
class Circle:
    """A disc around a pip, centred on its ring edges."""

    cx: float
    cy: float
    r: float


@dataclass(frozen=True)
class Outline:
    """The largest shape on the face whose box holds ``(x, y)``, grown by half
    of ``GROW`` so that the mask edge falls in the gap around the shape."""

    x: float
    y: float


@dataclass(frozen=True)
class Side:
    """The half of the face nearer ``(x, y)`` than ``(ox, oy)``, always cut out.

    Two pips that overlap share one outline, so a disc round either would take a
    crescent of the other; the line halfway between their centres takes neither.
    """

    x: float
    y: float
    ox: float
    oy: float


Region = Union[Circle, Outline, Side]

#: How far an outline mask is stroked past the shape, in viewBox units.
GROW = 0.4

#: Pip centres are the centres of the ring edges. A pip's outer radius is 4.5 in
#: the 2x2 block, where neighbours overlap, 4.0 on 9p and 5.0 on 3p and 5p; each
#: disc is 0.3-0.4 larger so that its edge lies in the white around the pip.
_LOWER_BLOCK: Tuple[Region, ...] = (
    Circle(10.5, 24.75, 4.9),
    Circle(18.5, 24.75, 4.9),
    Circle(10.5, 32.75, 4.9),
    Circle(18.5, 32.75, 4.9),
)
ACCENTS: Dict[str, Tuple[Tuple[str, Tuple[Region, ...]], ...]] = {
    "1p": ((RED, (Circle(14.5, 20.75, 7.6),)),),
    "3p": ((RED, (Circle(14.5, 20.75, 5.4),)),),
    "5p": ((RED, (Circle(14.5, 20.75, 5.4),)),),
    "6p": ((RED, _LOWER_BLOCK),),
    "7p": ((RED, _LOWER_BLOCK + (Side(21.25, 17.25, 18.5, 24.75),)),),  # the last blue pip overlaps the block
    "9p": ((RED, (Circle(7.5, 20.75, 4.3), Circle(14.5, 20.75, 4.3), Circle(21.5, 20.75, 4.3))),),
    "1s": ((FRAME, (Outline(14.5, 28.0),)), (RED, (Outline(11.45, 34.6), Outline(15.5, 34.6)))),
    "5s": ((RED, (Outline(14.5, 20.5),)),),
    "7s": ((RED, (Outline(14.5, 8.3),)),),
    "9s": ((RED, (Outline(14.5, 8.3), Outline(14.5, 20.5), Outline(14.5, 32.7))),),
}

NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")
#: Which arguments of each command are points, and how many arguments it takes.
POINTS = {"M": [(0, 1)], "L": [(0, 1)], "Q": [(0, 1), (2, 3)],
          "C": [(0, 1), (2, 3), (4, 5)], "A": [(5, 6)], "Z": []}
ARITY = {"M": 2, "L": 2, "Q": 4, "C": 6, "A": 7, "Z": 0}

Box = Tuple[float, float, float, float]


def _subpaths(d: str) -> List[str]:
    return [p.strip() for p in re.split(r"(?=M)", d.strip()) if p.strip()]


def _bbox(subpath: str) -> Box:
    """A box around the subpath's on-curve and control points.

    An arc can bulge past its endpoints, so this is a little tight in theory.
    It is only ever used to tell a frame from a character and a numeral from
    the 萬 below it, and those are far apart.
    """
    xs: List[float] = []
    ys: List[float] = []
    for letter, args in re.findall(r"([MLQCAZ])([^MLQCAZ]*)", subpath):
        arity = ARITY[letter]
        if arity == 0:
            continue
        nums = [float(n) for n in NUMBER.findall(args)]
        for start in range(0, len(nums) - arity + 1, arity):
            chunk = nums[start : start + arity]
            for xi, yi in POINTS[letter]:
                xs.append(chunk[xi])
                ys.append(chunk[yi])
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _view_box(svg: str) -> Box:
    raw = re.search(r'viewBox="([^"]*)"', svg)
    if raw is None:
        raise ValueError("tile has no viewBox")
    x, y, w, h = (float(v) for v in NUMBER.findall(raw.group(1)))
    return x, y, w, h


def _frame_indices(subs: List[str], width: float, height: float) -> List[int]:
    """The two subpaths that span the whole tile: the frame's outside and inside."""
    return [
        i
        for i, s in enumerate(subs)
        if (lambda b: (b[2] - b[0]) >= 0.85 * width and (b[3] - b[1]) >= 0.85 * height)(
            _bbox(s)
        )
    ]


def _face_ds(svg: str) -> List[str]:
    """The ``d`` of every drawn path, leaving out masks and the accents they cut."""
    body = re.sub(r"<defs>.*?</defs>", "", svg, flags=re.DOTALL)
    ds = []
    for tag in re.findall(r"<path\b[^>]*>", body):
        d = re.search(r'\sd="([^"]*)"', tag)
        if d is not None and 'mask="url(#accent-' not in tag:
            ds.append(d.group(1))
    return ds


def _glyph_colour(name: str, subpath: str, height: float) -> str:
    stem = name[:-4]  # drop ".svg"
    if stem.endswith("r"):  # 5mr, 5pr, 5sr
        return RED
    if stem == "rd":
        return RED
    if stem == "gd":
        return GREEN
    suit = stem[-1]
    if suit == "m" and stem[:-1].isdigit():
        box = _bbox(subpath)
        below = (box[1] + box[3]) / 2 >= MAN_SPLIT * height
        return RED if below else SUIT["m"]
    if suit in SUIT and stem[:-1].isdigit():
        return SUIT[suit]
    return FRAME  # 東南西北 and the blank 白


def _enclosing(glyph: Sequence[str], x: float, y: float) -> str:
    boxed = [(s, _bbox(s)) for s in glyph]
    inside = [(s, b) for s, b in boxed if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
    if not inside:
        raise ValueError(f"no shape holds ({x}, {y})")
    return max(inside, key=lambda sb: (sb[1][2] - sb[1][0]) * (sb[1][3] - sb[1][1]))[0]


def _half_plane(side: Side, reach: float = 100.0) -> str:
    """Polygon points covering the half-plane of ``side``, far past the tile."""
    dx, dy = side.x - side.ox, side.y - side.oy
    norm = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / norm, dy / norm  # towards the side that is cut
    vx, vy = -uy, ux  # along the dividing line
    mx, my = (side.x + side.ox) / 2, (side.y + side.oy) / 2
    corners = [
        (mx + vx * reach, my + vy * reach),
        (mx + (vx + ux) * reach, my + (vy + uy) * reach),
        (mx + (ux - vx) * reach, my + (uy - vy) * reach),
        (mx - vx * reach, my - vy * reach),
    ]
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in corners)


def _mask(
    mask_id: str,
    regions: Sequence[Region],
    glyph: Sequence[str],
    width: float,
    height: float,
    invert: bool = False,
) -> str:
    """A mask showing ``regions``, or with ``invert`` everything but them."""
    on, off = ("#000", "#fff") if invert else ("#fff", "#000")
    shapes = [f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff" stroke="none"/>'] if invert else []
    for region in regions:
        if isinstance(region, Circle):
            shapes.append(
                f'<circle cx="{region.cx}" cy="{region.cy}" r="{region.r}" fill="{on}" stroke="none"/>'
            )
        elif isinstance(region, Side):
            shapes.append(f'<polygon points="{_half_plane(region)}" fill="{off}" stroke="none"/>')
        else:
            d = _enclosing(glyph, region.x, region.y)
            shapes.append(f'<path d="{d}" fill="{on}" stroke="{on}" stroke-width="{GROW}"/>')
    return (
        f'<mask id="{mask_id}" maskUnits="userSpaceOnUse" x="0" y="0" '
        f'width="{width}" height="{height}">{"".join(shapes)}</mask>'
    )


def _path(d: str, colour: str, extra: str = "") -> str:
    return f'<path d="{d}" stroke="{colour}" fill="{colour}" style="stroke:{colour};fill:{colour}"{extra}/>'


def paint(path: Path) -> bool:
    """Rewrite one tile with a frame path, one path per glyph colour, and its accents."""
    svg = path.read_text(encoding="utf-8")
    ds = _face_ds(svg)
    if not ds:
        return False
    _, _, width, height = _view_box(svg)
    subs = _subpaths(" ".join(ds))
    frame = _frame_indices(subs, width, height)
    if len(frame) != 2:
        raise ValueError(f"{path.name}: found {len(frame)} frame subpaths, wanted 2")

    glyph = [sub for i, sub in enumerate(subs) if i not in frame]
    by_colour: Dict[str, List[str]] = {FRAME: [subs[i] for i in frame]}
    for sub in glyph:
        by_colour.setdefault(_glyph_colour(path.name, sub, height), []).append(sub)

    accented = ACCENTS.get(path.name[:-4], ())
    masks: List[str] = []
    accents: List[str] = []
    for k, (colour, regions) in enumerate(accented):
        mask_id = f"accent-{k}"
        masks.append(_mask(mask_id, regions, glyph, width, height))
        accents.append(_path(" ".join(glyph), colour, f' mask="url(#{mask_id})"'))
    base = ""
    if accented:
        # The suit colour is cut away wherever an accent goes. Left underneath,
        # its antialiased edge would show as a thin rim round every accent.
        every = [region for _, regions in accented for region in regions]
        masks.insert(0, _mask("base", every, glyph, width, height, invert=True))
        base = ' mask="url(#base)"'
    suit_colour = SUIT.get(path.name[:-4][-1], FRAME)

    head = re.sub(r"<defs>.*?</defs>", "", svg[: svg.index("<g")], flags=re.DOTALL)
    defs = f"<defs>{''.join(masks)}</defs>" if masks else ""
    # Keep the group tag exactly as it was. A path's own fill beats the one it
    # would inherit, so the colours below still win, and generate_en_tiles.py,
    # which copies this tag into the English set, keeps producing what it did.
    group = re.search(r"<g[^>]*>", svg).group(0)
    body = "".join(
        _path(" ".join(parts), colour, base if colour == suit_colour and colour != FRAME else "")
        for colour, parts in by_colour.items()
    )
    path.write_text(f"{head}{defs}{group}{body}{''.join(accents)}</g></svg>", encoding="utf-8")
    return True


def main(argv: List[str]) -> int:
    langs = argv[1:] or ["ja"]
    for lang in langs:
        folder = HERE / lang
        if not folder.is_dir():
            print(f"no such tile set: {lang}")
            return 1
        painted = 0
        for tile in sorted(folder.glob("*.svg")):
            if tile.name in SKIP:
                continue
            painted += bool(paint(tile))
        print(f"{lang}: painted {painted} faces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
