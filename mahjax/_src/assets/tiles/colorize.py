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
    python -m mahjax._src.assets.tiles.colorize ja en      # or name the sets

The faces are drawn as one ``<path>`` holding every subpath: the rounded frame
(an outer rectangle and an inset one, filled evenodd so only the ring shows)
followed by the character. One path means one colour, which is why the whole
set reads as black on white. Splitting the frame from the character lets the
character take the suit's colour while the frame stays ink, and a manzu tile
can keep a black numeral above its red 萬 the way a real tile does.

The colours are the ones ``generate_en_tiles.py`` already paints the English
set with, so the two sets agree. That set needs nothing from here: it is built
with its numerals already coloured.

The split is derived from the geometry every time, so running this twice is the
same as running it once.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

HERE = Path(__file__).resolve().parent

#: The frame is drawn in ink whatever the face says, so a red five reads as a
#: red character and not as a red tile.
FRAME = "#000"
RED = "#c62828"  # 萬, 中, and the red fives
GREEN = "#16703c"  # bamboo, and 發
BLUE = "#1f3f99"  # the circles of a pin tile
SUIT = {"m": FRAME, "p": BLUE, "s": GREEN}

#: Backs and the dealer marker are not faces and keep the colour they have.
SKIP = {"b.svg", "back.svg", "oya.svg"}

#: A manzu numeral sits in the top of the tile and 萬 fills the rest. Measured:
#: the lowest numeral stroke ends at 0.44 of the height, the highest 萬 stroke
#: starts at 0.43, so the midpoint of a subpath separates them cleanly.
MAN_SPLIT = 0.42

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


def paint(path: Path) -> bool:
    """Rewrite one tile with a frame path and one path per glyph colour."""
    svg = path.read_text(encoding="utf-8")
    ds = re.findall(r'\sd="([^"]*)"', svg)
    if not ds:
        return False
    _, _, width, height = _view_box(svg)
    subs = _subpaths(" ".join(ds))
    frame = _frame_indices(subs, width, height)
    if len(frame) != 2:
        raise ValueError(f"{path.name}: found {len(frame)} frame subpaths, wanted 2")

    by_colour: Dict[str, List[str]] = {FRAME: [subs[i] for i in frame]}
    for i, sub in enumerate(subs):
        if i in frame:
            continue
        by_colour.setdefault(_glyph_colour(path.name, sub, height), []).append(sub)

    head = svg[: svg.index("<g")]
    # Keep the group tag exactly as it was. A path's own fill beats the one it
    # would inherit, so the colours below still win, and generate_en_tiles.py,
    # which copies this tag into the English set, keeps producing what it did.
    group = re.search(r"<g[^>]*>", svg).group(0)
    body = "".join(
        f'<path d="{" ".join(parts)}" stroke="{colour}" fill="{colour}" '
        f'style="stroke:{colour};fill:{colour}"/>'
        for colour, parts in by_colour.items()
    )
    path.write_text(f"{head}{group}{body}</g></svg>", encoding="utf-8")
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
