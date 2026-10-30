"""The rule presets of the 4-player game."""

from __future__ import annotations

from mahjax.common.config import GameConfig

RED = GameConfig()
NO_RED = GameConfig(
    use_red_fives=False,
    allow_double_ron=False,
    enable_special_abortive_draw=False,
    enable_pao=False,
)
