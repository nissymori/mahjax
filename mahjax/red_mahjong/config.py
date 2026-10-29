from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class GameConfig:
    """Which optional rules are in play.

    The fields are plain Python bools, not traced values: the env holds one config
    and every function it compiles is built for that config.
    """

    use_red_fives: bool = True
    allow_double_ron: bool = True
    enable_special_abortive_draw: bool = True
    enable_pao: bool = True

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not isinstance(value, bool):
                raise TypeError(f"GameConfig.{field.name} must be a Python bool, got {type(value).__name__}")


RED = GameConfig()
NO_RED = GameConfig(
    use_red_fives=False,
    allow_double_ron=False,
    enable_special_abortive_draw=False,
    enable_pao=False,
)
