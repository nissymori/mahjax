from mahjax._src.types import Array, PRNGKey
from mahjax.common.action import Action
from mahjax.common.config import GameConfig
from mahjax.common.hand import Hand
from mahjax.common.meld import Meld
from mahjax.common.tile import River, Tile
from mahjax.common.yaku import Yaku
from mahjax.core import Env, EnvId, available_envs, make, rule_based_player

# rule_based_player imports these lazily, so they are imported here, before anything
# is traced: their constants are jnp arrays, and a module first imported inside a
# trace keeps that trace's tracers, which the next trace trips over.
from mahjax.mahjong_3p import players as _players_3p  # noqa: F401
from mahjax.mahjong_4p import players as _players_4p  # noqa: F401
from mahjax.mahjong_4p.config import NO_RED, RED
from mahjax.mahjong_4p.state import State, default_state

__version__ = "0.3.2"

__all__ = [
    # types
    "Array",
    "PRNGKey",
    "Action",
    "GameConfig",
    "RED",
    "NO_RED",
    "default_state",
    # v1 api components
    "State",
    "Env",
    "EnvId",
    "make",
    "available_envs",
    # main mahjong api
    "Tile",
    "River",
    "Hand",
    "Meld",
    "Yaku",
    "rule_based_player",
]
