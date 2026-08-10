"""Hong Kong Old Style mahjong rules and environment."""

from mahjax.hong_kong_mahjong.env import HongKongMahjong
from mahjax.hong_kong_mahjong.hand import Hand
from mahjax.hong_kong_mahjong.rules import HKOS_V1, Rules
from mahjax.hong_kong_mahjong.scoring import FaanResult, HongKongScoring
from mahjax.hong_kong_mahjong.tile import Tile

__all__ = ["FaanResult", "HKOS_V1", "Hand", "HongKongMahjong", "HongKongScoring", "Rules", "Tile"]
