import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.meld import Meld
from mahjax.red_mahjong.tile import Tile


def test_red_pon_meld_has_red_metadata() -> None:
    meld = Meld.init(Action.PON_RED, Tile.BLACK_FIVE["m"], 1)
    assert bool(Meld.is_pon(meld))
    assert bool(Meld.contains_red(meld))
    assert int(Meld.target(meld)) == Tile.BLACK_FIVE["m"]
    assert int(Meld.target_tile(meld)) == Tile.BLACK_FIVE["m"]


def test_red_chi_meld_keeps_action_and_target() -> None:
    meld = Meld.init(Action.CHI_M_RED, Tile.BLACK_FIVE["p"], 3)
    assert bool(Meld.is_chi(meld))
    assert bool(Meld.contains_red(meld))
    assert int(Meld.action(meld)) == Action.CHI_M_RED
    assert int(Meld.target(meld)) == Tile.BLACK_FIVE["p"]
    assert int(Meld.src(meld)) == 3


def test_open_kan_on_red_target_tracks_base_tile() -> None:
    meld = Meld.init(Action.OPEN_KAN, Tile.RED_FIVE["s"], 2)
    assert bool(Meld.is_kan(meld))
    assert int(Meld.target(meld)) == Tile.BLACK_FIVE["s"]
    assert int(Meld.target_tile(meld)) == Tile.RED_FIVE["s"]
    assert bool(Meld.contains_red(meld))


def test_empty_meld_roundtrip() -> None:
    empty = Meld.empty()
    assert bool(Meld.is_empty(empty))
    assert int(Meld.target(empty)) == -1
    assert int(Meld.src(empty)) == -1
    assert int(Meld.action(empty)) == -1
    assert int(Meld.fu(empty)) == 0


def test_chi_prohibitive_tile_type_behavior() -> None:
    assert int(Meld.prohibitive_tile_type_after_chi(Action.CHI_L, 0)) == 3
    assert int(Meld.prohibitive_tile_type_after_chi(Action.CHI_R, 8)) == 5
    assert int(Meld.prohibitive_tile_type_after_chi(Action.CHI_L, 6)) == -1
