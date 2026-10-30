import jax.numpy as jnp

from mahjax.common.action import Action
from mahjax.common.hand import Hand
from mahjax.common.tile import Tile
from mahjax.mahjong_4p.logic import (
    _claims_open_to,
    _advance_to_next_decision,
    _mask_for_chi,
    _next_meld_player,
    _next_ron_player,
    _replace_state,
)
from mahjax.mahjong_4p.state import default_state

from ..helpers import STEP_KEY, compiled_env


def test_mask_for_chi_forbids_forced_kuikae() -> None:
    # Regression test for issue #61: a CHI that leaves no legal discard after
    # kuikae (swap-calling) restrictions must not be offered as a legal action,
    # otherwise executing it produces an active state with an all-false mask.
    # Hand 1123444m (with two open melds held elsewhere). Calling 1m or 4m
    # consumes 2m3m and leaves only 1m/4m, all forbidden as discards by kuikae.
    hand = jnp.zeros(Tile.NUM_TILE_TYPE_WITH_RED, dtype=jnp.int8)
    hand = hand.at[0].set(2).at[1].set(1).at[2].set(1).at[3].set(3)  # 1123444m
    chi_slice = slice(Action.CHI_L, Action.CHI_R_RED + 1)
    # Chi 1m: only CHI_L (1m+23m) is possible; every remaining discard is kuikae.
    mask_1m = _mask_for_chi(hand, jnp.int8(0))
    assert not bool(jnp.any(mask_1m[chi_slice]))
    # Chi 4m: only CHI_R (234m) is possible; every remaining discard is kuikae.
    mask_4m = _mask_for_chi(hand, jnp.int8(3))
    assert not bool(jnp.any(mask_4m[chi_slice]))
    # Sanity: with a spare 9m in hand, chi 1m leaves a legal discard and is allowed.
    hand_ok = hand.at[3].set(2).at[8].set(1)  # 1123449m
    mask_ok = _mask_for_chi(hand_ok, jnp.int8(0))
    assert bool(mask_ok[Action.CHI_L])


def test_red_pon_of_a_five_forbids_both_copies_as_the_next_discard() -> None:
    """Kuikae after a pon is by tile type: calling a black five forbids the red one, and vice versa."""
    from mahjax.mahjong_4p import logic as m

    others = jnp.zeros(Tile.NUM_TILE_TYPE_WITH_RED, dtype=jnp.int8).at[9:17].set(1).at[18:20].set(1)  # 1p-8p 1s 2s
    base = default_state()
    for held, target in (
        (others.at[4].set(2).at[34].set(1), 4),  # holding 5m 5m 0m, a black 5m is called
        (others.at[4].set(3), 34),  # holding 5m 5m 5m, the red 0m is called
    ):
        state = m._replace_state(
            base,
            current_player=jnp.int8(1),
            last_player=jnp.int8(0),
            target=jnp.int8(target),
            hand_with_red=base.players.hand_with_red.at[1].set(held),
            hand=base.players.hand.at[1].set(Hand.to_34(held)),
        )

        mask = m._pon(state, jnp.int8(Action.PON)).legal_action_mask

        assert not bool(mask[4]) and not bool(mask[34])
        assert bool(mask[9])


def test_next_meld_player_prioritizes_ron_then_distance() -> None:
    legal = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    legal = legal.at[0, Action.PON].set(True)
    legal = legal.at[1, Action.RON].set(True)
    legal = legal.at[3, Action.RON].set(True)
    nxt, can_any = _next_meld_player(legal, jnp.int8(0))
    assert bool(can_any)
    assert int(nxt) == 1


def test_next_ron_player_returns_closest() -> None:
    legal = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    legal = legal.at[1, Action.RON].set(True)
    legal = legal.at[2, Action.RON].set(True)
    nxt, can_any = _next_ron_player(legal, jnp.int8(0))
    assert bool(can_any)
    assert int(nxt) == 1


def test_claims_open_to_holds_a_call_while_another_player_claims_higher() -> None:
    legal = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    ron_pon = legal.at[1, Action.RON].set(True).at[1, Action.PON].set(True).at[1, Action.PASS].set(True)
    # Nobody else claims: RON is still answered before the PON.
    offered = _claims_open_to(ron_pon, jnp.int8(1))
    assert {int(a) for a in jnp.flatnonzero(offered)} == {Action.RON, Action.PASS}
    # Another player can RON: the PON waits.
    offered = _claims_open_to(ron_pon.at[2, Action.RON].set(True), jnp.int8(1))
    assert bool(offered[Action.RON]) and bool(offered[Action.PASS])
    assert not bool(offered[Action.PON])
    # Another player can PON: the CHI waits, the RON does not.
    ron_chi = legal.at[1, Action.RON].set(True).at[1, Action.CHI_R].set(True).at[2, Action.PON].set(True)
    offered = _claims_open_to(ron_chi, jnp.int8(1))
    assert bool(offered[Action.RON])
    assert not bool(offered[Action.CHI_R])


def test_red_a_lower_call_waits_until_every_higher_claim_is_passed() -> None:
    """P0 has just discarded; each prompt is (player, offered calls) while everyone passes."""
    env = compiled_env(round_mode="single", round_end="auto")
    scenarios = [
        # Nothing conflicts: RON is still answered before the PON.
        ({1: [Action.RON, Action.PON]}, [(1, {Action.RON}), (1, {Action.PON})]),
        # P1 must not be able to pon over P2's ron.
        (
            {1: [Action.RON, Action.PON], 2: [Action.RON]},
            [(1, {Action.RON}), (2, {Action.RON}), (1, {Action.PON})],
        ),
        # P1 must not be able to chi over P2's pon.
        (
            {1: [Action.RON, Action.CHI_R], 2: [Action.PON]},
            [(1, {Action.RON}), (2, {Action.PON}), (1, {Action.CHI_R})],
        ),
    ]
    for rows, prompts in scenarios:
        base = default_state()
        legal = base.players.legal_action_mask.at[1, Action.PASS].set(True)
        for player, actions in rows.items():
            for action in actions:
                legal = legal.at[player, action].set(True)
        state = _advance_to_next_decision(
            _replace_state(
                base,
                current_player=jnp.int8(1),
                last_player=jnp.int8(0),
                target=jnp.int8(1),
                legal_action_mask=legal,
            )
        )
        for i, (player, offered) in enumerate(prompts):
            if i:
                state = env.step(state, jnp.int8(Action.PASS), STEP_KEY)
            assert int(state.current_player) == player
            assert {int(a) for a in jnp.flatnonzero(state.legal_action_mask)} == offered | {Action.PASS}


def test_double_ron_priority_is_left_to_discarder() -> None:
    legal = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    legal = legal.at[0, Action.RON].set(True)
    legal = legal.at[2, Action.RON].set(True)
    nxt, can_any = _next_meld_player(legal, jnp.int8(1))
    assert bool(can_any)
    assert int(nxt) == 2
