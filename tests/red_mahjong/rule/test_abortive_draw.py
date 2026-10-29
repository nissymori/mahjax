from __future__ import annotations

import jax
import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.config import GameConfig
from mahjax.red_mahjong.constants import FIRST_DRAW_IDX
from mahjax.red_mahjong.env import RedMahjong, _abortive_draw_normal, _draw, _init, _make_legal_action_mask_after_draw, _replace_state, _ron, _special_next_round, _step
from mahjax.red_mahjong.hand import Hand
from mahjax.red_mahjong.state import default_state
from mahjax.red_mahjong.tile import River, Tile

from ..helpers import STEP_KEY


def test_kyuushu_action_is_enabled_on_first_turn() -> None:
    state = default_state()
    hand_with_red = state.players.hand_with_red.at[0].set(
        jnp.zeros((37,), dtype=jnp.int8)
        .at[jnp.array([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 0])]
        .add(1)
    )
    state = state.replace(players=state.players.replace(hand_with_red=hand_with_red))

    mask = _make_legal_action_mask_after_draw(state, hand_with_red, jnp.int8(0), jnp.int8(0))

    assert bool(mask[Action.KYUUSHU])


def test_kyuushu_action_is_disabled_after_the_first_turn() -> None:
    state = default_state()
    hand_with_red = state.players.hand_with_red.at[0].set(
        jnp.zeros((37,), dtype=jnp.int8)
        .at[jnp.array([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 0])]
        .add(1)
    )
    state = state.replace(
        players=state.players.replace(hand_with_red=hand_with_red),
        round_state=state.round_state.replace(next_deck_ix=jnp.int32(FIRST_DRAW_IDX - 4)),
    )

    mask = _make_legal_action_mask_after_draw(state, hand_with_red, jnp.int8(0), jnp.int8(0))

    assert not bool(mask[Action.KYUUSHU])


def test_kyuushu_action_is_disabled_by_game_config() -> None:
    state = default_state()
    hand_with_red = state.players.hand_with_red.at[0].set(
        jnp.zeros((37,), dtype=jnp.int8)
        .at[jnp.array([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33, 0])]
        .add(1)
    )
    state = state.replace(players=state.players.replace(hand_with_red=hand_with_red))
    config = GameConfig(enable_special_abortive_draw=False)

    mask = _make_legal_action_mask_after_draw(
        state,
        hand_with_red,
        jnp.int8(0),
        jnp.int8(0),
        config,
    )

    assert not bool(mask[Action.KYUUSHU])


def test_four_winds_abortive_draw_sets_kyuushu_mask() -> None:
    state = default_state()
    river = state.players.river
    for player in range(4):
        river = River.add_discard(river, jnp.int8(27), jnp.int8(player), jnp.int8(0), False, False)
    state = state.replace(
        players=state.players.replace(
            river=river,
            discard_counts=jnp.ones((4,), dtype=jnp.int8),
        ),
        round_state=state.round_state.replace(next_deck_ix=jnp.int32(FIRST_DRAW_IDX)),
    )

    next_state = _draw(state)

    assert bool(next_state.legal_action_mask[Action.KYUUSHU])
    assert bool(next_state.players.legal_action_mask[:, Action.KYUUSHU].all())


def test_special_next_round_keeps_dealer_and_increments_honba() -> None:
    state = default_state().replace(
        current_player=jnp.int8(2),
        round_state=default_state().round_state.replace(
            dealer=jnp.int8(2),
            round=jnp.int8(3),
            honba=jnp.int8(2),
            kyotaku=jnp.int8(1),
            score=jnp.array([300, 250, 200, 250], dtype=jnp.int32),
        ),
    )

    next_state = _special_next_round(state, jax.random.PRNGKey(0))

    assert int(next_state.current_player) == 2
    assert int(next_state.round_state.dealer) == 2
    assert int(next_state.round_state.round) == 3
    assert int(next_state.round_state.honba) == 3
    assert int(next_state.round_state.kyotaku) == 1
    assert jnp.array_equal(next_state.round_state.score, jnp.array([300, 250, 200, 250], dtype=jnp.int32))


def test_triple_ron_abandons_the_round_before_anything_is_paid() -> None:
    base = default_state()
    legal_action_mask = base.players.legal_action_mask
    for player in range(3):
        legal_action_mask = legal_action_mask.at[player, Action.RON].set(True)
    ron_state = base.replace(
        current_player=jnp.int8(0),
        legal_action_mask=legal_action_mask[0],
        players=base.players.replace(
            legal_action_mask=legal_action_mask,
            fan=base.players.fan.at[:, 0].set(jnp.int32(5)),
            fu=base.players.fu.at[:, 0].set(jnp.int32(30)),
        ),
        round_state=base.round_state.replace(
            dealer=jnp.int8(3),
            last_player=jnp.int8(3),
            honba=jnp.int8(1),
            kyotaku=jnp.int8(2),
            score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        ),
    )
    config = GameConfig(allow_double_ron=True)

    after_first_ron = _ron(ron_state, config)
    after_second_ron = _ron(after_first_ron.replace(current_player=jnp.int8(1)), config)
    after_third_ron = _ron(after_second_ron.replace(current_player=jnp.int8(2)), config)

    # Nothing is paid while the others are still to answer ...
    assert jnp.array_equal(after_first_ron.round_state.score, ron_state.round_state.score)
    assert jnp.array_equal(after_second_ron.round_state.score, ron_state.round_state.score)
    # ... so the third ron just abandons the round: nobody wins a 三家和, and the
    # riichi sticks stay on the table.
    assert jnp.array_equal(after_third_ron.round_state.score, ron_state.round_state.score)
    assert not bool(after_third_ron.rewards.any())
    assert int(after_third_ron.round_state.kyotaku) == 2
    assert not bool(after_third_ron.players.has_won.any())
    assert not bool(after_third_ron.pending_rewards.any())
    assert bool(after_third_ron.players.legal_action_mask[:, Action.KYUUSHU].all())


def test_abortive_draw_payments_shape() -> None:
    state = default_state()
    state = state.replace(
        players=state.players.replace(
            can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[0].set(True)
        )
    )
    next_state = _abortive_draw_normal(state)
    assert bool(next_state.round_state.terminated_round)
    assert next_state.rewards.shape == (4,)


def test_red_a_wait_only_on_a_fifth_copy_is_noten_at_an_exhaustive_draw() -> None:
    """66 7777 99m EEE CC only 'waits' on a fifth 7m: nobody is tenpai, so nobody pays and the deal passes."""
    from mahjax.red_mahjong import env as m

    def counts(tiles):
        hand = jnp.zeros(Tile.NUM_TILE_TYPE_WITH_RED, dtype=jnp.int8)
        for tile in tiles:
            hand = hand.at[tile].add(1)
        return hand

    north = 30
    noten = counts([0, 2, 4, 9, 11, 13, 15, 17, 18, 20, 22, 24, 26])  # 135m 13579p 13579s
    dealer_hand = counts([5, 5, 6, 6, 6, 6, 8, 8, 27, 27, 27, 33, 33, north])  # 66 7777 99m EEE CC + N
    env = RedMahjong(round_mode="half")
    state = env.init(jax.random.PRNGKey(0))
    dealer = int(state.round_state.dealer)
    hands = jnp.stack([dealer_hand if p == dealer else noten for p in range(4)])
    hands34 = jax.vmap(Hand.to_34)(hands)
    state = _replace_state(
        state,
        current_player=jnp.int8(dealer),
        hand_with_red=hands,
        hand=hands34,
        can_win=m.v_can_win(hands34, m.TILE_RANGE),
        last_draw=jnp.int8(north),
        next_deck_ix=(state.round_state.last_deck_ix - 1).astype(jnp.int32),  # N was the last live tile
        is_haitei=jnp.bool_(True),
        has_nagashi_mangan=jnp.zeros(4, dtype=jnp.bool_),  # nobody is on course for a nagashi mangan
        legal_action_mask=jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_).at[dealer, north].set(True),
    )

    out = env.step(state, jnp.int32(north), STEP_KEY)

    assert bool(jnp.all(out.rewards == 0))
    assert int(out.round_state.dealer) == (dealer + 1) % 4
    assert int(out.round_state.honba) == 1


def test_red_four_kan_abortive_draw_after_a_passed_ron_keeps_the_riichi_stick() -> None:
    """After the fourth kan only a ron can take the discard; unclaimed, the round is drawn and a riichi on it stands."""
    from mahjax.red_mahjong import env as m

    five_s = 22

    def counts(tiles):
        hand = jnp.zeros(Tile.NUM_TILE_TYPE_WITH_RED, dtype=jnp.int8)
        for tile in tiles:
            hand = hand.at[tile].add(1)
        return hand

    def after_the_fourth_kan(p3_waits_on_5s: bool):
        hands = jnp.stack(
            [
                counts([five_s, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),  # P0 discards 5s declaring riichi
                counts([five_s, five_s, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32]),  # P1 could pon it
                counts([13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 33]),
                counts([0, 1, 2, 3, 4, 5, 15, 16, 17, 19, 20, 21, five_s if p3_waits_on_5s else 33]),  # P3 in riichi
            ]
        )
        hands34 = jax.vmap(Hand.to_34)(hands)
        return _replace_state(
            _init(jax.random.PRNGKey(3)),
            current_player=jnp.int8(0),
            dealer=jnp.int8(0),
            last_player=jnp.int8(3),
            hand_with_red=hands,
            hand=hands34,
            can_win=m.v_can_win(hands34, m.TILE_RANGE),
            n_kan=jnp.array([2, 2, 0, 0], dtype=jnp.int8),
            can_after_kan=jnp.bool_(True),
            riichi=jnp.array([False, False, False, True]),
            riichi_declared=jnp.array([True, False, False, False]),
            score=jnp.array([250, 250, 250, 240], dtype=jnp.int32),
            kyotaku=jnp.int8(1),
            next_deck_ix=jnp.int32(60),
        )

    step = jax.jit(_step)
    kyuushu_only = jnp.zeros((Action.NUM_ACTION,), dtype=jnp.bool_).at[Action.KYUUSHU].set(True)

    # Nobody can ron: drawn at once.
    drawn = step(after_the_fourth_kan(False), jnp.int8(five_s), STEP_KEY)
    # P3 can ron: only RON is offered, not P1's pon, and passing it draws the round.
    asked = step(after_the_fourth_kan(True), jnp.int8(five_s), STEP_KEY)
    assert int(asked.current_player) == 3
    assert {int(a) for a in jnp.flatnonzero(asked.legal_action_mask)} == {Action.RON, Action.PASS}
    assert not bool(asked.players.legal_action_mask[1, Action.PON])
    drawn_after_pass = step(asked, jnp.int8(Action.PASS), STEP_KEY)

    for state in (drawn, drawn_after_pass):
        assert jnp.array_equal(state.legal_action_mask, kyuushu_only)
        # P0's riichi on that discard stands: the stick goes on the table.
        assert bool(state.players.riichi[0])
        assert int(state.round_state.kyotaku) == 2
        assert int(state.round_state.score[0]) == 240
