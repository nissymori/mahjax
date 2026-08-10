import jax
import jax.numpy as jnp

import mahjax
from mahjax.hong_kong_mahjong.action import Action
from mahjax.hong_kong_mahjong.env import (
    _claim_masks,
    _draw_nonflower,
    _finish_round,
    _mask_after_draw,
    _next_claimant,
    _replace_state,
    _self_kong,
)
from mahjax.hong_kong_mahjong.meld import Meld
from mahjax.hong_kong_mahjong.tile import Tile


def test_public_factory_and_initial_deal_are_jittable():
    env = mahjax.make("hong_kong_mahjong", round_mode="single")
    state = jax.jit(env.init)(jax.random.PRNGKey(7))
    dealer = int(state.round_state.dealer)
    expected = jnp.full(4, 13).at[dealer].set(14)
    assert state.env_id == "hong_kong_mahjong"
    assert state.round_state.deck.shape == (144,)
    assert jnp.array_equal(state.players.hand.sum(axis=1), expected)
    assert state.players.flowers.sum() + state.players.hand.sum() == state.round_state.wall_index
    assert state.players.flowers.sum(axis=0).max() <= 1
    assert state.legal_action_mask[Action.TSUMOGIRI]


def test_jitted_discard_advances_game():
    env = mahjax.make("hong_kong_mahjong", round_mode="single")
    state = jax.jit(env.init)(jax.random.PRNGKey(11))
    player = state.current_player
    before = state.players.hand[player].sum()
    state = jax.jit(env.step)(state, jnp.int32(Action.TSUMOGIRI))
    assert state.step_count == 1
    assert state.players.hand[player].sum() == before - 1
    assert state.players.discard_counts[player] == 1


def test_nearest_winner_has_priority_and_only_one_winner_is_selected():
    masks = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    masks = masks.at[0, Action.RON].set(True).at[2, Action.RON].set(True)
    claimant, can_claim = _next_claimant(masks, jnp.int8(3))
    assert can_claim
    assert claimant == 0


def test_win_claim_precedes_kong_pon_and_chi():
    masks = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    masks = masks.at[0, Action.CHI_L].set(True)
    masks = masks.at[1, Action.PON].set(True)
    masks = masks.at[2, Action.OPEN_KAN].set(True)
    masks = masks.at[3, Action.RON].set(True)
    claimant, _ = _next_claimant(masks, jnp.int8(0))
    assert claimant == 3


def test_dealer_repeats_after_draw():
    env = mahjax.make("hong_kong_mahjong", round_mode="east")
    state = env.init(jax.random.PRNGKey(5))
    dealer = state.round_state.dealer
    ended = _replace_state(
        state,
        wall_index=jnp.int16(Tile.NUM_TILE_ID),
        terminated_round=jnp.bool_(True),
        legal_action_mask=jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_),
    )
    restarted = _finish_round(ended, env.rules)
    assert restarted.round_state.dealer == dealer
    assert restarted.round_state.round == 0


def test_dealer_repeats_after_dealer_win():
    env = mahjax.make("hong_kong_mahjong", round_mode="east")
    state = env.init(jax.random.PRNGKey(9))
    dealer = state.round_state.dealer
    ended = _replace_state(
        state,
        has_won=state.players.has_won.at[dealer].set(True),
        terminated_round=jnp.bool_(True),
        legal_action_mask=jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_),
    )
    restarted = _finish_round(ended, env.rules)
    assert restarted.round_state.dealer == dealer
    assert restarted.round_state.round == 0


def test_flower_draws_are_recorded_and_replaced_until_standard_tile():
    deck = jnp.zeros(144, dtype=jnp.int8).at[0].set(34).at[1].set(39).at[2].set(7)
    hand = jnp.zeros((4, 34), dtype=jnp.int8)
    flowers = jnp.zeros((4, 8), dtype=jnp.bool_)
    ix, hand, flowers, drawn = _draw_nonflower(deck, jnp.int16(0), hand, flowers, jnp.int8(2))
    assert ix == 3
    assert drawn == 7
    assert hand[2, 7] == 1
    assert flowers[2, 0]
    assert flowers[2, 5]


def test_closed_kong_draws_replacement_tile():
    env = mahjax.make("hong_kong_mahjong", round_mode="single")
    state = env.init(jax.random.PRNGKey(21))
    player = state.current_player
    ix = state.round_state.wall_index
    hand = jnp.zeros_like(state.players.hand).at[player, 0].set(4)
    deck = state.round_state.deck.at[ix].set(6)
    state = _replace_state(state, hand=hand, deck=deck)
    konged = _self_kong(state, jnp.int32(34), env.rules)
    assert konged.players.meld_counts[player] == 1
    assert konged.players.hand[player, 0] == 0
    assert konged.players.hand[player, 6] == 1
    assert konged.round_state.wall_index == ix + 1
    assert konged.round_state.after_kong


def test_added_kong_can_be_robbed_by_nearest_legal_winner():
    env = mahjax.make("hong_kong_mahjong", round_mode="single")
    state = env.init(jax.random.PRNGKey(22))
    player = jnp.int8(0)
    winner = jnp.int8(1)
    waiting = jnp.zeros(34, dtype=jnp.int8).at[0].set(1).at[jnp.array([1, 11, 12, 22])].set(3)
    hands = jnp.zeros_like(state.players.hand).at[player, 0].set(1).at[winner].set(waiting)
    melds = state.players.melds.at[player, 0].set(Meld.init(Action.PON, jnp.int8(0), jnp.int8(3)))
    state = _replace_state(
        state,
        current_player=player,
        hand=hands,
        melds=melds,
        meld_counts=state.players.meld_counts.at[player].set(1),
        pon=state.players.pon.at[player, 0].set(1),
        discard_counts=state.players.discard_counts.at[2].set(1),
    )
    robbed = _self_kong(state, jnp.int32(34), env.rules)
    assert robbed.round_state.robbing_kong
    assert robbed.current_player == winner
    assert robbed.legal_action_mask[Action.RON]
    assert robbed.legal_action_mask[Action.PASS]


def test_three_faan_minimum_controls_ron_and_self_draw_masks():
    env = mahjax.make("hong_kong_mahjong", round_mode="single")
    state = env.init(jax.random.PRNGKey(23))
    player = jnp.int8(1)
    pre_win = jnp.zeros(34, dtype=jnp.int8).at[jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12])].set(1).at[22].set(1)
    hands = jnp.zeros_like(state.players.hand).at[player].set(pre_win)
    state = _replace_state(
        state,
        current_player=player,
        hand=hands,
        dealer=jnp.int8(3),
        last_player=jnp.int8(0),
        discard_counts=state.players.discard_counts.at[0].set(2),
    )
    claims = _claim_masks(state, jnp.int8(0), jnp.int8(22), env.rules)
    assert not claims[player, Action.RON]

    complete = hands.at[player, 22].add(1)
    state = _replace_state(state, hand=complete, last_draw=jnp.int8(22))
    draw_mask = _mask_after_draw(state, player, jnp.int8(22), env.rules)
    assert draw_mask[Action.TSUMO]


def test_self_draw_step_settles_all_opponents_and_terminates_single_hand():
    env = mahjax.make("hong_kong_mahjong", round_mode="single")
    state = env.init(jax.random.PRNGKey(24))
    winner = jnp.int8(1)
    pre_win = jnp.zeros(34, dtype=jnp.int8).at[jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12])].set(1).at[22].set(1)
    complete = pre_win.at[22].add(1)
    state = _replace_state(
        state,
        current_player=winner,
        dealer=jnp.int8(3),
        hand=jnp.zeros_like(state.players.hand).at[winner].set(complete),
        last_draw=jnp.int8(22),
        discard_counts=state.players.discard_counts.at[0].set(2),
        legal_action_mask=jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_).at[winner, Action.TSUMO].set(True),
    )
    won = jax.jit(env.step)(state, jnp.int32(Action.TSUMO))
    assert won.terminated
    assert jnp.array_equal(won.rewards, jnp.array([-8, 32, -8, -16], dtype=jnp.float32))
    assert jnp.array_equal(won.round_state.score, won.rewards.astype(jnp.int32))
    assert won.rewards.sum() == 0


def test_discard_win_step_charges_only_discarder():
    env = mahjax.make("hong_kong_mahjong", round_mode="single")
    state = env.init(jax.random.PRNGKey(25))
    winner = jnp.int8(1)
    # Dragon pung + concealed + no flowers = exactly three faan.
    waiting = (
        jnp.zeros(34, dtype=jnp.int8).at[33].set(3).at[jnp.array([0, 1, 2, 3, 4, 5, 15, 16, 17])].set(1).at[22].set(1)
    )
    state = _replace_state(
        state,
        current_player=winner,
        dealer=jnp.int8(3),
        hand=jnp.zeros_like(state.players.hand).at[winner].set(waiting),
        last_player=jnp.int8(0),
        target=jnp.int8(22),
        discard_counts=state.players.discard_counts.at[0].set(2),
        legal_action_mask=jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_).at[winner, Action.RON].set(True),
    )
    won = jax.jit(env.step)(state, jnp.int32(Action.RON))
    assert won.terminated
    assert jnp.array_equal(won.rewards, jnp.array([-8, 8, 0, 0], dtype=jnp.float32))
    assert won.rewards.sum() == 0


def test_non_dealer_win_advances_dealer_and_round():
    env = mahjax.make("hong_kong_mahjong", round_mode="east")
    state = env.init(jax.random.PRNGKey(26))
    dealer = state.round_state.dealer
    winner = (dealer + 1) % 4
    ended = _replace_state(
        state,
        has_won=state.players.has_won.at[winner].set(True),
        terminated_round=jnp.bool_(True),
        legal_action_mask=jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_),
    )
    restarted = _finish_round(ended, env.rules)
    assert restarted.round_state.dealer == (dealer + 1) % 4
    assert restarted.round_state.round == 1
