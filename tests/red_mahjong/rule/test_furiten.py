import jax
import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.env import RedMahjong, _draw, _kan, _pass, _replace_state
from mahjax.red_mahjong.hand import Hand
from mahjax.red_mahjong.meld import Meld
from mahjax.red_mahjong.state import default_state
from mahjax.red_mahjong.tile import Tile

from ..helpers import STEP_KEY


def _furiten_by_pass_after_declining_then_an_unrelated_pass(m, Action):
    """X declines a winning discard, then passes on an unrelated PON before drawing."""
    base = m.default_state()
    x = 1
    mask = base.players.legal_action_mask
    declined = m._pass(
        m._replace_state(
            base,
            current_player=jnp.int8(x),
            last_player=jnp.int8(0),
            legal_action_mask=mask.at[x, Action.RON].set(True).at[x, Action.PASS].set(True),
        )
    )
    unrelated = m._pass(
        m._replace_state(
            declined,
            current_player=jnp.int8(x),
            last_player=jnp.int8(2),
            legal_action_mask=mask.at[x, Action.PON].set(True).at[x, Action.PASS].set(True),
        )
    )
    return x, declined, unrelated


def test_red_furiten_by_pass_survives_an_unrelated_pass() -> None:
    from mahjax.red_mahjong import env as m

    x, declined, unrelated = _furiten_by_pass_after_declining_then_an_unrelated_pass(m, Action)

    assert bool(declined.players.furiten_by_pass[x])
    assert bool(unrelated.players.furiten_by_pass[x])
    # It still clears on the player own next draw, unless they are in riichi.
    drawn = _draw(_replace_state(unrelated, current_player=jnp.int8(x), draw_next=True))
    assert not bool(drawn.players.furiten_by_pass[x])
    riichi = _draw(
        _replace_state(
            unrelated,
            current_player=jnp.int8(x),
            draw_next=True,
            riichi=unrelated.players.riichi.at[x].set(True),
        )
    )
    assert bool(riichi.players.furiten_by_pass[x])


def test_red_furiten_by_pass_clears_when_a_meld_gives_the_turn() -> None:
    """A pon takes the turn without a draw, so it must clear the furiten by pass."""
    from mahjax.red_mahjong import env as m

    x, declined, _ = _furiten_by_pass_after_declining_then_an_unrelated_pass(m, Action)
    hand = jnp.zeros((37,), dtype=jnp.int8).at[0].set(3).at[1].set(3).at[2].set(3).at[3].set(3)
    ponned = m._pon(
        _replace_state(
            declined,
            current_player=jnp.int8(x),
            last_player=jnp.int8(0),
            target=jnp.int8(0),
            hand_with_red=declined.players.hand_with_red.at[x].set(hand),
            hand=declined.players.hand.at[x].set(Hand.to_34(hand)),
        ),
        jnp.int8(Action.PON),
    )

    assert bool(declined.players.furiten_by_pass[x])
    assert not bool(ponned.players.furiten_by_pass[x])


def test_red_declining_a_robbing_kan_ron_makes_the_player_furiten() -> None:
    """Skipping a ron on an added kan is still skipping a win: furiten, for the round if in riichi."""
    from mahjax.red_mahjong import env as m

    six_s = 23

    def counts(tiles):
        hand = jnp.zeros(Tile.NUM_TILE_TYPE_WITH_RED, dtype=jnp.int8)
        for tile in tiles:
            hand = hand.at[tile].add(1)
        return hand

    filler = counts([27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33])
    hands = jnp.stack(
        [
            filler,
            counts([0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 18, 21, 22]),  # P1: 123m 456m 789p 11s 45s, waits 3s/6s
            counts([six_s, 6, 7, 8, 9, 10, 11, 12, 13, 14, 27]),  # P2 holds the fourth 6s
            filler,
        ]
    )
    hands34 = jax.vmap(Hand.to_34)(hands)
    base = default_state()
    for riichi in (False, True):
        state = _replace_state(
            base,
            current_player=jnp.int8(2),
            hand_with_red=hands,
            hand=hands34,
            can_win=m.v_can_win(hands34, m.TILE_RANGE),
            melds=base.players.melds.at[2, 0].set(Meld.init(Action.PON, six_s, 1)),
            meld_counts=base.players.meld_counts.at[2].set(1),
            pon=base.players.pon.at[2, six_s].set(jnp.int8(1 << 2)),
            riichi=base.players.riichi.at[1].set(riichi),
            next_deck_ix=jnp.int32(60),
        )

        robbable = _kan(state, jnp.int32(Tile.NUM_TILE_TYPE_WITH_RED + six_s))  # P2 adds the 6s
        declined = _pass(robbable)

        assert int(robbable.current_player) == 1
        assert bool(robbable.players.legal_action_mask[1, Action.RON])
        assert bool(declined.players.furiten_by_pass[1])


def test_red_letting_a_winning_tile_without_a_yaku_go_by_makes_the_player_furiten() -> None:
    """Not offered a ron for want of a yaku, the player still let a win go by: no ron until their own draw."""
    from mahjax.red_mahjong import env as m

    six_m, nine_m, one_p, nine_p = 5, 8, 9, 17

    def counts(tiles):
        hand = jnp.zeros(Tile.NUM_TILE_TYPE_WITH_RED, dtype=jnp.int8)
        for tile in tiles:
            hand = hand.at[tile].add(1)
        return hand

    hands = jnp.stack(
        [
            counts([nine_m, one_p, 27, 27, 27, 28, 28, 28, 29, 29, 29, 30, 30, 30]),  # P0 is about to discard
            counts([31, 31, 31, 32, 32, 32, 33, 33, 33, 18, 18, 18, 26]),
            counts([27, 28, 29, 30, 31, 32, 33, 24, 24, 24, 25, 25, 25]),
            counts([1, 2, 3, 13, 13, 13, 20, 21, 22, 23, 23, 6, 7]),  # P3: 234m 555p 345s 66s 78m, dama
        ]
    )  # P3 waits on 6m (tanyao) and 9m (no yaku)
    hands34 = jax.vmap(Hand.to_34)(hands)
    base = default_state()
    # The live wall in draw order: P1 draws 6m, P2 9p, P3 9p, P0 6m.
    deck = base.round_state.deck.at[60].set(six_m).at[59].set(nine_p).at[58].set(nine_p).at[57].set(six_m)
    step = jax.jit(RedMahjong(round_mode="half").step)

    def after_p1_discards_6m(p0_discard):
        state = _replace_state(
            base,
            current_player=jnp.int8(0),
            hand_with_red=hands,
            hand=hands34,
            can_win=m.v_can_win(hands34, m.TILE_RANGE),
            deck=deck,
            next_deck_ix=jnp.int32(60),
            legal_action_mask=base.players.legal_action_mask.at[0, p0_discard].set(True),
        )
        state = step(state, jnp.int32(p0_discard), STEP_KEY)
        assert int(state.current_player) == 1  # nobody is asked about P0's discard
        return step(state, jnp.int32(Action.TSUMOGIRI), STEP_KEY)

    # P0 discards 1p: P1's 6m is P3's ron.
    control = after_p1_discards_6m(one_p)
    assert int(control.current_player) == 3
    assert bool(control.legal_action_mask[Action.RON])

    # P0 discards 9m, a win for P3 with no yaku: P1's 6m in the same go-around is no ron either.
    state = after_p1_discards_6m(nine_m)
    assert not bool(state.players.legal_action_mask[3, Action.RON])
    assert bool(state.players.furiten_by_pass[3])
    assert int(state.current_player) == 2

    # P3's own draw ends it, so P0's 6m after that is a ron again.
    state = step(state, jnp.int32(Action.TSUMOGIRI), STEP_KEY)  # P2 discards 9p, P3 draws 9p
    assert int(state.current_player) == 3
    assert not bool(state.players.furiten_by_pass[3])
    state = step(state, jnp.int32(Action.TSUMOGIRI), STEP_KEY)  # P3 discards 9p, P0 draws 6m
    state = step(state, jnp.int32(Action.TSUMOGIRI), STEP_KEY)  # P0 discards 6m
    assert int(state.current_player) == 3
    assert bool(state.legal_action_mask[Action.RON])
