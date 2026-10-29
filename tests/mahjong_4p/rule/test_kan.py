from __future__ import annotations

import jax
import jax.numpy as jnp

from mahjax.mahjong_4p.action import Action
from mahjax.mahjong_4p.logic import _draw_rinshan, _init, _kan, _replace_state, _step
from mahjax.mahjong_4p.hand import Hand
from mahjax.mahjong_4p.meld import Meld
from mahjax.mahjong_4p.state import default_state
from mahjax.mahjong_4p.tile import Tile

from ..helpers import STEP_KEY, _post_ankan_state, compiled_env


def test_draw_after_kan_advances_only_the_dead_wall_boundary():
    """The premise the rinshan count rests on: 王牌繰り moves ``last_deck_ix``
    by exactly one and never touches ``next_deck_ix``."""
    state, _ = _post_ankan_state(6)
    out = _draw_rinshan(state)
    assert int(out.round_state.last_deck_ix) == int(state.round_state.last_deck_ix) + 1
    assert int(out.round_state.next_deck_ix) == int(state.round_state.next_deck_ix)


def test_draw_after_kan_advances_nothing_when_the_kan_dora_was_pre_flipped():
    """暗槓 reveals the kan dora inside ``_kan``, which already advanced
    ``last_deck_ix`` there; ``_draw_rinshan`` must not advance it again."""
    state, _ = _post_ankan_state(6, kan_dora_pre_flipped=True)
    out = _draw_rinshan(state)
    assert int(out.round_state.last_deck_ix) == int(state.round_state.last_deck_ix)
    assert int(out.round_state.next_deck_ix) == int(state.round_state.next_deck_ix)


def test_open_and_added_kan_dora_turns_over_at_the_discard():
    """A closed kan's new indicator shows at once; an open or added kan's only at the
    kanner's discard or next kan, so it is hidden while deciding on the rinshan tile."""
    from mahjax.mahjong_4p import logic as m
    from mahjax.mahjong_4p.meld import Meld
    from mahjax.mahjong_4p.tile import Tile

    kan = jax.jit(m._kan)
    discard = jax.jit(m._discard)
    draw_after_kan = jax.jit(_draw_rinshan)
    north, west, east = 30, 29, 27

    def counts(tiles):
        hand = jnp.zeros(37, dtype=jnp.int8)
        for tile in tiles:
            hand = hand.at[tile].add(1)
        return hand

    def revealed(state):
        return int((state.round_state.dora_indicators >= 0).sum())

    base = _init(jax.random.PRNGKey(2))
    others = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # Open kan: P0 discarded North and P1 holds NNN.
    open_kan = _replace_state(
        base,
        current_player=jnp.int8(1),
        last_player=jnp.int8(0),
        target=jnp.int8(north),
        hand_with_red=base.players.hand_with_red.at[1].set(counts([north] * 3 + others)),
        hand=base.players.hand.at[1].set(Hand.to_34(counts([north] * 3 + others))),
        n_kan=jnp.zeros(4, dtype=jnp.int8),
        discard_counts=base.players.discard_counts.at[0].set(1),
    )
    # Added kan: P1 has a pon of West and holds the fourth.
    added_kan = _replace_state(
        base,
        current_player=jnp.int8(1),
        hand_with_red=base.players.hand_with_red.at[1].set(counts([west] + others)),
        hand=base.players.hand.at[1].set(Hand.to_34(counts([west] + others))),
        n_kan=jnp.zeros(4, dtype=jnp.int8),
        melds=base.players.melds.at[1, 0].set(Meld.init(Action.PON, west, 1)),
        meld_counts=base.players.meld_counts.at[1].set(1),
        pon=base.players.pon.at[1, west].set(jnp.int8(1 << 2)),
    )

    for state, action in ((open_kan, Action.OPEN_KAN), (added_kan, Tile.NUM_TILE_TYPE_WITH_RED + west)):
        rinshan = draw_after_kan(kan(state, jnp.int32(action)))
        assert revealed(rinshan) == 1  # still only the round's first indicator
        assert revealed(discard(rinshan, jnp.int8(Action.TSUMOGIRI))) == 2

    # A closed kan right after the open kan turns over both indicators before its rinshan draw.
    rinshan = draw_after_kan(kan(open_kan, jnp.int32(Action.OPEN_KAN)))
    hand = rinshan.players.hand_with_red[1].at[east].set(4)
    second = _replace_state(
        rinshan,
        hand_with_red=rinshan.players.hand_with_red.at[1].set(hand),
        hand=rinshan.players.hand.at[1].set(Hand.to_34(hand)),
    )
    assert revealed(draw_after_kan(kan(second, jnp.int32(Tile.NUM_TILE_TYPE_WITH_RED + east)))) == 3


def test_a_robbed_added_kan_is_scored_before_the_previous_kans_indicator_turns_over():
    """Tenhou turns an open or added kan's indicator over at the kanner's discard or
    just before the next rinshan draw. When a second kan in a row is an added kan
    that is robbed, that draw never comes: the first kan's indicator stays face down,
    for the robbing player's view and for the ron's score. If nobody robs it, the
    indicator turns over before the rinshan draw and counts for a rinshan tsumo."""
    from mahjax.mahjong_4p import logic as m
    from mahjax.mahjong_4p.meld import Meld
    from mahjax.mahjong_4p.tile import Tile

    kan = jax.jit(m._kan)
    draw_after_kan = jax.jit(_draw_rinshan)
    west, seven_p, one_s, eight_m, eight_p, north, hatsu = 29, 15, 18, 7, 16, 30, 32

    def counts(tiles):
        hand = jnp.zeros(37, dtype=jnp.int8)
        for tile in tiles:
            hand = hand.at[tile].add(1)
        return hand

    def revealed(state):
        return int((state.round_state.dora_indicators >= 0).sum())

    # P0 waits on 4p-7p for tanyao. P1 has ponned West and 7p, holds the fourth of
    # each and 111m 999m, and draws hatsu from both rinshan draws: the second is a
    # rinshan tsumo (toitoi).
    others = [9, 10, 11, 16, 17, 22, 24, 25, 26, 27, 28, 30, 31]
    hands = jnp.stack([
        counts([19, 20, 21, 3, 4, 5, 7, 7, 7, 23, 23, 13, 14]),
        counts([west, seven_p, 0, 0, 0, 8, 8, 8]),
        counts(others),
        counts(others),
    ])
    base = _init(jax.random.PRNGKey(2))
    melds = base.players.melds.at[1, 0].set(Meld.init(Action.PON, west, 2))
    melds = melds.at[1, 1].set(Meld.init(Action.PON, seven_p, 2))

    def before_the_kans(first_kan_indicator):
        # The first kan's indicator is deck[7]: 1s makes P0's 2s a dora, 8m makes
        # P1's 999m dora, 8p does nothing.
        deck = base.round_state.deck.at[9].set(north).at[7].set(first_kan_indicator)
        deck = deck.at[10].set(hatsu).at[11].set(hatsu)
        hands34 = jax.vmap(Hand.to_34)(hands)
        return _replace_state(
            base,
            current_player=jnp.int8(1),
            deck=deck,
            dora_indicators=jnp.array([north, -1, -1, -1, -1], dtype=jnp.int8),
            ura_dora_indicators=jnp.array([deck[8], -1, -1, -1, -1], dtype=jnp.int8),
            n_kan_doras=jnp.int8(0),
            hand_with_red=hands,
            hand=hands34,
            can_win=m.v_can_win(hands34, m.TILE_RANGE),
            furiten_by_discard=jnp.zeros(4, dtype=jnp.bool_),
            furiten_by_pass=jnp.zeros(4, dtype=jnp.bool_),
            n_kan=jnp.zeros(4, dtype=jnp.int8),
            melds=melds,
            meld_counts=base.players.meld_counts.at[1].set(2),
            pon=base.players.pon.at[1, west].set(jnp.int8(2 << 2 | 0)).at[1, seven_p].set(jnp.int8(2 << 2 | 1)),
            is_hand_concealed=base.players.is_hand_concealed.at[1].set(False),
        )

    def robbable_second_kan(first_kan_indicator):
        state = draw_after_kan(kan(before_the_kans(first_kan_indicator), jnp.int32(Tile.NUM_TILE_TYPE_WITH_RED + west)))
        assert revealed(state) == 1  # the first kan's indicator waits for the discard or the next kan
        state = kan(state, jnp.int32(Tile.NUM_TILE_TYPE_WITH_RED + seven_p))
        assert int(state.current_player) == 0 and bool(state.players.legal_action_mask[0, Action.RON])
        return state

    windows = {indicator: robbable_second_kan(indicator) for indicator in (one_s, eight_m, eight_p)}
    for window in windows.values():
        assert revealed(window) == 1  # still face down while the added kan can be robbed
        # P0's non-dealer ron is tanyao + chankan, 2 han 40 fu, whatever the face-down indicator.
        assert m._ron(window).rewards.tolist() == [26.0, -26.0, 0.0, 0.0]

    # Nobody robs it: the first kan's indicator turns over before the rinshan draw,
    # and P1's rinshan tsumo counts it: 8m makes 999m three dora.
    drawn = {indicator: m._advance_to_next_decision(m._pass(window)) for indicator, window in windows.items()}
    for state in drawn.values():
        assert int(state.current_player) == 1 and int(state.round_state.last_draw) == hatsu
        assert revealed(state) == 2
        assert bool(state.players.legal_action_mask[1, Action.TSUMO])
    fan = {indicator: int(state.players.fan[1, 0]) for indicator, state in drawn.items()}
    assert fan[eight_m] == fan[eight_p] + 3 and fan[one_s] == fan[eight_p], fan


def test_no_kan_off_a_rinshan_draw_that_leaves_the_live_wall_empty() -> None:
    """A kan moves a live-wall tile to the dead wall, so with none left no further kan is offered."""
    from mahjax.mahjong_4p import logic as m

    east, south = 27, 28
    kan = jax.jit(m._kan)
    draw_after_kan = jax.jit(m._draw_rinshan)
    hand = jnp.zeros((37,), dtype=jnp.int8)
    for tile in [east] * 4 + [south] * 3 + [0, 1, 2, 3, 4, 5, 6]:  # EEEE SSS 123m 456m 7m
        hand = hand.at[tile].add(1)

    for riichi in (False, True):
        for live_draws_left, south_kan_offered in ((1, False), (2, True)):
            state = m._init(jax.random.PRNGKey(5))
            state = m._replace_state(
                state,
                current_player=jnp.int8(0),
                hand_with_red=state.players.hand_with_red.at[0].set(hand),
                hand=state.players.hand.at[0].set(Hand.to_34(hand)),
                riichi=state.players.riichi.at[0].set(riichi),
                deck=state.round_state.deck.at[10].set(south),  # the rinshan tile is the 4th South
                n_kan=jnp.zeros(4, dtype=jnp.int8),
            )
            # P0 has just drawn, with ``live_draws_left`` live-wall draws still to come.
            state = m._replace_state(
                state, next_deck_ix=jnp.int32(int(m._live_wall_end_ix(state)) + live_draws_left - 1)
            )

            after = draw_after_kan(kan(state, jnp.int32(Tile.NUM_TILE_TYPE_WITH_RED + east)))

            assert bool(after.players.legal_action_mask[0, Tile.NUM_TILE_TYPE_WITH_RED + south]) == south_kan_offered


def test_robbing_kan_mask_drops_stale_self_turn_actions() -> None:
    tile_ids = {
        **{f"{n}m": n - 1 for n in range(1, 10)},
        **{f"{n}p": 9 + n - 1 for n in range(1, 10)},
        **{f"{n}s": 18 + n - 1 for n in range(1, 10)},
    }

    def counts37(tiles: list[str]):
        counts = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8)
        for tile in tiles:
            counts = counts.at[tile_ids[tile]].add(1)
        return counts

    actor1_tiles = "3m 4m 5p 5p 6p 7p 8p 6s 6s 6s 7s 8s 9s".split()
    hand_with_red = jnp.zeros((4, Tile.NUM_TILE_TYPE_WITH_RED), dtype=jnp.int8)
    hand_with_red = hand_with_red.at[1].set(counts37(actor1_tiles))
    hand_with_red = hand_with_red.at[2].set(counts37(["2m"]))
    hand = jnp.stack([Hand.to_34(hand_with_red[player]) for player in range(4)])

    assert bool(Hand.can_ron(hand[1], tile_ids["2m"]))

    stale_actor1_actions = [
        tile_ids["3m"],
        tile_ids["4m"],
        tile_ids["5p"],
        tile_ids["6p"],
        tile_ids["7p"],
        tile_ids["8p"],
        tile_ids["6s"],
        tile_ids["7s"],
        tile_ids["8s"],
        tile_ids["9s"],
        Tile.RED_FIVE["p"],
        Action.TSUMOGIRI,
        Action.RIICHI,
    ]
    legal_action_mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
    for action_id in stale_actor1_actions:
        legal_action_mask = legal_action_mask.at[1, action_id].set(True)

    action = Tile.NUM_TILE_TYPE_WITH_RED + tile_ids["2m"]
    legal_action_mask = legal_action_mask.at[2, action].set(True)
    pon = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int32).at[2, tile_ids["2m"]].set(1)
    melds = default_state().players.melds.at[2, 1].set(Meld.init(Action.PON, tile_ids["2m"], 1))
    can_win = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[1, tile_ids["2m"]].set(True)

    state = _replace_state(
        default_state(),
        current_player=jnp.int8(2),
        legal_action_mask=legal_action_mask,
        hand=hand,
        hand_with_red=hand_with_red,
        pon=pon,
        melds=melds,
        can_win=can_win,
        last_draw=jnp.int8(tile_ids["2m"]),
        last_player=jnp.int8(2),
        deck=jnp.zeros((136,), dtype=jnp.int8).at[10].set(30),
        dora_indicators=jnp.array([tile_ids["5s"], -1, -1, -1, -1], dtype=jnp.int8),
    )

    kan_state = _kan(state, jnp.int32(action))

    expected = (
        jnp.zeros((Action.NUM_ACTION,), dtype=jnp.bool_)
        .at[Action.RON].set(True)
        .at[Action.PASS].set(True)
    )
    assert int(kan_state.current_player) == 1
    assert bool(kan_state.round_state.kan_declared)
    assert bool(jnp.all(kan_state.players.legal_action_mask[1] == expected))
    assert bool(jnp.all(kan_state.legal_action_mask == expected))


def test_robbing_kan_ron_finalizes_to_dummy_share_mask() -> None:
    tile_ids = {
        **{f"{n}m": n - 1 for n in range(1, 10)},
        **{f"{n}p": 9 + n - 1 for n in range(1, 10)},
        **{f"{n}s": 18 + n - 1 for n in range(1, 10)},
    }

    def counts37(tiles: list[str]):
        counts = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8)
        for tile in tiles:
            counts = counts.at[tile_ids[tile]].add(1)
        return counts

    actor1_tiles = "3m 4m 5p 5p 6p 7p 8p 6s 6s 6s 7s 8s 9s".split()
    hand_with_red = jnp.zeros((4, Tile.NUM_TILE_TYPE_WITH_RED), dtype=jnp.int8)
    hand_with_red = hand_with_red.at[1].set(counts37(actor1_tiles))
    hand_with_red = hand_with_red.at[2].set(counts37(["2m"]))
    hand = jnp.stack([Hand.to_34(hand_with_red[player]) for player in range(4)])
    action = Tile.NUM_TILE_TYPE_WITH_RED + tile_ids["2m"]

    state = _replace_state(
        default_state(),
        current_player=jnp.int8(2),
        legal_action_mask=jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_).at[2, action].set(True),
        hand=hand,
        hand_with_red=hand_with_red,
        pon=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int32).at[2, tile_ids["2m"]].set(1),
        melds=default_state().players.melds.at[2, 1].set(Meld.init(Action.PON, tile_ids["2m"], 1)),
        can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[1, tile_ids["2m"]].set(True),
        last_draw=jnp.int8(tile_ids["2m"]),
        last_player=jnp.int8(2),
        deck=jnp.zeros((136,), dtype=jnp.int8).at[10].set(30),
        dora_indicators=jnp.array([tile_ids["5s"], -1, -1, -1, -1], dtype=jnp.int8),
    )

    env = compiled_env(round_mode="half", round_end="confirm")
    kan_state = _kan(state, jnp.int32(action))
    ron_state = env.step(kan_state, jnp.int32(Action.RON), STEP_KEY)

    expected = jnp.zeros((Action.NUM_ACTION,), dtype=jnp.bool_).at[Action.DUMMY].set(True)
    assert bool(ron_state.round_state.terminated_round)
    assert bool(jnp.all(ron_state.legal_action_mask == expected))
    assert bool(jnp.all(ron_state.players.legal_action_mask[:, Action.DUMMY]))
    assert not bool(ron_state.round_state.kan_declared)
    assert int(ron_state.players.n_kan[1]) == 0
    assert int(ron_state.players.hand_with_red[1].sum()) == 13
    assert int(ron_state.round_state.n_kan_doras) == 0


def test_robbing_kan_second_candidate_pass_after_ron_does_not_draw_after_kan() -> None:
    tile_ids = {
        **{f"{n}m": n - 1 for n in range(1, 10)},
        **{f"{n}p": 9 + n - 1 for n in range(1, 10)},
        **{f"{n}s": 18 + n - 1 for n in range(1, 10)},
    }

    def counts37(tiles: list[str]):
        counts = jnp.zeros((Tile.NUM_TILE_TYPE_WITH_RED,), dtype=jnp.int8)
        for tile in tiles:
            counts = counts.at[tile_ids[tile]].add(1)
        return counts

    winner_tiles = "3m 4m 5p 5p 6p 7p 8p 6s 6s 6s 7s 8s 9s".split()
    hand_with_red = jnp.zeros((4, Tile.NUM_TILE_TYPE_WITH_RED), dtype=jnp.int8)
    hand_with_red = hand_with_red.at[1].set(counts37(winner_tiles))
    hand_with_red = hand_with_red.at[2].set(counts37(["2m"]))
    hand_with_red = hand_with_red.at[3].set(counts37(winner_tiles))
    hand = jnp.stack([Hand.to_34(hand_with_red[player]) for player in range(4)])
    action = Tile.NUM_TILE_TYPE_WITH_RED + tile_ids["2m"]

    state = _replace_state(
        default_state(),
        current_player=jnp.int8(2),
        legal_action_mask=jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_).at[2, action].set(True),
        hand=hand,
        hand_with_red=hand_with_red,
        pon=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int32).at[2, tile_ids["2m"]].set(1),
        melds=default_state().players.melds.at[2, 1].set(Meld.init(Action.PON, tile_ids["2m"], 1)),
        can_win=(
            jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_)
            .at[1, tile_ids["2m"]].set(True)
            .at[3, tile_ids["2m"]].set(True)
        ),
        last_draw=jnp.int8(tile_ids["2m"]),
        last_player=jnp.int8(2),
        deck=jnp.zeros((136,), dtype=jnp.int8).at[10].set(30),
        dora_indicators=jnp.array([tile_ids["5s"], -1, -1, -1, -1], dtype=jnp.int8),
    )

    env = compiled_env(round_mode="half", round_end="confirm")
    kan_state = _kan(state, jnp.int32(action))
    first_ron_state = env.step(kan_state, jnp.int32(Action.RON), STEP_KEY)
    pass_state = env.step(first_ron_state, jnp.int32(Action.PASS), STEP_KEY)

    expected = jnp.zeros((Action.NUM_ACTION,), dtype=jnp.bool_).at[Action.DUMMY].set(True)
    assert bool(pass_state.round_state.terminated_round)
    assert bool(jnp.all(pass_state.legal_action_mask == expected))
    assert bool(jnp.all(pass_state.players.legal_action_mask[:, Action.DUMMY]))
    assert not bool(pass_state.round_state.kan_declared)
    assert int(pass_state.players.n_kan[2]) == 0
    assert int(pass_state.players.hand_with_red[3].sum()) == 13
    assert int(pass_state.round_state.n_kan_doras) == 0


def test_red_added_kan_hides_whether_anyone_could_rob_it() -> None:
    """Neither ``last_player`` nor the action history may show that a chankan was declined."""
    from mahjax.mahjong_4p import logic as m

    six_s = 23

    def counts(tiles):
        hand = jnp.zeros(Tile.NUM_TILE_TYPE_WITH_RED, dtype=jnp.int8)
        for tile in tiles:
            hand = hand.at[tile].add(1)
        return hand

    filler = counts([27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33])
    waits_on_6s = counts([0, 1, 2, 3, 4, 5, 15, 16, 17, 18, 18, 21, 22])  # 123m 456m 789p 11s 45s
    declarer = counts([six_s, 6, 7, 8, 9, 10, 11, 12, 13, 14, 27])  # P2 holds the fourth 6s
    base = default_state()
    kan_action = jnp.int32(Tile.NUM_TILE_TYPE_WITH_RED + six_s)

    def after_kan(p1_hand):
        hands = jnp.stack([filler, p1_hand, declarer, filler])
        hands34 = jax.vmap(Hand.to_34)(hands)
        state = _replace_state(
            base,
            current_player=jnp.int8(2),
            last_player=jnp.int8(1),  # P1 discarded before P2 drew
            hand_with_red=hands,
            hand=hands34,
            can_win=m.v_can_win(hands34, m.TILE_RANGE),
            melds=base.players.melds.at[2, 0].set(Meld.init(Action.PON, six_s, 1)),
            meld_counts=base.players.meld_counts.at[2].set(1),
            pon=base.players.pon.at[2, six_s].set(jnp.int8(1 << 2)),
            legal_action_mask=base.players.legal_action_mask.at[2, kan_action].set(True),
            next_deck_ix=jnp.int32(60),
        )
        return _step(state, kan_action, STEP_KEY)

    robbable = after_kan(waits_on_6s)
    assert int(robbable.current_player) == 1
    declined = _step(robbable, jnp.int32(Action.PASS), STEP_KEY)
    not_robbable = after_kan(filler)

    assert int(declined.current_player) == int(not_robbable.current_player) == 2
    assert int(declined.round_state.last_player) == int(not_robbable.round_state.last_player) == 2
    assert int(declined.round_state.round_step) == int(not_robbable.round_state.round_step) == 1
    assert bool(jnp.all(declined.round_state.action_history == not_robbable.round_state.action_history))


def test_red_robbing_kan_on_a_red_five_keeps_the_red_dora() -> None:
    """A chankan must score and report the robbed five's redness."""
    from mahjax.mahjong_4p.meld import Meld as RedMeld
    from mahjax.mahjong_4p.config import GameConfig as RedGameConfig  # noqa: F401

    five_p, red_five_p = 13, Tile.RED_FIVE["p"]

    def rob(declarer_holds_red: bool):
        base = default_state()
        declarer, winner = 0, 1
        declarer_hand = jnp.zeros((37,), dtype=jnp.int8).at[
            red_five_p if declarer_holds_red else five_p
        ].set(1)
        winner_hand = jnp.zeros((37,), dtype=jnp.int8)
        for tile in (11, 12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 18, 18):
            winner_hand = winner_hand.at[tile].add(1)
        hand_with_red = base.players.hand_with_red.at[declarer].set(declarer_hand).at[
            winner
        ].set(winner_hand)
        hand = base.players.hand.at[declarer].set(Hand.to_34(declarer_hand)).at[winner].set(
            Hand.to_34(winner_hand)
        )
        can_win = jax.vmap(Hand.can_ron, in_axes=(None, 0))(hand[winner], jnp.arange(34))
        state = _replace_state(
            base,
            current_player=jnp.int8(declarer),
            hand=hand,
            hand_with_red=hand_with_red,
            pon=base.players.pon.at[(declarer, five_p)].set(1),
            melds=base.players.melds.at[declarer, 0].set(RedMeld.init(Action.PON, five_p, 1)),
            meld_counts=base.players.meld_counts.at[declarer].set(1),
            can_win=base.players.can_win.at[winner].set(can_win),
            deck=jnp.zeros((136,), dtype=jnp.int8).at[10].set(30),
            dora_indicators=jnp.full((5,), -1, dtype=jnp.int8),
            legal_action_mask=jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
            .at[declarer, Tile.NUM_TILE_TYPE_WITH_RED + five_p]
            .set(True),
        )
        out = _kan(state, jnp.int32(Tile.NUM_TILE_TYPE_WITH_RED + five_p))
        return int(out.round_state.target), int(out.players.fan[winner, 0])

    black_target, black_fan = rob(False)
    red_target, red_fan = rob(True)

    assert black_target == five_p
    assert red_target == red_five_p
    assert red_fan == black_fan + 1
