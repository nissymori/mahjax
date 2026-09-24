"""A ron window shows nothing until everyone who can ron the tile has answered.

P3 is the dealer unless a test says otherwise, and lets go of 4p. The others
hold one of these hands, and are asked in turn order from P3, so P0 goes first:
    A   1s-9s 5p6p SS     rons 4p, and as P0 can chi it
    B   1m-9m 2p3p WW     rons 4p
    S   1m-9m 4p4p EE     rons 4p, or pons it
    P   4p4p 7p8p9p ...   pons 4p but cannot ron it
    N   7p8p9p x2 ...     does nothing with 4p
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.env import TILE_RANGE, RedMahjong, _calc_wind, _replace_state, v_can_win
from mahjax.red_mahjong.hand import Hand
from mahjax.red_mahjong.meld import Meld
from mahjax.red_mahjong.state import default_state
from mahjax.red_mahjong.tile import Tile

KEY = jax.random.PRNGKey(0)
FOUR_P = 12
A = [*range(18, 27), 13, 14, 28, 28]
B = [*range(0, 9), 10, 11, 29, 29]
S = [*range(0, 9), FOUR_P, FOUR_P, 27, 27]
P = [FOUR_P, FOUR_P, 15, 16, 17, 21, 22, 23, 27, 27, 27, 32, 33]
N = [15, 16, 17, 15, 16, 17, 24, 25, 26, 27, 28, 29, 33]
DISCARDER = [FOUR_P, 30, 30, 30, 31, 31, 31, 32, 32, 32, 33, 33, 9, 9]
# For the robbing-kan window P3 has ponned 4p and holds the fourth.
KANNER = [FOUR_P, 30, 30, 30, 31, 31, 31, 32, 32, 32, 9]
ADDED_KAN_4P = Tile.NUM_TILE_TYPE_WITH_RED + FOUR_P

ENVS = {
    (mode, style): RedMahjong(round_mode=mode, next_round_style=style)
    for mode, style in [("single", "auto"), ("half", "auto"), ("half", "dummy_share")]
}
STEPS = {k: jax.jit(env.step) for k, env in ENVS.items()}
OBSERVE = jax.jit(ENVS["single", "auto"].observe)


def _counts(tiles):
    return jnp.zeros(Tile.NUM_TILE_TYPE_WITH_RED, dtype=jnp.int8).at[jnp.array(tiles)].add(1)


def before_the_discard(p0, p1, p2, *, dealer=3, riichi=False, robbing_kan=False):
    """P3 to let go of 4p, by discarding it or by adding it to their pon."""
    hands = jnp.stack([_counts(h) for h in (p0, p1, p2, KANNER if robbing_kan else DISCARDER)])
    hands34 = jax.vmap(Hand.to_34)(hands)
    melded = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.int8).at[FOUR_P].set(3 * robbing_kan)
    assert int((hands34.sum(axis=0) + melded).max()) <= 4
    base = default_state()
    state = _replace_state(
        base,
        current_player=jnp.int8(3),
        dealer=jnp.int8(dealer),
        seat_wind=_calc_wind(dealer),
        last_player=jnp.int8(2),
        hand_with_red=hands,
        hand=hands34,
        can_win=v_can_win(hands34, TILE_RANGE),
        riichi_declared=jnp.zeros(4, dtype=jnp.bool_).at[3].set(riichi),
        legal_action_mask=jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        .at[3, ADDED_KAN_4P if robbing_kan else FOUR_P]
        .set(True),
        honba=jnp.int8(2),
        kyotaku=jnp.int8(3),
    )
    if robbing_kan:
        state = _replace_state(
            state,
            pon=base.players.pon.at[3, FOUR_P].set(1 << 2),
            melds=base.players.melds.at[3, 0].set(Meld.init(Action.PON, FOUR_P, 1)),
            meld_counts=base.players.meld_counts.at[3].set(1),
            is_hand_concealed=base.players.is_hand_concealed.at[3].set(False),
        )
    return state


def play(state, *actions, env=("single", "auto")):
    for action in actions:
        assert bool(state.legal_action_mask[action]), (int(state.current_player), action)
        state = STEPS[env](state, jnp.int32(action), KEY)
    return state


def open_window(p0, p1, p2, *, env=("single", "auto"), **kwargs):
    state = before_the_discard(p0, p1, p2, **kwargs)
    return play(state, ADDED_KAN_4P if kwargs.get("robbing_kan") else FOUR_P, env=env)


def mangan(state):
    """Every ron on the window is worth 5 han 30 fu, so the payments are exact."""
    return state.replace(
        players=state.players.replace(
            fan=state.players.fan.at[:, 0].set(5), fu=state.players.fu.at[:, 0].set(30)
        )
    )


def offered(state):
    return {int(a) for a in jnp.flatnonzero(state.legal_action_mask)}


WINDOWS = {
    "double": (A, S, N),
    "triple": (A, B, S),
    "robbing_kan": (A, B, N),
}


@pytest.mark.parametrize("window", sorted(WINDOWS))
def test_a_later_ron_candidate_cannot_tell_an_earlier_ron_from_a_pass(window) -> None:
    state = open_window(*WINDOWS[window], robbing_kan=window == "robbing_kan")
    assert int(state.current_player) == 0
    assert offered(state) == {Action.RON, Action.PASS}
    # Every way the players before the last one can have answered.
    answers = [[Action.RON], [Action.PASS]]
    if window == "triple":
        answers = [first + [second] for first in answers for second in (Action.RON, Action.PASS)]
    views = [play(state, *answer) for answer in answers]

    last = len(answers[0])
    first_obs = OBSERVE(views[0])
    for view in views:
        assert int(view.current_player) == last
        assert offered(view) == {Action.RON, Action.PASS}
        assert not bool(view.round_state.terminated_round)
        np.testing.assert_array_equal(view.rewards, np.zeros(4))
        np.testing.assert_array_equal(view.legal_action_mask, views[0].legal_action_mask)
        # The stored rows too: a candidate reading their own row learns nothing either.
        np.testing.assert_array_equal(view.players.legal_action_mask, views[0].players.legal_action_mask)
        obs = OBSERVE(view)
        for field in first_obs:
            np.testing.assert_array_equal(obs[field], first_obs[field], err_msg=field)


@pytest.mark.parametrize(
    "window, answers, expected",
    [
        ("double", [Action.RON, Action.RON], [116, 80, 0, -166]),
        ("double", [Action.RON, Action.PASS], [116, 0, 0, -86]),
        ("triple", [Action.RON, Action.RON, Action.PASS], [116, 80, 0, -166]),
        ("triple", [Action.RON, Action.PASS, Action.RON], [116, 0, 80, -166]),
        ("triple", [Action.PASS, Action.RON, Action.RON], [0, 116, 80, -166]),
        ("robbing_kan", [Action.RON, Action.RON], [156, 120, 0, -246]),
        ("robbing_kan", [Action.RON, Action.PASS], [156, 0, 0, -126]),
    ],
    ids=["ron-ron", "ron-pass", "ron-ron-pass", "ron-pass-ron", "pass-ron-ron", "chankan-ron-ron", "chankan-ron-pass"],
)
def test_the_window_pays_every_ron_at_once_when_it_closes(window, answers, expected) -> None:
    # Mangan is 80 for a non-dealer, and robbing the kan adds a han for 120; the
    # head winner also takes the 2 honba (6) and the 3 riichi sticks (30), and
    # the dealer P3 pays for both.
    state = mangan(open_window(*WINDOWS[window], robbing_kan=window == "robbing_kan"))
    cursor = int(state.round_state.round_step)
    total = np.zeros(4)
    for i, answer in enumerate(answers):
        state = play(state, answer)
        total += np.asarray(state.rewards)
        if i < len(answers) - 1:
            np.testing.assert_array_equal(state.rewards, np.zeros(4))
            np.testing.assert_array_equal(state.round_state.score, [250] * 4)
            assert int(state.round_state.kyotaku) == 3
            assert int(state.round_state.round_step) == cursor
            assert not bool(state.players.has_won.any())

    winners = [seat for seat in range(3) if expected[seat] > 0]
    assert bool(state.terminated)
    np.testing.assert_array_equal(total, expected)
    assert total.sum() == 30  # the sticks came off the table, nothing else was created
    np.testing.assert_array_equal(state.round_state.score, 250 + np.asarray(expected))
    np.testing.assert_array_equal(state.players.has_won, [seat in winners for seat in range(4)])
    assert int(state.round_state.kyotaku) == 0
    assert int(state.round_state.honba) == 2
    assert int(state.players.n_kan.sum()) == 0
    assert not bool(state.pending_winners.any()) and not bool(state.pending_rewards.any())
    # The RONs are recorded now, in the order they were asked.
    assert int(state.round_state.round_step) == cursor + len(winners)
    history = np.asarray(state.round_state.action_history[:, cursor : cursor + len(winners)])
    np.testing.assert_array_equal(history, [winners, [Action.RON] * len(winners), [-1] * len(winners)])


def test_three_rons_abandon_the_round_without_paying_anything() -> None:
    state = mangan(open_window(*WINDOWS["triple"]))
    cursor = int(state.round_state.round_step)
    for _ in range(3):
        state = play(state, Action.RON)
        np.testing.assert_array_equal(state.rewards, np.zeros(4))
        np.testing.assert_array_equal(state.round_state.score, [250] * 4)
        assert int(state.round_state.kyotaku) == 3
        assert not bool(state.players.has_won.any())
    assert not bool(state.pending_winners.any()) and not bool(state.pending_rewards.any())
    assert bool(state.players.legal_action_mask[:, Action.KYUUSHU].all())
    history = np.asarray(state.round_state.action_history[:, cursor : cursor + 3])
    np.testing.assert_array_equal(history, [[0, 1, 2], [Action.RON] * 3, [-1] * 3])

    state = play(state, Action.KYUUSHU)
    assert int(state.round_state.dealer) == 3
    assert int(state.round_state.honba) == 3
    assert int(state.round_state.kyotaku) == 3
    np.testing.assert_array_equal(state.round_state.score, [250] * 4)


@pytest.mark.parametrize("style", ["auto", "dummy_share"])
@pytest.mark.parametrize(
    "second, expected, dealer, honba, kyoku",
    [
        # The dealer P1 wins too: they keep the deal and the honba goes up.
        (Action.RON, [116, 120, 0, -206], 1, 3, 0),
        # Only P0 wins: the deal moves on and the honba is cleared.
        (Action.PASS, [116, 0, 0, -86], 2, 0, 1),
    ],
    ids=["dealer_wins_too", "dealer_declines"],
)
def test_the_next_round_follows_the_closed_window(style, second, expected, dealer, honba, kyoku) -> None:
    env = ("half", style)
    state = mangan(open_window(A, S, N, dealer=1, env=env))
    state = play(state, Action.RON, env=env)
    # The first ron neither ends the round nor deals the next one.
    assert not bool(state.round_state.terminated_round)
    assert int(state.round_state.round) == 0
    assert int(state.current_player) == 1
    np.testing.assert_array_equal(state.round_state.score, [250] * 4)

    state = play(state, second, env=env)
    np.testing.assert_array_equal(state.rewards, expected)
    if style == "dummy_share":
        assert bool(state.round_state.terminated_round)
        np.testing.assert_array_equal(state.players.has_won, np.asarray(expected) > 0)
        state = play(state, *[Action.DUMMY] * 4, env=env)
    assert not bool(state.terminated)
    assert not bool(state.round_state.terminated_round)
    assert int(state.round_state.dealer) == dealer
    assert int(state.round_state.honba) == honba
    assert int(state.round_state.round) == kyoku
    assert int(state.round_state.kyotaku) == 0
    np.testing.assert_array_equal(state.round_state.score, 250 + np.asarray(expected))


@pytest.mark.parametrize(
    "second, score",
    [(Action.RON, [366, 330, 250, 84]), (Action.PASS, [366, 250, 250, 164])],
    ids=["ron-ron", "ron-pass"],
)
def test_a_ron_on_a_riichi_declaration_leaves_the_riichi_unmade(second, score) -> None:
    ronned = play(mangan(open_window(A, S, N, riichi=True)), Action.RON, second)
    assert bool(ronned.terminated)
    assert not bool(ronned.players.riichi[3])
    assert int(ronned.round_state.kyotaku) == 0
    np.testing.assert_array_equal(ronned.round_state.score, score)


def test_a_riichi_declaration_everyone_passes_stands_at_the_next_draw() -> None:
    # P0 and P1 decline their rons, P1 its pon and P0 its chi.
    state = play(open_window(A, S, N, riichi=True), Action.PASS, Action.PASS)
    assert offered(state) == {Action.PON, Action.PASS}
    state = play(state, Action.PASS)
    assert int(state.current_player) == 0
    assert offered(state) == {Action.CHI_L, Action.PASS}
    assert not bool(state.players.riichi[3])
    state = play(state, Action.PASS)
    assert int(state.current_player) == 0
    assert bool(state.legal_action_mask[Action.TSUMOGIRI])
    assert bool(state.players.riichi[3])
    assert int(state.round_state.kyotaku) == 4
    np.testing.assert_array_equal(state.round_state.score, [250, 250, 250, 240])


@pytest.mark.parametrize(
    "hands, first", [((N, S, N), []), ((A, S, N), [Action.PASS])], ids=["alone", "after_another_pass"]
)
def test_a_declined_ron_still_leaves_the_pon(hands, first) -> None:
    state = play(open_window(*hands), *first)
    assert int(state.current_player) == 1
    assert offered(state) == {Action.RON, Action.PASS}
    state = play(state, Action.PASS)
    assert int(state.current_player) == 1
    assert offered(state) == {Action.PON, Action.PASS}
    state = play(state, Action.PON)
    assert int(state.players.meld_counts[1]) == 1
    assert int(state.current_player) == 1


@pytest.mark.parametrize(
    "p0, others",
    [
        # P0 can ron or pon: whether P1 can ron too must not show.
        (S, [(A, N), (N, N)]),
        # P0 can ron or chi: whether P1 can pon must not show.
        (A, [(P, N), (N, N)]),
    ],
    ids=["ron_or_pon", "ron_or_chi"],
)
def test_a_ron_candidate_is_offered_the_same_calls_whoever_else_can_claim(p0, others) -> None:
    views = [open_window(p0, p1, p2) for p1, p2 in others]
    for view in views:
        assert int(view.current_player) == 0
        assert offered(view) == {Action.RON, Action.PASS}
    first_obs = OBSERVE(views[0])
    for view in views[1:]:
        obs = OBSERVE(view)
        for field in first_obs:
            np.testing.assert_array_equal(obs[field], first_obs[field], err_msg=field)


def test_a_closed_window_does_not_show_a_declined_ron() -> None:
    env = ("half", "dummy_share")
    declined = play(open_window(A, S, N, env=env), Action.RON, Action.PASS, env=env)
    alone = play(open_window(A, N, N, env=env), Action.RON, env=env)
    for state in (declined, alone):
        assert bool(state.round_state.terminated_round)
        assert int(state.current_player) == 0
        assert int(state.round_state.target) == FOUR_P
    np.testing.assert_array_equal(declined.rewards, alone.rewards)
    # P1 knows what they held and declined; nobody else can tell.
    for seat in (0, 2, 3):
        a = OBSERVE(declined.replace(current_player=jnp.int8(seat)))
        b = OBSERVE(alone.replace(current_player=jnp.int8(seat)))
        for field in a:
            np.testing.assert_array_equal(a[field], b[field], err_msg=f"seat {seat}: {field}")
