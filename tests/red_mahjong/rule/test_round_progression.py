import jax
import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.env import RedMahjong, _advance_to_next_round_auto, _replace_state
from mahjax.red_mahjong.state import default_state

from ..helpers import STEP_KEY


# ----------------- next_round_style tests -----------------


def _ron_legal_mask(ron_player: int = 0):
    return (
        jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        .at[ron_player, Action.RON].set(True)
    )


def test_red_auto_single_mode_terminates_like_legacy() -> None:
    env_auto = RedMahjong(round_mode="single", next_round_style="auto")
    state = env_auto.init(jax.random.PRNGKey(11))
    state = _replace_state(
        state,
        legal_action_mask=_ron_legal_mask(0),
        current_player=jnp.int8(0),
    )
    next_state = env_auto.step(state, jnp.int32(Action.RON), STEP_KEY)
    assert bool(next_state.terminated)


# ---------------- parity: auto vs dummy_share ----------------
#
# These tests assert that ``auto`` mode collapses the dummy_share rotation
# phase into a single env.step while producing the same end state.


def _force_ron_state(env: RedMahjong, key, ron_player: int = 0):
    state = env.init(key)
    return _replace_state(
        state,
        legal_action_mask=_ron_legal_mask(ron_player),
        current_player=jnp.int8(ron_player),
    )


def test_red_auto_matches_dummy_share_at_mid_game_round_transition() -> None:
    """auto's 1-step round transition == dummy_share's 5-step (RON + 4 DUMMY)
    transition, modulo:
    - ``step_count`` (auto +1, share +5)
    - ``rewards`` (auto preserves the round-end vector from the RON step;
      dummy_share resets it to zero in the ``_make_state``-based init).
    """
    env_auto = RedMahjong(round_mode="half", next_round_style="auto")
    env_share = RedMahjong(round_mode="half", next_round_style="dummy_share")
    key = jax.random.PRNGKey(2026)

    state_auto = env_auto.step(_force_ron_state(env_auto, key), jnp.int32(Action.RON), STEP_KEY)
    state_share = env_share.step(_force_ron_state(env_share, key), jnp.int32(Action.RON), STEP_KEY)
    rewards_at_ron = state_share.rewards  # round-end reward delivered at RON step in dummy_share
    for _ in range(4):
        state_share = env_share.step(state_share, jnp.int32(Action.DUMMY), STEP_KEY)

    # Both must land in the next-round init: mid-game ⇒ not terminated, not terminated_round.
    assert not bool(state_auto.terminated)
    assert not bool(state_share.terminated)
    assert not bool(state_auto.round_state.terminated_round)
    assert not bool(state_share.round_state.terminated_round)
    assert int(state_auto.round_state.dummy_count) == 0
    assert int(state_share.round_state.dummy_count) == 0

    rs_a, rs_s = state_auto.round_state, state_share.round_state
    assert int(state_auto.current_player) == int(state_share.current_player)
    assert int(rs_a.dealer) == int(rs_s.dealer)
    assert int(rs_a.round) == int(rs_s.round)
    assert int(rs_a.honba) == int(rs_s.honba)
    assert int(rs_a.kyotaku) == int(rs_s.kyotaku)
    assert bool(jnp.all(rs_a.score == rs_s.score))
    assert bool(jnp.all(rs_a.deck == rs_s.deck))
    assert bool(jnp.all(rs_a.dora_indicators == rs_s.dora_indicators))
    assert int(rs_a.next_deck_ix) == int(rs_s.next_deck_ix)
    assert int(rs_a.last_draw) == int(rs_s.last_draw)

    ps_a, ps_s = state_auto.players, state_share.players
    assert bool(jnp.all(ps_a.hand == ps_s.hand))
    assert bool(jnp.all(ps_a.has_won == ps_s.has_won))
    assert bool(jnp.all(ps_a.legal_action_mask == ps_s.legal_action_mask))
    assert bool(jnp.all(state_auto.legal_action_mask == state_share.legal_action_mask))

    # auto preserves the round-end rewards; dummy_share's were delivered at the
    # RON step (captured in ``rewards_at_ron``) and zeroed afterwards.
    assert bool(jnp.all(state_auto.rewards == rewards_at_ron))


def test_red_auto_matches_dummy_share_at_game_end() -> None:
    """When RON ends the game, auto terminates after the RON step; dummy_share
    terminates one step later (DUMMY 1 detects ``_is_game_end`` at ``dc==0``).
    Compare the two terminal states: same ``terminated``, same final ``score``
    (the expected one, uma included), same ``rewards``.
    """
    env_auto = RedMahjong(round_mode="half", next_round_style="auto", order_points=[300, 100, -100, -300])
    env_share = RedMahjong(round_mode="half", next_round_style="dummy_share", order_points=[300, 100, -100, -300])
    key = jax.random.PRNGKey(2026)
    forced = dict(
        dealer=jnp.int8(0),
        score=jnp.array([310, 310, 190, 190], dtype=jnp.int32),
        init_wind=jnp.array([0, 1, 2, 3], dtype=jnp.int8),
        round=jnp.int8(7),
        round_limit=jnp.int8(7),
        kyotaku=jnp.int8(3),
        has_won=jnp.array([False, False, False, False], dtype=jnp.bool_),
    )

    state_auto = _replace_state(_force_ron_state(env_auto, key), **forced)
    state_share = _replace_state(_force_ron_state(env_share, key), **forced)

    state_auto = env_auto.step(state_auto, jnp.int32(Action.RON), STEP_KEY)
    total_auto = state_auto.rewards
    state_share = env_share.step(state_share, jnp.int32(Action.RON), STEP_KEY)
    total_share = state_share.rewards
    state_share = env_share.step(state_share, jnp.int32(Action.DUMMY), STEP_KEY)
    total_share = total_share + state_share.rewards

    assert bool(state_auto.terminated)
    assert bool(state_share.terminated)
    expected = jnp.array([640, 410, 90, -110], dtype=jnp.int32)
    assert bool(jnp.all(state_auto.round_state.score == expected)), (
        f"got {state_auto.round_state.score}, expected {expected}"
    )
    assert bool(jnp.all(state_auto.round_state.score == state_share.round_state.score))
    assert bool(jnp.all(total_auto == total_share))


def test_calc_wind_assigns_east_to_dealer() -> None:
    """``seat_wind`` is indexed BY PLAYER, so the dealer must read 0 (East).

    Nothing tracked enforced this contract, and no_red shipped the
    inverse permutation (``[(dealer + i) % 4]``) for a long time. That form agrees
    only when the dealer sits at seat 0 or 2; at seats 1 and 3 it reports the dealer
    as West and some other seat as East, which mislabels seat-wind yakuhai and fu.
    """
    from mahjax.red_mahjong.env import _calc_wind

    assert _calc_wind(jnp.int32(0)).tolist() == [0, 1, 2, 3]
    assert _calc_wind(jnp.int32(1)).tolist() == [3, 0, 1, 2]
    assert _calc_wind(jnp.int32(2)).tolist() == [2, 3, 0, 1]
    assert _calc_wind(jnp.int32(3)).tolist() == [1, 2, 3, 0]
    # The invariant the four literals encode.
    for dealer in range(4):
        assert int(_calc_wind(jnp.int32(dealer))[dealer]) == 0


def _end_of_round_state(env, **overrides):
    state = env.init(jax.random.PRNGKey(7))
    mask = jnp.zeros_like(state.players.legal_action_mask).at[:, Action.DUMMY].set(True)
    fields = dict(
        dealer=jnp.int8(0),
        current_player=jnp.int8(0),
        round=jnp.int8(7),
        round_limit=jnp.int8(7),
        honba=jnp.int32(0),
        kyotaku=jnp.int8(0),
        dummy_count=jnp.int8(0),
        has_won=jnp.zeros(4, dtype=jnp.bool_),
        can_win=jnp.zeros_like(state.players.can_win),
        terminated_round=True,
        draw_next=False,
        legal_action_mask=mask,
    )
    fields.update(overrides)
    return _replace_state(state, **fields)


def test_red_half_game_is_eight_kyoku_long() -> None:
    assert int(RedMahjong(round_mode="half").round_limit) == 7
    assert int(RedMahjong(round_mode="east").round_limit) == 3


def test_red_final_round_ends_the_game_once_someone_has_30000() -> None:
    env = RedMahjong(round_mode="half", next_round_style="auto")
    state = _end_of_round_state(env, score=jnp.array([300, 250, 240, 210], dtype=jnp.int32))

    out = _advance_to_next_round_auto(state, jax.random.PRNGKey(0))

    assert bool(out.terminated)


def test_red_final_round_runs_on_when_nobody_has_30000() -> None:
    env = RedMahjong(round_mode="half", next_round_style="auto")
    state = _end_of_round_state(env, score=jnp.array([280, 250, 240, 230], dtype=jnp.int32))

    out = _advance_to_next_round_auto(state, jax.random.PRNGKey(0))

    assert not bool(out.terminated)
    assert int(out.round_state.round) == 8


def test_red_sudden_death_ends_at_the_last_extra_round() -> None:
    env = RedMahjong(round_mode="half", next_round_style="auto")
    state = _end_of_round_state(
        env, round=jnp.int8(11), score=jnp.array([280, 250, 240, 230], dtype=jnp.int32)
    )

    out = _advance_to_next_round_auto(state, jax.random.PRNGKey(0))

    assert bool(out.terminated)


def test_red_sudden_death_runs_for_exactly_four_extra_rounds() -> None:
    env = RedMahjong(round_mode="half", next_round_style="auto")
    below = jnp.array([280, 250, 240, 230], dtype=jnp.int32)
    terminated = [
        bool(
            _advance_to_next_round_auto(
                _end_of_round_state(env, round=jnp.int8(r), score=below), jax.random.PRNGKey(0)
            ).terminated
        )
        for r in range(7, 12)
    ]

    assert terminated == [False, False, False, False, True]


def test_red_last_extra_round_continues_when_the_dealer_keeps_the_deal() -> None:
    """West-4 is the last round, but a dealer's renchan still comes first."""
    env = RedMahjong(round_mode="half", next_round_style="auto")
    state = _end_of_round_state(
        env,
        round=jnp.int8(11),
        score=jnp.array([280, 250, 240, 230], dtype=jnp.int32),
        can_win=jnp.zeros_like(env.init(jax.random.PRNGKey(7)).players.can_win).at[0, 0].set(True),  # dealer tenpai
    )

    out = _advance_to_next_round_auto(state, jax.random.PRNGKey(0))

    assert not bool(out.terminated)
    assert int(out.round_state.round) == 11
    assert int(out.round_state.dealer) == 0
    assert int(out.round_state.honba) == 1


def test_red_the_dealer_keeps_the_deal_at_eight_honba() -> None:
    """There is no renchan limit: a dealer win or a tenpai draw keeps the deal however many honba are up."""
    env = RedMahjong(round_mode="half", next_round_style="auto")
    dealer_won = dict(has_won=jnp.zeros(4, dtype=jnp.bool_).at[0].set(True))
    dealer_tenpai = dict(can_win=jnp.zeros_like(env.init(jax.random.PRNGKey(7)).players.can_win).at[0, 0].set(True))
    for renchan in (dealer_won, dealer_tenpai):
        state = _end_of_round_state(
            env, round=jnp.int8(0), honba=jnp.int8(8), score=jnp.array([250, 250, 250, 250], dtype=jnp.int32), **renchan
        )

        out = _advance_to_next_round_auto(state, jax.random.PRNGKey(0))

        assert int(out.round_state.round) == 0
        assert int(out.round_state.dealer) == 0
        assert int(out.round_state.honba) == 9


def test_red_a_dealer_win_in_second_place_does_not_end_south_four_at_eight_honba() -> None:
    """P1 is past 30000, but the dealer is not top, so the dealer's win is a renchan and the game goes on."""
    for style in ("auto", "dummy_share"):
        env = RedMahjong(round_mode="half", next_round_style=style)
        state = _end_of_round_state(
            env,
            honba=jnp.int8(8),
            score=jnp.array([290, 320, 200, 190], dtype=jnp.int32),
            has_won=jnp.zeros(4, dtype=jnp.bool_).at[0].set(True),
        )

        if style == "auto":
            out = _advance_to_next_round_auto(state, jax.random.PRNGKey(0))
        else:
            step = jax.jit(env.step)
            out = state
            for _ in range(4):  # the DUMMY share phase
                out = step(out, jnp.int32(Action.DUMMY), STEP_KEY)

        assert not bool(out.terminated)
        assert int(out.round_state.round) == 7
        assert int(out.round_state.dealer) == 0
        assert int(out.round_state.honba) == 9


def test_red_honba_does_not_wrap_after_127() -> None:
    """With no renchan limit honba can keep growing, and a renchan at 127 honba makes 128, not -128."""
    for style in ("auto", "dummy_share"):
        env = RedMahjong(round_mode="half", next_round_style=style)
        state = _end_of_round_state(
            env,
            round=jnp.int8(0),
            # In the env's own stored dtype, as honba would have reached 127 in play.
            honba=jnp.asarray(127, env.init(jax.random.PRNGKey(7)).round_state.honba.dtype),
            score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
            can_win=jnp.zeros_like(env.init(jax.random.PRNGKey(7)).players.can_win).at[0, 0].set(True),
        )

        if style == "auto":
            out = _advance_to_next_round_auto(state, jax.random.PRNGKey(0))
        else:
            step = jax.jit(env.step)
            out = state
            for _ in range(4):  # the DUMMY share phase
                out = step(out, jnp.int32(Action.DUMMY), STEP_KEY)

        assert int(out.round_state.dealer) == 0
        assert int(out.round_state.honba) == 128


def test_red_forty_three_honba_do_not_wrap_negative() -> None:
    """43 honba are +12900 points; stored as int8, ``honba * 3`` wrapped them."""
    from mahjax.red_mahjong import env as m

    base = default_state()
    state = m._replace_state(
        base,
        current_player=jnp.int8(1),
        last_player=jnp.int8(2),
        honba=jnp.asarray(43, base.round_state.honba.dtype),  # the env's own stored dtype
        next_deck_ix=jnp.int32(50),
        score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        fan=base.players.fan.at[1, 0].set(jnp.int32(1)),
        fu=base.players.fu.at[1, 0].set(jnp.int32(30)),
    )

    assert float(m._ron(state).rewards[1]) == 10 + 129  # 1 han 30 fu ron + 300 a honba
    assert float(m._tsumo(state).rewards[1]) == 11 + 129  # 300/500 tsumo + 100 a honba from each


def test_red_uma_is_added_to_the_final_score() -> None:
    env = RedMahjong(round_mode="half", next_round_style="auto", order_points=[300, 100, -100, -300])
    before = jnp.array([310, 300, 200, 190], dtype=jnp.int32)
    state = _end_of_round_state(env, score=before)

    out = _advance_to_next_round_auto(state, jax.random.PRNGKey(0))

    assert bool(out.terminated)
    assert jnp.array_equal(out.round_state.score - before, jnp.array([300, 100, -100, -300]))


def test_red_dummy_share_applies_uma_when_a_non_dealer_won_the_final_round() -> None:
    env = RedMahjong(round_mode="half", next_round_style="dummy_share", order_points=[300, 100, -100, -300])
    state = _end_of_round_state(
        env,
        score=jnp.array([200, 300, 260, 240], dtype=jnp.int32),
        kyotaku=jnp.int8(2),
        has_won=jnp.zeros(4, dtype=jnp.bool_).at[1].set(True),
        can_win=jnp.zeros_like(env.init(jax.random.PRNGKey(7)).players.can_win).at[0, 0].set(True),
    )

    out = env.step(state, jnp.int8(Action.DUMMY), jax.random.PRNGKey(0))

    assert bool(out.terminated)
    assert jnp.array_equal(out.round_state.score, jnp.array([-100, 620, 360, 140]))


def test_red_uma_scales_a_custom_order_points() -> None:
    env = RedMahjong(round_mode="half", next_round_style="auto", order_points=[200, 50, -50, -200])
    before = jnp.array([310, 300, 200, 190], dtype=jnp.int32)
    state = _end_of_round_state(env, score=before)

    out = _advance_to_next_round_auto(state, jax.random.PRNGKey(0))

    assert jnp.array_equal(out.round_state.score - before, jnp.array([200, 50, -50, -200]))


def test_red_tied_players_are_ranked_by_their_starting_wind() -> None:
    """On equal score East-1's seat order decides, not the player index."""
    from mahjax.red_mahjong import env as m

    # P2 was the first dealer: P2 started East, P3 South, P0 West, P1 North.
    round_state = default_state().round_state.replace(
        init_wind=jnp.array([2, 3, 0, 1], dtype=jnp.int8),
        score=jnp.array([180, 320, 180, 320], dtype=jnp.int32),
        order_points=jnp.array([300, 100, -100, -300], dtype=jnp.int32),
        kyotaku=jnp.int8(1),
        round=jnp.int8(7),
        round_limit=jnp.int8(7),
        dealer=jnp.int8(1),
    )

    # P3 outranks P1 and P2 outranks P0, so P3 takes the top uma and the stick.
    assert jnp.array_equal(m._final_score(round_state), jnp.array([180 - 300, 320 + 100, 180 - 100, 320 + 300 + 10]))
    # The dealer P1 is not top on a tie with P3, so the dealer's renchan keeps the game going.
    assert not bool(m._is_game_end(round_state, jnp.bool_(True)))

    # The starting winds survive a round change.
    env = RedMahjong(round_mode="half", next_round_style="auto")
    state = _end_of_round_state(
        env,
        round=jnp.int8(0),
        init_wind=jnp.array([2, 3, 0, 1], dtype=jnp.int8),
        score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
    )
    out = _advance_to_next_round_auto(state, jax.random.PRNGKey(0))
    assert int(out.round_state.round) == 1
    assert jnp.array_equal(out.round_state.init_wind, jnp.array([2, 3, 0, 1]))
