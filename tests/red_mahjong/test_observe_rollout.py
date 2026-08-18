"""Rollout invariants for ``_observe_dict``.

``test_observe.py`` builds one hand-crafted state and observes it once, so it can
only check plumbing: it never steps the env, never crosses a round boundary and
never sees ``step_count > 0``. Both of the observation's real defects lived
exactly there --

  * ``action_history`` was indexed by the hanchan-global ``step_count`` while the
    buffer itself was rebuilt each round, so it silently went all -1;
  * ``shanten_current_player`` was reset to 0 (== tenpai) by the round-advance
    path and never recomputed.

-- and neither was reachable without stepping. These tests drive a real rollout
in ``round_mode='half'``, which is ``RedMahjong``'s own default and the mode both
bugs required.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mahjax.red_mahjong.env import RedMahjong, _observe_dict
from mahjax.red_mahjong.shanten import Shanten
from mahjax.red_mahjong.constants import MAX_DORA_INDICATORS


# Long enough to cross the old 200-slot ceiling and several round boundaries.
ROLLOUT_STEPS = 420


def _rollout(seed: int, num_steps: int = ROLLOUT_STEPS):
    """Random-policy rollout. Yields (state, obs) before each action."""
    env = RedMahjong(round_mode="half")
    step = jax.jit(env.step)
    observe = jax.jit(_observe_dict)

    key = jax.random.PRNGKey(seed)
    key, sub = jax.random.split(key)
    state = env.init(sub)

    out = []
    for _ in range(num_steps):
        if bool(state.terminated):
            break
        out.append((state, observe(state)))
        legal = np.flatnonzero(np.asarray(state.legal_action_mask))
        assert legal.size > 0, "no legal action available"
        key, pick_key, step_key = jax.random.split(key, 3)
        action = jnp.int32(np.asarray(jax.random.choice(pick_key, legal)))
        state = step(state, action, step_key)
    return out


@pytest.fixture(scope="module")
def rollout():
    return _rollout(seed=0)


def test_rollout_actually_crosses_the_old_ceiling(rollout):
    """Guard the guard: the other tests only mean something if we get far enough."""
    max_step_count = max(int(s.step_count) for s, _ in rollout)
    rounds = {(int(s.round_state.round), int(s.round_state.honba)) for s, _ in rollout}
    assert max_step_count > 250, f"rollout stopped at step_count={max_step_count}"
    assert len(rounds) >= 3, f"rollout saw only {len(rounds)} round(s)"


def test_action_history_cursor_matches_buffer_contents(rollout):
    """The per-round cursor and the buffer must never disagree.

    Under the old ``step_count`` indexing this failed the moment ``step_count``
    outran the round: the cursor kept climbing while every write past slot 199
    was dropped by JAX's out-of-bounds scatter.
    """
    for state, obs in rollout:
        history = np.asarray(obs["action_history"])
        occupied = int((history[1] >= 0).sum())
        assert occupied == int(state.round_state.round_step)
        # Entries are dense from 0: no holes, no writes past the cursor.
        assert bool((history[1][:occupied] >= 0).all())
        assert bool((history[1][occupied:] == -1).all())


def test_action_history_is_populated_in_every_round(rollout):
    """After the first action of a round the history must not be empty.

    This is the regression itself: from round 3 onward the old code handed the
    policy an all -1 tensor.
    """
    empty_but_should_not_be = [
        (int(s.step_count), int(s.round_state.round))
        for s, o in rollout
        if int(s.round_state.round_step) > 0
        and int((np.asarray(o["action_history"])[1] >= 0).sum()) == 0
    ]
    assert empty_but_should_not_be == []


def test_action_history_resets_at_round_boundaries(rollout):
    """A new round starts from an empty buffer, not from wherever the last one ended."""
    seen_reset = False
    prev_key, prev_step = None, None
    for state, obs in rollout:
        round_key = (int(state.round_state.round), int(state.round_state.honba))
        cursor = int(state.round_state.round_step)
        if prev_key is not None and round_key != prev_key:
            assert cursor < prev_step, (
                f"cursor did not reset at the {prev_key} -> {round_key} boundary "
                f"({prev_step} -> {cursor})"
            )
            seen_reset = True
        prev_key, prev_step = round_key, cursor
    assert seen_reset, "rollout never crossed a round boundary"


def test_history_never_overflows_under_random_play(rollout):
    """200 slots hold a random-play round; if this ever trips, widen the buffer."""
    assert not any(bool(s.round_state.history_overflow) for s, _ in rollout)


def test_relative_player_channel_is_self_consistent(rollout):
    """Channel 0 is the actor's seat relative to the observer, 0 == me."""
    for state, obs in rollout:
        history = np.asarray(obs["action_history"])
        raw = np.asarray(state.round_state.action_history)
        valid = raw[0] >= 0
        expected = (raw[0][valid].astype(np.int32) - int(state.current_player)) % 4
        np.testing.assert_array_equal(history[0][valid], expected.astype(np.int8))
        np.testing.assert_array_equal(history[0][~valid], raw[0][~valid])


def test_shanten_count_is_never_stale(rollout):
    """``shanten_count`` must describe the hand the observer is actually holding.

    The round-advance path rebuilt RoundState from defaults, so this used to read
    0 (== tenpai) on the first observation of every round after the first.
    """
    for state, obs in rollout:
        hand = state.players.hand[state.current_player]
        expected = int(Shanten.number(hand))
        assert int(obs["shanten_count"]) == expected, (
            f"step_count={int(state.step_count)} round={int(state.round_state.round)}: "
            f"obs said {int(obs['shanten_count'])}, hand is {expected}-shanten"
        )


def test_shanten_count_is_in_range(rollout):
    for _, obs in rollout:
        assert -1 <= int(obs["shanten_count"]) <= 6


def test_discard_shanten_agrees_with_shanten_discard(rollout):
    """Column-wise min over the three hand shapes must reproduce ``Shanten.discard``."""
    for state, obs in rollout:
        hand = np.asarray(state.players.hand[state.current_player])
        detailed = np.asarray(obs["discard_shanten"])
        assert detailed.shape == (34, 3)
        held = hand > 0
        reference = np.asarray(Shanten.discard(state.players.hand[state.current_player]))
        np.testing.assert_array_equal(detailed[held].min(axis=1), reference[held])


def test_discard_shanten_sentinel_is_unambiguous(rollout):
    """``NOT_IN_HAND`` must never collide with a real value."""
    for state, obs in rollout:
        hand = np.asarray(state.players.hand[state.current_player])
        detailed = np.asarray(obs["discard_shanten"])
        held = hand > 0
        assert bool((detailed[~held] == Shanten.NOT_IN_HAND).all())
        # Real entries stay strictly below the sentinel in every column.
        assert bool((detailed[held] < Shanten.NOT_IN_HAND).all())
        assert bool((detailed[held] >= 0).all())
        # Per-column bounds. Note the normal column is NOT capped at 6: the overall
        # shanten is, but only because seven pairs / thirteen orphans undercut it on
        # the hands where the normal decomposition is worst. Thirteen orphans is the
        # widest column and is what NOT_IN_HAND has to clear.
        assert bool((detailed[held][:, 0] <= 8).all())
        assert bool((detailed[held][:, 1] <= 13).all())
        assert bool((detailed[held][:, 2] <= 13).all())


def test_is_discard_node_matches_the_legal_actions(rollout):
    for state, obs in rollout:
        legal = np.asarray(state.legal_action_mask)
        expected = bool(legal[:37].any() or legal[71])  # discards, or tsumogiri
        assert bool(obs["is_discard_node"]) == expected


def test_last_draw_is_never_another_players_draw(rollout):
    """``last_draw`` must be the observer's own tile or -1, never someone else's."""
    for state, obs in rollout:
        shown = int(obs["last_draw"])
        if shown < 0:
            continue
        # If it is reported, the observer must be able to act on it, and it must
        # actually be in their concealed hand.
        assert bool(obs["is_discard_node"]) or bool(state.legal_action_mask[73])
        hand = np.asarray(state.players.hand_with_red[state.current_player])
        assert hand[shown] > 0, (
            f"last_draw={shown} reported to seat {int(state.current_player)} "
            f"who does not hold it"
        )


def test_last_draw_is_masked_during_a_robbing_kan_window():
    """The kan declarer's private draw must not leak to the chankan responder.

    ``round_state.last_draw`` is round-level and is not cleared when the added-kan
    branch switches ``current_player`` to the ron candidate, so passing it through
    would show the responder a tile only the declarer has drawn.
    """
    from mahjax.red_mahjong.state import default_state

    declarer, responder = 0, 1
    private_draw = 30  # a tile only the declarer has seen
    state = default_state()
    hand_with_red = state.players.hand_with_red.at[responder, 5].set(1)
    legal = state.players.legal_action_mask.at[responder, 74].set(True)  # RON
    legal = legal.at[responder, 84].set(True)  # PASS
    state = state.replace(
        current_player=jnp.int8(responder),
        legal_action_mask=legal[responder],
        players=state.players.replace(hand_with_red=hand_with_red, legal_action_mask=legal),
        round_state=state.round_state.replace(
            last_draw=jnp.int8(private_draw),
            last_player=jnp.int8(declarer),
            target=jnp.int8(9),
            kan_declared=jnp.bool_(True),
        ),
    )

    obs = _observe_dict(state)
    assert not bool(obs["is_discard_node"]), "chankan node is not a discard node"
    assert int(obs["last_draw"]) == -1, (
        f"leaked the declarer's draw ({private_draw}) to the responder"
    )
    # The public part of the same node is still reported.
    assert int(obs["target"]) == 9
    assert int(obs["last_player"]) == (declarer - responder) % 4


def test_last_draw_is_reported_to_the_player_who_drew_it():
    """The mask must not cost the drawer their own information."""
    env = RedMahjong(round_mode="half")
    state = env.init(jax.random.PRNGKey(11))
    obs = _observe_dict(state)
    assert bool(obs["is_discard_node"]), "the dealer's opening node should allow a discard"
    assert int(obs["last_draw"]) == int(state.round_state.last_draw)
    assert int(obs["last_draw"]) >= 0


def test_is_discard_node_is_false_at_terminal_states():
    """``step`` sets ``legal_action_mask`` to all-True once the game is over.

    Read naively that makes every terminal observation look like a discard node,
    with a ``discard_shanten`` describing a hand nobody will ever play.
    """
    env = RedMahjong(round_mode="single")
    step = jax.jit(env.step)
    key = jax.random.PRNGKey(3)
    key, sub = jax.random.split(key)
    state = env.init(sub)

    for _ in range(400):
        if bool(state.terminated):
            break
        legal = np.flatnonzero(np.asarray(state.legal_action_mask))
        key, pick_key, step_key = jax.random.split(key, 3)
        action = jnp.int32(np.asarray(jax.random.choice(pick_key, legal)))
        state = step(state, action, step_key)

    assert bool(state.terminated), "rollout never terminated"
    assert bool(np.asarray(state.legal_action_mask).all()), "expected the all-True mask"
    assert not bool(_observe_dict(state)["is_discard_node"])


def test_is_discard_node_is_true_for_3n_plus_2_hands(rollout):
    """At a discard node the concealed hand is 3n+2, which is what makes
    ``discard_shanten`` mean 'shanten if I discard this'."""
    for state, obs in rollout:
        n_tiles = int(np.asarray(state.players.hand[state.current_player]).sum())
        if bool(obs["is_discard_node"]):
            assert n_tiles % 3 == 2, f"discard node with a {n_tiles}-tile hand"


def test_target_and_last_player_are_relative_and_consistent(rollout):
    for state, obs in rollout:
        assert int(obs["target"]) == int(state.round_state.target)
        assert -1 <= int(obs["target"]) <= 36
        raw_last = int(state.round_state.last_player)
        if raw_last < 0:
            assert int(obs["last_player"]) == -1
        else:
            assert int(obs["last_player"]) == (raw_last - int(state.current_player)) % 4
            assert 0 <= int(obs["last_player"]) <= 3


def test_target_is_present_whenever_a_call_is_legal(rollout):
    """A PON/CHI/OPEN_KAN/RON decision is meaningless without the tile it is about."""
    call_actions = [74, 75, 76, 77, 78, 79, 80, 81, 82, 83]  # RON, PON.., CHI..
    for state, obs in rollout:
        legal = np.asarray(state.legal_action_mask)
        if legal[call_actions].any():
            assert int(obs["target"]) >= 0, "call node with no target tile"
            assert int(obs["last_player"]) >= 0, "call node with no discarder"


def test_river_is_rotated_and_matches_the_raw_state(rollout):
    from mahjax.red_mahjong.tile import River

    for state, obs in rollout:
        c_p = int(state.current_player)
        order = (np.arange(4) + c_p) % 4
        expected = np.asarray(River.decode_river(state.players.river[order]))
        river = np.asarray(obs["river"])
        assert river.shape == (6, 4, 24)
        np.testing.assert_array_equal(river, expected.astype(np.int8))
        # Row 0 is the observer.
        own = np.asarray(River.decode_river(state.players.river[c_p : c_p + 1]))
        np.testing.assert_array_equal(river[:, 0, :], own[:, 0, :].astype(np.int8))


def test_river_tiles_match_the_discard_counts(rollout):
    for state, obs in rollout:
        c_p = int(state.current_player)
        river = np.asarray(obs["river"])
        counts = np.asarray(state.players.discard_counts)[(np.arange(4) + c_p) % 4]
        for row in range(4):
            occupied = int((river[0, row] >= 0).sum())
            assert occupied == int(counts[row])
            # Dense from 0, like the history buffer.
            assert bool((river[0, row][:occupied] >= 0).all())


def test_river_src_frame_recovers_the_actual_caller(rollout):
    """``src`` is (discarder - caller) mod 4 stored on the DISCARDER's row.

    Getting this frame backwards is silent and plausible-looking, so check it
    against the melds, which record the same call from the caller's side.
    """
    for state, obs in rollout:
        river = np.asarray(obs["river"])
        melds = np.asarray(obs["melds"])
        tile, called_away, src, meld_type = river[0], river[2], river[4], river[5]
        for row in range(4):
            for slot in range(24):
                if not called_away[row, slot]:
                    continue
                caller = (row - int(src[row, slot])) % 4
                assert caller != row, "a player cannot call their own discard"
                # The caller must own a meld formed from this tile, and that meld's
                # own src must point back at this row.
                owned = [
                    m
                    for m in range(4)
                    if melds[0, caller, m] >= 0
                    and int(melds[1, caller, m]) == int(tile[row, slot])
                    and (caller + int(melds[2, caller, m])) % 4 == row
                ]
                assert owned, (
                    f"river says seat {caller} called tile {int(tile[row, slot])} from "
                    f"seat {row} (meld_type={int(meld_type[row, slot])}) but that seat "
                    f"has no matching meld: {melds[:, caller]}"
                )


def test_melds_match_the_raw_state_and_counts(rollout):
    from mahjax.red_mahjong.meld import Meld

    for state, obs in rollout:
        c_p = int(state.current_player)
        order = (np.arange(4) + c_p) % 4
        raw = state.players.melds[order]
        melds = np.asarray(obs["melds"])
        assert melds.shape == (3, 4, 4)
        np.testing.assert_array_equal(melds[0], np.asarray(Meld.action(raw)).astype(np.int8))
        np.testing.assert_array_equal(melds[1], np.asarray(Meld.target_tile(raw)).astype(np.int8))
        np.testing.assert_array_equal(melds[2], np.asarray(Meld.src(raw)).astype(np.int8))
        counts = np.asarray(state.players.meld_counts)[order]
        for row in range(4):
            assert int((melds[0, row] >= 0).sum()) == int(counts[row])


def test_own_melds_are_visible_even_though_hand_excludes_them(rollout):
    """The concealed hand shrinks by 3 per meld; the melds themselves must show up."""
    saw_a_meld = False
    for state, obs in rollout:
        melds = np.asarray(obs["melds"])
        n_own = int((melds[0, 0] >= 0).sum())
        if n_own == 0:
            continue
        saw_a_meld = True
        n_concealed = int(np.asarray(state.players.hand[state.current_player]).sum())
        assert n_concealed % 3 == (2 if bool(obs["is_discard_node"]) else 1)
        assert n_concealed <= 14 - 3 * n_own
    if not saw_a_meld:
        pytest.skip("random rollout produced no melds")


def test_meld_actions_are_valid_call_ids(rollout):
    for _, obs in rollout:
        melds = np.asarray(obs["melds"])
        actions = melds[0][melds[0] >= 0]
        # 37-70 self-kan, 75-77 pon/pon_red/open_kan, 78-83 chi variants.
        assert bool((((actions >= 37) & (actions <= 70)) | ((actions >= 75) & (actions <= 83))).all()), actions
        srcs = melds[2][melds[0] >= 0]
        assert bool(((srcs >= 0) & (srcs <= 3)).all())




def test_wall_remaining_starts_at_seventy():
    """A fresh round has exactly 70 drawable tiles (136 - 14 dead wall - 13*4 hands)."""
    env = RedMahjong(round_mode="half")
    state = env.init(jax.random.PRNGKey(5))
    # init() already dealt and drew for the dealer, so one tile is off the wall.
    assert int(_observe_dict(state)["wall_remaining"]) == 69


def test_wall_remaining_matches_the_visualization_formula(rollout):
    """Single source of truth: the same count the board renderer prints."""
    for state, obs in rollout:
        expected = max(
            int(state.round_state.next_deck_ix) - int(state.round_state.last_deck_ix) + 1, 0
        )
        assert int(obs["wall_remaining"]) == expected


def test_wall_remaining_is_in_range_and_decreases_within_a_round(rollout):
    prev_key, prev_wall = None, None
    saw_a_decrease = False
    for state, obs in rollout:
        wall = int(obs["wall_remaining"])
        assert 0 <= wall <= 70, wall
        key = (int(state.round_state.round), int(state.round_state.honba))
        if prev_key == key:
            # Within a round the live wall only shrinks (kan advances the haitei
            # line, which shrinks it too).
            assert wall <= prev_wall, f"wall grew {prev_wall} -> {wall} inside a round"
            if wall < prev_wall:
                saw_a_decrease = True
        prev_key, prev_wall = key, wall
    assert saw_a_decrease


def test_wall_remaining_zero_agrees_with_the_envs_haitei_flag(rollout):
    """``is_haitei`` is set on the state produced by drawing the last live tile, and
    that same draw is what takes the count to 0 -- so the flag implies an empty wall.
    """
    saw_haitei = False
    for state, obs in rollout:
        if bool(state.round_state.is_haitei):
            saw_haitei = True
            assert int(obs["wall_remaining"]) == 0, (
                f"is_haitei with {int(obs['wall_remaining'])} tiles left"
            )
    if not saw_haitei:
        pytest.skip("random rollout never reached haitei")


def _expected_tiles_seen(state) -> np.ndarray:
    """Independent recount of visible tiles, written from the rules rather than by
    mirroring the implementation.

    Walks the raw packed state via the public decoders and expands each meld by its
    action id, so a bug in the observation's own reconstruction cannot cancel out.
    """
    from mahjax.red_mahjong.meld import EMPTY_MELD, Meld
    from mahjax.red_mahjong.tile import River, Tile

    seen = np.zeros(34, dtype=np.int32)
    seen += np.asarray(state.players.hand[state.current_player]).astype(np.int32)

    decoded = np.asarray(River.decode_river(state.players.river))  # (6, 4, 24)
    tiles, called_away = decoded[0], decoded[2]
    for seat in range(4):
        for slot in range(24):
            t = int(tiles[seat, slot])
            if t < 0 or called_away[seat, slot]:
                continue  # called tiles are counted on the meld side
            seen[int(Tile.to_tile_type(t))] += 1

    for seat in range(4):
        for slot in range(4):
            m = state.players.melds[seat, slot]
            if int(m) == int(EMPTY_MELD):
                continue
            action = int(Meld.action(m))
            target = int(Meld.target(m))
            if 78 <= action <= 83:  # chi variants
                low = target - int(Meld._chi_index(action))
                for off in range(3):
                    seen[low + off] += 1
            elif action == 77 or 37 <= action <= 70:  # open / closed / added kan
                seen[target] += 4
            else:  # pon / pon_red
                seen[target] += 3

    for ind in np.asarray(state.round_state.dora_indicators):
        if int(ind) >= 0:
            seen[int(Tile.to_tile_type(int(ind)))] += 1
    return seen


def test_tiles_seen_matches_an_independent_recount(rollout):
    for state, obs in rollout:
        np.testing.assert_array_equal(
            np.asarray(obs["tiles_seen"]).astype(np.int32), _expected_tiles_seen(state)
        )


def test_tiles_seen_never_exceeds_four(rollout):
    """The hard invariant: only four of each tile exist.

    Deliberately not clipped in the implementation, so a double count (the obvious
    failure mode here -- a called tile lives in both a river and a meld) surfaces
    as a test failure rather than being silently capped.
    """
    for state, obs in rollout:
        seen = np.asarray(obs["tiles_seen"])
        assert seen.min() >= 0
        assert seen.max() <= 4, (
            f"tile type {int(seen.argmax())} counted {int(seen.max())} times"
        )


def test_tiles_seen_counts_a_called_tile_exactly_once(rollout):
    """Regression for the double-count: rivers keep called tiles, melds repeat them."""
    saw_a_call = False
    for state, obs in rollout:
        river = np.asarray(obs["river"])
        if not river[2].any():
            continue
        saw_a_call = True
        np.testing.assert_array_equal(
            np.asarray(obs["tiles_seen"]).astype(np.int32), _expected_tiles_seen(state)
        )
        # And the naive sum (rivers in full) would have over-counted.
        naive_extra = int(river[2].sum())
        assert naive_extra > 0
    if not saw_a_call:
        pytest.skip("random rollout produced no calls")


def test_tiles_seen_includes_own_hand_and_dora(rollout):
    for state, obs in rollout:
        seen = np.asarray(obs["tiles_seen"]).astype(np.int32)
        hand = np.asarray(state.players.hand[state.current_player]).astype(np.int32)
        # Every concealed tile the observer holds must be accounted for.
        assert bool((seen >= hand).all())
    # Dora indicators are visible too.
    state, obs = rollout[-1]
    from mahjax.red_mahjong.tile import Tile

    for ind in np.asarray(state.round_state.dora_indicators):
        if int(ind) >= 0:
            assert int(np.asarray(obs["tiles_seen"])[int(Tile.to_tile_type(int(ind)))]) >= 1


def test_scores_are_rotated_to_the_observer(rollout):
    for state, obs in rollout:
        c_p = int(state.current_player)
        raw = np.asarray(state.round_state.score)
        np.testing.assert_array_equal(np.asarray(obs["scores"]), np.roll(raw, -c_p))


def test_observation_keys_and_shapes_are_stable(rollout):
    expected = {
        "hand": (14,),
        "last_draw": (),
        "action_history": (3, 200),
        "shanten_count": (),
        "discard_shanten": (34, 3),
        "is_discard_node": (),
        "furiten": (),
        "target": (),
        "last_player": (),
        "river": (6, 4, 24),
        "melds": (3, 4, 4),
        "tiles_seen": (34,),
        "scores": (4,),
        "round": (),
        "honba": (),
        "kyotaku": (),
        "wall_remaining": (),
        "prevalent_wind": (),
        "seat_wind": (),
        "dora_indicators": (MAX_DORA_INDICATORS,),
    }
    for _, obs in rollout:
        assert set(obs) == set(expected)
        for key, shape in expected.items():
            assert jnp.shape(obs[key]) == shape, f"{key}: {jnp.shape(obs[key])} != {shape}"


def test_observation_module_imports_standalone():
    """``observation`` must not require ``env`` to have been imported first.

    The circular import cost a real benchmark job (speed_benchmark job 9343550
    died at ``from mahjax.red_mahjong.observation import _observe_dict``).
    """
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", "import mahjax.red_mahjong.observation as m; assert m._observe_dict"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_observation_shape_property_does_not_raise():
    env = RedMahjong(round_mode="half")
    shapes = env.observation_shape
    assert shapes["discard_shanten"] == (34, 3)
    assert shapes["target"] == ()
