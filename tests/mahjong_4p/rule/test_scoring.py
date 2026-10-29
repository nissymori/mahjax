from __future__ import annotations

import jax
import jax.numpy as jnp

from mahjax.mahjong_4p.action import Action
from mahjax.mahjong_4p.config import GameConfig
from mahjax.mahjong_4p.logic import _abortive_draw_normal, _mangan_tsumo, _pao, _pass, _ron
from mahjax.mahjong_4p.meld import Meld
from mahjax.mahjong_4p.state import default_state
from mahjax.mahjong_4p.tile import Tile
from mahjax.mahjong_4p.yaku import Yaku

from ..helpers import STEP_KEY, compiled_env


def test_pao_and_nagashi_mangan_scoring() -> None:
    base = default_state()
    melds = base.players.melds.at[0, 0].set(Meld.init(Action.PON, 31, 1))
    melds = melds.at[0, 1].set(Meld.init(Action.PON, 32, 3))
    melds = melds.at[0, 2].set(Meld.init(Action.OPEN_KAN, 33, 2))
    ron_state = base.replace(
        current_player=jnp.int8(0),
        players=base.players.replace(
            melds=melds,
            meld_counts=base.players.meld_counts.at[0].set(3),
            fan=base.players.fan.at[0, 0].set(jnp.int32(5)),
            fu=base.players.fu.at[0, 0].set(jnp.int32(30)),
        ),
        round_state=base.round_state.replace(
            dealer=jnp.int8(1),
            last_player=jnp.int8(1),
            honba=jnp.int8(1),
            score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        ),
    )

    is_pao, pao_player = _pao(ron_state, jnp.int8(0))
    ron_after = _ron(ron_state)
    basic = Yaku.score(jnp.int32(5), jnp.int32(30))
    ron_payment = jnp.ceil(basic * 4 / 100)
    # P1 discarded and P2 is liable: the points are split, the honba is P2's alone.
    expected_ron = jnp.array(
        [ron_payment + 3, -ron_payment / 2, -ron_payment / 2 - 3, 0],
        dtype=jnp.float32,
    )

    assert bool(is_pao)
    assert int(pao_player) == 2
    assert jnp.array_equal(ron_after.rewards, expected_ron)

    nagashi_state = base.replace(
        players=base.players.replace(
            has_nagashi_mangan=jnp.array([True, False, False, False], dtype=jnp.bool_)
        ),
        round_state=base.round_state.replace(
            dealer=jnp.int8(0),
            honba=jnp.int8(1),
            score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        ),
    )
    nagashi_after = _abortive_draw_normal(nagashi_state)
    expected_nagashi = _mangan_tsumo(jnp.int8(0), jnp.int8(0), jnp.int8(0))

    assert jnp.array_equal(nagashi_after.rewards, expected_nagashi.astype(jnp.float32))


def test_double_ron_config_chains_ron_and_pass_resolution() -> None:
    base = default_state()
    legal_action_mask = base.players.legal_action_mask
    legal_action_mask = legal_action_mask.at[0, Action.RON].set(True)
    legal_action_mask = legal_action_mask.at[1, Action.RON].set(True)
    ron_state = base.replace(
        current_player=jnp.int8(0),
        legal_action_mask=legal_action_mask[0],
        players=base.players.replace(
            legal_action_mask=legal_action_mask,
            fan=base.players.fan.at[0, 0].set(jnp.int32(5)),
            fu=base.players.fu.at[0, 0].set(jnp.int32(30)),
        ),
        round_state=base.round_state.replace(
            dealer=jnp.int8(2),
            last_player=jnp.int8(3),
            honba=jnp.int8(1),
            kyotaku=jnp.int8(2),
            score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        ),
    )
    config = GameConfig(allow_double_ron=True)

    after_first_ron = _ron(ron_state, config)

    assert not bool(after_first_ron.round_state.terminated_round)
    assert int(after_first_ron.current_player) == 1
    assert bool(after_first_ron.players.legal_action_mask[1, Action.RON])
    assert bool(after_first_ron.players.legal_action_mask[1, Action.PASS])
    # Nothing is paid until P1 has answered too, so P1 cannot tell whether P0 won.
    assert int(after_first_ron.round_state.kyotaku) == 2
    assert jnp.array_equal(after_first_ron.round_state.score, ron_state.round_state.score)
    assert not bool(after_first_ron.players.has_won.any())

    after_pass = _pass(after_first_ron, config)

    assert bool(after_pass.round_state.terminated_round)
    assert bool(after_pass.players.legal_action_mask[:, Action.DUMMY].all())
    assert int(after_pass.round_state.kyotaku) == 0
    # Declining the second ron closes the window, and P0's ron is paid then.
    assert jnp.array_equal(after_pass.rewards, jnp.array([103, 0, 0, -83], dtype=jnp.float32))
    assert jnp.array_equal(
        after_pass.round_state.score, jnp.int32(ron_state.round_state.score + after_pass.rewards)
    )
    assert not bool(after_pass.pending_rewards.any())


def test_single_ron_when_double_ron_is_disabled() -> None:
    base = default_state()
    legal_action_mask = base.players.legal_action_mask
    legal_action_mask = legal_action_mask.at[0, Action.RON].set(True)
    legal_action_mask = legal_action_mask.at[1, Action.RON].set(True)
    ron_state = base.replace(
        current_player=jnp.int8(0),
        legal_action_mask=legal_action_mask[0],
        players=base.players.replace(
            legal_action_mask=legal_action_mask,
            fan=base.players.fan.at[0, 0].set(jnp.int32(5)),
            fu=base.players.fu.at[0, 0].set(jnp.int32(30)),
        ),
        round_state=base.round_state.replace(
            dealer=jnp.int8(2),
            last_player=jnp.int8(3),
            honba=jnp.int8(1),
            score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        ),
    )
    config = GameConfig(allow_double_ron=False)

    after_ron = _ron(ron_state, config)

    assert bool(after_ron.round_state.terminated_round)
    assert bool(after_ron.players.legal_action_mask[:, Action.DUMMY].all())


def test_pao_can_be_disabled_by_game_config() -> None:
    base = default_state()
    melds = base.players.melds.at[0, 0].set(Meld.init(Action.PON, 31, 1))
    melds = melds.at[0, 1].set(Meld.init(Action.PON, 32, 3))
    melds = melds.at[0, 2].set(Meld.init(Action.OPEN_KAN, 33, 2))
    ron_state = base.replace(
        current_player=jnp.int8(0),
        players=base.players.replace(
            melds=melds,
            meld_counts=base.players.meld_counts.at[0].set(3),
            fan=base.players.fan.at[0, 0].set(jnp.int32(5)),
            fu=base.players.fu.at[0, 0].set(jnp.int32(30)),
        ),
        round_state=base.round_state.replace(
            dealer=jnp.int8(1),
            last_player=jnp.int8(1),
            honba=jnp.int8(1),
            score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        ),
    )
    config = GameConfig(enable_pao=False)
    basic = Yaku.score(jnp.int32(5), jnp.int32(30))
    ron_payment = jnp.ceil(basic * 4 / 100)

    after_ron = _ron(ron_state, config)

    assert jnp.array_equal(
        after_ron.rewards,
        jnp.array([ron_payment + 3, -ron_payment - 3, 0, 0], dtype=jnp.float32),
    )


def test_pao_survives_an_added_kan() -> None:
    """Adding the fourth tile to a called dragon pon does not change who is liable."""
    base = default_state()
    srcs = (1, 3, 2)  # White from P1, Green from P3, Red (the third) from P2
    for kanned in range(3):
        melds = base.players.melds
        for i, src in enumerate(srcs):
            tile = 31 + i
            action = Tile.NUM_TILE_TYPE_WITH_RED + tile if i == kanned else Action.PON
            melds = melds.at[0, i].set(Meld.init(action, tile, src))
        state = base.replace(
            players=base.players.replace(melds=melds, meld_counts=base.players.meld_counts.at[0].set(3))
        )

        is_pao, pao_player = _pao(state, jnp.int8(0))

        assert bool(is_pao)
        assert int(pao_player) == 2


def test_red_thirteen_riichi_sticks_do_not_wrap_negative() -> None:
    """13 sticks are +13000 points; multiplying the int8 ``kyotaku`` wrapped them."""
    from mahjax.mahjong_4p import logic as m

    base = default_state()
    state = m._replace_state(
        base,
        current_player=jnp.int8(1),
        last_player=jnp.int8(2),
        kyotaku=jnp.int8(13),
        next_deck_ix=jnp.int32(50),
        score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
        fan=base.players.fan.at[1, 0].set(jnp.int32(1)),
        fu=base.players.fu.at[1, 0].set(jnp.int32(30)),
    )

    assert float(m._ron(state).rewards[1]) == 10 + 130  # 1 han 30 fu ron + the sticks
    assert float(m._tsumo(state).rewards[1]) == 11 + 130  # 300/500 tsumo + the sticks
    assert int(m._final_score(state.round_state)[0]) == 250 + 130


def test_red_limit_hand_rons_pay_the_limit() -> None:
    """3 fan 70 fu is a mangan and 23 fan 100 fu a counted yakuman, not an int32 overflow."""
    from mahjax.mahjong_4p import logic as m

    base = default_state()
    for fan, fu, points in ((3, 70, 80), (23, 100, 320)):
        state = m._replace_state(
            base,
            current_player=jnp.int8(1),
            last_player=jnp.int8(2),
            next_deck_ix=jnp.int32(50),
            score=jnp.array([250, 250, 250, 250], dtype=jnp.int32),
            fan=base.players.fan.at[1, 0].set(jnp.int32(fan)),
            fu=base.players.fu.at[1, 0].set(jnp.int32(fu)),
        )

        assert m._ron(state).rewards.tolist() == [0, points, -points, 0]


def test_red_rewards_report_only_the_current_transition() -> None:
    env = compiled_env(round_mode="east", next_round_style="dummy_share")
    step = env.step
    key = jax.random.PRNGKey(11)
    key, sub = jax.random.split(key)
    state = env.init(sub)

    for _ in range(1500):
        before = state.round_state.score
        key, action_key, step_key = jax.random.split(key, 3)
        action = jax.random.categorical(
            action_key, jnp.log(jnp.where(state.legal_action_mask, 1.0, 0.0))
        )
        state = step(state, action.astype(jnp.int8), step_key)
        delta = (state.round_state.score - before).astype(jnp.float32)
        assert jnp.allclose(state.rewards, delta), (
            f"rewards {state.rewards} but the score moved by {delta}"
        )
        if bool(state.terminated) or bool(state.truncated):
            break


def test_red_multi_ron_reward_stream_matches_the_score_movement() -> None:
    env = compiled_env(round_mode="single", next_round_style="auto")
    base = default_state()
    legal_action_mask = base.players.legal_action_mask
    for player in range(3):
        legal_action_mask = legal_action_mask.at[player, Action.RON].set(True)
    state = base.replace(
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
    start = state.round_state.score
    total = jnp.zeros(4, dtype=jnp.float32)

    for action in (Action.RON, Action.PASS):
        state = env.step(state, jnp.int8(action), STEP_KEY)
        total = total + state.rewards

    assert jnp.allclose(total, (state.round_state.score - start).astype(jnp.float32))
