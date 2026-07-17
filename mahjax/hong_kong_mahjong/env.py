"""Hong Kong Old Style mahjong environment."""

from typing import Dict, Literal, Optional, Tuple

import jax
import jax.numpy as jnp

from mahjax._src.types import Array, PRNGKey
from mahjax.core import Env
from mahjax.hong_kong_mahjong.action import Action
from mahjax.hong_kong_mahjong.hand import Hand
from mahjax.hong_kong_mahjong.meld import Meld
from mahjax.hong_kong_mahjong.rules import HKOS_V1, Rules
from mahjax.hong_kong_mahjong.scoring import HongKongScoring
from mahjax.hong_kong_mahjong.state import PlayerStateArrays, RoundState, State
from mahjax.hong_kong_mahjong.tile import Tile

FALSE = jnp.bool_(False)
TRUE = jnp.bool_(True)
ZERO_MASK = jnp.zeros(Action.NUM_ACTION, dtype=jnp.bool_)
ZERO_MASK_4P = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)


def _replace_state(state: State, **updates) -> State:
    player_names = set(PlayerStateArrays.__annotations__)
    round_names = set(RoundState.__annotations__)
    player_updates = {k: v for k, v in updates.items() if k in player_names}
    round_updates = {k: v for k, v in updates.items() if k in round_names}
    env_updates = {k: v for k, v in updates.items() if k not in player_names | round_names}
    players = state.players.replace(**player_updates) if player_updates else state.players
    round_state = state.round_state.replace(**round_updates) if round_updates else state.round_state
    current = env_updates.get("current_player", state.current_player)
    if "legal_action_mask" in env_updates and getattr(env_updates["legal_action_mask"], "ndim", 0) == 2:
        players = players.replace(legal_action_mask=env_updates.pop("legal_action_mask"))
    if "legal_action_mask" not in env_updates:
        env_updates["legal_action_mask"] = players.legal_action_mask[current]
    return state.replace(players=players, round_state=round_state, **env_updates)


def _seat_winds(dealer: Array) -> Array:
    return ((jnp.arange(4) - dealer) % 4).astype(jnp.int8)


def _draw_nonflower(
    deck: Array, wall_index: Array, hand: Array, flowers: Array, player: Array
) -> Tuple[Array, Array, Array, Array]:
    """Draw through any flowers and return the first standard tile."""
    found = FALSE
    drawn = jnp.int8(-1)

    def body(_, carry):
        ix, h, f, done, result = carry
        available = ix < Tile.NUM_TILE_ID
        tile = jnp.where(available, deck[jnp.minimum(ix, Tile.NUM_TILE_ID - 1)], -1)
        active = available & ~done
        flower = active & Tile.is_flower(tile)
        flower_ix = jnp.clip(tile - Tile.FLOWER_START, 0, 7)
        f = f.at[player, flower_ix].set(f[player, flower_ix] | flower)
        standard = active & Tile.is_standard(tile)
        safe_tile = jnp.clip(tile, 0, 33)
        h = h.at[player, safe_tile].add(standard.astype(jnp.int8))
        result = jnp.where(standard, tile, result)
        return ix + active.astype(jnp.int16), h, f, done | standard, result

    wall_index, hand, flowers, found, drawn = jax.lax.fori_loop(0, 9, body, (wall_index, hand, flowers, found, drawn))
    return wall_index, hand, flowers, drawn


def _deal(deck: Array, dealer: Array) -> Tuple[Array, Array, Array, Array]:
    hand = jnp.zeros((4, 34), dtype=jnp.int8)
    flowers = jnp.zeros((4, 8), dtype=jnp.bool_)
    wall_index = jnp.int16(0)

    def deal_one(i, carry):
        ix, h, f, _ = carry
        player = jnp.int8(i // 13)
        ix, h, f, tile = _draw_nonflower(deck, ix, h, f, player)
        return ix, h, f, tile

    wall_index, hand, flowers, _ = jax.lax.fori_loop(0, 52, deal_one, (wall_index, hand, flowers, jnp.int8(-1)))
    wall_index, hand, flowers, drawn = _draw_nonflower(deck, wall_index, hand, flowers, dealer)
    return wall_index, hand, flowers, drawn


def _append_meld(state: State, player: Array, action: Array, target: Array, src: Array) -> State:
    meld = Meld.init(action, target, src)
    melds = state.players.melds.at[player, state.players.meld_counts[player]].set(meld)
    counts = state.players.meld_counts.at[player].add(1)
    return _replace_state(state, melds=melds, meld_counts=counts)


def _score_result(state: State, player: Array, tile: Array, is_self_draw: Array, rules: Rules = HKOS_V1):
    # Drawn tiles are already represented in state.hand, while the pattern
    # evaluator accepts the pre-win hand and adds winning_tile itself.
    scoring_hand = jax.lax.cond(
        is_self_draw,
        lambda: Hand.sub(state.players.hand[player], tile),
        lambda: state.players.hand[player],
    )
    return HongKongScoring.judge(
        scoring_hand,
        state.players.melds[player],
        state.players.meld_counts[player],
        tile,
        state.players.flowers[player],
        state.round_state.prevalent_wind,
        state.round_state.seat_wind[player],
        is_self_draw=is_self_draw,
        is_robbing_kong=state.round_state.robbing_kong,
        is_kong_replacement=state.round_state.after_kong,
        is_last_tile=state.round_state.wall_index >= Tile.NUM_TILE_ID,
        is_heavenly_hand=(
            is_self_draw
            & (player == state.round_state.dealer)
            & (state.players.discard_counts.sum() == 0)
            & (state.players.meld_counts.sum() == 0)
        ),
        is_earthly_hand=(
            ~is_self_draw
            & (player != state.round_state.dealer)
            & (state.round_state.last_player == state.round_state.dealer)
            & (state.players.discard_counts.sum() == 1)
            & (state.players.meld_counts.sum() == 0)
        ),
        rules=rules,
    )


def _mask_after_draw(state: State, player: Array, drawn: Array, rules: Rules) -> Array:
    hand = state.players.hand[player]
    mask = ZERO_MASK.at[:34].set(hand > 0)
    mask = mask.at[Action.TSUMOGIRI].set(drawn >= 0)
    self_kong = jax.vmap(lambda t: (hand[t] == 4) | ((hand[t] > 0) & (state.players.pon[player, t] > 0)))(
        jnp.arange(34)
    )
    mask = mask.at[34:68].set(self_kong)
    result = _score_result(state, player, drawn, TRUE, rules)
    can_shape = (drawn >= 0) & Hand.can_tsumo(hand)
    mask = mask.at[Action.TSUMO].set(can_shape & (result.faan >= rules.minimum_faan))
    return mask


def _claim_masks(state: State, discarder: Array, tile: Array, rules: Rules) -> Array:
    def one(player):
        hand = state.players.hand[player]
        mask = ZERO_MASK
        src = (discarder - player) % 4
        chi = jax.vmap(lambda a: Hand.can_chi(hand, tile, a))(jnp.arange(Action.CHI_L, Action.CHI_R + 1)) & (src == 3)
        mask = mask.at[Action.CHI_L : Action.CHI_R + 1].set(chi)
        mask = mask.at[Action.PON].set(Hand.can_pon(hand, tile))
        mask = mask.at[Action.OPEN_KAN].set(Hand.can_open_kan(hand, tile))
        result = _score_result(state, player, tile, FALSE, rules)
        ron = Hand.can_ron(hand, tile) & (result.faan >= rules.minimum_faan)
        mask = mask.at[Action.RON].set(ron)
        return jnp.where(player == discarder, ZERO_MASK, mask)

    return jax.vmap(one)(jnp.arange(4))


def _next_claimant(masks: Array, discarder: Array) -> Tuple[Array, Array]:
    ron = masks[:, Action.RON]
    kong = masks[:, Action.OPEN_KAN]
    pon = masks[:, Action.PON]
    chi = masks[:, Action.CHI_L : Action.CHI_R + 1].any(axis=1)
    priority = jnp.where(ron, 3, jnp.where(kong, 2, jnp.where(pon, 1, jnp.where(chi, 0, -1))))
    best = priority.max()
    candidates = priority == best
    distance = (jnp.arange(4) - discarder) % 4
    claimant = jnp.argmin(jnp.where(candidates, distance, 5)).astype(jnp.int8)
    return claimant, best >= 0


def _draw(state: State, player: Array, rules: Rules, *, after_kong: Array = FALSE) -> State:
    ix, hand, flowers, drawn = _draw_nonflower(
        state.round_state.deck,
        state.round_state.wall_index,
        state.players.hand,
        state.players.flowers,
        player,
    )
    state = _replace_state(
        state,
        current_player=player,
        hand=hand,
        flowers=flowers,
        wall_index=ix,
        last_draw=drawn,
        draw_next=FALSE,
        after_kong=after_kong,
        robbing_kong=FALSE,
        pending_kong_player=jnp.int8(-1),
        legal_action_mask=ZERO_MASK_4P,
    )
    mask = _mask_after_draw(state, player, drawn, rules)
    state = _replace_state(state, legal_action_mask=state.players.legal_action_mask.at[player].set(mask))
    return jax.lax.cond(drawn < 0, lambda: _end_draw(state, rules), lambda: state)


def _discard(state: State, action: Array, rules: Rules) -> State:
    player = state.current_player
    tile = jnp.where(action == Action.TSUMOGIRI, state.round_state.last_draw, action).astype(jnp.int8)
    hand = state.players.hand.at[player, tile].add(-1)
    river = state.players.river.at[player, state.players.discard_counts[player]].set(tile)
    discard_counts = state.players.discard_counts.at[player].add(1)
    state = _replace_state(
        state,
        hand=hand,
        river=river,
        discard_counts=discard_counts,
        last_player=player,
        target=tile,
        last_draw=jnp.int8(-1),
        after_kong=FALSE,
    )
    masks = _claim_masks(state, player, tile, rules)
    claimant, can_claim = _next_claimant(masks, player)
    masks = masks.at[claimant, Action.PASS].set(can_claim)
    claimed = _replace_state(state, current_player=claimant, legal_action_mask=masks)
    return jax.lax.cond(can_claim, lambda: claimed, lambda: _draw(state, (player + 1) % 4, rules))


def _pass(state: State, rules: Rules) -> State:
    masks = state.players.legal_action_mask.at[state.current_player].set(ZERO_MASK)
    claimant, can_claim = _next_claimant(masks, state.round_state.last_player)
    masks = masks.at[claimant, Action.PASS].set(can_claim)
    claimed = _replace_state(state, current_player=claimant, legal_action_mask=masks)
    next_player = jnp.where(
        state.round_state.robbing_kong,
        state.round_state.pending_kong_player,
        (state.round_state.last_player + 1) % 4,
    )
    drawn = _draw(state, next_player, rules, after_kong=state.round_state.robbing_kong)
    return jax.lax.cond(can_claim, lambda: claimed, lambda: drawn)


def _pon(state: State) -> State:
    player, tile, discarder = state.current_player, state.round_state.target, state.round_state.last_player
    src = (discarder - player) % 4
    state = _append_meld(state, player, jnp.int8(Action.PON), tile, src)
    hand = state.players.hand.at[player, tile].add(-2)
    pon = state.players.pon.at[player, tile].set(state.players.meld_counts[player])
    mask = ZERO_MASK.at[:34].set(hand[player] > 0).at[tile].set(FALSE)
    return _replace_state(
        state, hand=hand, pon=pon, target=jnp.int8(-1), legal_action_mask=ZERO_MASK_4P.at[player].set(mask)
    )


def _chi(state: State, action: Array) -> State:
    player, tile = state.current_player, state.round_state.target
    state = _append_meld(state, player, action, tile, jnp.int8(3))
    start = tile - (action - Action.CHI_L)
    consume = jnp.array([start, start + 1, start + 2], dtype=jnp.int8)
    hand = state.players.hand.at[player, consume].add(-1).at[player, tile].add(1)
    mask = ZERO_MASK.at[:34].set(hand[player] > 0).at[tile].set(FALSE)
    return _replace_state(state, hand=hand, target=jnp.int8(-1), legal_action_mask=ZERO_MASK_4P.at[player].set(mask))


def _open_kong(state: State, rules: Rules) -> State:
    player, tile, discarder = state.current_player, state.round_state.target, state.round_state.last_player
    state = _append_meld(state, player, jnp.int8(Action.OPEN_KAN), tile, (discarder - player) % 4)
    state = _replace_state(state, hand=state.players.hand.at[player, tile].add(-3), target=jnp.int8(-1))
    return _draw(state, player, rules, after_kong=TRUE)


def _self_kong(state: State, action: Array, rules: Rules) -> State:
    player = state.current_player
    tile = (action - 34).astype(jnp.int8)
    is_added = state.players.pon[player, tile] > 0

    def added(s):
        idx = s.players.pon[player, tile] - 1
        melds = s.players.melds.at[player, idx].set(Meld.init(action, tile, Meld.src(s.players.melds[player, idx])))
        s = _replace_state(s, hand=s.players.hand.at[player, tile].add(-1), melds=melds)
        masks = jax.vmap(
            lambda p: ZERO_MASK.at[Action.RON].set(
                (p != player)
                & Hand.can_ron(s.players.hand[p], tile)
                & (
                    _score_result(_replace_state(s, robbing_kong=TRUE), p, tile, FALSE, rules).faan
                    >= rules.minimum_faan
                )
            )
        )(jnp.arange(4))
        claimant, can_rob = _next_claimant(masks, player)
        masks = masks.at[claimant, Action.PASS].set(can_rob)
        waiting = _replace_state(
            s,
            current_player=claimant,
            last_player=player,
            target=tile,
            robbing_kong=TRUE,
            pending_kong_player=player,
            legal_action_mask=masks,
        )
        return jax.lax.cond(
            can_rob & rules.robbing_a_kong, lambda: waiting, lambda: _draw(s, player, rules, after_kong=TRUE)
        )

    def closed(s):
        s = _append_meld(s, player, action, tile, jnp.int8(0))
        s = _replace_state(s, hand=s.players.hand.at[player, tile].add(-4))
        return _draw(s, player, rules, after_kong=TRUE)

    return jax.lax.cond(is_added, lambda: added(state), lambda: closed(state))


def _win(state: State, self_draw: Array, rules: Rules) -> State:
    winner = state.current_player
    tile = jnp.where(self_draw, state.round_state.last_draw, state.round_state.target)
    result = _score_result(state, winner, tile, self_draw, rules)
    rewards, _ = HongKongScoring.settle(
        winner,
        state.round_state.last_player,
        result.faan,
        self_draw,
        state.round_state.dealer,
        rules,
    )
    score = state.round_state.score + rewards
    state = _replace_state(
        state,
        rewards=rewards.astype(jnp.float32),
        score=score,
        has_won=state.players.has_won.at[winner].set(TRUE),
        terminated_round=TRUE,
        legal_action_mask=ZERO_MASK_4P,
    )
    return _finish_round(state, rules)


def _end_draw(state: State, rules: Rules) -> State:
    state = _replace_state(state, terminated_round=TRUE, rewards=jnp.zeros(4), legal_action_mask=ZERO_MASK_4P)
    return _finish_round(state, rules)


def _finish_round(state: State, rules: Rules) -> State:
    dealer_won = state.players.has_won[state.round_state.dealer]
    is_draw = ~state.players.has_won.any()
    repeats = (dealer_won & rules.dealer_repeats_on_win) | (is_draw & rules.dealer_repeats_on_draw)
    next_round = state.round_state.round + (~repeats).astype(jnp.int8)
    game_over = next_round > state.round_state.round_limit

    def finish():
        return _replace_state(state, terminated=TRUE)

    def restart():
        dealer = jnp.where(repeats, state.round_state.dealer, (state.round_state.dealer + 1) % 4)
        rng, key = jax.random.split(state.round_state.rng_key)
        new_state = _init_round(
            key,
            dealer,
            next_round,
            state.round_state.round_limit,
            state.round_state.score,
            rules,
        )
        return _replace_state(new_state, rewards=state.rewards, step_count=state.step_count)

    return jax.lax.cond(game_over, finish, restart)


def _init_round(
    key: PRNGKey, dealer: Array, round_number: Array, round_limit: Array, score: Array, rules: Rules
) -> State:
    key, shuffle_key = jax.random.split(key)
    deck = Tile.from_tile_id_to_tile(jax.random.permutation(shuffle_key, jnp.arange(Tile.NUM_TILE_ID)))
    wall_index, hand, flowers, drawn = _deal(deck, dealer)
    state = State(
        current_player=dealer,
        players=PlayerStateArrays(hand=hand, flowers=flowers),
        round_state=RoundState(
            rng_key=key,
            deck=deck,
            wall_index=wall_index,
            dealer=dealer,
            round=round_number,
            round_limit=round_limit,
            score=score,
            seat_wind=_seat_winds(dealer),
            prevalent_wind=(round_number // 4).astype(jnp.int8),
            last_draw=drawn,
        ),
    )
    mask = _mask_after_draw(state, dealer, drawn, rules)
    return _replace_state(state, legal_action_mask=state.players.legal_action_mask.at[dealer].set(mask))


class HongKongMahjong(Env):
    def __init__(self, round_mode: Literal["single", "east", "half"] = "half", rules: Rules = HKOS_V1):
        if round_mode not in ("single", "east", "half"):
            raise ValueError("round_mode must be 'single', 'east', or 'half'")
        self.round_mode = round_mode
        self.rules = rules
        # A negative limit makes the first completed hand terminal even when
        # the dealer would otherwise repeat after a win or draw.
        self.round_limit = {"single": -1, "east": 3, "half": 7}[round_mode]

    def init(self, key: PRNGKey) -> State:
        key, dealer_key = jax.random.split(key)
        dealer = jax.random.randint(dealer_key, (), 0, 4, dtype=jnp.int8)
        return _init_round(
            key, dealer, jnp.int8(0), jnp.int8(self.round_limit), jnp.zeros(4, dtype=jnp.int32), self.rules
        )

    def step(self, state: State, action: Array, key: Optional[Array] = None) -> State:
        del key
        illegal = ~state.legal_action_mask[action]

        def apply(s):
            funcs = [
                lambda: _discard(s, action, self.rules),
                lambda: _self_kong(s, action, self.rules),
                lambda: s,
                lambda: _win(s, FALSE, self.rules),
                lambda: _win(s, TRUE, self.rules),
                lambda: _pon(s),
                lambda: _open_kong(s, self.rules),
                lambda: _chi(s, action),
                lambda: _pass(s, self.rules),
            ]
            kind = jnp.where(
                (action < 34) | (action == Action.TSUMOGIRI),
                0,
                jnp.where(
                    (action >= 34) & (action < 68),
                    1,
                    jnp.where(
                        action == Action.RON,
                        3,
                        jnp.where(
                            action == Action.TSUMO,
                            4,
                            jnp.where(
                                action == Action.PON,
                                5,
                                jnp.where(
                                    action == Action.OPEN_KAN,
                                    6,
                                    jnp.where((action >= Action.CHI_L) & (action <= Action.CHI_R), 7, 8),
                                ),
                            ),
                        ),
                    ),
                ),
            )
            return jax.lax.switch(kind, funcs)

        def penalize(s):
            rewards = jnp.ones(4, dtype=jnp.float32).at[s.current_player].set(-3)
            return _replace_state(s, rewards=rewards, terminated=TRUE)

        already_done = state.terminated | state.truncated
        next_state = jax.lax.cond(
            already_done,
            lambda: _replace_state(state, rewards=jnp.zeros(4)),
            lambda: jax.lax.cond(illegal, lambda: penalize(state), lambda: apply(state)),
        )
        return _replace_state(
            next_state,
            step_count=jnp.where(already_done, state.step_count, state.step_count + 1),
        )

    def observe(self, state: State) -> Dict[str, Array]:
        player = state.current_player
        order = (jnp.arange(4) + player) % 4
        return {
            "hand": state.players.hand[player],
            "flowers": state.players.flowers[player],
            "melds": state.players.melds[player],
            "river": state.players.river[order],
            "scores": state.round_state.score[order],
            "seat_wind": state.round_state.seat_wind[player],
            "prevalent_wind": state.round_state.prevalent_wind,
            "wall_remaining": Tile.NUM_TILE_ID - state.round_state.wall_index,
        }

    @property
    def id(self) -> str:
        return "hong_kong_mahjong"

    @property
    def version(self) -> str:
        return "v1"

    @property
    def num_players(self) -> int:
        return 4

    @property
    def num_actions(self) -> int:
        return Action.NUM_ACTION

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return (34,)

    @property
    def _illegal_action_penalty(self) -> float:
        return -3.0
