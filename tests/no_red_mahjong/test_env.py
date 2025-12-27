import unittest
import jax
import jax.numpy as jnp
from mahjax.no_red_mahjong.tile import Tile
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.state import FIRST_DRAW_IDX
from mahjax.no_red_mahjong.env import (
    _init,
    _step,
    _make_legal_action_mask_after_draw,
    _make_legal_action_mask_after_draw_w_riichi,
    _dora_array,
    _discard,
    _make_legal_action_mask_after_discard,
    _next_meld_player,
    _next_ron_player,
    _selfkan,
    _closed_kan,
    _added_kan,
    _open_kan,
    _pon,
    _chi,
    _pass,
    _riichi,
    _ron,
    _tsumo,
    _accept_riichi,
    _append_meld,
    _draw,
    _draw_after_kan,
    _pass,
    _make_legal_action_mask_after_chi,
    _abortive_draw_normal,
    _next_round,
    _kan,
    NoRedMahjong,
)
env = NoRedMahjong()

jitted_init = jax.jit(_init)
jitted_step = jax.jit(_step)
jitted_draw = jax.jit(_draw)
jitted_draw_after_kan = jax.jit(_draw_after_kan)
jitted_make_legal_action_mask_after_draw = jax.jit(_make_legal_action_mask_after_draw)
jitted_make_legal_action_mask_after_draw_w_riichi = jax.jit(_make_legal_action_mask_after_draw_w_riichi)
jitted_dora_array = jax.jit(_dora_array)
jitted_discard = jax.jit(_discard)
jitted_make_legal_action_mask_after_discard = jax.jit(_make_legal_action_mask_after_discard)
jitted_next_meld_player = jax.jit(_next_meld_player)
jitted_next_ron_player = jax.jit(_next_ron_player)
jitted_accept_riichi = jax.jit(_accept_riichi)
jitted_append_meld = jax.jit(_append_meld)
jitted_kan = jax.jit(_kan)
jitted_selfkan = jax.jit(_selfkan)
jitted_closed_kan = jax.jit(_closed_kan)
jitted_added_kan = jax.jit(_added_kan)
jitted_open_kan = jax.jit(_open_kan)
jitted_pon = jax.jit(_pon)
jitted_chi = jax.jit(_chi)
jitted_make_legal_action_mask_after_chi = jax.jit(_make_legal_action_mask_after_chi)
jitted_pass = jax.jit(_pass)
jitted_riichi = jax.jit(_riichi)
jitted_ron = jax.jit(_ron)
jitted_tsumo = jax.jit(_tsumo)
jitted_abortive_draw_normal = jax.jit(_abortive_draw_normal)
jitted_next_round = jax.jit(_next_round)

IDX_AFTER_FIRST_DRAW = FIRST_DRAW_IDX - 1

def act_randomly(rng, legal_action_mask) -> int:
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits=logits)

def _advance_after_dummy(state, steps: int = 4):
    """Advance the state after the dummy sharing is complete."""
    # If the dummy count is 0, the dummy sharing is complete, so the next round is called.
    for _ in range(steps):
        state = jitted_next_round(state)
        if int(state._dummy_count) == 0:
            # The dummy sharing is complete, so the next round is called.
            break
    return state


class TestEnv(unittest.TestCase):
    def setUp(self):
        rng = jax.random.PRNGKey(1)
        self.state = jitted_init(rng)

    def set_state(self, state, **kwargs):
        for k, v in kwargs.items():
            state = state.replace(  # type:ignore
                **{k: v}
            )
        return state

    def test_init(self):
        # Verify initial state deck/hand layout and zeroed flags.
        state = self.state
        self.assertEqual(jnp.all(state.rewards == 0), True)
        self.assertEqual(state.terminated, 0)
        self.assertEqual(state.truncated, 0)
        self.assertEqual(state._next_deck_ix, IDX_AFTER_FIRST_DRAW)  # the first tile to draw is the 135 - 13 * 4 - 1th tile
        # deck is correctly generated?
        self.assertEqual(state._deck.shape, (136,))
        for i in range(Tile.NUM_TILE_TYPE):
            self.assertEqual((state._deck == i).sum(), 4)
        # hand is correctly generated?
        self.assertEqual(state._hand.shape, (4, Tile.NUM_TILE_TYPE))
        hand = state._hand

        for i in range(136 - (4 * 13), 136):
            player = (i - 136 + (4 * 13)) // 13
            tile = state._deck[i]
            self.assertEqual(hand[player, tile] > 0, True)

    def test_discard(self):
        # Ensure _discard removes exactly one tile from current hand.
        state = self.state
        c_p = state.current_player
        hand = state._hand[c_p]
        logit = jnp.where(hand > 0, 0, -jnp.inf)
        discard_tile = jax.random.categorical(jax.random.PRNGKey(1), logit)
        state = jitted_discard(state, discard_tile)
        self.assertEqual(state._hand[c_p, discard_tile], hand[discard_tile] - 1)

    def test_draw(self):
        # Cover draw-state updates for deck index, target reset, and furiten clearing rules.
        state = self.state
        state = jitted_draw(state)
        self.assertEqual(state._next_deck_ix, IDX_AFTER_FIRST_DRAW -1) # Draw the first tile
        self.assertEqual(state._target, -1) # Target is -1
        # Furiten by pass without riichi
        state = self.set_state(state, current_player=jnp.int8(0), _furiten_by_pass=state._furiten_by_pass.at[0].set(True))
        state = jitted_draw(state)
        self.assertEqual(state._next_deck_ix, IDX_AFTER_FIRST_DRAW - 2)
        self.assertEqual(state._furiten_by_pass[0], False) # Furiten by pass is released
        # Furiten by pass with riichi
        state = self.set_state(state, current_player=jnp.int8(0), _riichi=state._riichi.at[0].set(True), _furiten_by_pass=state._furiten_by_pass.at[0].set(True))
        state = jitted_draw(state)
        self.assertEqual(state._next_deck_ix, IDX_AFTER_FIRST_DRAW - 3)
        self.assertEqual(state._furiten_by_pass[0], True) # Furiten by pass after riichi is not released.
        # Furiten by discard
        state = self.set_state(state, current_player=jnp.int8(0), _riichi=state._riichi.at[0].set(True), _furiten_by_discard=state._furiten_by_discard.at[0].set(True))
        state = jitted_draw(state)
        self.assertEqual(state._next_deck_ix, IDX_AFTER_FIRST_DRAW - 4)
        self.assertEqual(state._furiten_by_discard[0], True) # Furiten by discard is not released.

    def test_make_legal_action_mask_after_draw(self):
        # Ensure legal actions after draws respect haitei, yaku, riichi, and kan constraints.
        hand = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int8)
        hand = hand.at[0].set(
            jnp.array(
                [
                    4, 1, 1, 1, 1, 1, 1, 1, 0,
                    2, 1, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0
                ],
                dtype=jnp.int8,
            )
        )
        state = self.set_state(self.state, _hand=hand, _is_hand_concealed=jnp.ones(4, dtype=jnp.bool_))
        # After drawing 1m
        legal_action_mask = jitted_make_legal_action_mask_after_draw(state, hand, c_p=0, new_tile=0)
        self.assertEqual(jnp.all(legal_action_mask[Action.TSUMOGIRI]), True)
        self.assertEqual(jnp.all(legal_action_mask[1:8]), True) # 1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m
        self.assertEqual(jnp.all(legal_action_mask[9]), True) # 1p
        self.assertEqual(jnp.all(legal_action_mask[Tile.NUM_TILE_TYPE]), True) # Closed kan is allowed for 1m
        self.assertEqual(jnp.all(legal_action_mask[Action.RIICHI]), True) # Riichi is allowed for 2p
        # After drawing 2m
        state = self.set_state(self.state, _hand=hand, _is_hand_concealed=jnp.ones(4, dtype=jnp.bool_))
        legal_action_mask = jitted_make_legal_action_mask_after_draw(state, hand, c_p=0, new_tile=0) # After drawing 1m, the legal actions are generated
        self.assertEqual(jnp.all(legal_action_mask[Action.TSUMOGIRI]), True)
        self.assertEqual(jnp.all(legal_action_mask[1:8]), True) # 1m, 3m, 4m, 5m, 6m, 7m, 8m, we cannot discard 2m because it is not in the hand originaly
        self.assertEqual(jnp.all(legal_action_mask[9]), True) # 1p
        self.assertEqual(jnp.all(legal_action_mask[Tile.NUM_TILE_TYPE]), True) # Closed kan is allowed for 1m
        self.assertEqual(jnp.all(legal_action_mask[Action.RIICHI]), True) # Riichi is allowed for 2p
        # Not bottom of the sea
        state = self.set_state(
            self.state,
            _hand=hand,
            current_player=jnp.int8(0),
            _is_haitei=jnp.bool_(False),
            _is_hand_concealed=jnp.zeros(4, dtype=jnp.bool_),
            _can_after_kan=jnp.bool_(False),
            _can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[0].set(True), # Can win by tile combination
            _has_yaku=jnp.zeros((4, 2), dtype=jnp.bool_) # No yaku
        )
        legal_action_mask = jitted_make_legal_action_mask_after_draw(state, hand, c_p=0, new_tile=0) # Can win by tile combination, but no yaku
        self.assertEqual(legal_action_mask[Action.TSUMO], False) # No yaku, so cannot tsumo
        # Bottom of the sea
        state = self.set_state(
            self.state,
            current_player=jnp.int8(0),
            _next_deck_ix=jnp.int32(14),
            _last_deck_ix=jnp.int32(14),
            _is_hand_concealed=jnp.zeros(4, dtype=jnp.bool_),
            _can_after_kan=jnp.bool_(False),
            _is_haitei=jnp.bool_(True),
            _can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[0].set(True), # Can win by tile combination
            _has_yaku=jnp.zeros((4, 2), dtype=jnp.bool_) # No yaku
        )
        legal_action_mask = jitted_make_legal_action_mask_after_draw(state, hand, c_p=0, new_tile=0) # Can win by tile combination, but no yaku
        self.assertEqual(jnp.all(legal_action_mask[Tile.NUM_TILE_TYPE]), False) # Cannot closed kan on bottom of the sea
        self.assertEqual(jnp.all(legal_action_mask[Action.RIICHI]), False) # Cannot riichi on bottom of the sea
        self.assertEqual(jnp.all(legal_action_mask[Action.TSUMO]), True) # Can tsumo on bottom of the sea
        # No next draw turn No riichi
        state = self.set_state(
            state,
            _next_deck_ix=jnp.int32(15),
            _last_deck_ix=jnp.int32(14),
        )
        legal_action_mask = jitted_make_legal_action_mask_after_draw(state, hand, c_p=0, new_tile=0) # No next draw turn, so cannot riichi
        self.assertEqual(jnp.all(legal_action_mask[Action.RIICHI]), False) # No next draw turn, so cannot riichi

    def test_make_legal_action_mask_after_discard(self):
        # Ensure discard responses (chi/pon/kan/ron) obey seat priority plus riichi/haitei/furiten/yaku rules.
        hand1 = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.int8).at[:8].set(1) # 1m, 2m, 3m, 4m, 5m, 6m, 7m, 8m
        hand2 = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.int8).at[4].set(3) # 5m
        hand3 = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.int8).at[:14].set(1) # 123456789m, 1234p tenpai
        hand4 = jnp.zeros(Tile.NUM_TILE_TYPE, dtype=jnp.int8).at[0].set(1).at[2:12].set(1) # 1113456789m
        # Riichi, no closed_kan
        state = self.state
        basic_state = self.set_state(
            state,
            current_player=jnp.int8(3),
            _is_haitei=jnp.bool_(False),
            _furiten_by_discard=jnp.zeros(4, dtype=jnp.bool_),
            _furiten_by_pass=jnp.zeros(4, dtype=jnp.bool_),
            _riichi=jnp.zeros(4, dtype=jnp.bool_),
            _is_hand_concealed=jnp.ones(4, dtype=jnp.bool_),
            _can_win=jnp.ones((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_), # Can win by tile combination
            _has_yaku=jnp.ones((4, 2), dtype=jnp.bool_), # Has yaku
        )
        state = basic_state
        legal_action_mask_1 = jitted_make_legal_action_mask_after_discard(state, hand1, c_p=0, tile=2) # For 3m
        self.assertEqual(jnp.all(legal_action_mask_1[Action.CHI_L:Action.CHI_R]), True)  # Can chi for left neighbor
        legal_action_mask_2 = jitted_make_legal_action_mask_after_discard(state, hand1, c_p=1, tile=2)
        self.assertEqual(jnp.all(legal_action_mask_2[Action.CHI_L:Action.CHI_R]), False)  # Cannot chi when not the left neighbor
        legal_action_mask_3 = jitted_make_legal_action_mask_after_discard(state, hand2, c_p=0, tile=4) # For 5m
        self.assertEqual(jnp.all(legal_action_mask_3[Action.PON]), True)  # Can pon
        self.assertEqual(jnp.all(legal_action_mask_3[Action.OPEN_KAN]), True)  # Can open kan
        legal_action_mask_4 = jitted_make_legal_action_mask_after_discard(state.replace(_n_kan=jnp.array([2, 2, 0, 0], dtype=jnp.int8)), hand2, c_p=0, tile=4) # For 5m, n_kan is 4, so cannot open kan
        self.assertEqual(jnp.all(legal_action_mask_4[Action.PON]), True)  # Can pon
        self.assertEqual(jnp.all(legal_action_mask_4[Action.OPEN_KAN]), False)  # Cannot open kan when n_kan is 4
        legal_action_mask_5 = jitted_make_legal_action_mask_after_discard(state, hand3, c_p=0, tile=9) # For 1p
        self.assertEqual(jnp.all(legal_action_mask_5[Action.CHI_L]), True) # Can chi for left neighbor
        self.assertEqual(jnp.all(legal_action_mask_5[Action.RON]), True) # Has yaku of pure straight
        legal_action_mask_6 = jitted_make_legal_action_mask_after_discard(state, hand3, c_p=1, tile=9) # For 1m
        self.assertEqual(jnp.all(legal_action_mask_6[Action.CHI_L]), False)  # Cannot chi when not the left neighbor
        # Riichi
        state = self.set_state(
            basic_state,
            current_player=jnp.int8(0),
            _riichi=jnp.ones(4, dtype=jnp.bool_),
        )
        legal_action_mask_1 = jitted_make_legal_action_mask_after_discard(state, hand1, c_p=0, tile=2) # For 3m
        self.assertEqual(jnp.all(legal_action_mask_1[Action.CHI_L:Action.CHI_R]), False)  # cannot chi when riichi
        legal_action_mask_2 = jitted_make_legal_action_mask_after_discard(state, hand2, c_p=0, tile=4) # For 5m
        self.assertEqual(jnp.all(legal_action_mask_2[Action.PON]), False)  # cannot pon when riichi
        self.assertEqual(jnp.all(legal_action_mask_2[Action.OPEN_KAN]), False)  # cannot open kan when riichi
        legal_action_mask_3 = jitted_make_legal_action_mask_after_discard(state, hand3, c_p=0, tile=9) # For 1p
        self.assertEqual(jnp.all(legal_action_mask_3[Action.CHI_L]), False)  # Cannot chi when riichi
        self.assertEqual(jnp.all(legal_action_mask_3[Action.RON]), True)  # can ron for tenpai hand
        # Tile at the bottom of the sea
        state = self.set_state(
            basic_state,
            _is_haitei=True
        )
        legal_action_mask_1 = jitted_make_legal_action_mask_after_discard(state, hand1, c_p=0, tile=2) # For 3m
        self.assertEqual(jnp.all(legal_action_mask_1[Action.CHI_L:Action.CHI_R]), False)
        legal_action_mask_2 = jitted_make_legal_action_mask_after_discard(state, hand2, c_p=0, tile=4) # For 5m
        self.assertEqual(jnp.all(legal_action_mask_2[Action.PON]), False)
        self.assertEqual(jnp.all(legal_action_mask_2[Action.OPEN_KAN]), False)
        legal_action_mask_3 = jitted_make_legal_action_mask_after_discard(state, hand3, c_p=0, tile=9) # For 1p
        self.assertEqual(jnp.all(legal_action_mask_3[Action.CHI_L]), False)
        self.assertEqual(jnp.all(legal_action_mask_3[Action.RON]), True)
        # Furiten by discard
        state = self.set_state(
            basic_state,
            _furiten_by_discard=jnp.ones(4, dtype=jnp.bool_),
        )
        legal_action_mask_3 = jitted_make_legal_action_mask_after_discard(state, hand3, c_p=0, tile=9) # For 1p
        self.assertEqual(jnp.all(legal_action_mask_3[Action.CHI_L]), True)
        self.assertEqual(jnp.all(legal_action_mask_3[Action.RON]), False)  # cannot ron when furiten
        # Furiten by pass
        state = self.set_state(
            basic_state,
            _furiten_by_pass=jnp.ones(4, dtype=jnp.bool_),
        )
        legal_action_mask_3 = jitted_make_legal_action_mask_after_discard(state, hand3, c_p=0, tile=9) # For 1p
        self.assertEqual(jnp.all(legal_action_mask_3[Action.CHI_L]), True)
        self.assertEqual(jnp.all(legal_action_mask_3[Action.RON]), False)  # cannot ron when furiten
        # We can win by tile combination, but no yaku
        state = self.set_state(
            basic_state,
            _is_haitei=True,
            _can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[0].set(True), # Can win by tile combination
            _has_yaku=jnp.zeros((4, 2), dtype=jnp.bool_) # No yaku
        )
        legal_action_mask_4 = jitted_make_legal_action_mask_after_discard(state, hand4, c_p=0, tile=2) # For 3m
        self.assertEqual(jnp.all(legal_action_mask_4[Action.CHI_L:Action.CHI_R]), False)
        legal_action_mask_5 = jitted_make_legal_action_mask_after_discard(state.replace(_is_haitei=False), hand4, c_p=0, tile=2) # For 3m, not haitei
        self.assertEqual(jnp.all(legal_action_mask_5[Action.RON]), False) # Cannot ron without yaku

    def test_next_meld_player(self):
        # Check meld priority order (ron>pon>chi) and seat rotation logic.
        # current player = 2, discarded player = 0
        # ron vs pon
        legal_action_mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        legal_action_mask = legal_action_mask.at[0, Action.PON].set(True).at[0, Action.CHI_L].set(True)
        legal_action_mask = legal_action_mask.at[1, Action.RON].set(True).at[1, Action.PON].set(True)
        next_player, can_any = jitted_next_meld_player(legal_action_mask, 0)
        self.assertEqual(next_player, 1, f"{next_player}")
        self.assertEqual(can_any, True)
        # pon vs chi
        legal_action_mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        legal_action_mask = legal_action_mask.at[0, Action.PON].set(True).at[0, Action.CHI_L].set(True)
        legal_action_mask = legal_action_mask.at[1, Action.CHI_L].set(True)
        next_player, can_any = jitted_next_meld_player(legal_action_mask, 0)
        self.assertEqual(next_player, 0, f"{next_player}")
        self.assertEqual(can_any, True)
        # ron vs ron (left first)
        legal_action_mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        legal_action_mask = legal_action_mask.at[1, Action.RON].set(True).at[0, Action.PON].set(True)
        legal_action_mask = legal_action_mask.at[3, Action.RON].set(True).at[1, Action.PON].set(True)
        next_player, can_any = jitted_next_meld_player(legal_action_mask, 2)  # p3 discarded. p4 is more prioritized over p2.
        self.assertEqual(next_player, 3, f"{next_player}")  # p4 is the next player.
        self.assertEqual(can_any, True)
        # ron vs ron (left first)
        legal_action_mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        legal_action_mask = legal_action_mask.at[0, Action.RON].set(True).at[0, Action.PON].set(True)
        legal_action_mask = legal_action_mask.at[2, Action.RON].set(True).at[1, Action.PON].set(True)
        legal_action_mask = legal_action_mask.at[3, Action.RON].set(True).at[3, Action.PON].set(True)
        next_player, can_any = jitted_next_meld_player(legal_action_mask, 1)  # p2 discarded. p3 is more prioritized over p4.
        self.assertEqual(next_player, 2, f"{next_player}")  # p2 is the next player.
        self.assertEqual(can_any, True)
        # no meld player
        legal_action_mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        next_player, can_any = jitted_next_meld_player(legal_action_mask, 0)
        self.assertEqual(can_any, False)

    def test_next_ron_player(self):
        # Confirm ron claims iterate correctly over candidates or report none.
        # p0 discarded single ron
        legal_action_mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        legal_action_mask = legal_action_mask.at[0, Action.RON].set(True).at[0, Action.PON].set(True)
        next_player, can_any = jitted_next_ron_player(legal_action_mask, 3)
        self.assertEqual(next_player, 0, f"{next_player}")
        self.assertEqual(can_any, True)
        # p0 discarded double ron
        legal_action_mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        legal_action_mask = legal_action_mask.at[1, Action.RON].set(True).at[0, Action.PON].set(True)
        legal_action_mask = legal_action_mask.at[2, Action.RON].set(True).at[1, Action.PON].set(True)
        next_player, can_any = jitted_next_ron_player(legal_action_mask, 0)
        self.assertEqual(next_player, 1, f"{next_player}")
        # no ron player
        legal_action_mask = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        next_player, can_any = jitted_next_ron_player(legal_action_mask, 0)
        self.assertEqual(can_any, False)

    def test_accept_riichi(self):
        # Validate riichi acceptance toggles flags, deducts points, and only sets double riichi when allowed.
        state = self.set_state(
            self.state,
            _riichi_declared=True,
            _riichi=jnp.zeros(4, dtype=jnp.bool_),
            _last_player=jnp.int8(0),
            _next_deck_ix=jnp.int8(FIRST_DRAW_IDX - 10),
        )
        state = jitted_accept_riichi(state)
        self.assertEqual(state._riichi_declared, False) # riichi is already declared
        self.assertEqual(state._riichi[0], True)
        self.assertEqual(state._score[0], 240) # reduce score for riichi declaration
        self.assertEqual(state._double_riichi[0], False) # not double riichi
        self.assertEqual(state._ippatsu[0], True) # ippatsu is enabled
        # double riichi
        state = self.set_state(
            self.state,
            _riichi_declared=True,
            _riichi=jnp.zeros(4, dtype=jnp.bool_),
            _last_player=jnp.int8(0),
            _next_deck_ix=jnp.int8(FIRST_DRAW_IDX - 2),
        )
        state = jitted_accept_riichi(state)
        self.assertEqual(state._riichi_declared, False) # riichi is already declared
        self.assertEqual(state._riichi[0], True)
        self.assertEqual(state._score[0], 240) # reduce score for riichi declaration
        self.assertEqual(state._double_riichi[0], True) # double riichi
        self.assertEqual(state._ippatsu[0], True) # ippatsu is enabled
        # if meld, double riichi is not enabled
        state = self.set_state(
            self.state,
            _riichi_declared=True,
            _riichi=jnp.zeros(4, dtype=jnp.bool_),
            _last_player=jnp.int8(0),
            _next_deck_ix=jnp.int8(FIRST_DRAW_IDX - 2),
            _n_meld=jnp.array([1, 0, 0, 0], dtype=jnp.int8),
        )
        state = jitted_accept_riichi(state)
        self.assertEqual(state._riichi_declared, False) # riichi is already declared
        self.assertEqual(state._riichi[0], True)
        self.assertEqual(state._score[0], 240) # reduce score for riichi declaration
        self.assertEqual(state._double_riichi[0], False) # not double riichi

    def test_draw_after_kan(self):
        # Ensure rinshan draws increment kan counts, enable after-kan actions, and clear ippatsu.
        state = self.state
        state = self.set_state(
            state,
            current_player=jnp.int8(0),
            _n_kan=jnp.array([0, 0, 0, 0], dtype=jnp.int8),
            _n_kan_doras=jnp.int8(0),
            _deck=jnp.arange(136, dtype=jnp.int8).at[10].set(2).at[11].set(1),
            _kan_declared=True,
            _ippatsu=jnp.ones(4, dtype=jnp.bool_),
            _double_riichi=jnp.ones(4, dtype=jnp.bool_),
            _hand=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int8),
        )
        kan_state = jitted_draw_after_kan(state)
        self.assertEqual(kan_state._can_after_kan, True) # accept after kan
        self.assertEqual(jnp.all(kan_state._n_kan[0] == 1), True) # n_kan is increased by 1
        self.assertEqual(kan_state._hand[0, 2], 1) # Draw a tile after kan
        self.assertEqual(jnp.all(kan_state._ippatsu == False), True) # ippatsu is disabled
        self.assertEqual(kan_state._kan_declared, False) # kan is not declared
        kan_state = jitted_draw_after_kan(state.replace(_n_kan=jnp.array([1, 0, 0, 0], dtype=jnp.int8)))
        self.assertEqual(kan_state._can_after_kan, True) # accept after kan
        self.assertEqual(jnp.all(kan_state._n_kan[0] == 2), True) # n_kan is increased by 1
        self.assertEqual(kan_state._hand[0, 1], 1) # rinshan tile is drawn
        self.assertEqual(kan_state._kan_declared, False) # kan is not declared

    def test_selfkan(self):
        # Exercise closed and added kan flows consuming tiles and updating meld/pon structures.
        state = self.state
        # closed kan
        hand = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int8)
        hand = hand.at[0, 0].set(4)  # 1m x 4
        old_deck_ix = 100
        deck = jnp.arange(136, dtype=jnp.int8)  # dummy deck
        state = self.set_state(
            state,
            _hand=hand,
            current_player=jnp.int8(0),
            _next_deck_ix=old_deck_ix,
            _n_kan=jnp.array([0, 0, 0, 0], dtype=jnp.int8),
            _n_meld=jnp.array([0, 0, 0, 0], dtype=jnp.int8),
            _deck=deck,
            _can_after_kan=False,
            _n_kan_doras=jnp.int8(0),
            _pon=jnp.zeros((4, 34), dtype=jnp.int32)
        )
        state = jitted_closed_kan(state, 0)
        # 1m is consumed
        self.assertEqual(jnp.all(state._hand[0, 0] == 0), True)
        # n_meld is increased by 1
        self.assertEqual(jnp.all(state._n_meld[0] == 1), True)
        # added kan
        hand = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int8)
        hand = hand.at[0, 4].set(1)  # 5m x 1
        # existing melds
        melds = jnp.zeros((4, 4), dtype=jnp.int32)
        n_meld = jnp.zeros(4, dtype=jnp.int8).at[0].set(1)  # player0 has 1 pon
        pon = jnp.zeros((4, 34), dtype=jnp.int32).at[0, 4].set(0 << 2 | 1)  # player0 pon 5m (second)
        state = self.set_state(
            state,
            _hand=hand,
            current_player=jnp.int8(0),
            _n_kan=jnp.array([0, 0, 0, 0], dtype=jnp.int8),
            _n_kan_doras=jnp.int8(0),
            _can_after_kan=False,
            _n_meld=n_meld,
            _melds=melds,
            _pon=pon
        )
        state = jitted_added_kan(state, 4)
        # 5m is consumed
        self.assertEqual(jnp.all(state._hand[0, 4] == 0), True)
        # pon is removed
        self.assertEqual(jnp.all(state._pon[0, 4] == 0), True)

    def test_open_kan(self):
        # Ensure open kan consumes tiles, uses discard target, and increments meld count.
        state = self.state
        hand = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int8)
        hand = hand.at[0, 3].set(3) # 4m x 3
        state = self.set_state(
            state,
            _hand=hand,
            _target=jnp.int8(3),
            _is_hand_concealed=jnp.array([True, True, True, True], dtype=jnp.bool_),
            _n_meld=jnp.array([0, 0, 0, 0], dtype=jnp.int8),
            _n_kan=jnp.array([0, 0, 0, 0], dtype=jnp.int8),
            _n_kan_doras=jnp.int8(0),
            _can_after_kan=False,
            current_player=jnp.int8(0)
        )
        state = jitted_open_kan(state)
        self.assertEqual(jnp.all(state._hand[0, 3] == 0), True) # 4m is consumed
        self.assertEqual(jnp.all(state._n_meld[0] == 1), True) # n_meld is increased by 1

    def test_after_kan(self):
        # Check kan plus rinshan sequence removes kan tiles, draws rinshan, and unlocks tsumo.
        state = self.state
        hand = jnp.array(
            [
                1, 1, 0, 1, 1, 1, 1, 1, 1,
                2, 4, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0
            ],
            dtype=jnp.int8,
        )
        state = self.set_state(
            state,
            current_player=jnp.int8(0),
            _n_kan=jnp.array([0, 0, 0, 0], dtype=jnp.int8),
            _n_kan_doras=jnp.int8(0),
            _deck=jnp.arange(136, dtype=jnp.int8).at[10].set(2),
            _hand=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int8).at[0].set(hand), # player0 has 2p x 4
        )
        state = jitted_kan(state, 10) # player0 closed kan 2p
        state = jitted_draw_after_kan(state)
        self.assertEqual(state._can_after_kan, True) # accept after kan
        self.assertEqual(jnp.all(state._n_kan[0] == 1), True) # n_kan is increased by 1
        self.assertEqual(state._hand[0, 2], 1) # rinshan tile is drawn
        self.assertEqual(state._hand[0, 10], 0) # kan is closed, so 2p is removed
        self.assertEqual(state._legal_action_mask_4p[0, Action.TSUMO], True) # kan is closed, so tsumo is possible
        self.assertEqual(state._kan_declared, False) # kan is not declared

    def test_robbing_kan(self):
        # Verify robbing-kan priority and action masks for ron/pass without boosting kan counts.
        hand = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int8)
        hand = hand.at[0].set(
            jnp.array(
                [
                    1, 1, 1, 1, 1, 1, 1, 1, 1,
                    2, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0
                ],
                dtype=jnp.int8,
            )
        )  # (player 0 can win with 1s, 4s)
        hand = hand.at[1].set(
            jnp.array(
                [
                    0, 1, 1, 1, 1, 1, 1, 1, 1,
                    2, 2, 0, 0, 0, 0, 0, 0, 0,
                    1, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0
                ],
                dtype=jnp.int8,
            )
        )  # (player 1 added kan 1s)
        pon = jnp.zeros((4, 34), dtype=jnp.int32).at[1, 18].set(1) # player 1 pon 1s
        base_state = self.set_state(
            self.state,
            _hand=hand,
            _pon=pon,
            current_player=jnp.int8(1),
            _can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[0, 18].set(True),
        )
        action = 18 + Tile.NUM_TILE_TYPE  # play added kan 1s
        kan_state = jitted_kan(base_state, action)
        self.assertEqual(kan_state._kan_declared, True) # kan is declared
        self.assertEqual(kan_state._hand[1, 18], 0) # 1s is consumed
        self.assertEqual(kan_state._pon[1, 18], 0) # pon is removed
        self.assertEqual(kan_state._n_kan[1], 0) # n_kan is not increased
        self.assertEqual(kan_state.current_player, 0) # next player is player 0
        self.assertEqual(kan_state._last_player, 1) # last player is player 1
        self.assertEqual(kan_state._legal_action_mask_4p[0, Action.RON], True) # player 0 can ron
        self.assertEqual(kan_state._legal_action_mask_4p[0, Action.PASS], True) # player 0 can pass
        # left first for robbing kan
        hand = hand.at[2].set(
            jnp.array(
                [
                    1, 1, 1, 1, 1, 1, 1, 1, 1,
                    2, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 1, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0
                ],
                dtype=jnp.int8,
            )
        )  # player 2 also can win with 1s
        robbing_kan_two_player_state = self.set_state(
            base_state,
            _hand=hand,
            _can_win=base_state._can_win.at[2, 18].set(True),
        )
        kan_state = jitted_kan(robbing_kan_two_player_state, action)
        self.assertEqual(kan_state._kan_declared, True) # kan is declared
        self.assertEqual(kan_state._hand[2, 18], 0) # 1s is consumed
        self.assertEqual(kan_state._pon[2, 18], 0) # pon is removed
        self.assertEqual(kan_state._n_kan[2], 0) # n_kan is not increased
        self.assertEqual(kan_state.current_player, 2) # next player is player 2
        self.assertEqual(kan_state._last_player, 1) # last player is player 1
        self.assertEqual(kan_state._legal_action_mask_4p[[0, 2], Action.RON].all(), True) # player 0 and player 2 can ron
        self.assertEqual(kan_state._legal_action_mask_4p[[2], Action.PASS].all(), True) # player 2 can pass

    def test_pon(self):
        # Ensure pon consumes tiles, opens the hand, updates legal discards, and retains the turn.
        state = self.state
        hand = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int8)
        hand = hand.at[0].set(
            jnp.array(
                [
                    1, 1, 1, 1, 3, 1, 1, 1, 0,
                    2, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0
                ],
                dtype=jnp.int8,
            )
        )
        state = self.set_state(
            state,
            _hand=hand,
            _target=jnp.int8(4),
            _is_hand_concealed=jnp.array([True, True, True, True], dtype=jnp.bool_),
            _n_meld=jnp.array([0, 0, 0, 0], dtype=jnp.int8),
            current_player=jnp.int8(0),
        )
        state = jitted_pon(state, Action.PON) # player 0 pon 5m
        self.assertEqual(jnp.all(state._hand[0, 4] == 1), True) # 5m is consumed
        self.assertEqual(jnp.all(state._n_meld[0] == 1), True)
        self.assertEqual(jnp.all(state._is_hand_concealed[0] == False), True)
        self.assertEqual(jnp.all(state._legal_action_mask_4p[0, :Tile.NUM_TILE_TYPE] == (hand[0] > 0).at[4].set(False)), True) # player can discard other than target tile (5m)
        self.assertEqual(state.current_player, 0)

    def test_chi(self):
        # Ensure chi removes the appropriate sequence tiles and records the meld.
        hand = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int8)
        hand = hand.at[0].set(
            jnp.array(
                [
                    4, 1, 1, 0, 1, 1, 1, 1, 0,
                    2, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0
                ],
                dtype=jnp.int8,
            )
        )
        state = self.set_state(
            self.state,
            _hand=hand,
            _target=jnp.int8(3),
            _is_hand_concealed=jnp.array([True, True, True, True], dtype=jnp.bool_),
            _n_meld=jnp.array([0, 0, 0, 0], dtype=jnp.int8),
            current_player=jnp.int8(0),
        )
        state = jitted_chi(state, Action.CHI_M) # player 0 chi 4m (mid)
        self.assertEqual(jnp.all(state._hand[0, 2] == 0), True) # 3m is consumed
        self.assertEqual(jnp.all(state._hand[0, 4] == 0), True) # 5m is consumed
        self.assertEqual(jnp.all(state._n_meld[0] == 1), True)
        self.assertEqual(jnp.all(state._is_hand_concealed[0] == False), True)

    def test_legal_action_mask_after_chi(self):
        # Confirm post-chi legal masks forbid discarding the just-called tile combinations.
        hand = jnp.ones((4, Tile.NUM_TILE_TYPE), dtype=jnp.int8)
        base_legal_action_mask = jnp.ones((Action.NUM_ACTION), dtype=jnp.bool_).at[Tile.NUM_TILE_TYPE:].set(False)
        # player 0 chi 2m (left)
        legal_action_mask = jitted_make_legal_action_mask_after_chi(self.state, hand, 0, 1, Action.CHI_L)
        expected_legal_action_mask = base_legal_action_mask.at[4].set(False).at[1].set(False)  # swap calling
        self.assertEqual(jnp.all(legal_action_mask[0] == expected_legal_action_mask), True)
        # player 0 chi 2m (mid)
        legal_action_mask = jitted_make_legal_action_mask_after_chi(self.state, hand, 0, 1, Action.CHI_M)
        expected_legal_action_mask = base_legal_action_mask.at[1].set(False) # target tile is prohibited in mid chi
        self.assertEqual(jnp.all(legal_action_mask[0] == expected_legal_action_mask), True)
        # player 0 chi 7m (left)
        legal_action_mask = jitted_make_legal_action_mask_after_chi(self.state, hand, 0, 6, Action.CHI_L)
        expected_legal_action_mask = base_legal_action_mask.at[6].set(False) # target tile is prohibited
        self.assertEqual(jnp.all(legal_action_mask[0] == expected_legal_action_mask), True)
        # player 0 chi 3m (right)
        legal_action_mask = jitted_make_legal_action_mask_after_chi(self.state, hand, 0, 2, Action.CHI_R)
        expected_legal_action_mask = base_legal_action_mask.at[2].set(False) # target tile is prohibited
        self.assertEqual(jnp.all(legal_action_mask[0] == expected_legal_action_mask), True)

    def test_pass(self):
        # Check pass handling for queued callers, furiten-by-pass, and target clearing.
        legal_action_mask_4p = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        legal_action_mask_4p = legal_action_mask_4p.at[1, Action.CHI_L].set(True)  # player 1 can chi
        legal_action_mask_4p = legal_action_mask_4p.at[2, Action.PON].set(True)  # player 2 can pon
        state = self.set_state(
            self.state,
            current_player=jnp.int8(0),
            _last_player=jnp.int8(2),
            _target=jnp.int8(1), # 2m is discarded
            _legal_action_mask_4p=legal_action_mask_4p
        )
        state = jitted_pass(state)  # execute pass
        # player 0 passed, so next player is player 2 (pon)
        self.assertEqual(jnp.all(state.current_player == 2), True)
        # pass is added to legal action mask
        self.assertEqual(jnp.all(state._legal_action_mask_4p[2, Action.PASS]), True)
        self.assertEqual(jnp.all(state._target == 1), True)
        self.assertEqual(jnp.all(state._last_player == 2), True)
        state = self.set_state(
            state,
            _legal_action_mask_4p=jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_).at[0, Action.PON].set(True),  # player who can pon passes
            current_player=jnp.int8(0),
        )
        state = jitted_pass(state)  # execute pass
        # no player can action, so next player is player 3
        self.assertEqual(jnp.all(state.current_player == 3), True)
        # pass is added to legal action mask
        self.assertEqual(jnp.all(state._target == -1), True) # target is -1 because player passed
        self.assertEqual(jnp.all(state._last_player == 2), True) # last_player is 2
        # player who can ron passes
        state = self.state
        legal_action_mask_4p = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        legal_action_mask_4p = legal_action_mask_4p.at[1, Action.RON].set(True)  # player 1 can ron
        state = self.set_state(
            state,
            current_player=jnp.int8(1),
            _legal_action_mask_4p=legal_action_mask_4p,
            _last_player=jnp.int8(2),
            _furiten_by_pass=jnp.array([False, False, False, False], dtype=jnp.bool_),
        )
        state = jitted_pass(state)
        self.assertEqual(state._furiten_by_pass[1], True)  # player who can ron passed, so furiten by pass

    def test_riichi(self):
        # Ensure riichi declaration restricts discards to tenpai-safe tiles.
        state = self.state
        # setup hand (tenpai state)
        hand = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int8)
        hand = hand.at[0].set(
            jnp.array(
                [
                    1, 1, 1, 1, 2, 1, 1, 1, 1,
                    2, 1, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0
                ],
                dtype=jnp.int8,
            )
        )
        legal_action_mask_4p = jnp.zeros((4, Action.NUM_ACTION), dtype=jnp.bool_)
        state = self.set_state(
            state,
            _hand=hand,
            current_player=jnp.int8(0),
            _is_hand_concealed=jnp.array([True, True, True, True], dtype=jnp.bool_),
            _riichi=jnp.array([False, False, False, False], dtype=jnp.bool_),
            _last_draw=0,  # 1m is drawn
            _legal_action_mask_4p=legal_action_mask_4p
        )
        state = jitted_riichi(state)  # declare riichi
        # after riichi, only tiles that can be discarded are allowed
        self.assertTrue(jnp.all(state._legal_action_mask_4p == legal_action_mask_4p.at[0, [4, 9]].set(True)), True)  # only 5m and 1m are legal
        hand = jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.int8)
        hand = hand.at[0].set(
            jnp.array(
                [
                    1, 1, 1, 1, 1, 1, 1, 1, 0,
                    0, 0, 0, 0, 0, 1, 1, 2, 0,
                    0, 0, 0, 0, 0, 0, 2, 0, 0,
                    0, 0, 0, 0, 0, 0, 0
                ],
                dtype=jnp.int8,
            )
        )
        state = self.set_state(
            state,
            current_player=jnp.int8(0),
            _hand=hand,
            _last_draw=24,
        )
        state = jitted_riichi(state)
        self.assertEqual(jnp.all(state._legal_action_mask_4p == legal_action_mask_4p.at[0, 16].set(True)), True)

    def test_double_riichi(self):
        # Show double riichi only applies with no prior melds while clearing the declaration flag.
        state = self.state
        state = self.set_state(
            state,
            _last_player=jnp.int8(0),
            _n_meld=jnp.zeros(4, dtype=jnp.int8),
            _riichi_declared=True,
            _riichi=jnp.zeros(4, dtype=jnp.bool_),
        )
        state = jitted_accept_riichi(state)
        self.assertEqual(state._double_riichi[0], True) # no meld in first round, so double riichi is established
        self.assertEqual(state._riichi_declared, False) # riichi declaration is not valid

        state = self.set_state(
            state,
            _n_meld=jnp.ones(4, dtype=jnp.int8),
            _riichi_declared=True,
            _riichi=jnp.zeros(4, dtype=jnp.bool_),
        )
        state = jitted_accept_riichi(state)
        self.assertEqual(state._double_riichi[0], False) # meld is in, so double riichi is not established
        self.assertEqual(state._riichi_declared, False) # riichi declaration is not valid

    def test_ron(self):
        # Verify ron scoring and rewards including honba and kyotaku payouts.
        # yaku test is done in other test, so here we test if the score is correct
        # one fan 30 fu, target is player 1, current_player is player 0
        fan = jnp.zeros((4, 2), dtype=jnp.int32).at[0, 0].set(1)
        fu = jnp.zeros((4, 2), dtype=jnp.int32).at[0, 0].set(30)
        current_player = jnp.int8(0)
        last_player = jnp.int8(1)
        target = jnp.int8(2)  # 3m
        score = jnp.array([250, 250, 250, 250], dtype=jnp.float32)
        dealer = jnp.int8(0)
        basic_state = self.set_state(
            self.state,
            _fan=fan,
            _fu=fu,
            _target=target,
            _last_player=last_player,
            _kyotaku=jnp.int8(0),
            current_player=current_player,
            _score=score,
            _dealer=dealer,
        )
        # parent wins 1500 points
        state = basic_state
        state = jitted_ron(state)
        self.assertEqual(jnp.all(state._score == jnp.array([265, 235, 250, 250], dtype=jnp.float32)), True)
        self.assertEqual(jnp.all(state.rewards == jnp.array([15, -15, 0, 0], dtype=jnp.float32)), True)
        # with honba and kyotaku
        state = self.set_state(
            basic_state,
            _honba=jnp.int8(1),
            _kyotaku=jnp.int8(1),
            _riichi=jnp.array([True, False, False, True], dtype=jnp.bool_),
            _score=jnp.array([250, 250, 250, 240], dtype=jnp.float32),
        )
        state = jitted_ron(state)
        self.assertEqual(jnp.all(state._score == jnp.array([278, 232, 250, 240], dtype=jnp.float32)), True)
        self.assertEqual(jnp.all(state.rewards == jnp.array([28, -18, 0, 0], dtype=jnp.float32)), True)
        self.assertEqual(state._kyotaku, 0)
        # yakuman test
        state = self.set_state(
            basic_state,
            _fan=jnp.zeros((4, 2), dtype=jnp.int32).at[0, 0].set(1),
            _fu=jnp.zeros((4, 2), dtype=jnp.int32).at[0, 0].set(0),
            _kan_declared=jnp.bool_(True),
        )
        state = jitted_ron(state)
        self.assertEqual(jnp.all(state._score == jnp.array([250 + 480, 250 - 480, 250, 250], dtype=jnp.float32)), True)  # robbing_kan are not counted
        self.assertEqual(jnp.all(state.rewards == jnp.array([480, -480, 0, 0], dtype=jnp.float32)), True)
        # double_riichi robbing_kan test
        state = self.set_state(
            basic_state,
            _fu=jnp.zeros((4, 2), dtype=jnp.int32).at[0, 0].set(40),
            _double_riichi=jnp.array([True, False, False, True], dtype=jnp.bool_),
            _riichi=jnp.array([True, False, False, False], dtype=jnp.bool_),
            _kyotaku=jnp.int8(0),
            _kan_declared=jnp.bool_(True),
            _dealer=jnp.int8(3),
        )
        state = jitted_ron(state)
        self.assertEqual(jnp.all(state._score == jnp.array([302, 198, 250, 250], dtype=jnp.float32)), True)  # double_riichi + robbing_kan = 3 fans
        self.assertEqual(jnp.all(state.rewards == jnp.array([52, -52, 0, 0], dtype=jnp.float32)), True)
        # ippatsu haitei test
        state = self.set_state(
            basic_state,
            _fu=jnp.zeros((4, 2), dtype=jnp.int32).at[0, 0].set(40),
            _riichi=jnp.array([True, False, False, False], dtype=jnp.bool_),
            _ippatsu=jnp.array([True, False, False, False], dtype=jnp.bool_),
            _kyotaku=jnp.int8(0),
            _is_haitei=jnp.bool_(True),
            _dealer=jnp.int8(3),
        )
        state = jitted_ron(state)
        self.assertEqual(jnp.all(state._score == jnp.array([302, 198, 250, 250], dtype=jnp.float32)), True)  # riichi + ippatsu + haitei = 3 fans
        self.assertEqual(jnp.all(state.rewards == jnp.array([52, -52, 0, 0], dtype=jnp.float32)), True)

    def test_tsumo(self):
        # Verify tsumo scoring/rewards for parent/child plus honba/kyotaku adjustments.
        """
        - yaku test is done in other test, so here we test if the score is correct
        - one fan 30 fu, target is player 1, current_player is player 0
        """
        fan = jnp.zeros((4, 2), dtype=jnp.int32).at[0, 0].set(2)
        fu = jnp.zeros((4, 2), dtype=jnp.int32).at[0, 0].set(30)
        current_player = jnp.int8(0)
        target = jnp.int8(2)  # 3m
        score = jnp.array([250, 250, 250, 250], dtype=jnp.float32)
        basic_state = self.set_state(
            self.state,
            _fan=fan,
            _fu=fu,
            _target=target,
            _score=score,
            current_player=current_player,
        )
        # child wins 300 500
        state = self.set_state(
            basic_state,
            current_player=jnp.int8(0),
            _dealer=jnp.int8(1),
            _next_deck_ix=FIRST_DRAW_IDX - 8,
        )
        state = jitted_tsumo(state)
        self.assertEqual(jnp.all(state._score == jnp.array([270, 240, 245, 245], dtype=jnp.float32)), True, f"{state._score}")
        self.assertEqual(jnp.all(state.rewards == jnp.array([20, -10, -5, -5], dtype=jnp.float32)), True, f"{state.rewards}")
        # parent wins 1500 points, kyotaku 2, honba 1
        state = self.set_state(
            basic_state,
            _dealer=jnp.int8(0),
            current_player=jnp.int8(0),
            _honba=jnp.int8(1),
            _next_deck_ix=FIRST_DRAW_IDX - 8,
            _kyotaku=jnp.int8(2),
            _riichi=jnp.array([False, False, True, True], dtype=jnp.bool_),
            _score=jnp.array([250, 250, 240, 240], dtype=jnp.float32),
        )
        state = jitted_tsumo(state)
        self.assertEqual(jnp.all(state._score == jnp.array([303, 239, 229, 229], dtype=jnp.float32)), True, f"{state._score}")
        self.assertEqual(jnp.all(state.rewards == jnp.array([53, -11, -11, -11], dtype=jnp.float32)), True, f"{state.rewards}")
        self.assertEqual(state._kyotaku, 0)
        # test for yakuman
        state = self.set_state(
            basic_state,
            _fan=jnp.zeros((4, 2), dtype=jnp.int32).at[0, 0].set(1),
            _fu=jnp.zeros((4, 2), dtype=jnp.int32),
            _can_after_kan=jnp.bool_(True),
            _dealer=jnp.int8(3),
            current_player=jnp.int8(0),
            _next_deck_ix=FIRST_DRAW_IDX - 8,  # not blessing of heaven
        )
        state = jitted_tsumo(state)
        self.assertEqual(jnp.all(state._score == jnp.array([570, 170, 170, 90], dtype=jnp.float32)), True)  # after_kan is not counted
        self.assertEqual(jnp.all(state.rewards == jnp.array([320, -80, -80, -160], dtype=jnp.float32)), True)

    def test_abortive_draw_normal(self):
        # Ensure drawn-game payouts depend on the number of tenpai players.
        """
        - test the score distribution
        """
        state = self.state

        # no one is tenpai
        state = self.set_state(
            state,
            _can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_),
        )
        state = jitted_abortive_draw_normal(state)
        self.assertEqual(state._terminated_round, True)
        self.assertEqual(jnp.all(state.rewards.astype(jnp.int32) == jnp.array(
            [0, 0, 0, 0], dtype=jnp.int32
        )), True, f"{state.rewards}")
        # one player is tenpai
        state = self.set_state(
            state,
            _can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[0].set(True),
        )
        state = jitted_abortive_draw_normal(state)
        self.assertEqual(jnp.all(state.rewards.astype(jnp.int32) == jnp.array(
            [30, -10, -10, -10], dtype=jnp.int32
        )), True, f"{state.rewards}")
        # two players are tenpai
        state = self.set_state(
            state,
            _can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[0].set(True).at[1].set(True),
        )
        state = jitted_abortive_draw_normal(state)
        self.assertEqual(jnp.all(state.rewards.astype(jnp.int32) == jnp.array(
            [15, 15, -15, -15], dtype=jnp.int32
        )), True, f"{state.rewards}")
        # three players are tenpai
        state = self.set_state(
            state,
            _can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[0].set(True).at[1].set(True).at[2].set(True),
        )
        state = jitted_abortive_draw_normal(state)
        self.assertEqual(jnp.all(state.rewards.astype(jnp.int32) == jnp.array(
            [10, 10, 10, -30], dtype=jnp.int32
        )), True, f"{state.rewards}")
        # four players are tenpai
        state = self.set_state(
            state,
            _can_win=jnp.zeros((4, Tile.NUM_TILE_TYPE), dtype=jnp.bool_).at[0].set(True).at[1].set(True).at[2].set(True).at[3].set(True),
        )
        state = jitted_abortive_draw_normal(state)
        self.assertEqual(jnp.all(state.rewards.astype(jnp.int32) == jnp.array(
            [0, 0, 0, 0], dtype=jnp.int32
        )), True)

    def test_next_round(self):
        # Exercise dealer/round/honba/kyotaku transitions and termination conditions.
        """
        - test the next round (after dummy sharing)
        - test the honba process
        - test the parent movement
        - test the round process
        - test the kyotaku process
        """
        state = self.state
        # dealer wins, so the round continues
        state = self.set_state(
            state,
            _honba=jnp.int8(0),
            _kyotaku=jnp.int8(0),
            _dealer=jnp.int8(0),
            current_player=jnp.int8(0),
            _has_won=jnp.array([True, False, False, False], dtype=jnp.bool_),
            _round=jnp.int8(0),
            _dummy_count=jnp.int8(0),
        )
        state = _advance_after_dummy(state)
        self.assertEqual(state._honba, 1)          # honba is 1
        self.assertEqual(state._dealer, 0)         # dealer is kept
        self.assertEqual(state._round, 0)          # round is kept
        self.assertEqual(state.current_player, 0)  # after dummy sharing, current player is dealer
        self.assertEqual(state._dummy_count, 0)    # dummy count is 0
        # dealer goes bankrupt, so the round ends
        state = self.set_state(
            state,
            _honba=jnp.int8(1),
            _dealer=jnp.int8(0),
            current_player=jnp.int8(0),
            _has_won=jnp.array([False, False, False, False], dtype=jnp.bool_),
            _round=jnp.int8(0),
            _dummy_count=jnp.int8(0),
        )
        state = _advance_after_dummy(state)
        self.assertEqual(state._honba, 2)          # honba+1
        self.assertEqual(state._dealer, 1)         # new dealer is dealer+1
        self.assertEqual(state._round, 1)          # round+1
        self.assertEqual(state.current_player, 1)  # after dummy sharing, current player is new dealer
        self.assertEqual(state._dummy_count, 0)
        # last round, dealer goes bankrupt, so the game ends
        state = self.set_state(
            state,
            _honba=jnp.int8(0),
            _dealer=jnp.int8(0),
            current_player=jnp.int8(0),
            _has_won=jnp.array([False, False, False, False], dtype=jnp.bool_),
            _score=jnp.array([320, 180, 250, 250], dtype=jnp.int32),
            _round=jnp.int8(7),   # final round
            _dummy_count=jnp.int8(0),
        )
        state = _advance_after_dummy(state)
        self.assertTrue(state.terminated)
        # one player goes bankrupt, so the game ends
        state = self.set_state(
            state,
            _honba=jnp.int8(0),
            _dealer=jnp.int8(0),
            current_player=jnp.int8(0),
            _has_won=jnp.array([False, False, False, False], dtype=jnp.bool_),
            _round=jnp.int8(0),
            _score=jnp.array([510, 250, 250, -100], dtype=jnp.int32),
            _dummy_count=jnp.int8(0),
        )
        state = _advance_after_dummy(state)
        self.assertTrue(state.terminated)
        # final round, parent is top, same score wind order (kyotaku top)
        state = self.state
        state = self.set_state(
            state,
            _dealer=jnp.int8(0),
            current_player=jnp.int8(0),
            _init_wind=jnp.array([0, 1, 2, 3], dtype=jnp.int8),
            _score=jnp.array([310, 310, 190, 190], dtype=jnp.int32),
            _has_won=jnp.array([False, False, False, False], dtype=jnp.bool_),
            _round=jnp.int8(7),   # final round
            _kyotaku=jnp.int8(3),
            _dummy_count=jnp.int8(0),
        )
        state = _advance_after_dummy(state)
        self.assertTrue(state.terminated)
        self.assertTrue(
            jnp.all(state._score == jnp.array([370, 320, 180, 160], dtype=jnp.int32)),
            f"{state._score}",
        )

    def test_dummy_phase_progression_rotation(self):
        # Check current_player rotation during the dummy sharing phase.
        """
        check if the current_player rotation is correct during dummy sharing phase
        共有前: cp=dealer
        after first call: (cp+1)%4
        after dummy sharing, +1 rotation every step
        """
        state = self.state
        state = self.set_state(
            state,
            _honba=jnp.int8(0),
            _kyotaku=jnp.int8(0),
            _dealer=jnp.int8(2),
            current_player=jnp.int8(2),
            _has_won=jnp.array([False, False, False, False], dtype=jnp.bool_),
            _round=jnp.int8(0),
            _dummy_count=jnp.int8(0),
        )

        cps = [2]
        for _ in range(3):
            state = jitted_next_round(state)
            cps.append(int(state.current_player))

        # 2 -> 3 -> 0 -> 1 rotation
        self.assertEqual(cps, [2, 3, 0, 1], f"unexpected rotation: {cps}")


    def test_dora_array(self):
        # Ensure _dora_array produces the correct dora/ura-dora indicator masks.
        state = self.state
        rng = jax.random.PRNGKey(1)
        dora_indicators = jnp.array([1, 2, 8, -1, -1], dtype=jnp.int8) # dora indicators (3m, 4m, 1m)
        ura_dora_indicators = jnp.array([2, 3, -1, -1, -1], dtype=jnp.int8) # ura dora indicators (4m, 5m, -)
        state = self.set_state(
            state,
            _dora_indicators=dora_indicators,
            _ura_dora_indicators=ura_dora_indicators,
        )
        # no riichi (only dora indicators)
        dora_array = _dora_array(state)
        self.assertEqual(dora_array.shape, (2, 34))
        self.assertEqual(dora_array[0, 2] == 1, True)
        self.assertEqual(dora_array[0, 3] == 1, True)
        self.assertEqual(dora_array[0, 0] == 1, True)

        self.assertEqual(dora_array[1, 3] == 1, True)
        self.assertEqual(dora_array[1, 4] == 1, True)

    def test_action_history(self):
        # Confirm action history stores every (player, action) pair up to step_count.
        state = self.state
        rng = jax.random.PRNGKey(1)
        action_history = []
        current_player_history = []
        jitted_env_step = jax.jit(env.step)
        while not state._terminated_round:
            action = act_randomly(rng, state.legal_action_mask)
            action_history.append(action)
            current_player_history.append(int(state.current_player))
            state = jitted_env_step(state, action)
            rng, rng_sub = jax.random.split(rng)

        final_step_count = state._step_count
        self.assertEqual(
            jnp.all(state._action_history[0, :final_step_count] == jnp.array(current_player_history, dtype=jnp.int8)),
            True,
            f"{state._action_history[0, :final_step_count]} != {current_player_history}"
        )
        self.assertEqual(
            jnp.all(state._action_history[1, :final_step_count] == jnp.array(action_history, dtype=jnp.int8)),
            True,
            f"{state._action_history[1, :final_step_count]} != {action_history}"
        )
        self.assertEqual(final_step_count, (state._action_history[0] != -1).sum(), f"{final_step_count} != {(state._action_history[0] != -1).sum()}")


if __name__ == "__main__":
    unittest.main()
