import unittest
import jax
import jax.numpy as jnp
from mahjax.no_red_mahjong.tile import Tile, River
from mahjax.no_red_mahjong.meld import Meld
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.state import FIRST_DRAW_IDX
from mahjax.no_red_mahjong.env import _init, _step, _dora_array
from mahjax.no_red_mahjong.hand import Hand
from mahjax.no_red_mahjong.meld import Meld
from mahjax.no_red_mahjong.tile import Tile, River
from mahjax.no_red_mahjong.meld import EMPTY_MELD
from mahjax.no_red_mahjong.shanten import Shanten
from mahjax.no_red_mahjong.yaku import Yaku
from mahjax.no_red_mahjong.players import rule_based_player

jitted_init = jax.jit(_init)
jitted_step = jax.jit(_step)
jitted_shanten_discard = jax.jit(Shanten.discard)
jitted_rule_based_player = jax.jit(rule_based_player)
jitted_yaku_judge = jax.jit(Yaku.judge)

# ========= debug helpers =========

_TILE_NAMES = [
    "1m","2m","3m","4m","5m","6m","7m","8m","9m",
    "1p","2p","3p","4p","5p","6p","7p","8p","9p",
    "1s","2s","3s","4s","5s","6s","7s","8s","9s",
    "東","南","西","北","白","發","中",
]

_ACTION_NAMES = {
    **{i: f"DISCARD({ _TILE_NAMES[i] })" for i in range(Tile.NUM_TILE_TYPE)},
    **{i + Tile.NUM_TILE_TYPE: f"SELF-KAN({ _TILE_NAMES[i] })" for i in range(Tile.NUM_TILE_TYPE)},
    Action.TSUMOGIRI: "TSUMOGIRI",
    Action.RIICHI: "RIICHI",
    Action.TSUMO: "TSUMO",
    Action.RON: "RON",
    Action.PON: "PON",
    Action.OPEN_KAN: "open_kan",
    Action.CHI_L: "CHI_L",
    Action.CHI_M: "CHI_M",
    Action.CHI_R: "CHI_R",
    Action.PASS: "PASS",
    Action.DUMMY: "DUMMY",
}


def act_randomly(rng, legal_action_mask) -> int:
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits=logits)

def _act_name(a:int) -> str:
    return _ACTION_NAMES.get(int(a), f"UNKNOWN({int(a)})")

def _hand_summary(row: jnp.ndarray) -> str:
    # "exists/not exists" of tiles only in a short form. If detailed information is needed, count is also added.
    parts = []
    for i, c in enumerate(list(map(int, row))):
        if c > 0:
            parts.append(f"{_TILE_NAMES[i]}x{c}")
    return " ".join(parts) if parts else "-"

def _safe_meld_str(m):
    m = jnp.uint16(m)
    # EMPTY check is more explicit than looking at action/target/src directly
    if int(m) == int(EMPTY_MELD):  # or Meld.is_empty(m) to bool(int(...)) is also OK
        return "-"
    a = int(Meld.action(m))
    t = int(Meld.target(m))
    s = int(Meld.src(m))
    # Human readable minimal display (alternative to to_str)
    kind = (
        "PON" if a == Action.PON else
        "open_kan" if a == Action.OPEN_KAN else
        "CHI_L" if a == Action.CHI_L else
        "CHI_M" if a == Action.CHI_M else
        "CHI_R" if a == Action.CHI_R else
        "SELFKAN" if Action.is_selfkan(a) else
        f"a{a}"
    )
    return f"{kind}(t={t},src={s})"

def _melds_summary(meld_row):
    out = []
    for m in list(map(int, list(meld_row))):
        out.append(_safe_meld_str(m))
    return "[" + ", ".join(out) + "]"


def _legal_summary(mask: jnp.ndarray) -> str:
    mask = jnp.array(mask)
    picks = []
    # Discard (0..33)
    disc = jnp.where(mask[:Tile.NUM_TILE_TYPE])[0]
    if disc.size > 0:
        picks.append("DISCARD{" + ",".join(_TILE_NAMES[int(i)] for i in list(map(int, disc))) + "}")
    # Self kan
    sk = jnp.where(mask[Tile.NUM_TILE_TYPE:Action.TSUMOGIRI])[0]
    if sk.size > 0:
        tiles = [ _TILE_NAMES[int(i)] for i in list(map(int, sk)) ]
        picks.append("SELF-KAN{" + ",".join(tiles) + "}")
    # Single action
    for k, name in [(Action.TSUMOGIRI,"TSUMOGIRI"),(Action.RIICHI,"RIICHI"),
                    (Action.TSUMO,"TSUMO"),(Action.RON,"RON"),
                    (Action.PON,"PON"),(Action.OPEN_KAN,"open_kan"),
                    (Action.CHI_L,"CHI_L"),(Action.CHI_M,"CHI_M"),(Action.CHI_R,"CHI_R"),
                    (Action.PASS,"PASS"),(Action.DUMMY,"DUMMY")]:
        if bool(mask[k]):
            picks.append(name)
    return " ".join(picks) if picks else "-"

# --- Safe tile name conversion (number tiles/honors) ---
def _tile_name(t: int) -> str:
    if not (0 <= t < Tile.NUM_TILE_TYPE):
        return "__"  # Empty
    # Suit tiles
    if t < 27:
        suit = "mps"[t // 9]
        num = (t % 9) + 1
        return f"{num}{suit}"
    # Honors
    honors = ["E", "S", "W", "N", "P", "F", "C"]  # East, South, West, North, White, Green, Charcoal
    return honors[t - 27]

# Safe river display (1 line)
# n is the number of slots to display. If None, scan the entire river (display "__" for -1)
def _river_one_line(river, p: int, n: int | None = None) -> str:
    dec = River.decode_river(river)  # (6,4,18)
    tiles = dec[0, p]       # (18,) -1 is empty
    riichi = dec[1, p]      # (18,) 0/1
    gray = dec[2, p]        # (18,) 0/1
    tsumogiri = dec[3, p]   # (18,) 0/1
    src = dec[4, p]         # (18,) 0..3
    mt = dec[5, p]          # (18,) 0:none, 1:pon, 2:open_kan, 3:chi_l, 4:chi_m, 5:chi_r

    # If possible, display the actual number of discarded tiles (the most reliable way is to pass cs.players.discard_counts[p] from the test caller)
    length = int(n) if n is not None else 18

    out = []
    for i in range(length):
        t = int(tiles[i])
        token = _tile_name(t)

        flags = ""
        if int(riichi[i]):    flags += "R"
        if int(tsumogiri[i]): flags += "T"

        # Visualize gray (called) in parentheses
        if int(gray[i]):
            token = f"({token})"

        # If you want to display meld_type and src, add it here
        # Example: if mt>0, add :mt
        if int(mt[i]) > 0:
            token += f":m{int(mt[i])}"

        out.append(token + (f":{flags}" if flags else ""))

    return " ".join(out)


def dump_debug(ls, cs, action, step:int):
    # Action name
    a = int(action)
    print("\n" + "="*90)
    print(f"[STEP {step}] ACTION: {a} ({_act_name(a)}) C_P: {int(cs.current_player)}")
    print("-"*90)
    # Current player etc.
    print(f"last_player(ls->cs): {int(ls.current_player)} -> {int(cs.current_player)}   /  discarder: {int(cs.round_state.last_player)}")
    print(f"target: {int(cs.round_state.target)} ({_TILE_NAMES[int(cs.round_state.target)] if int(cs.round_state.target)>=0 else '-'})")
    print(f"last_draw(ls,cs): {int(ls.round_state.last_draw)}->{int(cs.round_state.last_draw)}")
    print(f"kan_declared: {bool(cs.round_state.kan_declared)}  rinshan: {bool(cs.round_state.can_after_kan)}  haitei: {bool(cs.round_state.is_haitei)}")
    print(f"riichi_declared: {bool(cs.players.riichi_declared.any())}  terminated_round: {bool(cs.round_state.terminated_round)}  terminated: {bool(cs.terminated)}")
    print(f"honba: {int(cs.round_state.honba)}  kyotaku: {int(cs.round_state.kyotaku)}  round: {int(cs.round_state.round)}  dealer: {int(cs.round_state.dealer)}")
    print("-"*90)
    # Status of each player
    for p in range(4):
        hand14 = int(cs.players.hand[p].sum())
        print(f"[P{p}] hand={hand14}  riichi={bool(cs.players.riichi[p])}  furiten_by_river={bool(cs.players.furiten_by_discard[p])}  furiten_by_pass_ron={bool(cs.players.furiten_by_pass[p])}  menzen={bool(cs.players.is_hand_concealed[p])}  ippatsu={bool(cs.players.ippatsu[p])}  dbl_riichi={bool(cs.players.double_riichi[p])}")
        print(f"      melds: {_melds_summary(cs.players.melds[p])}")
        print(f"      hand:  {_hand_summary(cs.players.hand[p])}")
    print("-"*90)
    # River
    for p in range(4):
        print(f"RIVER P{p}: {_river_one_line(cs.players.river, p)}")
    print("-"*90)
    # Legal actions (current player and discarder's surroundings)
    cp = int(cs.current_player)
    lp = int(cs.round_state.last_player)
    print(f"LEGAL P{cp}: {_legal_summary(cs.players.legal_action_mask[cp])}")
    if 0 <= lp < 4:
        print(f"LEGAL P{lp}: {_legal_summary(cs.players.legal_action_mask[lp])}")
    # Yaku judgment (current tile/next tile)
    print("-"*90)
    for p in range(4):
        hy = list(map(bool, cs.players.has_yaku[p]))
        fan = list(map(int, cs.players.fan[p]))
        fu  = list(map(int, cs.players.fu[p]))
        print(f"YAKU P{p}: has={hy}  fan={fan}  fu={fu}")
    print("="*90 + "\n")


IDX_AFTER_FIRST_DRAW = FIRST_DRAW_IDX - 1
NUM_TILE_TYPE = Tile.NUM_TILE_TYPE

MELD_ACTIONS = jnp.array([
    Action.CHI_L,
    Action.CHI_M,
    Action.CHI_R,
    Action.PON,
    Action.OPEN_KAN,
    Action.RON,
    Action.PASS,
])

def _tile_from_action(ls, action):
    """Return the tile that actually decreases from the discard action (TSUMOGIRI support)"""
    if int(action) == Action.TSUMOGIRI:
        return int(ls.round_state.last_draw)
    return int(action)

def _hand_sum(hand_row):
    return int(hand_row.sum())

def _bool(x):  # JAX bool → Python bool
    return bool(jnp.array(x))

def _any(x):
    return _bool(jnp.any(x))

def _all(x):
    return _bool(jnp.all(x))

def test_step(ls, action, cs):
    """行動タイプに応じてそれぞれの詳細テストに振り分ける"""
    a = int(action)
    if a == Action.DUMMY:
        return test_dummy(ls, action, cs)
    if a == Action.RIICHI:
        return test_riichi(ls, action, cs)
    if a == Action.RON:
        return test_ron(ls, action, cs)
    if a == Action.TSUMO:
        return test_tsumo(ls, action, cs)
    if a == Action.PASS:
        return test_pass(ls, action, cs)
    if Action.CHI_L <= a <= Action.CHI_R:
        return test_chi(ls, action, cs)
    if a == Action.PON:
        return test_pon(ls, action, cs)
    if a == Action.OPEN_KAN:
        return test_open_kan(ls, action, cs)
    if Action.is_selfkan(a):
        return test_selfkan(ls, action, cs)
    # Otherwise, discard (0..33 or TSUMOGIRI)
    return test_discard(ls, action, cs)

# ========== Additional helpers ==========

def _expect_tile_mask_after_draw(cs, cp):
    """Expected discard mask after draw (False for direct draw tile, True for other tiles)"""
    last_draw = int(cs.round_state.last_draw)
    hand = cs.players.hand[cp]
    mask = (hand > 0).astype(jnp.bool_).at[last_draw].set(hand[last_draw] >= 2)
    player_mask = jnp.zeros(cs.players.legal_action_mask.shape[1], dtype=jnp.bool_)
    player_mask = player_mask.at[:Tile.NUM_TILE_TYPE].set(mask)
    player_mask = player_mask.at[Action.TSUMOGIRI].set(True)
    return player_mask

def _meld_head(ls, cs, p):
    """Estimate and return the location where meld (addition or replacement) was performed in this step"""
    nm0 = int(ls.players.meld_counts[p])
    nm1 = int(cs.players.meld_counts[p])
    # Added case (closed kan/open kan/pon/chi)
    if nm1 == nm0 + 1:
        idx = nm1 - 1
        return idx, int(cs.players.melds[p, idx])
    # Replacement (added kan)
    for i in range(4):
        if int(ls.players.melds[p, i]) != int(cs.players.melds[p, i]):
            return i, int(cs.players.melds[p, i])
    return -1, -1

def _meld_decode(m):
    m = jnp.uint16(m)
    return int(Meld.action(m)), int(Meld.target(m)), int(Meld.src(m))

def _hand_delta(ls, cs, p):
    return (cs.players.hand[p] - ls.players.hand[p]).astype(jnp.int32)

# ========== Detailed test_* ==========

def test_discard(ls, action, cs):
    l_p = int(ls.current_player)
    tile = _tile_from_action(ls, action)
    # Hand sum and number of tiles
    assert _hand_sum(ls.players.hand[l_p]) + ls.players.meld_counts[l_p] * 3 == 14, "before discard: 14 tiles"
    assert int(ls.players.hand[l_p, tile]) >= 1, "must have the tile to discard"
    assert _hand_sum(cs.players.hand[l_p]) + cs.players.meld_counts[l_p] * 3 == 13, "after discard: 13 tiles"
    assert int(cs.players.hand[l_p, tile]) == int(ls.players.hand[l_p, tile]) - 1, "tile count must decrease by 1"
    assert int(cs.round_state.last_player) == l_p, "_last_player should be discarder"

    # River update (1 tile added)
    r_idx = int(cs.players.discard_counts[l_p]) - 1
    dec = River.decode_river(cs.players.river)  # [tile, riichi, gray, tsumogiri, src, meld_type]
    assert r_idx >= 0, "river index must be >= 0"
    assert int(dec[0, l_p, r_idx]) == tile, "river must record discarded tile"
    assert bool(dec[3, l_p, r_idx]) == (int(action) == Action.TSUMOGIRI), "tsumogiri flag mismatch"
    assert bool(dec[1, l_p, r_idx]) == bool(ls.players.riichi_declared[l_p]), "riichi flag mismatch"
    assert int(dec[5, l_p, r_idx]) == 0, "meld_type must be 0 on pure discard"

    # Meld window
    is_ended = _bool(cs.round_state.terminated_round)
    has_meld_window = (int(cs.round_state.target) == tile) and _any(cs.players.legal_action_mask[:, Action.PASS])

    if is_ended:
        assert _all(cs.players.legal_action_mask[:, Action.DUMMY]), "after ryukyoku, only DUMMY"
        return

    if not has_meld_window:
        cp = int(cs.current_player)
        assert cp == (l_p + 1) % 4, "next player must be (l_p+1)%4"
        lm = cs.legal_action_mask
        last_draw = int(cs.round_state.last_draw)
        if not _bool(cs.players.riichi[cp]):
            expected = _expect_tile_mask_after_draw(cs, cp)
            # Compare tile area and TSUMOGIRI (other options are given by the environment)
            assert _all(lm[:Tile.NUM_TILE_TYPE] == expected[:Tile.NUM_TILE_TYPE]), "tile mask mismatch after draw"
            assert _bool(lm[Action.TSUMOGIRI]), "tsumogiri must be legal after draw"
            assert cs.players.hand[cp,int(cs.round_state.last_draw)]>0, "_last_draw should be in the hand of the current player"
            can_tsumo = _bool(cs.players.can_win[cp, last_draw]) and (_bool(cs.players.is_hand_concealed[cp]) or _bool(cs.round_state.can_after_kan) or _bool(cs.round_state.is_haitei) or _bool(cs.players.has_yaku[cp, 1]))
            # last_draw cannot be discarded
            assert _bool(lm[Action.TSUMO]) == can_tsumo, "tsumo legality mismatch"
        else:
            assert _bool(lm[Action.TSUMOGIRI]), "tsumogiri must be allowed under riichi"
            last_draw = int(cs.round_state.last_draw)
            if cs.players.hand[cp, last_draw] < 2:
                assert not _bool(lm[last_draw]), "tile action for last_draw must be False in riichi mask"
            else:
                if _bool(lm[last_draw]):
                    assert _bool(Hand.is_tenpai(Hand.sub(cs.players.hand[cp], last_draw))), "tile action for last_draw must keep tenpai"

            # Each discard that is True must keep tenpai
            hand14 = cs.players.hand[cp]
            for i in range(Tile.NUM_TILE_TYPE):
                if _bool(lm[i]):
                    assert _bool(Hand.is_tenpai(Hand.sub(hand14, i))), f"discard {i} must keep tenpai"

            # tsumogiri can only be discarded if it keeps tenpai
            if _bool(lm[Action.TSUMOGIRI]):
                assert _bool(Hand.is_tenpai(Hand.sub(cs.players.hand[cp], last_draw))), "tsumogiri must keep tenpai"

            # Hand must be tenpai after draw
            assert _bool(Hand.is_tenpai(Hand.sub(cs.players.hand[cp], last_draw))), "hand must be tenpai after draw"
    else:
        t = int(cs.round_state.target)
        assert t == tile, "target must equal discarded tile"
        mp = int(cs.current_player)
        assert _bool(cs.players.legal_action_mask[mp, Action.PASS]), "PASS must be allowed for taker"
        meld_any = (
            _any(cs.players.legal_action_mask[:, Action.CHI_L:Action.CHI_R+1])
            or _bool(cs.players.legal_action_mask[:, Action.PON].any())
            or _bool(cs.players.legal_action_mask[:, Action.OPEN_KAN].any())
            or _bool(cs.players.legal_action_mask[:, Action.RON].any())
        )
        assert meld_any, "some meld/ron must be available in window"
        assert int(cs.round_state.last_draw) == -1, "_last_draw should be cleared after discard without draw"

def test_selfkan(ls, action, cs):
    cp = int(ls.current_player)   # Kan executor = ls.current_player
    tile = int(action) - Tile.NUM_TILE_TYPE
    was_pon = int(ls.players.pon[(cp, tile)]) != 0
    d = _hand_delta(ls, cs, cp)

    # Common: kan flow
    assert _bool(cs.round_state.kan_declared) or _bool(cs.round_state.can_after_kan), "kan flow must be in effect (declared or rinshan)"

    if not was_pon:
        # Closed kan
        assert int(d[tile]) == -4, "closed_kan consumes 4 tiles"
        # n_meld +1 & recent meld contents
        idx, m = _meld_head(ls, cs, cp)
        assert idx >= 0, "closed_kan should append a meld"
        act, tgt, src = _meld_decode(m)
        assert Action.is_selfkan(act), "closed_kan meld must be selfkan"
        assert tgt == tile, "closed_kan target mismatch"
        assert src == 0, "closed_kan src must be 0 (self)"
    else:
        # Added kan
        assert int(d[tile]) == -1, "added_kan consumes 1 tile"
        # _n_meld does not increase (replacement)
        assert int(cs.players.meld_counts[cp]) == int(ls.players.meld_counts[cp]), "added_kan should not change n_meld"
        # pon information is cleared to 0
        assert int(cs.players.pon[(cp, tile)]) == 0, "_pon must be cleared on added_kan"
        # Replaced meld contents
        idx, m = _meld_head(ls, cs, cp)
        assert idx >= 0, "added_kan must replace an existing peng meld"
        act, tgt, src = _meld_decode(m)
        assert Action.is_selfkan(act), "added_kan meld must be selfkan"
        assert tgt == tile, "added_kan target mismatch"

    # Is there a chan kan reception?
    robbing_kan_open = _any(cs.players.legal_action_mask[:, Action.RON])
    if robbing_kan_open:
        mp = int(cs.current_player)
        assert _bool(cs.players.legal_action_mask[mp, Action.PASS]), "robbing_kan window must include PASS"
        assert int(cs.round_state.target) == tile, "robbing_kan target must be added_kan tile"
        assert not _bool(cs.round_state.can_after_kan), "rinshan draw should not have happened yet"
    else:
        # Already processed rinshan
        assert _bool(cs.round_state.can_after_kan), "rinshan must be active when no robbing_kan"
        assert not _bool(cs.round_state.kan_declared), "kan_declared should be cleared after draw_after_kan"
        # Self kan is allowed
        assert _bool(cs.players.legal_action_mask[int(cs.current_player), Action.TSUMOGIRI]), "tsumogiri should be legal after rinshan"

def test_open_kan(ls, action, cs):
    cp = int(cs.current_player)   # The player who鸣いた is ls.current_player
    lp = int(cs.round_state.last_player)
    tile = int(ls.round_state.target)
    d = _hand_delta(ls, cs, cp)

    assert not _bool(cs.players.is_hand_concealed[cp]), "open_kan breaks menzen"
    assert int(d[tile]) == -3, "open_kan consumes 3 tiles"

    # Graying of the river + meld_type=2
    r_idx = int(cs.players.discard_counts[lp]) - 1
    dec = River.decode_river(cs.players.river)
    assert r_idx >= 0, "river index must be valid"
    assert _bool(dec[2, lp, r_idx]), "river tile must be grayed after open_kan"
    assert int(dec[5, lp, r_idx]) == 2, "meld_type must be 2 for open_kan"

    # Additional meld contents
    idx, m = _meld_head(ls, cs, cp)
    assert idx >= 0, "open_kan should append a meld"
    act, tgt, src = _meld_decode(m)
    assert act == Action.OPEN_KAN, "meld.action must be open_kan"
    assert tgt == tile, "meld.target must equal target tile"
    assert src == (lp - cp) % 4, "meld.src must be relative position"

def test_pon(ls, action, cs):
    cp = int(cs.current_player)
    lp = int(cs.round_state.last_player)
    tile = int(ls.round_state.target)
    d = _hand_delta(ls, cs, cp)

    assert not _bool(cs.players.is_hand_concealed[cp]), "pon breaks menzen"
    assert int(d[tile]) == -2, "pon consumes 2 tiles"
    assert _hand_sum(cs.players.hand[cp]) == _hand_sum(ls.players.hand[cp]) - 2, "hand sum must decrease by 2"

    # River update
    r_idx = int(cs.players.discard_counts[lp]) - 1
    dec = River.decode_river(cs.players.river)
    assert r_idx >= 0, "river index must be valid"
    assert _bool(dec[2, lp, r_idx]), "river tile must be grayed after pon"
    assert int(dec[5, lp, r_idx]) == 1, "meld_type=1 for PON"

    # Additional meld
    idx, m = _meld_head(ls, cs, cp)
    assert idx >= 0, "pon must append a meld"
    act, tgt, src = _meld_decode(m)
    print("act", act, "tgt", tgt, "src", src, "tile", tile)
    assert act == Action.PON and tgt == tile, "pon meld mismatch"
    assert src == (lp - cp) % 4, "src must be relative"

    # _pon bookkeeping
    expect_idx = int(cs.players.meld_counts[cp]) - 1
    expect = (src << 2) | expect_idx
    assert int(cs.players.pon[(cp, tile)]) == expect, "_pon bookkeeping mismatch"

    # Legal discard is hand>0 (14→12, so immediate discard is allowed)
    lm = cs.players.legal_action_mask[cp, :Tile.NUM_TILE_TYPE]
    hand_mask = (cs.players.hand[cp] > 0).astype(jnp.bool_)
    hand_mask = hand_mask.at[tgt].set(False)  # Prohibitive tile exchange
    assert _all(lm == hand_mask), "discard mask must match hand (>0)"

def test_chi(ls, action, cs):
    cp = int(ls.current_player)
    tar_p = int(ls.round_state.last_player)
    tile = int(ls.round_state.target)

    # Face-up collapse
    assert not _bool(cs.players.is_hand_concealed[cp]), "chi breaks menzen"

    # River update (gray, mt=3/4/5, src=upper seat=3)
    r_idx = int(cs.players.discard_counts[tar_p]) - 1
    src = (tar_p - cp) % 4
    dec = River.decode_river(cs.players.river)
    print("src", src)
    assert r_idx >= 0, "river index must be valid"
    assert _bool(dec[2, tar_p, r_idx]), "river tile must be grayed after chi"
    mt = int(dec[5, tar_p, r_idx])
    assert mt in (3,4,5), "meld_type must indicate CHI"
    assert int(dec[4, tar_p, r_idx]) == 3, "src must be upper seat (3)"
    assert src == 3, "chi src must be upper seat (=3)"

    # Hand difference (target does not decrease, other 2 tiles are -1)
    d = _hand_delta(ls, cs, cp)
    chi_idx = int(action) - Action.CHI_L
    start = tile - chi_idx
    need = [start, start+1, start+2]
    need.remove(tile)
    for t in need:
        assert int(d[t]) == -1, "chi must consume the two non-target tiles"
    assert int(d[tile]) == 0, "target is taken from discard, not from hand"
    assert _hand_sum(cs.players.hand[cp]) == _hand_sum(ls.players.hand[cp]) - 2, "hand sum must decrease by 2"

    # Additional meld
    idx, m = _meld_head(ls, cs, cp)
    assert idx >= 0, "chi must append a meld"
    act, tgt, src = _meld_decode(m)
    assert act == int(action) and tgt == tile, "chi meld mismatch"
    assert src == 3, "chi src must be upper seat (=3)"

    # Prohibitive tile exchange
    prohib = int(Meld.prohibitive_tile_type_after_chi(int(action), tile))
    if prohib >= 0:
        assert not _bool(cs.players.legal_action_mask[cp, prohib]), "prohibitive tile must not be legal right after chi"


def test_dummy(ls, action, cs):
    """
    After DUMMY (= immediately after _next_round call) comprehensive check (shared phase version).
    """
    dc = int(cs.round_state.dummy_count)

    # ----------------------------
    # During shared phase (state-only)
    # ----------------------------
    if dc != 0 and not _bool(cs.terminated):
        # 1) current_player is rotated by +1
        expected_cp = int((ls.current_player + 1) % 4)
        assert int(cs.current_player) == expected_cp, \
            f"during DUMMY sharing, cp must rotate by +1 (expected {expected_cp}, got {int(cs.current_player)})"

        # 2) During sharing, (dealer, round, honba, kyotaku) must not change
        assert int(cs.round_state.dealer)     == int(ls.round_state.dealer),     "dealer must not change during sharing"
        assert int(cs.round_state.round)   == int(ls.round_state.round),   "round must not change during sharing"
        assert int(cs.round_state.honba)   == int(ls.round_state.honba),   "honba must not change during sharing"
        assert int(cs.round_state.kyotaku) == int(ls.round_state.kyotaku), "kyotaku must carry over unchanged during sharing"

        # 3) Scores:
        #   - During the first shared phase (dc==1), the previous round settlement may be reflected at this point
        #   - During the second and third shared phases (dc in {2,3}), the previous state must be unchanged
        if dc in (1,2, 3) and not _bool(ls.terminated):
            assert _all(cs.round_state.score == ls.round_state.score), \
                "scores must carry over unchanged during sharing steps dc=2/3"
        # dc==1 is also acceptable (allow one-time settlement fluctuation)

        # Initialization of tile distribution, river, etc. is checked after "determined"
        return

    # ----------------------------
    # Endgame branch
    # ----------------------------
    if _bool(cs.terminated):
        assert cs.round_state.score.dtype == jnp.int32 and _all(jnp.isfinite(cs.round_state.score)), \
            "final scores must be finite float32"
        return

    # ----------------------------
    # Endgame branch (game ended when determined)
    # ----------------------------
    if _bool(cs.terminated):
        assert cs.round_state.score.dtype == jnp.int32 and _all(jnp.isfinite(cs.round_state.score)), \
            "final scores must be finite float32"
        return

    # ----------------------------
    # Hereafter the full check for "determined" (new round initialized)
    # dc == 0 かつ not terminated
    # ----------------------------

    # Initialization of round termination flag, etc. (as before)
    # assert not _bool(cs.round_state.terminated_round), "after DUMMY, round should be reset (not _terminated_round)"
    assert int(cs.round_state.target) == -1, "target should be reset to -1 on new round"
    assert not _bool(cs.round_state.kan_declared), "kan_declared should be False at new round start"
    assert not _bool(cs.round_state.can_after_kan),   "can_after_kan should be False at new round start"
    assert not _bool(cs.round_state.is_haitei),    "is_haitei should be False at new round start"

    # --- Parent, round, honba update rules (_next_round and equivalent; refer to ls as previous round state) ---
    dealer_before   = int(ls.round_state.dealer)
    ls_tenpai    = jnp.any(ls.players.can_win, axis=-1)  # (4,)
    ls_has_won      = ls.players.has_won                        # (4,)
    will_dealer_continue = _bool(ls_tenpai[dealer_before]) or _bool(ls_has_won[dealer_before])

    expected_round   = int(ls.round_state.round if will_dealer_continue else (ls.round_state.round + 1))
    expected_dealer     = int(dealer_before if will_dealer_continue else (dealer_before + 1) % 4)
    expected_honba   = int(0 if _any(ls_has_won) else (ls.round_state.honba + 1))
    expected_kyotaku = int(ls.round_state.kyotaku)

    assert int(cs.round_state.round)   == expected_round, f"round must update correctly (expected {expected_round}, got {int(cs.round_state.round)})"
    assert int(cs.round_state.dealer)     == expected_dealer,   f"dealer must update correctly (expected {expected_dealer}, got {int(cs.round_state.dealer)})"
    assert int(cs.round_state.honba)   == expected_honba, f"honba must update correctly (expected {expected_honba}, got {int(cs.round_state.honba)})"
    assert int(cs.round_state.kyotaku) == expected_kyotaku, "kyotaku should be carried over on next round"

    # Wind array is updated to the new parent standard
    assert _all(cs.round_state.seat_wind == jnp.array(
        [expected_dealer, (expected_dealer+1)%4, (expected_dealer+2)%4, (expected_dealer+3)%4], dtype=jnp.int8)), \
        "_seat_wind should match new dealer order"

    # Scores carry over unchanged from ls.round_state.score at the determined time (assuming the yaku calculation is done on the ls side)
    assert _all(cs.round_state.score == ls.round_state.score), "scores should carry over unchanged into new round"

    # --- Basic properties after the first draw ---
    hand_sums = jnp.sum(cs.players.hand, axis=1).astype(jnp.int32)
    idx_14 = int(jnp.argmax(hand_sums))
    n_14   = int((hand_sums == 14).astype(jnp.int32).sum())
    assert n_14 == 1, f"exactly one player must have 14 tiles at new round start, got counts={list(map(int, hand_sums))}"
    assert idx_14 == int(cs.current_player), "the player with 14 tiles must be the current player"

    for p in range(4):
        if p == idx_14:
            continue
        assert int(hand_sums[p]) == 13, f"non-current players must have 13 tiles, but P{p} has {int(hand_sums[p])}"

    active_rows = jnp.any(cs.players.legal_action_mask, axis=1).astype(jnp.int32)
    assert int(active_rows.sum()) >= 1, "at least one player should have actions at new round start"
    assert _bool(jnp.any(cs.players.legal_action_mask[idx_14])), "current player must have legal actions at new round start"

    if _bool(cs.players.legal_action_mask[idx_14, Action.TSUMOGIRI]):
        last_draw = int(cs.round_state.last_draw)
        if cs.players.hand[idx_14, last_draw] <= 1:
            assert not _bool(cs.players.legal_action_mask[idx_14, last_draw]), "cannot discard just-drawn tile as a tile action"

        lm = cs.players.legal_action_mask[idx_14, :Tile.NUM_TILE_TYPE]
        if 0 <= last_draw < Tile.NUM_TILE_TYPE:
            expect_mask = (cs.players.hand[idx_14] > 0).astype(jnp.bool_)
            expect_mask = expect_mask.at[last_draw].set(cs.players.hand[idx_14, last_draw] > 1)
        assert _all(lm == expect_mask), "discard mask should equal (hand>0) except for just-drawn tile"

    # Initial state of the river
    assert _all(cs.players.discard_counts == jnp.zeros(4, dtype=jnp.int8)), "_n_river should be zeroed at new round"
    dec_new = River.decode_river(cs.players.river)
    for p in range(4):
        n = int(cs.players.discard_counts[p])
        if n == 0:
            continue
        tiles = dec_new[0, p]
        assert _all(tiles[:n] >= 0) and _all(tiles[n:] == -1), "river decode shape should match n_river"


def test_riichi(ls, action, cs):
    cp = int(ls.current_player)
    assert int(cs.current_player) == cp, "riichi does not pass turn immediately"
    assert not _bool(cs.players.riichi[cp]), "riichi is accepted on draw, not immediately"


def test_ron(ls, action, cs):
    cp = int(cs.current_player)
    assert _bool(cs.round_state.terminated_round), "round must be terminated after RON"
    assert _all(cs.players.legal_action_mask[:, Action.DUMMY]), "only DUMMY should be legal after RON"
    assert int(cs.round_state.kyotaku) == 0, "kyotaku should be cleared to 0 after RON"
    assert _bool(cs.players.has_won[cp]), "winner's _has_won must be True"
    # Yaku and score calculation
    prevalent_wind = ls.round_state.round % 4
    seat_wind = ls.round_state.seat_wind[ls.current_player]
    yaku, fan, fu = jitted_yaku_judge(
        ls.players.hand[cp],
        ls.players.melds[cp],
        ls.players.meld_counts[cp],
        ls.round_state.target,
        ls.players.riichi[cp],
        True,
        prevalent_wind,
        seat_wind,
        _dora_array(ls),
    )
    # Check fan, fu
    assert int(ls.players.can_win[ls.current_player, ls.round_state.target]), "ron should be allowed"
    assert int(ls.players.fan[ls.current_player, 0]) == fan
    assert int(ls.players.fu[ls.current_player, 0]) == fu


def test_tsumo(ls, action, cs):
    cp = int(cs.current_player)
    assert _bool(cs.round_state.terminated_round), "round must be terminated after TSUMO"
    assert _all(cs.players.legal_action_mask[:, Action.DUMMY]), "only DUMMY should be legal after TSUMO"
    assert int(cs.round_state.kyotaku) == 0, "kyotaku should be cleared to 0 after TSUMO"
    assert _bool(cs.players.has_won[cp]), "winner's _has_won must be True"
    # Yaku and score calculation
    prevalent_wind = ls.round_state.round % 4
    seat_wind = ls.round_state.seat_wind[ls.current_player]
    yaku, fan, fu = jitted_yaku_judge(
        ls.players.hand[cp].at[ls.round_state.last_draw].set(ls.players.hand[cp][ls.round_state.last_draw] -1),
        ls.players.melds[cp],
        ls.players.meld_counts[cp],
        ls.round_state.last_draw,
        ls.players.riichi[cp],
        False,
        prevalent_wind,
        seat_wind,
        _dora_array(ls),
    )
    # Check fan, fu
    assert int(ls.players.can_win[ls.current_player, ls.round_state.last_draw]), "tsumo should be allowed"
    assert int(ls.players.fan[ls.current_player, 0]) == fan
    assert int(ls.players.fu[ls.current_player, 0]) == fu


def test_pass(ls, action, cs):
    cp = int(cs.current_player)
    was_robbing_kan = _bool(ls.round_state.kan_declared)
    could_ron = _bool(ls.players.legal_action_mask[cp, Action.RON])

    if not was_robbing_kan and could_ron:
        assert _bool(cs.players.furiten_by_pass[cp]), "passing RON (non-robbing_kan) should set furiten"

    someone_has_pass = _any(ls.players.legal_action_mask[:, Action.PASS])
    still_window = _any(cs.players.legal_action_mask[:, Action.PASS])
    if someone_has_pass and not still_window:
        if not was_robbing_kan:
            # Self-draw progression
            assert cp == int(ls.round_state.last_player + 1) % 4, "current player should be the next player"
            np_c = int(cs.current_player)
            h = _hand_sum(cs.players.hand[np_c])
            assert h + cs.players.meld_counts[np_c] * 3 == 14, f"post-pass draw path: hand sum sanity, got {h}"
        else:
            # Chan kan path: move to rinshan tsumo, so check the consistency of _can_after_kan/kan_declared
            assert cs.players.n_kan == ls.players.n_kan + 1, "n_kan should be incremented by 1"
            assert cp == int(ls.round_state.last_player), "last_player should be the current player"


def act_randomly(rng, legal_action_mask) -> int:
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    return jax.random.categorical(rng, logits=logits)


def play_according_to_shanten(rng, hand, melds, n_meld, legal_action_mask):
    flatten_hand = Yaku.flatten(hand, melds, n_meld)
    best_action = jnp.argmin(jitted_shanten_discard(flatten_hand))
    is_legal = legal_action_mask[best_action]
    logits = jnp.log(legal_action_mask.astype(jnp.float32))
    random_action = jax.random.categorical(rng, logits=logits)
    action = jnp.where(is_legal, best_action, random_action)
    # If riichi is possible, riichi
    can_riichi = legal_action_mask[Action.RIICHI]
    action = jnp.where(can_riichi, Action.RIICHI, action)
    # If tsumo is possible, tsumo
    can_tsumo = legal_action_mask[Action.TSUMO]
    action = jnp.where(can_tsumo, Action.TSUMO, action)
    # If ron is possible, ron
    can_ron = legal_action_mask[Action.RON]
    action = jnp.where(can_ron, Action.RON, action)
    return action

def play_kan_orientedly(rng, hand, legal_action_mask):
    # Convenience shortcut
    A = legal_action_mask.shape[0]

    # If it is not determined here, the discard action
    # "Keep tiles with 2 or more" = discard candidates are tiles <= 1
    # Explicitly specify the index set of the discard action (example)
    DISCARD_START = 0
    DISCARD_END   = Tile.NUM_TILE_TYPE  # Discard is 0..NUM_TILE_TYPE-1

    # Discard legal & discard candidate both mask
    discard_legal = legal_action_mask[DISCARD_START:DISCARD_END]  # shape [T]
    keep_mask = (hand > 1)  # Keep tiles (don't want to discard)
    discard_candidate = discard_legal & (~keep_mask)

    # If all "keep", (if discard is needed, sample from legal discard)
    no_candidate = ~jnp.any(discard_candidate)
    final_discard_mask = jnp.where(no_candidate, discard_legal, discard_candidate)

    # Create logits and sample with categorical (-inf to disable)
    logits = jnp.log(final_discard_mask.astype(jnp.float32))

    rng, sub = jax.random.split(rng)
    sampled_discard = jax.random.categorical(sub, logits=logits)  # 0..T-1 (discard index)

    sampled_discard = jnp.int32(DISCARD_START) + sampled_discard

    action = sampled_discard

    # First priority is ron
    can_ron   = legal_action_mask[Action.RON]
    can_tsumo = legal_action_mask[Action.TSUMO]

    action = jnp.where(can_ron,   jnp.int32(Action.RON),   action)
    action = jnp.where(can_tsumo, jnp.int32(Action.TSUMO), action)

    # If there is no ron, the kan system
    can_open_kan = legal_action_mask[Action.OPEN_KAN]
    action = jnp.where(can_open_kan, jnp.int32(Action.OPEN_KAN), action)

    # Other kan groups (adjust the index according to the environment)
    KAN_START = Tile.NUM_TILE_TYPE             # Example: This is the start of the kan system
    KAN_END   = Action.TSUMOGIRI               # Example: This is the end of the kan system (TSUMOGIRI is not included)

    can_kan_slice = legal_action_mask[KAN_START:KAN_END]  # shape [K]
    has_kan = jnp.any(can_kan_slice)

    # If there are multiple kan, the first True (if needed, score with policy)
    first_kan_offset = jnp.argmax(can_kan_slice)  # 0..K-1
    chosen_kan = jnp.int32(KAN_START) + jnp.int32(first_kan_offset)
    action = jnp.where(has_kan, chosen_kan, action)

    # Last insurance: if not determined, sample from legal actions uniformly
    fallback_logits = jnp.log(legal_action_mask.astype(jnp.float32))
    rng, sub2 = jax.random.split(rng)
    fallback = jax.random.categorical(sub2, logits=fallback_logits)
    action = jnp.where(legal_action_mask[action], action, fallback)
    return action.astype(jnp.int32)  # Return rng if needed



class TestPlay(unittest.TestCase):
    def setup(self):
        pass

    def test_random_play(self):
        """
        Randomly execute and test each method.
        """
        for i in range(10):
            rng = jax.random.PRNGKey(i)
            state = jitted_init(rng)
            max_steps = 2000
            steps = 0
            while not bool(state.terminated):
                ls = state
                action = act_randomly(rng, state.legal_action_mask)
                rng, rng_sub = jax.random.split(rng)
                state_next = jitted_step(state, action)
                assert state_next.step_count <= len(state_next.round_state.action_history[0]), "step_count should be less than the length of action_history"
                print("seed", i, "step", steps, "current_player", int(ls.current_player), "action", action, "next_deck_ix", int(ls.round_state.next_deck_ix), "remaining_deck_ix", int(ls.round_state.next_deck_ix - ls.round_state.last_deck_ix + 1))
                try:
                    test_step(ls, action, state_next)
                except AssertionError as e:
                    # Visualize and re-throw
                    dump_debug(ls, state_next, action, steps)
                    raise

                state = state_next
                steps += 1
                if steps > max_steps:
                    dump_debug(ls, state_next, action, steps)
                    self.fail("Exceeded max steps without termination (possible loop)")

    def test_shanten_play(self):
        """
        Play to minimize the shanten number.
        """
        for i in range(5):
            rng = jax.random.PRNGKey(i)
            state = jitted_init(rng)
            max_steps = 2000
            steps = 0

            while not bool(state.terminated):
                ls = state
                action = play_according_to_shanten(
                    rng,
                    ls.players.hand[ls.current_player],
                    ls.players.melds[ls.current_player],
                    ls.players.meld_counts[ls.current_player],
                    state.legal_action_mask
                )
                rng, rng_sub = jax.random.split(rng)
                state_next = jitted_step(state, action)
                assert state_next.step_count <= len(state_next.round_state.action_history[0]), "step_count should be less than the length of action_history"
                print("seed", i, "step", steps, "current_player", int(ls.current_player), "action", action)
                try:
                    test_step(ls, action, state_next)
                except AssertionError as e:
                    # Visualize and re-throw
                    dump_debug(ls, state_next, action, steps)
                    raise

                state = state_next
                steps += 1
                if steps > max_steps:
                    dump_debug(ls, state_next, action, steps)
                    self.fail("Exceeded max steps without termination (possible loop)")

    def test_kan_oriented_play(self):
        """
        Play to prioritize kan.
        """
        for i in range(10):
            rng = jax.random.PRNGKey(i)
            state = jitted_init(rng)
            max_steps = 2000
            steps = 0

            while not bool(state.terminated):
                ls = state
                action = play_kan_orientedly(rng, ls.players.hand[ls.current_player], state.legal_action_mask)
                rng, rng_sub = jax.random.split(rng, 2)
                state_next = jitted_step(state, action)
                assert state_next.step_count <= len(state_next.round_state.action_history[0]), "step_count should be less than the length of action_history"
                print(
                    "seed", i,
                    "step", steps,
                    "current_player", int(ls.current_player),
                    "action", action,
                )
                if Tile.NUM_TILE_TYPE <= action < Action.TSUMOGIRI:
                    print("KAN!!!!!!")
                try:
                    test_step(ls, action, state_next)
                except AssertionError as e:
                    # Visualize and re-throw
                    dump_debug(ls, state_next, action, steps)
                    raise

                state = state_next
                steps += 1
                if steps > max_steps:
                    dump_debug(ls, state_next, action, steps)
                    self.fail("Exceeded max steps without termination (possible loop)")

    def test_rule_based_play(self):
        """
        Test a somewhat reasonable rule-based player.
        """
        for i in range(10):
            rng = jax.random.PRNGKey(i)
            state = jitted_init(rng)
            max_steps = 2000
            steps = 0

            while not bool(state.terminated):
                ls = state
                action = jitted_rule_based_player(state, rng)
                rng, rng_sub = jax.random.split(rng, 2)
                state_next = jitted_step(state, action)
                assert state_next.step_count <= len(state_next.round_state.action_history[0]), "step_count should be less than the length of action_history"
                print(
                    "seed", i,
                    "step", steps,
                    "current_player", int(ls.current_player),
                    "action", action,
                )
                try:
                    test_step(ls, action, state_next)
                except AssertionError as e:
                    # Visualize and re-throw
                    dump_debug(ls, state_next, action, steps)
                    raise

                state = state_next
                steps += 1
                if steps > max_steps:
                    dump_debug(ls, state_next, action, steps)
                    self.fail("Exceeded max steps without termination (possible loop)")

