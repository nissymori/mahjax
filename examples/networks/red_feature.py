"""Observation -> token sequence for ``mahjax.red_mahjong``.

This module knows about mahjong and about red fives. It turns the observation dict
into a flat sequence of tokens plus a validity mask, and nothing else. ``network.py``
holds the sequence models, which are env-agnostic and consume whatever tokenizer they
are handed -- so ``no_red_feature.py`` is the same contract without the red-five
machinery.

Every observation key is REQUIRED. There are no ``if key in obs`` fallbacks: a
missing key raises a KeyError here rather than silently removing a feature branch and
training a blinded network on a subtly different input. If the observation changes,
this file should break loudly.

Sequence layout, mirroring the observation's own grouping:

    [CLS_hand]   hand tokens      (34, one per tile TYPE)
    [CLS_meld]   meld tokens      (16 = 4 seats x 4 slots)
    [CLS_hist]   history tokens   (200)
    [CLS_glob]   global token     (1)
                                  = 255

Hand tokens are per tile TYPE, not per physical tile. ``discard_shanten`` and
``can_win`` are 34-indexed and RED-INCLUSIVE (they derive from ``state.players.hand``,
which is ``Hand.to_34(hand_with_red)``), so one token per type carries the count
alongside those rows with no gather. ``tiles_seen`` is also 34-indexed but is table
state rather than hand state, so it goes to the GLOBAL segment, not onto these tokens.

Red fives ride as a 3-dim flag on the 5m/5p/5s tokens rather than as three extra
tokens. That is not cosmetic: ``hand_with_red[4]`` counts BLACK 5m only, with the red
at index 34, so aligning a raw 37-wide vector against the 34-indexed features would
under-count every five -- a hand with one red 5p and no black 5p would report "no 5p"
on one channel while ``discard_shanten`` said otherwise.
Folding reds into their type and flagging them separately keeps index t meaning the
same thing on every channel.
"""

from typing import Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.shanten import Shanten
from mahjax.red_mahjong.state import RoundState
from mahjax.red_mahjong.tile import Tile

try:
    from .transformer import orthogonal_init
except ImportError:
    from networks.transformer import orthogonal_init

NUM_PLAYERS = 4
NUM_TILE_TYPE = Tile.NUM_TILE_TYPE  # 34
NUM_TILE_TYPE_WITH_RED = Tile.NUM_TILE_TYPE_WITH_RED  # 37
MAX_MELDS_PER_PLAYER = 4
# Read off the buffer it indexes rather than copied: a positional-embedding table
# that silently disagrees with ``action_history`` is the kind of drift that shows up
# as garbage gradients, not as an error.
MAX_HISTORY_LENGTH = int(RoundState().action_history.shape[1])  # 200

# Token-type ids. CLS tokens get their own so a readout head is distinguishable from
# the content it reads -- and so the four CLS rows are not bitwise identical, which
# would make their outputs identical too.
TOK_CLS_HAND, TOK_CLS_MELD, TOK_CLS_HIST, TOK_CLS_GLOB = 0, 1, 2, 3
TOK_HAND, TOK_MELD, TOK_HIST, TOK_GLOB = 4, 5, 6, 7
NUM_TOKEN_TYPES = 8

SCORE_OFFSET = 250.0
SCORE_SCALE = 1250.0
MAX_ROUND_VALUE = 8.0
MAX_HONBA = 10.0
MAX_KYOTAKU = 10.0
MAX_WALL_REMAINING = 70.0
MAX_WIND_VALUE = 3.0

NOT_IN_HAND = Shanten.NOT_IN_HAND
# One embedding row per reachable discard-shanten value plus NOT_IN_HAND itself.
# Columns reach [0,8] / [0,13] / [0,13], so 0..13 are real and 14 means "not held".
SHANTEN_VOCAB = int(NOT_IN_HAND) + 1  # 15
# shanten_count is documented [-1, 6] (-1 == complete). Sized to [-1, 8] so that a
# widening of the bound clips at the extreme instead of indexing out of the table.
SHANTEN_COUNT_VOCAB = 10
# Red-five ids 34/35/36 fold onto tile types 4/13/22.
RED_FIVE_TYPES = jnp.array([4, 13, 22], dtype=jnp.int32)


def _batch(x: jnp.ndarray, *, base_ndim: int) -> jnp.ndarray:
    arr = jnp.asarray(x)
    return arr.reshape((1,) + arr.shape) if arr.ndim == base_ndim else arr


class ObservationTokenizer(nn.Module):
    """``obs -> (tokens, mask)``, shapes (B, NUM_TOKENS, D_MODEL) and (B, NUM_TOKENS).

    The class attributes below are the contract ``network.py`` reads; the no-red
    tokenizer exposes the same names with its own values.
    """

    D_MODEL = 128
    NUM_ACTIONS = Action.NUM_ACTION  # 87
    NUM_HAND_TOKENS = NUM_TILE_TYPE  # 34
    NUM_MELD_TOKENS = NUM_PLAYERS * MAX_MELDS_PER_PLAYER  # 16
    NUM_HIST_TOKENS = MAX_HISTORY_LENGTH  # 200
    NUM_GLOBAL_TOKENS = 1
    NUM_TOKENS = 4 + NUM_HAND_TOKENS + NUM_MELD_TOKENS + NUM_HIST_TOKENS + NUM_GLOBAL_TOKENS
    CLS_POSITIONS = (
        0,
        1 + NUM_HAND_TOKENS,
        2 + NUM_HAND_TOKENS + NUM_MELD_TOKENS,
        3 + NUM_HAND_TOKENS + NUM_MELD_TOKENS + NUM_HIST_TOKENS,
    )

    @nn.compact
    def __call__(self, obs: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        D = self.D_MODEL
        hand_ids = _batch(obs["hand"], base_ndim=1).astype(jnp.int32)
        B = hand_ids.shape[0]

        tile_emb = nn.Embed(NUM_TILE_TYPE + 1, D, embedding_init=orthogonal_init())
        type_emb = nn.Embed(NUM_TOKEN_TYPES, D, embedding_init=orthogonal_init())
        seat_emb = nn.Embed(NUM_PLAYERS, D, embedding_init=orthogonal_init())

        globals_vec = self._globals(obs, B)
        hand_tok, hand_mask = self._hand_tokens(obs, hand_ids, B, D, tile_emb, type_emb)
        meld_tok, meld_mask = self._meld_tokens(obs, B, D, tile_emb, type_emb, seat_emb)
        hist_tok, hist_mask = self._history_tokens(obs, B, D, type_emb)

        glob_tok = jnp.broadcast_to(
            type_emb(jnp.full((1,), TOK_GLOB))[None]
            + nn.Dense(D, kernel_init=orthogonal_init())(globals_vec)[:, None, :],
            (B, self.NUM_GLOBAL_TOKENS, D),
        )

        def cls(type_id):
            return jnp.broadcast_to(type_emb(jnp.full((1,), type_id))[None], (B, 1, D))

        ones = jnp.ones((B, 1), jnp.float32)
        tokens = jnp.concatenate(
            [cls(TOK_CLS_HAND), hand_tok, cls(TOK_CLS_MELD), meld_tok,
             cls(TOK_CLS_HIST), hist_tok, cls(TOK_CLS_GLOB), glob_tok],
            axis=1,
        )
        mask = jnp.concatenate(
            [ones, hand_mask, ones, meld_mask, ones, hist_mask,
             ones, jnp.ones((B, self.NUM_GLOBAL_TOKENS), jnp.float32)],
            axis=1,
        )
        return tokens * mask[..., None], mask

    # ---- segments ----------------------------------------------------------

    def _hand_tokens(self, obs, hand_ids, B, D, tile_emb, type_emb):
        """One token per tile TYPE, built by SUMMING embedding lookups.

        Every feature here is a small integer code, so each value gets its own
        learned vector instead of being normalized to a float and pushed through a
        shared projection. A Dense over ``[count, reds, shanten, ...]`` forces one
        direction in D to serve all of them, and it encodes "shanten 6" as six times
        "shanten 1" -- these are ordered labels, not a ratio scale, and the step from
        tenpai to 1-away is not the same size as the step from 5-away to 6-away.

        Only hand-related keys appear here. ``tiles_seen`` is table state that
        happens to be indexed by tile type; gathering it onto hand tokens would put
        a global fact inside the hand segment and defeat the segmentation.
        """
        held = hand_ids >= 0
        ids = jnp.where(held, jnp.clip(hand_ids, 0, NUM_TILE_TYPE_WITH_RED - 1), 0)
        is_red = ids >= NUM_TILE_TYPE
        types = jnp.where(is_red, RED_FIVE_TYPES[jnp.clip(ids - NUM_TILE_TYPE, 0, 2)], ids)

        rows = jnp.arange(B)[:, None]
        zeros = jnp.zeros((B, NUM_TILE_TYPE), jnp.int32)
        counts = zeros.at[rows, types].add(held.astype(jnp.int32))
        reds = zeros.at[rows, types].add((held & is_red).astype(jnp.int32))

        # (B,34,3), where 14 == NOT_IN_HAND. The per-column offset gives each
        # decomposition its own table, so "2 away by seven pairs" and "2 away
        # normally" are different vectors. Row 14 IS the not-held marker, which is
        # why no separate in-hand flag is needed any more.
        ds = jnp.clip(
            _batch(obs["discard_shanten"], base_ndim=2).astype(jnp.int32), 0, NOT_IN_HAND
        )
        ds_tok = nn.Embed(3 * SHANTEN_VOCAB, D, embedding_init=orthogonal_init())(
            ds + jnp.arange(3, dtype=jnp.int32) * SHANTEN_VOCAB
        ).sum(axis=-2)

        # Hand-wide scalars, added to all 34 tokens so each tile is read in the
        # context of the hand holding it.
        whole = (
            nn.Embed(SHANTEN_COUNT_VOCAB, D, embedding_init=orthogonal_init())(
                jnp.clip(_batch(obs["shanten_count"], base_ndim=0).astype(jnp.int32), -1, 8) + 1
            )
            + nn.Embed(2, D, embedding_init=orthogonal_init())(
                _batch(obs["furiten"], base_ndim=0).astype(jnp.int32)
            )
            + nn.Embed(2, D, embedding_init=orthogonal_init())(
                _batch(obs["is_discard_node"], base_ndim=0).astype(jnp.int32)
            )
        )[:, None, :]

        tok = (
            # +1 to match the meld segment, which reserves row 0 for "no tile" and
            # stores tile type t at row t+1. Both segments share this table so that a
            # 5m in hand and a 5m in someone's pon are the SAME vector; indexing it
            # with two conventions silently breaks exactly that.
            tile_emb(jnp.arange(NUM_TILE_TYPE) + 1)[None]
            + type_emb(jnp.full((1,), TOK_HAND))[None]
            + nn.Embed(5, D, embedding_init=orthogonal_init())(jnp.clip(counts, 0, 4))
            + nn.Embed(2, D, embedding_init=orthogonal_init())(jnp.clip(reds, 0, 1))
            + ds_tok
            + nn.Embed(2, D, embedding_init=orthogonal_init())(
                _batch(obs["can_win"], base_ndim=1).astype(jnp.int32)
            )
            + whole
        )
        # Every tile type is a valid token: "I hold none of these" is as informative
        # as holding two.
        return tok, jnp.ones((B, self.NUM_HAND_TOKENS), jnp.float32)

    def _meld_tokens(self, obs, B, D, tile_emb, type_emb, seat_emb):
        melds = _batch(obs["melds"], base_ndim=3).astype(jnp.int32)  # (B,3,4,4)
        action, tile, src = melds[:, 0], melds[:, 1], melds[:, 2]
        valid = (action >= 0).astype(jnp.float32)
        tok = (
            nn.Embed(self.NUM_ACTIONS + 1, D, embedding_init=orthogonal_init())(action + 1)
            + tile_emb(jnp.clip(self.to_type(tile), -1, NUM_TILE_TYPE - 1) + 1)
            + type_emb(jnp.full((1,), TOK_MELD))
            + seat_emb(jnp.arange(NUM_PLAYERS))[None, :, None, :]
            + nn.Dense(D, kernel_init=orthogonal_init())(
                jax.nn.one_hot(src, NUM_PLAYERS, dtype=jnp.float32)
            )
        )
        return tok.reshape(B, self.NUM_MELD_TOKENS, D), valid.reshape(B, self.NUM_MELD_TOKENS)

    def _history_tokens(self, obs, B, D, type_emb):
        """One token per recorded action, in order.

        With ``river`` gone this is the only discard channel, so it carries the
        actor's relative seat, the action (a tile for discards, a raw action id
        otherwise), the tsumogiri flag that tells those apart, and a position.
        """
        hist = _batch(obs["action_history"], base_ndim=2).astype(jnp.int32)  # (B,3,200)
        actor, act, tsumogiri = hist[:, 0], hist[:, 1], hist[:, 2]
        valid = (act >= 0).astype(jnp.float32)
        pos = nn.Embed(MAX_HISTORY_LENGTH, D, embedding_init=orthogonal_init())(
            jnp.arange(MAX_HISTORY_LENGTH)
        )
        tok = (
            nn.Embed(self.NUM_ACTIONS + 1, D, embedding_init=orthogonal_init())(act + 1)
            + nn.Embed(NUM_PLAYERS + 1, D, embedding_init=orthogonal_init())(actor + 1)
            + nn.Embed(3, D, embedding_init=orthogonal_init())(tsumogiri + 1)
            + type_emb(jnp.full((1,), TOK_HIST))[None]
            + pos[None]
        )
        return tok, valid

    def _globals(self, obs, B) -> jnp.ndarray:
        """Everything that is not about one tile or one event, as a flat vector."""
        b = _batch
        g = [
            (b(obs["scores"], base_ndim=1).astype(jnp.float32) + SCORE_OFFSET) / SCORE_SCALE,
            # Table state, not hand state: how much of each type is already visible
            # is true regardless of what I hold, so it is read by the global CLS.
            b(obs["tiles_seen"], base_ndim=1).astype(jnp.float32) / 4.0,
            (b(obs["round"], base_ndim=0).astype(jnp.float32) / MAX_ROUND_VALUE)[..., None],
            (b(obs["round_limit"], base_ndim=0).astype(jnp.float32) / MAX_ROUND_VALUE)[..., None],
            (b(obs["honba"], base_ndim=0).astype(jnp.float32) / MAX_HONBA)[..., None],
            (b(obs["kyotaku"], base_ndim=0).astype(jnp.float32) / MAX_KYOTAKU)[..., None],
            (b(obs["wall_remaining"], base_ndim=0).astype(jnp.float32) / MAX_WALL_REMAINING)[..., None],
            (b(obs["prevalent_wind"], base_ndim=0).astype(jnp.float32) / MAX_WIND_VALUE)[..., None],
            (b(obs["seat_wind"], base_ndim=0).astype(jnp.float32) / MAX_WIND_VALUE)[..., None],
            b(obs["ippatsu"], base_ndim=1).astype(jnp.float32),
            b(obs["riichi"], base_ndim=1).astype(jnp.float32),
            b(obs["is_hand_concealed"], base_ndim=1).astype(jnp.float32),
            # Relative seat one-hot over [none, me, right, across, left]; a linear
            # index would imply an ordering that is not there.
            jax.nn.one_hot(
                b(obs["last_player"], base_ndim=0).astype(jnp.int32) + 1,
                NUM_PLAYERS + 1, dtype=jnp.float32,
            ),
        ]
        # Tiles the decision is literally about: last_draw grounds TSUMOGIRI, target
        # grounds PON/CHI/RON/PASS.
        ctx = jnp.stack(
            [b(obs["last_draw"], base_ndim=0).astype(jnp.int32),
             b(obs["target"], base_ndim=0).astype(jnp.int32)],
            axis=1,
        )
        g.append(
            jax.nn.one_hot(self.to_type(ctx) + 1, NUM_TILE_TYPE + 1, dtype=jnp.float32).reshape(B, -1)
        )
        dora = b(obs["dora_indicators"], base_ndim=1).astype(jnp.int32)
        g.append(
            (
                jax.nn.one_hot(self.to_type(dora), NUM_TILE_TYPE, dtype=jnp.float32)
                * (dora >= 0).astype(jnp.float32)[..., None]
            ).sum(axis=1)
        )
        return jnp.concatenate(g, axis=-1)

    @staticmethod
    def to_type(tile_ids: jnp.ndarray) -> jnp.ndarray:
        """Red-aware id [0-36] (or -1) -> tile type [0-33] (or -1)."""
        return jnp.where(
            tile_ids >= NUM_TILE_TYPE,
            RED_FIVE_TYPES[jnp.clip(tile_ids - NUM_TILE_TYPE, 0, 2)],
            tile_ids,
        )
