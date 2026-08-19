from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp

from mahjax.red_mahjong.action import Action
from mahjax.red_mahjong.constants import MAX_DISCARDS_PER_PLAYER
from mahjax.red_mahjong.shanten import Shanten
from mahjax.red_mahjong.tile import Tile
try:
    from .transformer import TransformerBlock, orthogonal_init
except ImportError:
    from networks.transformer import TransformerBlock, orthogonal_init

NUM_PLAYERS = 4
MAX_HISTORY_LENGTH = 200
NUM_ACTIONS = Action.NUM_ACTION
NUM_TILE_TYPE = Tile.NUM_TILE_TYPE  # 34
NUM_TILE_TYPE_WITH_RED = Tile.NUM_TILE_TYPE_WITH_RED  # 37

HAND_EMB_SIZE = 128
HISTORY_EMB_SIZE = 192
GLOBAL_EMB_SIZE = 64
FINAL_MLP_DIM = 256
TRANFORMER_MLP_DIM = 256
NUM_HAND_LAYER = 3
NUM_HISTORY_LAYER = 3

MAX_SHANTEN = 6.0
SCORE_OFFSET = 250.0
SCORE_SCALE = 1250.0
# ``round`` runs [0, round_limit]; round_limit is 8 for 'single'/'half' and 4 for
# 'east' (RedMahjong.__init__). The old value of 12 came from a stale docstring.
MAX_ROUND_VALUE = 8.0
MAX_HONBA = 10.0
MAX_KYOTAKU = 10.0
# Live wall at the start of a round: 136 - 14 dead wall - 13*4 dealt hands.
MAX_WALL_REMAINING = 70.0
MAX_WIND_VALUE = 3.0

# obs["discard_shanten"] is (34, 3) = [normal, seven pairs, thirteen orphans] in
# standard notation, with Shanten.NOT_IN_HAND (14) for tile types the hand does not
# hold. The three columns have different ranges, so they are scaled separately.
NOT_IN_HAND = Shanten.NOT_IN_HAND
DISCARD_SHANTEN_SCALE = jnp.array([8.0, 12.0, 13.0], dtype=jnp.float32)
# Hand tiles are red-aware ids [0-36]; 34/35/36 are the red fives, which map onto
# tile types 4/13/22 for anything indexed by tile type (shanten does not care about
# red, but the tile identity still has to line up).
RED_ID_TO_TILE_TYPE = jnp.array(list(range(NUM_TILE_TYPE)) + [4, 13, 22], dtype=jnp.int32)

# obs["river"] is (6, 4, 24) and obs["melds"] is (3, 4, 4), both [channel, seat, slot]
# with seat 0 == the observer.
RIVER_EMB_SIZE = 64
MELD_EMB_SIZE = 64
NUM_MELD_TYPES = 6  # none / pon / open kan / chi left / chi middle / chi right


class FeatureExtractor(nn.Module):
    @nn.compact
    def __call__(self, obs: Dict[str, jnp.ndarray]):
        hand = self._ensure_batch_dim(obs["hand"], base_ndim=1).astype(jnp.int32)
        action_history = self._ensure_batch_dim(obs["action_history"], base_ndim=2).astype(
            jnp.int32
        )
        shanten = self._ensure_batch_dim(obs["shanten_count"], base_ndim=0).astype(
            jnp.float32
        )
        furiten = self._ensure_batch_dim(obs["furiten"], base_ndim=0).astype(jnp.float32)
        scores = self._ensure_batch_dim(obs["scores"], base_ndim=1).astype(jnp.float32)
        round_number = self._ensure_batch_dim(obs["round"], base_ndim=0).astype(
            jnp.float32
        )
        honba = self._ensure_batch_dim(obs["honba"], base_ndim=0).astype(jnp.float32)
        kyotaku = self._ensure_batch_dim(obs["kyotaku"], base_ndim=0).astype(jnp.float32)
        if "round_wind" in obs:
            round_wind = self._ensure_batch_dim(obs["round_wind"], base_ndim=0).astype(
                jnp.float32
            )
        else:
            round_wind = self._ensure_batch_dim(
                obs["prevalent_wind"], base_ndim=0
            ).astype(jnp.float32)
        seat_wind = self._ensure_batch_dim(obs["seat_wind"], base_ndim=0).astype(
            jnp.float32
        )
        dora_indicators = self._ensure_batch_dim(
            obs["dora_indicators"], base_ndim=1
        ).astype(jnp.int32)
        # Fields added alongside the per-discard shanten feature. The guards below let
        # this network still BUILD on an older observation dict (e.g. a dataset
        # collected before these keys existed). They do NOT make old checkpoints
        # loadable: `last_draw` is read unconditionally and is new, and the table /
        # shanten branches add parameters, so the param tree differs from any
        # previously saved red_mahjong checkpoint either way. Retrain or re-run BC.
        last_draw = self._ensure_batch_dim(obs["last_draw"], base_ndim=0).astype(jnp.int32)
        has_discard_shanten = "discard_shanten" in obs
        if has_discard_shanten:
            discard_shanten = self._ensure_batch_dim(
                obs["discard_shanten"], base_ndim=2
            ).astype(jnp.float32)
            is_discard_node = self._ensure_batch_dim(
                obs["is_discard_node"], base_ndim=0
            ).astype(jnp.float32)
        has_call_context = "target" in obs
        if has_call_context:
            target = self._ensure_batch_dim(obs["target"], base_ndim=0).astype(jnp.int32)
            last_player = self._ensure_batch_dim(obs["last_player"], base_ndim=0).astype(
                jnp.int32
            )
        has_wall = "wall_remaining" in obs
        if has_wall:
            wall_remaining = self._ensure_batch_dim(
                obs["wall_remaining"], base_ndim=0
            ).astype(jnp.float32)
        has_table = "river" in obs
        has_tiles_seen = "tiles_seen" in obs
        if has_tiles_seen:
            tiles_seen = self._ensure_batch_dim(obs["tiles_seen"], base_ndim=1).astype(
                jnp.float32
            )
        if has_table:
            # (B, 6, 4, 24) and (B, 3, 4, 4), rows ordered (me, right, across, left).
            river = self._ensure_batch_dim(obs["river"], base_ndim=3).astype(jnp.int32)
            melds = self._ensure_batch_dim(obs["melds"], base_ndim=3).astype(jnp.int32)

        hand_emb = nn.Embed(
            Tile.NUM_TILE_TYPE_WITH_RED + 1,
            HAND_EMB_SIZE,
            embedding_init=orthogonal_init(),
        )(hand + 1)
        hand_mask = (hand >= 0).astype(jnp.float32)

        if has_discard_shanten:
            # Split "unavailable" out of the magnitude so the sentinel is never read
            # as a shanten value, then rescale each column by its own range.
            held = (discard_shanten < NOT_IN_HAND).astype(jnp.float32)  # (B, 34, 3)
            magnitude = jnp.where(held > 0, discard_shanten, 0.0) / DISCARD_SHANTEN_SCALE
            plane = jnp.concatenate([magnitude, held], axis=-1)  # (B, 34, 6)

            # Per-token: "what happens to my shanten if I discard THIS tile". The hand
            # encoder has no positional embedding, so a per-token attribute is exactly
            # the right shape for it.
            slot_type = RED_ID_TO_TILE_TYPE[jnp.clip(hand, 0, NUM_TILE_TYPE_WITH_RED - 1)]
            slot_plane = jnp.take_along_axis(plane, slot_type[..., None], axis=1)
            hand_emb = hand_emb + nn.Dense(
                HAND_EMB_SIZE, kernel_init=orthogonal_init()
            )(slot_plane)

        x_hand = hand_emb * hand_mask[..., None]
        for _ in range(NUM_HAND_LAYER):
            x_hand = TransformerBlock(
                HAND_EMB_SIZE, num_heads=4, mlp_dim=TRANFORMER_MLP_DIM
            )(x_hand, mask=hand_mask)
        token_count = jnp.maximum(hand_mask.sum(axis=1, keepdims=True), 1.0)
        hand_feature = (x_hand * hand_mask[..., None]).sum(axis=1) / token_count

        if action_history.shape[1] == MAX_HISTORY_LENGTH:
            players = action_history[:, :, 0]
            actions = action_history[:, :, 1]
            is_tsumogiri = action_history[:, :, 2]
        else:
            players = action_history[:, 0, :]
            actions = action_history[:, 1, :]
            is_tsumogiri = action_history[:, 2, :]

        hist_player_emb = nn.Embed(
            NUM_PLAYERS + 1, HISTORY_EMB_SIZE, embedding_init=orthogonal_init()
        )(players + 1)
        hist_action_emb = nn.Embed(
            NUM_ACTIONS + 1, HISTORY_EMB_SIZE, embedding_init=orthogonal_init()
        )(actions + 1)
        hist_is_tsumogiri_emb = nn.Embed(
            2 + 1, HISTORY_EMB_SIZE, embedding_init=orthogonal_init()
        )(is_tsumogiri + 1)
        positions = jnp.arange(MAX_HISTORY_LENGTH)[None, :]
        hist_pos_emb = nn.Embed(
            MAX_HISTORY_LENGTH, HISTORY_EMB_SIZE, embedding_init=orthogonal_init()
        )(positions)
        x_hist = hist_player_emb + hist_action_emb + hist_is_tsumogiri_emb + hist_pos_emb
        hist_mask = (actions >= 0).astype(jnp.float32)
        x_hist = x_hist * hist_mask[..., None]
        for _ in range(NUM_HISTORY_LAYER):
            x_hist = TransformerBlock(
                HISTORY_EMB_SIZE, num_heads=4, mlp_dim=TRANFORMER_MLP_DIM
            )(x_hist, mask=hist_mask)
        hist_token_count = jnp.maximum(hist_mask.sum(axis=1, keepdims=True), 1.0)
        history_feature = (x_hist * hist_mask[..., None]).sum(axis=1) / hist_token_count

        shanten_feature = (shanten / MAX_SHANTEN)[..., None]
        furiten_feature = furiten[..., None]
        score_feature = (scores + SCORE_OFFSET) / SCORE_SCALE
        round_feature = (round_number / MAX_ROUND_VALUE)[..., None]
        honba_feature = (honba / MAX_HONBA)[..., None]
        kyotaku_feature = (kyotaku / MAX_KYOTAKU)[..., None]
        round_wind_feature = (round_wind / MAX_WIND_VALUE)[..., None]
        seat_wind_feature = (seat_wind / MAX_WIND_VALUE)[..., None]
        scalar_parts = [
            score_feature,
            shanten_feature,
            furiten_feature,
            round_feature,
            honba_feature,
            kyotaku_feature,
            round_wind_feature,
            seat_wind_feature,
        ]
        if has_discard_shanten:
            # Tells the net whether discard_shanten answers "if I discard this"
            # (3n+2 hand) or is a floater map one discard further out (3n+1).
            scalar_parts.append(is_discard_node[..., None])
        if has_wall:
            # The round's clock. Normalized by the full live wall so 1.0 is a fresh
            # round and 0.0 is the exhaustive draw.
            scalar_parts.append((wall_remaining / MAX_WALL_REMAINING)[..., None])
        if has_call_context:
            # Relative seat of whoever acted last, one-hot over [none, me, right,
            # across, left]; a linear seat index would imply an ordering that is not
            # there.
            scalar_parts.append(
                jax.nn.one_hot(last_player + 1, NUM_PLAYERS + 1, dtype=jnp.float32)
            )
        global_scalar = jnp.concatenate(scalar_parts, axis=-1)

        dora_mask = (dora_indicators >= 0).astype(jnp.float32)
        dora_emb = nn.Embed(
            Tile.NUM_TILE_TYPE_WITH_RED + 1,
            HAND_EMB_SIZE,
            embedding_init=orthogonal_init(),
        )(dora_indicators + 1)
        dora_emb = dora_emb * dora_mask[..., None]
        dora_denom = jnp.maximum(dora_mask.sum(axis=1, keepdims=True), 1.0)
        dora_summary = dora_emb.sum(axis=1) / dora_denom
        dora_feat = nn.Dense(GLOBAL_EMB_SIZE, kernel_init=orthogonal_init())(dora_summary)

        # Tiles that name a specific card the decision is about. ``last_draw`` was in
        # the observation all along but the network never read it, so TSUMOGIRI (which
        # discards exactly that tile) had no grounding. ``target`` is the tile a
        # pending pon/chi/kan/ron is about.
        tile_emb = nn.Embed(
            Tile.NUM_TILE_TYPE_WITH_RED + 1,
            HAND_EMB_SIZE,
            embedding_init=orthogonal_init(),
        )
        context_tiles = [last_draw]
        if has_call_context:
            context_tiles.append(target)
        context_stack = jnp.stack(context_tiles, axis=1)  # (B, n)
        context_emb = tile_emb(context_stack + 1)
        context_emb = context_emb * (context_stack >= 0).astype(jnp.float32)[..., None]
        context_feat = nn.Dense(GLOBAL_EMB_SIZE, kernel_init=orthogonal_init())(
            context_emb.reshape((context_emb.shape[0], -1))
        )

        global_parts = [global_scalar, dora_feat, context_feat]
        if has_tiles_seen:
            # Dead-tile counts per tile type, /4. A spatial (34,) feature, so it goes
            # in as a plane rather than a scalar: "how many 5p are gone" is per-tile.
            global_parts.append(
                nn.Dense(GLOBAL_EMB_SIZE, kernel_init=orthogonal_init())(tiles_seen / 4.0)
            )
        if has_table:
            global_parts.append(TableEncoder()(river, melds))
        if has_discard_shanten:
            # Whole-plane view, tile-type indexed, so the net can also reason about
            # tiles it does not hold (kabe / what would help).
            global_parts.append(
                nn.Dense(GLOBAL_EMB_SIZE, kernel_init=orthogonal_init())(
                    plane.reshape((plane.shape[0], -1))
                )
            )
        global_in = jnp.concatenate(global_parts, axis=-1)
        global_out = nn.Dense(GLOBAL_EMB_SIZE, kernel_init=orthogonal_init())(global_in)
        global_out = nn.relu(global_out)
        global_out = nn.Dense(GLOBAL_EMB_SIZE, kernel_init=orthogonal_init())(global_out)
        return jnp.concatenate([hand_feature, history_feature, global_out], axis=-1)

    @staticmethod
    def _ensure_batch_dim(x: jnp.ndarray, *, base_ndim: int) -> jnp.ndarray:
        arr = jnp.asarray(x)
        if arr.ndim == base_ndim:
            return arr.reshape((1,) + arr.shape)
        return arr


class TableEncoder(nn.Module):
    """Encode the four discard piles and the four meld sets.

    ``river`` is (B, 6, 4, 24) and ``melds`` is (B, 3, 4, 4), both laid out
    [channel, seat, slot] with seat 0 == the observer. The seat axis is kept as an
    explicit dimension rather than pooled away: "who" is the whole point of a discard
    pile, and the observation has already rotated the seats into the observer's frame,
    so seat 1 always means the player to the right.
    """

    @nn.compact
    def __call__(self, river: jnp.ndarray, melds: jnp.ndarray) -> jnp.ndarray:
        tile, riichi, called_away, tsumogiri, src, meld_type = (river[:, i] for i in range(6))
        valid = (tile >= 0).astype(jnp.float32)  # (B, 4, 24)

        tile_emb = nn.Embed(
            NUM_TILE_TYPE_WITH_RED + 1, RIVER_EMB_SIZE, embedding_init=orthogonal_init()
        )(tile + 1)
        flags = jnp.stack(
            [
                riichi.astype(jnp.float32),
                called_away.astype(jnp.float32),
                tsumogiri.astype(jnp.float32),
            ],
            axis=-1,
        )
        # src and meld_type are categorical, not ordinal: a linear "3" is not three
        # times a "1". src is only meaningful where called_away is set.
        call_info = jnp.concatenate(
            [
                jax.nn.one_hot(src, NUM_PLAYERS, dtype=jnp.float32)
                * called_away.astype(jnp.float32)[..., None],
                jax.nn.one_hot(meld_type, NUM_MELD_TYPES, dtype=jnp.float32),
            ],
            axis=-1,
        )
        # Discard order matters (early vs late reads differently), and unlike the
        # action history this axis is short enough to keep positionally.
        turn_emb = nn.Embed(
            MAX_DISCARDS_PER_PLAYER, RIVER_EMB_SIZE, embedding_init=orthogonal_init()
        )(jnp.arange(MAX_DISCARDS_PER_PLAYER))
        x = tile_emb + turn_emb + nn.Dense(
            RIVER_EMB_SIZE, kernel_init=orthogonal_init()
        )(jnp.concatenate([flags, call_info], axis=-1))
        x = nn.relu(x) * valid[..., None]
        # Pool within each seat, keeping the seat axis.
        denom = jnp.maximum(valid.sum(axis=-1, keepdims=True), 1.0)
        river_per_seat = x.sum(axis=2) / denom  # (B, 4, RIVER_EMB_SIZE)

        m_action, m_tile, m_src = melds[:, 0], melds[:, 1], melds[:, 2]
        m_valid = (m_action >= 0).astype(jnp.float32)  # (B, 4, 4)
        m = nn.Embed(NUM_ACTIONS + 1, MELD_EMB_SIZE, embedding_init=orthogonal_init())(
            m_action + 1
        ) + nn.Embed(
            NUM_TILE_TYPE_WITH_RED + 1, MELD_EMB_SIZE, embedding_init=orthogonal_init()
        )(
            m_tile + 1
        ) + nn.Dense(
            MELD_EMB_SIZE, kernel_init=orthogonal_init()
        )(
            jax.nn.one_hot(m_src, NUM_PLAYERS, dtype=jnp.float32)
        )
        m = nn.relu(m) * m_valid[..., None]
        # Sum, not mean: three melds is a materially different hand from one, and a
        # mean would erase that.
        melds_per_seat = m.sum(axis=2)  # (B, 4, MELD_EMB_SIZE)

        table = jnp.concatenate([river_per_seat, melds_per_seat], axis=-1)
        table = nn.Dense(GLOBAL_EMB_SIZE, kernel_init=orthogonal_init())(table)
        table = nn.relu(table)
        return table.reshape((table.shape[0], -1))  # keep the seat axis, flattened


class ACNet(nn.Module):
    def setup(self):
        self.policy_extractor = FeatureExtractor()
        self.critic_extractor = FeatureExtractor()
        self.policy_mlp = nn.Sequential(
            [
                nn.Dense(FINAL_MLP_DIM, kernel_init=orthogonal_init()),
                nn.relu,
                nn.Dense(NUM_ACTIONS, kernel_init=orthogonal_init(0.01)),
            ]
        )
        self.value_critic_mlp = nn.Sequential(
            [
                nn.Dense(FINAL_MLP_DIM, kernel_init=orthogonal_init()),
                nn.relu,
                nn.Dense(1, kernel_init=orthogonal_init()),
            ]
        )

    def __call__(self, obs):
        return self.get_action_logits(obs), self.get_value(obs)

    def get_action_logits(self, obs):
        features = self.policy_extractor(obs)
        return self.policy_mlp(features)

    def get_value(self, obs):
        features = self.critic_extractor(obs)
        return self.value_critic_mlp(features).squeeze(-1)
