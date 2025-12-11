from typing import Dict, Any, Tuple
from functools import partial
from pathlib import Path
import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
import orbax.checkpoint as ocp
from mahjax.no_red_mahjong.action import Action
from mahjax.no_red_mahjong.tile import Tile
from typing import Literal

# Definitions
NUM_PLAYERS = 4
HAND_SIZE = 14
MAX_HISTORY_LENGTH = 200
NUM_ACTIONS = Action.NUM_ACTION

HAND_EMB_SIZE = 128
HISTORY_EMB_SIZE = 192
GLOBAL_EMB_SIZE = 64
FEATURE_SIZE = HAND_EMB_SIZE + HISTORY_EMB_SIZE + GLOBAL_EMB_SIZE

TRANFORMER_MLP_DIM = 256
FINAL_MLP_DIM = 256
NUM_HAND_LAYER = 2
NUM_HISTORY_LAYER = 2

MAX_SHANTEN = 6.0
SCORE_OFFSET = 250.0
SCORE_SCALE = 1250.0
MAX_ROUND_VALUE = 12.0
MAX_HONBA = 10.0
MAX_KYOTAKU = 10.0
MAX_WIND_VALUE = 3.0

NEG = -1e9

# Initialization function
def orthogonal_init(scale: float = jnp.sqrt(2.0)):
    return nn.initializers.orthogonal(scale)

class TransformerBlock(nn.Module):
    features: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x, mask=None):
        # Attention Block (Pre-Norm)
        y = nn.LayerNorm()(x)
        
        # Mask shape adjustment for MultiHeadDotProductAttention
        # (Batch, SeqLen) -> (Batch, 1, 1, SeqLen)
        if mask is not None and mask.ndim == 2:
            mask = mask[:, None, None, :]
        
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=orthogonal_init(),
            deterministic=True
        )(y, mask=mask)
        x = x + y

        # MLP Block (Pre-Norm)
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim, kernel_init=orthogonal_init())(y)
        y = nn.relu(y)
        y = nn.Dense(self.features, kernel_init=orthogonal_init())(y)
        x = x + y
        
        return x

class FeatureExtractor(nn.Module):
    @nn.compact
    def __call__(self, obs: Dict[str, jnp.ndarray]):
        # --- Prepare inputs ---
        hand = self._ensure_batch_dim(obs["hand"], base_ndim=1).astype(jnp.int32)
        action_history = self._ensure_batch_dim(obs["action_history"], base_ndim=2).astype(jnp.int32)
        
        # Debug: check the shape (displayed at compile time)
        # print(f"DEBUG: action_history shape = {action_history.shape}")

        shanten = self._ensure_batch_dim(obs["shanten_count"], base_ndim=0).astype(jnp.float32)
        furiten = self._ensure_batch_dim(obs["furiten"], base_ndim=0).astype(jnp.float32)
        scores = self._ensure_batch_dim(obs["scores"], base_ndim=1).astype(jnp.float32)
        round_number = self._ensure_batch_dim(obs["round"], base_ndim=0).astype(jnp.float32)
        honba = self._ensure_batch_dim(obs["honba"], base_ndim=0).astype(jnp.float32)
        kyotaku = self._ensure_batch_dim(obs["kyotaku"], base_ndim=0).astype(jnp.float32)
        
        if "round_wind" in obs:
            round_wind = self._ensure_batch_dim(obs["round_wind"], base_ndim=0).astype(jnp.float32)
        else:
            round_wind = self._ensure_batch_dim(obs["prevalent_wind"], base_ndim=0).astype(jnp.float32)
            
        seat_wind = self._ensure_batch_dim(obs["seat_wind"], base_ndim=0).astype(jnp.float32)
        dora_indicators = self._ensure_batch_dim(obs["dora_indicators"], base_ndim=1).astype(jnp.int32)

        # --- Encode Hand ---
        # hand: (Batch, 14)
        hand_emb = nn.Embed(Tile.NUM_TILE_TYPE + 1, HAND_EMB_SIZE, embedding_init=orthogonal_init())(hand + 1)
        hand_mask = (hand >= 0).astype(jnp.float32) # (Batch, 14)
        
        x_hand = hand_emb * hand_mask[..., None]
        for _ in range(NUM_HAND_LAYER):
            x_hand = TransformerBlock(HAND_EMB_SIZE, num_heads=4, mlp_dim=TRANFORMER_MLP_DIM)(x_hand, mask=hand_mask)
            
        token_count = jnp.maximum(hand_mask.sum(axis=1, keepdims=True), 1.0)
        hand_feature = (x_hand * hand_mask[..., None]).sum(axis=1) / token_count 

        # --- Encode History ---
        # action_history: (Batch, 200, 2) 想定
        # [Important correction point: check if shape[1] is 200 (Time) and slice accordingly]
        # Check if shape[1] is 200 (Time) and slice accordingly
        # This allows it to work with both (Batch, 200, 2) and (Batch, 2, 200)
        
        if action_history.shape[1] == MAX_HISTORY_LENGTH:
             # (Batch, 200, 2) -> slice by feature dimension (2)
             players = action_history[:, :, 0]
             actions = action_history[:, :, 1]
        elif action_history.shape[2] == MAX_HISTORY_LENGTH:
             # (Batch, 2, 200) -> slice by feature dimension (1)
             players = action_history[:, 0, :]
             actions = action_history[:, 1, :]
        else:
             # fallback (previous correction version)
             players = action_history[:, :, 0]
             actions = action_history[:, :, 1]
        
        hist_player_emb = nn.Embed(NUM_PLAYERS + 1, HISTORY_EMB_SIZE, embedding_init=orthogonal_init())(players + 1)
        hist_action_emb = nn.Embed(NUM_ACTIONS + 1, HISTORY_EMB_SIZE, embedding_init=orthogonal_init())(actions + 1)
        
        # Positional Embedding
        positions = jnp.arange(MAX_HISTORY_LENGTH)[None, :] # (1, 200)
        hist_pos_emb = nn.Embed(MAX_HISTORY_LENGTH, HISTORY_EMB_SIZE, embedding_init=orthogonal_init())(positions)
        
        # check if the shape is consistent
        # players: (Batch, 200), hist_player_emb: (Batch, 200, 192)
        # positions: (1, 200), hist_pos_emb: (1, 200, 192)
        x_hist = hist_player_emb + hist_action_emb + hist_pos_emb
        
        hist_mask = (actions >= 0).astype(jnp.float32) # (Batch, 200)
        
        x_hist = x_hist * hist_mask[..., None]
        for _ in range(NUM_HISTORY_LAYER):
            x_hist = TransformerBlock(HISTORY_EMB_SIZE, num_heads=4, mlp_dim=TRANFORMER_MLP_DIM)(x_hist, mask=hist_mask)
            
        hist_token_count = jnp.maximum(hist_mask.sum(axis=1, keepdims=True), 1.0)
        history_feature = (x_hist * hist_mask[..., None]).sum(axis=1) / hist_token_count

        # --- Encode Global ---
        shanten_feature = (shanten / MAX_SHANTEN)[..., None]
        furiten_feature = furiten[..., None]
        score_feature = (scores + SCORE_OFFSET) / SCORE_SCALE
        round_feature = (round_number / MAX_ROUND_VALUE)[..., None]
        honba_feature = (honba / MAX_HONBA)[..., None]
        kyotaku_feature = (kyotaku / MAX_KYOTAKU)[..., None]
        round_wind_feature = (round_wind / MAX_WIND_VALUE)[..., None]
        seat_wind_feature = (seat_wind / MAX_WIND_VALUE)[..., None]

        global_scalar = jnp.concatenate([
            score_feature, shanten_feature, furiten_feature,
            round_feature, honba_feature, kyotaku_feature,
            round_wind_feature, seat_wind_feature
        ], axis=-1)

        # Dora embedding
        dora_mask = (dora_indicators >= 0).astype(jnp.float32)
        dora_emb = nn.Embed(Tile.NUM_TILE_TYPE + 1, HAND_EMB_SIZE, embedding_init=orthogonal_init())(dora_indicators + 1)
        dora_emb = dora_emb * dora_mask[..., None]
        dora_denom = jnp.maximum(dora_mask.sum(axis=1, keepdims=True), 1.0)
        dora_summary = dora_emb.sum(axis=1) / dora_denom
        
        dora_feat = nn.Dense(GLOBAL_EMB_SIZE, kernel_init=orthogonal_init())(dora_summary)
        
        global_in = jnp.concatenate([global_scalar, dora_feat], axis=-1)
        
        global_out = nn.Dense(GLOBAL_EMB_SIZE, kernel_init=orthogonal_init())(global_in)
        global_out = nn.relu(global_out)
        global_out = nn.Dense(GLOBAL_EMB_SIZE, kernel_init=orthogonal_init())(global_out)

        # --- Fuse All ---
        return jnp.concatenate([hand_feature, history_feature, global_out], axis=-1)

    @staticmethod
    def _ensure_batch_dim(x: jnp.ndarray, *, base_ndim: int) -> jnp.ndarray:
        arr = jnp.asarray(x)
        if arr.ndim == base_ndim:
            return arr.reshape((1,) + arr.shape)
        return arr

        
# --- Full Network (Policy + Critic) ---
class ACNet(nn.Module):
    """Combines Policy and Critic Feature Extractors and Heads."""
    
    def setup(self):
        self.policy_extractor = FeatureExtractor()
        self.critic_extractor = FeatureExtractor()
        
        self.policy_mlp = nn.Sequential([
            nn.Dense(FINAL_MLP_DIM, kernel_init=orthogonal_init()),
            nn.relu,
            nn.Dense(NUM_ACTIONS, kernel_init=orthogonal_init(0.01)) # policy head (small init)
        ])
        
        self.value_critic_mlp = nn.Sequential([
            nn.Dense(FINAL_MLP_DIM, kernel_init=orthogonal_init()),
            nn.relu,
            nn.Dense(1, kernel_init=orthogonal_init())
        ])

    def __call__(self, obs):
        return self.get_action_logits(obs), self.get_value(obs)

    def get_action_logits(self, obs):
        features = self.policy_extractor(obs)
        return self.policy_mlp(features)

    def get_value(self, obs):
        features = self.critic_extractor(obs)
        return self.value_critic_mlp(features).squeeze(-1)


