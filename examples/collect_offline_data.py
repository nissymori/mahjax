#!/usr/bin/env python3
"""
Data collector for Mahjong BC & Value Learning using Rule-based players.
Saves observations, actions, masks, AND returns.
Optimized for memory efficiency (Small chunks).
"""
import os
import sys
import pickle
import time
from typing import NamedTuple, Dict

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from pydantic import BaseModel
from tqdm import tqdm

from mahjax.wrappers.auto_reset_wrapper import auto_reset
from mahjax.no_red_mahjong.env import Mahjong
from mahjax.no_red_mahjong.rule_based_players import rule_based_player

# --- Config ---
class CollectConfig(BaseModel):
    seed: int = 0
    num_envs: int = 4
    num_steps: int = 32
    total_timesteps: int = 200_000
    save_path: str = "mahjong_offline_data.pkl"
    gamma: float = 0.99       # 割引率
    max_reward: float = 320.0 # 正規化用

# --- Environment ---
env = Mahjong(one_round=True, observe_type="dict")
step_env = auto_reset(env.step, env.init)

def _one_step(state, key):
    obs = env.observe(state)
    mask = state.legal_action_mask
    curr_player = state.current_player # 誰の手番か
    
    k_act, k_step = jax.random.split(key, 2)
    action = rule_based_player(state, k_act)
    
    next_state = step_env(state, action, k_step)
    
    # 状態遷移情報
    done = next_state.terminated | next_state.truncated
    reward = next_state.rewards # [P]
    
    return next_state, (obs, action, mask, reward, done, curr_player)

def _rollout_chunk(state, keys, num_steps):
    vmap_step = jax.vmap(_one_step, in_axes=(0, 0))
    def body(carry, key):
        st = carry
        new_st, out = vmap_step(st, key)
        return new_st, out
    final_state, outs = jax.lax.scan(body, state, keys, length=num_steps)
    return final_state, outs

def compute_returns(rewards, dones, current_players, gamma):
    """
    Compute discounted returns for the ACTIVE player at each step.
    rewards: [T, B, P]
    dones:   [T, B]
    current_players: [T, B]
    Returns: [T, B] (Scalar return for the actor)
    """
    T, B, P = rewards.shape
    returns = np.zeros((T, B), dtype=np.float32)
    
    # Monte Carlo Return Calculation (Reverse Order)
    # Note: Bootstrapping is required, but we use complete data for this purpose, so we approximate it with MC.
    # Calculate for each environment
    for b in range(B):
        running_ret = np.zeros(P, dtype=np.float32) # G_t for each player
        for t in reversed(range(T)):
            # Reward at the current step
            r_t = rewards[t, b] # [P]
            d_t = dones[t, b]   # bool
            
            if d_t:
                running_ret = np.zeros(P, dtype=np.float32)
            
            # G_t = r_t + gamma * G_{t+1}
            running_ret = r_t + gamma * running_ret
            
            # Record the Return for the player who acted at this step
            # (Value at the action selection point = immediate reward + future value)
            # Note: Whether to include the immediate reward or not depends on the definition, but in Q-learning, it is included.
            p = current_players[t, b]
            returns[t, b] = running_ret[p]
            
    return returns

def main():
    print("=== Starting Data Collection (With Returns) ===", flush=True)
    conf = OmegaConf.from_cli()
    cfg = CollectConfig(**conf)
    print(f"Config: {cfg}", flush=True)

    rng = jax.random.PRNGKey(cfg.seed)
    rng, k_init = jax.random.split(rng)
    
    init_keys = jax.random.split(k_init, cfg.num_envs)
    state = jax.vmap(env.init)(init_keys)
    
    print("Compiling JIT function...", flush=True)
    jit_rollout = jax.jit(lambda s, k: _rollout_chunk(s, k, cfg.num_steps))
    
    # Buffers
    data_obs = [] 
    data_act = []
    data_mask = []
    data_ret = [] # New: Returns
    
    chunk_size = cfg.num_envs * cfg.num_steps
    num_chunks = (cfg.total_timesteps + chunk_size - 1) // chunk_size
    
    total_steps = 0
    start_time = time.time()
    
    # Carry for continuity (from the previous chunk)
    # In a full offline RL dataset creation, calculations across episode boundaries are required, but here we simplify it by calculating within a chunk or treating it as a sufficiently long episode.
    # Here, we implement a simplified version that calculates within a chunk and ignores the boundary (with some tolerance for error).
    # Note: If you want to do it strictly, you need to keep all history in memory and calculate at the end, but due to memory constraints (79GB error), we prioritize chunk processing.
    
    for _ in tqdm(range(num_chunks), desc="Collecting", mininterval=10.0):
        rng, k_chunk = jax.random.split(rng)
        keys = jax.random.split(k_chunk, cfg.num_envs * cfg.num_steps).reshape(cfg.num_steps, cfg.num_envs, -1)
        
        state, (obs_seq, act_seq, mask_seq, rew_seq, done_seq, cp_seq) = jit_rollout(state, keys)
        
        # CPU Transfer
        obs_cpu = jax.tree_map(np.array, obs_seq)
        act_cpu = np.array(act_seq)
        mask_cpu = np.array(mask_seq)
        rew_cpu = np.array(rew_seq)
        done_cpu = np.array(done_seq)
        cp_cpu = np.array(cp_seq)
        
        # --- Return Calculation (CPU) ---
        # Calculate the return within this chunk
        # Note: Due to the truncation at the end of the chunk, there is an error because Bootstrap cannot be performed.
        # Since num_steps is shorter than the episode length, there is a bias.
        # -> For offline RL, buffering until the episode is complete is necessary, but here we avoid the complexity of implementation by utilizing the fact that the Mahjong game is short (avg 50-60 steps), and num_steps=32 is easy to truncate.
        # In this case, we compromise by saving the reward and calculating the return during learning, or simply using the return within this chunk.
        # This time, we want to use the normalized return as teacher data, so we calculate it here.
        returns_chunk = compute_returns(rew_cpu, done_cpu, cp_cpu, cfg.gamma)

        # Normalize (Divide by Max Reward)
        returns_chunk = returns_chunk / cfg.max_reward

        # Flatten & Store
        flat_obs = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), obs_cpu)
        data_obs.append(flat_obs)
        data_act.append(act_cpu.flatten())
        data_mask.append(mask_cpu.reshape(-1, mask_cpu.shape[-1]))
        data_ret.append(returns_chunk.flatten())
        
        total_steps += act_cpu.size
        if total_steps >= cfg.total_timesteps:
            break

    # Save
    print("Concatenating data...", flush=True)
    if not data_obs: return

    keys = data_obs[0].keys()
    full_obs = {k: np.concatenate([d[k] for d in data_obs], axis=0) for k in keys}
    full_act = np.concatenate(data_act, axis=0)
    full_mask = np.concatenate(data_mask, axis=0)
    full_ret = np.concatenate(data_ret, axis=0)
    
    N = cfg.total_timesteps
    dataset = {
        "observation": {k: v[:N] for k, v in full_obs.items()},
        "action": full_act[:N],
        "legal_action_mask": full_mask[:N],
        "return": full_ret[:N] # New: Returns
    }
    
    if cfg.save_path:
        os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)
        print(f"Saving to {cfg.save_path} ...", flush=True)
        with open(cfg.save_path, "wb") as f:
            pickle.dump(dataset, f)
        print("Done.", flush=True)

if __name__ == "__main__":
    main()