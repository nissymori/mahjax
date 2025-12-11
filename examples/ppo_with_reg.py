"""
PPO with Regularization trainer for MahJax.
"""

import sys
import time
from typing import Dict, Literal, NamedTuple, Any, Optional
import pickle

import jax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax
from omegaconf import OmegaConf
from pydantic import BaseModel
import distrax
import wandb

from mahjax.wrappers.auto_reset_wrapper import auto_reset
from mahjax.no_red_mahjong.env import Mahjong

from network import ACNet
from bc import visualize_game

# Constants
MAX_REWARD = 320.0  # Normalization factor for reward
NEG = -1e9  # Negative infinity for masking
Observation = Dict[str, jnp.ndarray]  # Observation type


class PPOWithRegArgs(BaseModel):
    algo: str = "ppo_with_reg"
    # Environment
    env_name: str = "mahjax/no_red_mahjong"
    one_round: bool = True
    seed: int = 0
    # Training setup
    num_envs: int = 32
    num_steps: int = 128
    total_timesteps: int = 4e7
    update_epochs: int = 4
    minibatch_size: int = 1024
    # PPO hyperparameters
    gamma: float = 1.0
    gae_lambda: float = 0.95
    lr: float = 3e-4
    ent_coef: float = 0.01
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    # Magnet hyperparameters
    mag_coef: float = 0.2
    mag_divergence_type: Literal["kl", "l2"] = "kl"
    pretrained_path: Optional[str] = None
    # For logging and saving
    wandb_project: str = "mahjax-ppo-with-reg"
    save_model: bool = True
    do_eval: bool = True
    eval_interval: int = 50
    eval_num_envs: int = 500
    viz_max_steps: int = 1000
    class args: extra = "forbid"

args = PPOWithRegArgs(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args, file=sys.stderr)

BASE_ENV = Mahjong(one_round=args.one_round, observe_type="dict")
step_fn = auto_reset(BASE_ENV.step, BASE_ENV.init)
NUM_PLAYERS, NUM_UPDATES = BASE_ENV.num_players, int(args.total_timesteps // (args.num_envs * args.num_steps))
BATCH_SIZE = args.num_envs * args.num_steps
if BATCH_SIZE % args.minibatch_size != 0: raise ValueError("minibatch_size error")
NUM_MINIBATCHES = BATCH_SIZE // args.minibatch_size


class Transition(NamedTuple):
    is_new_episode: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    observation: Observation
    action_mask: jnp.ndarray
    current_player: jnp.ndarray

def masked_mean(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    return (x * mask.astype(jnp.float32)).sum() / jnp.maximum(mask.astype(jnp.float32).sum(), 1.0)

def collect_rollout(network: nn.Module, params, env_state, key):
    def step_fn_scan(carry, _):
        state, rng = carry
        rng, action_key, env_key = jax.random.split(rng, 3)
        # PREPARE observation and action mask
        observation = BASE_ENV.observe(state)
        action_mask = state.legal_action_mask.astype(jnp.bool_)
        current_player = jnp.asarray(state.current_player, dtype=jnp.int32)
        # Check if the episode is new
        is_new_episode = jnp.asarray(state.terminated | state.truncated, dtype=jnp.bool_)

        # STEP ENV
        logits, value = network.apply(params, observation)
        logits = jnp.where(action_mask, logits, NEG)
        dist = distrax.Categorical(logits=logits)
        action, log_prob = dist.sample_and_log_prob(seed=action_key)
        next_state = step_fn(state, action, env_key)
        # Reward comes from the transition to the next state
        reward = jnp.asarray(next_state.rewards, dtype=jnp.float32) / MAX_REWARD  # (B, 4)

        transition = Transition(
            is_new_episode=is_new_episode, action=action, value=jnp.squeeze(value), reward=reward,
            log_prob=log_prob, observation=observation, action_mask=action_mask, current_player=current_player
        )
        return (next_state, rng), transition

    batched_rollout = jax.vmap(lambda c, x: lax.scan(step_fn_scan, c, None, length=args.num_steps))
    keys = jax.random.split(key, args.num_envs)
    (env_state, _), transitions = batched_rollout((env_state, keys), None)
    return env_state, transitions

def calculate_gae(transitions: Transition):
    """
    Calculate Generalized Advantage Estimation (GAE)
    """
    def single_env_gae(transition: Transition):
        def scan_fn(carry, t: Transition):
            gae, next_value, reward_accum, has_next_value, is_new_episode, next_valid_mask = carry
            player, reward, value, done = t.current_player, t.reward, t.value, t.is_new_episode
            # Reset accumulators on episode boundary
            gae = jnp.where(done, 0, gae)
            reward_accum = jnp.where(done, 0, reward_accum)
            has_next_value = jnp.where(done, 0, has_next_value)
            next_value = jnp.where(done, 0, next_value)
            
            reward_accum = reward_accum + reward
            player_reward = reward_accum[player]
            reward_accum = reward_accum.at[player].set(0.0)
            
            td_error = player_reward + args.gamma * next_value[player] - value
            new_gae = td_error + args.gamma * args.gae_lambda * gae[player]
            gae = gae.at[player].set(new_gae)
            
            is_valid = has_next_value[player] | done | next_valid_mask[player]
            advantage = jnp.where(is_valid, new_gae, 0.0)
            target = jnp.where(is_valid, advantage + value, value)
            
            new_carry = (
                gae, next_value.at[player].set(value), reward_accum,
                has_next_value.at[player].set(True), done, next_valid_mask.at[player].set(is_valid) | done
            )
            output = (
                jnp.zeros(NUM_PLAYERS).at[player].set(advantage),
                jnp.zeros(NUM_PLAYERS).at[player].set(target),
                jnp.zeros(NUM_PLAYERS, dtype=bool).at[player].set(is_valid)
            )
            return new_carry, output

        init = (jnp.zeros(NUM_PLAYERS), jnp.zeros(NUM_PLAYERS), jnp.zeros(NUM_PLAYERS),
                jnp.zeros(NUM_PLAYERS, dtype=bool), False, jnp.zeros(NUM_PLAYERS, dtype=bool))
        _, (adv, targets, valid_mask) = lax.scan(scan_fn, init, transition, reverse=True)
        return adv, targets, valid_mask
    return jax.vmap(single_env_gae)(transitions)


def make_update_fn(network: nn.Module, magnet_params: Optional[Dict[str, Any]] = None):
    def update_step(runner_state, update_idx):
        train_state, env_state, rng = runner_state
        rng, key = jax.random.split(rng)
        
        env_state, transitions = collect_rollout(network, train_state.params, env_state, key)
        advantages, targets, valid_mask = calculate_gae(transitions)
        
        # Flatten (B, T, ...) -> (B*T, ...)
        flat_transitions = jax.tree.map(lambda x: x.reshape((BATCH_SIZE,) + x.shape[2:]), transitions)
        advantages = advantages.reshape((BATCH_SIZE, NUM_PLAYERS))
        targets = targets.reshape((BATCH_SIZE, NUM_PLAYERS))
        valid_mask = valid_mask.reshape((BATCH_SIZE, NUM_PLAYERS))

        mask_float = valid_mask.astype(jnp.float32)
        # Normalize Advantage
        advantages = (advantages - masked_mean(advantages, mask_float)) / (jnp.sqrt(masked_mean((advantages - masked_mean(advantages, mask_float))**2, mask_float)) + 1e-8)

        def train_epoch(epoch_carry, _):
            train_state_inner, rng_inner = epoch_carry
            rng_inner, perm_key = jax.random.split(rng_inner)
            permutation = jax.random.permutation(perm_key, BATCH_SIZE)

            batch_shuffled = (
                jax.tree.map(lambda x: x[permutation], flat_transitions),
                advantages[permutation], targets[permutation], valid_mask[permutation]
            )  # Shuffled batch
            minibatches = jax.tree.map(lambda x: x.reshape((NUM_MINIBATCHES, args.minibatch_size) + x.shape[1:]), batch_shuffled)

            def train_minibatch(t_state, minibatch):
                transition_mb: Transition; adv_mb, target_mb, mask_mb = minibatch[1], minibatch[2], minibatch[3]
                transition_mb = minibatch[0]

                def loss_fn(params):
                    # ACTOR LOSS (Policy Gradient Loss)
                    logits, values = network.apply(params, transition_mb.observation)
                    logits = jnp.where(transition_mb.action_mask, logits, NEG)
                    dists = distrax.Categorical(logits=logits)
                    log_ratio = dists.log_prob(transition_mb.action) - transition_mb.log_prob
                    ratio = jnp.exp(log_ratio)[..., None]

                    ppo_loss = -masked_mean(jnp.minimum(ratio * adv_mb, jnp.clip(ratio, 1-args.clip_eps, 1+args.clip_eps) * adv_mb), mask_mb)

                    # REGULARIZATION LOSS
                    mag_kl = 0.0
                    if magnet_params is not None:
                        mag_logits, _ = network.apply(magnet_params, transition_mb.observation)
                        mag_dists = distrax.Categorical(logits=jnp.where(transition_mb.action_mask, mag_logits, NEG))
                        if args.mag_divergence_type == "kl":
                            vals = dists.kl_divergence(mag_dists)
                        else:
                            vals = 0.5 * jnp.sum((dists.probs - mag_dists.probs)**2, axis=-1)
                        mag_kl = masked_mean(vals[..., None], mask_mb)

                    entropy = masked_mean(dists.entropy()[..., None], mask_mb)
                    loss_actor = ppo_loss - args.ent_coef * entropy + args.mag_coef * mag_kl

                    # CRITIC LOSS
                    value_clipped = transition_mb.value[..., None] + jnp.clip(values[..., None] - transition_mb.value[..., None], -args.clip_eps, args.clip_eps)
                    loss_critic = 0.5 * args.vf_coef * masked_mean(jnp.maximum((values[..., None] - target_mb)**2, (value_clipped - target_mb)**2), mask_mb)

                    # FOR LOGGING
                    approx_kl = masked_mean((ratio - 1.0) - log_ratio[..., None], mask_mb)
                    clip_frac = masked_mean((jnp.abs(ratio - 1.0) > args.clip_eps).astype(jnp.float32), mask_mb)
                    explained_var = jnp.maximum(1 - masked_mean((target_mb - values[..., None])**2, mask_mb) / (masked_mean((target_mb - masked_mean(target_mb, mask_mb))**2, mask_mb)+1e-8), 0)

                    return loss_actor + loss_critic, {
                        "total_loss": loss_actor + loss_critic, "actor_loss": loss_actor, "critic_loss": loss_critic,
                        "entropy": entropy, "approx_kl": approx_kl, "mag_kl": mag_kl, "clip_frac": clip_frac, "explained_var": explained_var
                    }

                grads, metrics = jax.grad(loss_fn, has_aux=True)(t_state.params)
                return t_state.apply_gradients(grads=grads), metrics

            new_train_state, metrics = lax.scan(train_minibatch, train_state_inner, minibatches)
            return (new_train_state, rng_inner), jax.tree.map(jnp.mean, metrics)

        (train_state, rng), metrics = lax.scan(train_epoch, (train_state, rng), None, length=args.update_epochs)
        
        stats = {"avg_episode_len": jnp.mean(transitions.is_new_episode.astype(jnp.float32)), "avg_reward": jnp.mean(transitions.reward[..., 0])}
        return (train_state, env_state, rng), (jax.tree.map(jnp.mean, metrics), stats)
    return update_step


def make_evaluator(network: nn.Module, num_eval_envs, baseline_params):
    def evaluate(params, key):
        def play_episode(state, seat_policy_ids, player_params, opponent_type, key_episode):
            def body_fn(carry, _):
                current_state, rng = carry
                rng, agent_key, opp_key, step_key = jax.random.split(rng, 4)
                # AGENT ACTION (Policy)
                logits, _ = network.apply(player_params, BASE_ENV.observe(current_state))
                agent_action = jnp.argmax(jnp.where(current_state.legal_action_mask, logits, NEG))
                # OPPOENT ACTION (Baseline)
                baseline_logits, _ = network.apply(baseline_params, BASE_ENV.observe(current_state))
                baseline_action = jnp.argmax(jnp.where(current_state.legal_action_mask, baseline_logits, NEG))
                rand_logits = jnp.where(current_state.legal_action_mask, 0.0, NEG)
                random_action = jax.random.categorical(opp_key, rand_logits)

                final_action = jnp.where(seat_policy_ids[current_state.current_player] == 0, agent_action, jnp.where(opponent_type == 0, random_action, baseline_action))
                next_state = lax.cond(current_state.terminated | current_state.truncated, lambda x: x, lambda x: BASE_ENV.step(x, final_action, step_key), current_state)
                return (next_state, rng), None
            (final_state, _), _ = lax.scan(body_fn, (state, key_episode), None, length=200)
            return final_state

        def evaluate_vs_opponent(key_run, opponent_type):
            key_run, key_init, key_assign, key_play = jax.random.split(key_run, 4)
            seats = jnp.arange(NUM_PLAYERS)[None, :]
            solo_seat_idx = jax.random.randint(key_assign, (num_eval_envs,), 0, NUM_PLAYERS)
            seat_policy_ids = jnp.where(seats == solo_seat_idx[:, None], 0, 1) # 0=Agent
            
            final_states = jax.vmap(play_episode, (0, 0, None, None, 0))(
                jax.vmap(BASE_ENV.init)(jax.random.split(key_init, num_eval_envs)), seat_policy_ids, params, opponent_type, jax.random.split(key_play, num_eval_envs)
            )
            
            is_agent = (seat_policy_ids == 0)
            scores = final_states._score
            agent_scores = (scores * is_agent).sum(axis=1)
            opponent_scores_avg = (scores * (~is_agent)).sum(axis=1) / 3.0
            return {
                "win_rate": (agent_scores > (scores * (~is_agent)).max(axis=1)).mean(),
                "avg_margin": (agent_scores - opponent_scores_avg).mean(),
                "agent_score": agent_scores.mean(),
                "opponent_score": opponent_scores_avg.mean(),
                "avg_rank": (1 + (scores > agent_scores[:, None]).sum(axis=1) + 0.5 * ((scores == agent_scores[:, None]).sum(axis=1) - 1)).mean(),
                "hora_rate": ((final_states._has_won & is_agent).any(axis=1)).mean(),
                "riichi_rate": ((final_states._riichi & is_agent).any(axis=1)).mean(),
                "meld_rate": ((final_states._n_meld > 0) & is_agent).any(axis=1).mean()
            }

        key_ret, key_rand, key_base = jax.random.split(key, 3)
        log = {f"vs_rand/{k}": v for k, v in evaluate_vs_opponent(key_rand, 0).items()}
        log.update({f"vs_baseline/{k}": v for k, v in evaluate_vs_opponent(key_base, 1).items()})
        return key_ret, log
    return evaluate


def train(rng_key):
    rng, key_net, key_reset = jax.random.split(rng_key, 3)
    # Network Initialization
    network = ACNet()
    dummy_obs = BASE_ENV.observe(BASE_ENV.init(jax.random.PRNGKey(0)))
    params = network.init(key_net, dummy_obs)
    # Initialize train state
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optax.adamw(args.lr, eps=1e-5)
    )
    # Load baseline parameters
    if args.pretrained_path:
        print(f"Loading anchor: {args.pretrained_path}", flush=True)
        with open(args.pretrained_path, "rb") as f: magnet_params = pickle.load(f)
        if not isinstance(magnet_params, dict): magnet_params = {"params": magnet_params}
    else:
        print("Using random anchor.", flush=True); magnet_params = params
    # Initialize environment states
    env_state = jax.vmap(BASE_ENV.init)(jax.random.split(key_reset, args.num_envs))
    # Runner state: (train_state, magnet_params, env_state, rng)
    runner_state = (train_state, magnet_params, env_state, rng)
    # JIT functions
    update_step_fn = jax.jit(make_update_fn(network, magnet_params))
    evaluator_fn = jax.jit(make_evaluator(network, args.eval_num_envs, magnet_params))
    steps, start_time = 0, time.time()

    def eval_and_log(rng, steps, update_idx, params):
        rng, eval_logs = evaluator_fn(params, rng)
        eval_logs = {k: float(v) for k, v in eval_logs.items()}
        wandb.log({"steps": steps, "update": update_idx, **eval_logs}); print({"steps": steps, **eval_logs}, flush=True)
        return rng

    if args.do_eval:
        runner_state = (*runner_state[:-1], eval_and_log(runner_state[-1], steps, 0, runner_state[0].params))

    for i in range(NUM_UPDATES):
        runner_state, (loss_metrics, stats) = update_step_fn(runner_state, jnp.array(i, dtype=jnp.int32))
        steps += BATCH_SIZE
        inv_len = stats["avg_episode_len"]
        wandb.log({
            "steps": steps, "update": i + 1,
            "train/loss_total": float(loss_metrics["total_loss"]),
            "train/loss_actor": float(loss_metrics["actor_loss"]),
            "train/loss_critic": float(loss_metrics["critic_loss"]),
            "train/entropy": float(loss_metrics["entropy"]),
            "train/approx_kl": float(loss_metrics["approx_kl"]),
            "train/mag_kl": float(loss_metrics["mag_kl"]),
            "train/clip_frac": float(loss_metrics["clip_frac"]),
            "train/explained_var": float(loss_metrics["explained_var"]),
            "train/avg_reward": float(stats["avg_reward"]),
            "train/avg_eps_len": float(1.0/inv_len) if inv_len > 0 else float('nan'),
            "train/avg_return": float(stats["avg_reward"]/inv_len) if inv_len > 0 else float('nan')
        })

        if args.do_eval and ((i + 1) % args.eval_interval == 0 or i + 1 == NUM_UPDATES):
            runner_state = (*runner_state[:-1], eval_and_log(runner_state[-1], steps, i + 1, runner_state[0].params))

    wandb.log({"train_time": time.time() - start_time, "steps": steps})
    return runner_state[0]

if __name__ == "__main__":
    wandb.init(project=args.wandb_project, config=args.dict())
    final_state = train(jax.random.PRNGKey(args.seed))
    if args.save_model:
        with open(f"{args.env_name}-seed={args.seed}.ckpt", "wb") as f: pickle.dump(final_state.params, f)
    visualize_game(args, final_state)