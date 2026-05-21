"""Evaluation utilities for MahJax example agents."""

from typing import Callable, Dict

import jax
import jax.numpy as jnp
from jax import lax

from mahjax.core import Env


NEG = -1e9
STARTING_SCORE = 250  # init score in mahjax (1 unit = 100 points)


def make_eval_fn(env: Env, num_eval_envs: int, max_steps: int = 200):
    """Build a rich, JIT-able evaluator that pits two policies in a 1-vs-3 setup.

    Tested under ``round_mode='single'``. In ``round_mode='east'/'half'`` with
    ``next_round_style='auto'``, per-hand stats only capture the last hand of each
    game because the env auto-resets and overwrites the round-end ``players`` info;
    use ``next_round_style='dummy_share'`` to capture every hand.
    """
    init_fn = env.init
    step_fn = env.step
    NUM_PLAYERS = env.num_players

    def one_vs_three(
        actor_fn_1: Callable,
        actor_fn_2: Callable,
    ):
        """Return ``eval_fn(key) -> Dict[str, jnp.ndarray]``.

        ``actor_fn_i(state, key) -> action_idx`` (int scalar).
        For each env, one randomly-chosen seat is filled by ``actor_fn_1``; the
        other three seats are filled by ``actor_fn_2``. The same trajectory is
        scored from both perspectives (``agent/*`` for actor_fn_1, ``baseline/*``
        for actor_fn_2 averaged over its three seats), so a single eval call
        yields both rows of statistics.
        """

        def play_one(init_state, seat_is_a1, key):
            def step_body(carry, _):
                state, rng = carry
                rng, k1, k2, kstep = jax.random.split(rng, 4)
                a1 = actor_fn_1(state, k1)
                a2 = actor_fn_2(state, k2)
                cp = state.current_player
                action = jnp.where(seat_is_a1[cp], a1, a2)
                done = state.terminated | state.truncated
                next_state = lax.cond(
                    done,
                    lambda: state,
                    lambda: step_fn(state, action, kstep),
                )

                # Rising edge of terminated_round → a hand just resolved.
                hand_ended = (
                    next_state.round_state.terminated_round
                    & ~state.round_state.terminated_round
                )
                p = next_state.players
                won = p.has_won
                riichi = p.riichi
                melded = p.meld_counts > 0
                tenpai = p.can_win.any(axis=-1)
                rewards = next_state.rewards
                is_hora = won.any()
                # Ron leaves exactly one player at a large negative reward
                # (the discarder, == last_player). Tsumo has 2 or 3 negatives.
                is_ron = is_hora & ((rewards < 0).sum() == 1)
                is_ryuukyoku = ~is_hora
                last_player = next_state.round_state.last_player
                dealt_in_mask = (
                    jnp.zeros(NUM_PLAYERS, dtype=jnp.bool_)
                    .at[last_player]
                    .set(is_ron)
                )

                event = {
                    "hand_ended": hand_ended.astype(jnp.int32),
                    "hora_hand": (hand_ended & is_hora).astype(jnp.int32),
                    "ryuu_hand": (hand_ended & is_ryuukyoku).astype(jnp.int32),
                    "won": (hand_ended & won).astype(jnp.int32),
                    "dealt_in": (hand_ended & dealt_in_mask).astype(jnp.int32),
                    "riichi": (hand_ended & riichi).astype(jnp.int32),
                    "melded": (hand_ended & melded).astype(jnp.int32),
                    "tenpai_ryuu": (hand_ended & is_ryuukyoku & tenpai).astype(jnp.int32),
                }
                return (next_state, rng), event

            (final_state, _), events = lax.scan(
                step_body, (init_state, key), None, length=max_steps
            )
            stats = jax.tree.map(lambda x: x.sum(axis=0), events)

            # Final game stats — average rank handles ties symmetrically.
            score = final_state.round_state.score.astype(jnp.float32)
            higher = (score[None, :] > score[:, None]).sum(axis=1)
            ties_inc_self = (score[None, :] == score[:, None]).sum(axis=1)
            rank = (
                1.0
                + higher.astype(jnp.float32)
                + 0.5 * (ties_inc_self.astype(jnp.float32) - 1.0)
            )
            gain = score - jnp.float32(STARTING_SCORE)
            return stats, rank, gain

        def eval_fn(key) -> Dict[str, jnp.ndarray]:
            k_seat, k_init, k_play = jax.random.split(key, 3)
            agent_seat = jax.random.randint(
                k_seat, (num_eval_envs,), 0, NUM_PLAYERS
            )
            seats = jnp.arange(NUM_PLAYERS)
            seat_is_a1 = seats[None, :] == agent_seat[:, None]  # (N, P)

            init_states = jax.vmap(init_fn)(
                jax.random.split(k_init, num_eval_envs)
            )
            play_keys = jax.random.split(k_play, num_eval_envs)
            stats, ranks, gains = jax.vmap(play_one)(
                init_states, seat_is_a1, play_keys
            )

            n_hands = stats["hand_ended"].astype(jnp.float32)        # (N,)
            n_hora = stats["hora_hand"].astype(jnp.float32)           # (N,)
            n_ryuu = stats["ryuu_hand"].astype(jnp.float32)           # (N,)
            total_hands = n_hands.sum()

            def safe_div(num, den):
                return num / jnp.maximum(den, 1.0)

            def per_view(prefix, mask):
                mf = mask.astype(jnp.float32)             # (N, P)
                player_hands = (n_hands[:, None] * mf).sum()
                player_ryuu = (n_ryuu[:, None] * mf).sum()
                n_seats = mf.sum()
                w = lambda k: (stats[k].astype(jnp.float32) * mf).sum()
                return {
                    f"{prefix}/hora_rate":           safe_div(w("won"),         player_hands),
                    f"{prefix}/deal_in_rate":        safe_div(w("dealt_in"),    player_hands),
                    f"{prefix}/riichi_rate":         safe_div(w("riichi"),      player_hands),
                    f"{prefix}/meld_rate":           safe_div(w("melded"),      player_hands),
                    f"{prefix}/tenpai_at_ryuu_rate": safe_div(w("tenpai_ryuu"), player_ryuu),
                    f"{prefix}/avg_rank":            safe_div((ranks * mf).sum(), n_seats),
                    f"{prefix}/avg_gain":            safe_div((gains * mf).sum(), n_seats),
                }

            metrics = {
                "hand/hora_finish_rate": safe_div(n_hora.sum(), total_hands),
                "hand/ryuukyoku_rate":   safe_div(n_ryuu.sum(), total_hands),
                "hand/total":            total_hands,
            }
            metrics.update(per_view("agent", seat_is_a1))
            metrics.update(per_view("baseline", ~seat_is_a1))
            return metrics

        return eval_fn

    return one_vs_three
