# API

## Install

```
pip install mahjax
```

## Usage

We follow almost the same API as [pgx](https://github.com/sotetsuk/pgx). Below is an example of MahJax usage.

```py
import jax
import jax.numpy as jnp
import mahjax


batch_size = 10
rng = jax.random.PRNGKey(0)

# Initialize environment and state
env = mahjax.make(
    "red_mahjong",
    round_mode="single",   # "single", "east" (tonpuusen), or "half" (hanchan)
    next_round_style="auto",  # "auto" (default, RL) or "dummy_share" (interactive / mjai)
    order_points=[30, 10, -10, -30],
)
step_fn = jax.jit(jax.vmap(env.step))
obs_fn = jax.jit(jax.vmap(env.observe))

# Initialize state
rng, subrng = jax.random.split(rng)
rngs = jax.random.split(subrng, batch_size)
state = jax.jit(jax.vmap(env.init))(rngs)

# Tsumogiri play
while not state.terminated.all():
    rng, subrng = jax.random.split(rng)
    obs = obs_fn(state)  # (batch_size, ...) access to the observation.
    rngs = jax.random.split(subrng, batch_size)
    action = jnp.full((batch_size,), mahjax.Action.TSUMOGIRI, dtype=jnp.int32)
    state = step_fn(state, action, rngs)
    reward = state.rewards  # (batch_size, 4) access to the reward.
```

## State

Both `no_red_mahjong` and `red_mahjong` share the same nested state layout. The state is an immutable JAX dataclass (`EnvState`) made of:

- a small set of **top-level fields** (the standard RL handles)
- a per-player array group (`state.players`, a `PlayerStateArrays`)
- a round-level array group (`state.round_state`, a `RoundState`)

`red_mahjong` adds a few extra fields on top of this — those are listed in [Red Mahjong](red_mahjong.md). The common fields are the same.

### Top-level fields (`state`)

| Field | Shape | Type | Meaning |
| :--- | :---: | :--- | :--- |
| `current_player` | `()` | `int8` | Player whose turn it is. The legal action mask exposed at the top level always belongs to this player. |
| `legal_action_mask` | `(num_actions,)` | `bool` | Legal action mask for `current_player`. At terminal states this is set to all-`True` to avoid zero-division when normalizing action probabilities. |
| `rewards` | `(4,)` | `float32` | 4-player reward vector for the **step that just ran**. Mahjong rewards are score deltas in hundreds of points (e.g. ron payments, tsumo payments, tenpai/noten settlement, illegal-action penalty). Zero in steps that produced no scoring event. |
| `terminated` | `()` | `bool` | Game terminal. `True` once for the step that ends the game; remains `True` afterwards. After it goes `True`, subsequent `env.step` calls return the state unchanged with zero rewards. |
| `truncated` | `()` | `bool` | Reserved for external truncation wrappers (e.g. `TimeLimit`). MahJax does not itself produce truncated episodes. |
| `step_count` | `()` | `int32` | Total `env.step` calls applied so far. |
| `players` | nested | `PlayerStateArrays` | Per-player arrays — see below. |
| `round_state` | nested | `RoundState` | Round-level arrays — see below. |

### `state.players` — per-player state

All entries are leading-axis-`4` arrays indexed by absolute player id (seat 0..3).

| Field | Shape | Type | Meaning |
| :--- | :---: | :--- | :--- |
| `hand` | `(4, 34)` | `int8` | Tile-type histogram of each player's concealed hand. |
| `legal_action_mask` | `(4, num_actions)` | `bool` | Per-player legal action mask. The 1-D `state.legal_action_mask` is `players.legal_action_mask[current_player]`. |
| `can_win` | `(4, 34)` | `bool` | For each player and tile type, whether the player can win on that tile. |
| `has_yaku` | `(4, 2)` | `bool` | Whether each player has a valid yaku, for RON / TSUMO respectively. |
| `fan` | `(4, 2)` | `int32` | Fan count, for RON / TSUMO respectively. |
| `fu` | `(4, 2)` | `int32` | Fu count, for RON / TSUMO respectively. |
| `melds` | `(4, 4)` | `uint16` | Packed meld records (action / target / src). |
| `meld_counts` | `(4,)` | `int8` | Number of melds claimed by each player. |
| `river` | `(4, 24)` | `uint16` | Discard rivers; packed tile + tsumogiri flag. |
| `discard_counts` | `(4,)` | `int8` | Number of discards in each player's river. |
| `riichi` | `(4,)` | `bool` | Whether each player is in riichi. |
| `riichi_declared` | `(4,)` | `bool` | Whether each player has declared riichi this turn (latch cleared on accept). |
| `double_riichi` | `(4,)` | `bool` | Double-riichi flag per player. |
| `ippatsu` | `(4,)` | `bool` | Ippatsu still live per player. |
| `furiten_by_discard` | `(4,)` | `bool` | Furiten because of own discard. |
| `furiten_by_pass` | `(4,)` | `bool` | Furiten because of passed RON. |
| `is_hand_concealed` | `(4,)` | `bool` | Hand still closed (no open melds). |
| `pon` | `(4, 34)` | `int32` | Bookkeeping for pon-related calls per tile type. |
| `has_won` | `(4,)` | `bool` | Whether each player won this round (set on RON / TSUMO). |
| `n_kan` | `(4,)` | `int8` | Number of kan declared by each player this round. |

### `state.round_state` — round-level state

Held common across all four players.

| Field | Shape | Type | Meaning |
| :--- | :---: | :--- | :--- |
| `rng_key` | `(2,)` | `uint32` | JAX PRNG key used to deal the current round. |
| `action_history` | `(3, 200)` | `int8` | Per-step action history. Row 0 acting player, row 1 action payload (discarded tile for discards, raw action id otherwise), row 2 tsumogiri flag. |
| `round` | `()` | `int8` | Round index (`0`-based). |
| `round_limit` | `()` | `int8` | Round limit derived from `round_mode`: `4` for `east`, `8` for `half`. |
| `terminated_round` | `()` | `bool` | `True` on the step that ends the round (RON / TSUMO / 流局). See the round-transition section for how `auto` vs `dummy_share` expose this. |
| `honba` | `()` | `int8` | Honba count (renchan counter). |
| `kyotaku` | `()` | `int8` | Number of unclaimed riichi sticks on the table. |
| `init_wind` | `(4,)` | `int8` | Initial seat winds at the start of the game. |
| `seat_wind` | `(4,)` | `int8` | Current seat wind per player. |
| `dealer` | `()` | `int8` | Current dealer. |
| `order_points` | `(4,)` | `int32` | Placement bonus / uma. |
| `score` | `(4,)` | `int32` | Per-player score, in hundreds of points. |
| `deck` | `(136,)` | `int8` | Current round's shuffled wall (tile types). |
| `next_deck_ix` | `()` | `int32` | Next index to draw from in `deck`. |
| `last_deck_ix` | `()` | `int8` | End of the live wall. |
| `last_draw` | `()` | `int8` | Most recently drawn tile type. |
| `last_player` | `()` | `int8` | Player who took the previous action (used for discard-target resolution). |
| `dora_indicators` | `(5,)` | `int8` | Dora indicator tile types; `-1` for unrevealed slots. |
| `ura_dora_indicators` | `(5,)` | `int8` | Ura-dora indicators; `-1` for unrevealed slots. |
| `is_abortive_draw_normal` | `()` | `bool` | True once the wall has been exhausted (流局). |
| `is_haitei` | `()` | `bool` | Currently on the last live-wall tile (海底). |
| `target` | `()` | `int8` | Tile currently being targeted by callers. |
| `n_kan_doras` | `()` | `int8` | Number of kan dora flipped so far this round. |
| `kan_declared` | `()` | `bool` | Kan declared this step. |
| `can_after_kan` | `()` | `bool` | After-kan / 嶺上開花 still possible. |
| `can_robbing_kan` | `()` | `bool` | A robbing-a-kan / 槍槓 window is open. |
| `draw_next` | `()` | `bool` | Internal flag: the env should draw a tile next. |
| `dummy_count` | `()` | `int8` | Counter for the DUMMY-rotation phase. Only used by `next_round_style="dummy_share"` (see below); always `0` under `auto`. |

## Round Transition Style (`next_round_style`)

In multi-round modes (`east` / `half`) the env can either advance to the next round **automatically inside a single `env.step`**, or expose an explicit DUMMY-action phase that lets every seat observe the round-end state before the next round begins. This is controlled by `next_round_style`:

```py
env = mahjax.make("red_mahjong", round_mode="half", next_round_style="auto")  # default
env = mahjax.make("red_mahjong", round_mode="half", next_round_style="dummy_share")
```

| Style | Default | When to use |
| :--- | :--- | :--- |
| `auto` | ✅ | RL / training. One round transition = one `env.step`. |
| `dummy_share` |  | Interactive UI, mjai-compatible replays, anything that needs per-player round-end observation. |

In `single` mode the two styles are identical — there is no "next round."

### How a round-end step looks in each style

When the agent plays the action that ends a round (RON / TSUMO / exhaustive draw):

**`auto`** — collapse the round transition into a single step:

```
   t      action      terminated_round   terminated   rewards
   k       RON              False           False        X        ← state for round k+1's first turn,
                                                                    but `rewards` carries the round-end reward
```

The state returned by `env.step` is already the **next round's init state** (new deck, new dealer if needed, fresh hands, `legal_action_mask` for the new dealer). `terminated` is `False` until the game itself ends; on the game-ending round it is `True` and the score is updated with rank points + kyotaku bonuses.

**`dummy_share`** — make the round-end phase explicit. After the winning action, the env exposes a state with `legal_action_mask` containing only `DUMMY` for every seat. Each of the four players must then play `DUMMY`; the fourth `DUMMY` advances to the next round.

```
   t      action      terminated_round   dummy_count   rewards
   k       RON              True              0           X
   k+1     DUMMY            True              1           X      (player k+1)
   k+2     DUMMY            True              2           X      (player k+2)
   k+3     DUMMY            True              3           X      (player k+3)
   k+4     DUMMY            False             0           0      ← next round's init state
```

This matches the "次の局へ" / "next round" button in interactive UIs.

State equivalence (proved by `TestAutoDummyShareParity` in `tests/`): for the same initial state, the next-round init produced by **one `auto` step** equals the next-round init produced by **five `dummy_share` steps (RON + 4 × DUMMY)**, comparing every round-level and player-level field. The only intentional differences are `step_count` (which advances per `env.step` call) and `rewards` (preserved by `auto` on the transition step; delivered by `dummy_share` on the RON step and zeroed thereafter).

### Using `auto` rewards in RL

Mahjong reward is sparse: it lands on the steps that end a round (RON / TSUMO / 流局). Under `auto`, that reward vector is **carried on the same step that produces the next round's observation**. This is the simplest interface for turn-based MARL because each `env.step` corresponds to exactly one logical decision and exactly one reward delivery.

Note that turn-based mahjong is not a synchronous MARL game: only `current_player` acts at each step. The reward, however, is a 4-player vector — a player whose seat is not `current_player` may still receive a non-zero reward (for example, the discarder of a winning tile). The GAE has to account for "I received reward while I was not on turn." The standard trick used by the PPO example in `examples/ppo_with_reg.py` (and by NashPG / Pgx-style turn-based RL in general) is a **per-player reward accumulator**.

#### GAE for turn-based MARL with per-player reward accumulators

Rollout each step records, for the agent that just acted:

```py
Transition(
    is_new_episode,           # 1 if this step is the first step of a fresh episode
    action,                   # what the acting player did
    value,                    # V(s_t) for the acting player's network
    reward,                   # 4-vector of rewards delivered at this step
    log_prob,                 # log π(a_t | s_t)
    observation,              # observation from current_player's perspective
    action_mask,              # legal action mask shown to current_player
    current_player,           # which seat acted
)
```

Backward scan to compute advantages: for each player keep a running accumulator. When we visit a step whose `current_player == p`, that's where player `p`'s next decision happens — we settle `p`'s accumulated reward into a single per-decision delta:

```py
def gae_backward(advantage_next, value_next, reward_accum, transition):
    p = transition.current_player                # who acted at this step
    reward_p = reward_accum[p] + transition.reward[p]  # everything p has accumulated since last on-turn
    delta = reward_p + gamma * value_next * (1 - transition.is_new_episode) - transition.value
    advantage = delta + gamma * gae_lambda * (1 - transition.is_new_episode) * advantage_next

    # Reset p's accumulator (consumed). Continue accumulating other players' rewards.
    reward_accum = reward_accum.at[p].set(0.0)
    reward_accum = reward_accum + transition.reward          # still owe other players
    reward_accum = reward_accum.at[p].set(0.0)               # but not p again on this step

    return advantage, value, reward_accum
```

Round transitions under `auto` slot into this naturally:

- The RON step delivers a non-zero `reward` vector; the next observation is already the next round's init.
- The discarder seat that paid out is *not* `current_player` on this step (the RON player is). The discarder's reward sits in its accumulator until the discarder's next on-turn step (which is now in the new round).
- This is consistent with how mahjong logically works: the discarder learns about the loss on the same observation that opens its next turn.

With `dummy_share` instead, the rewarded step (RON) is followed by four DUMMY steps with zero reward, and the next-round init shows up only at step `k+4`. RL-wise this is wasteful (one logical decision = five env steps, four of them no-op) and clutters the trajectory with "DUMMY" actions that should not appear in a policy's action distribution. That is why `auto` is the default — it gives the same state equivalence as `dummy_share` but exposes a clean one-step-per-decision API.

### When to choose `dummy_share`

- Driving an interactive UI (the user clicks "next round" after seeing the round-end summary).
- Round-end packet replay against external logs (e.g. mjai / mjlog) where every seat is expected to observe the round-end state.

For everything else, use `auto`.
