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
| `action_history` | `(3, 200)` | `int8` | Per-step action history for the **current round**. Row 0 acting player, row 1 action payload (discarded tile for discards, raw action id otherwise), row 2 tsumogiri flag. Unused slots are `-1`. |
| `round_step` | `()` | `int32` | Write cursor into `action_history`, reset to 0 each round. Do **not** index the history with `state.step_count`: that counter is hanchan-global, so it walks past the end of this per-round buffer and JAX drops the out-of-bounds scatter silently. |
| `history_overflow` | `()` | `bool` | `True` once a round produced more actions than `action_history` can hold; the newest action then overwrites the last slot. Makes truncation observable instead of silent. |
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

## Observation (`env.observe`)

`observe_type="dict"` is the only supported mode. `env.observe(state)` returns a dict of
arrays describing the table **from the current player's seat**. Everything in it is either
that player's own private information or public information — no opponent hands, no wall,
no ura dora. Scalars are 0-d arrays, not length-1 vectors. `env.observation_shape` returns
the per-key shapes.

Fields with a seat axis are rotated so index 0 is the observer and 1/2/3 are the players to
the right / across / left.

| Key | Shape | Type | Meaning |
| :--- | :---: | :--- | :--- |
| `hand` | `(34,)` | `int8` | **Count** of each tile type in the concealed hand, `[0, 4]`. Red fives fold into their type; `is_red` carries the redness. Melded tiles are **not** here — see `melds`. |
| `is_red` | `(34,)` | `bool` | Whether the concealed hand holds the **red** five of that type. Only indices 4 / 13 / 22 can be `True`. **`red_mahjong` only** — `no_red_mahjong` omits the key rather than emitting an always-zero column. |
| `last_draw` | `()` | `int8` | The observer's own most recent draw `[0-36]`, `-1` if none. This is the tile `TSUMOGIRI` discards. Masked: the underlying state field is round-level, so during a robbing-a-kan window it would otherwise expose the kan declarer's private draw to the responder. |
| `shanten_count` | `()` | `int8` | Shanten of the concealed hand, `[-1, 6]` (`-1` complete, `0` tenpai). |
| `discard_shanten` | `(34, 3)` | `int8` | Shanten after discarding each tile **type**, split `[normal, seven pairs, thirteen orphans]`, per-column ranges `[0,8]`/`[0,13]`/`[0,13]`. `Shanten.NOT_IN_HAND` (14) marks types the hand does not hold. At a 3n+1 response node these describe the 3n hands one further discard out — see `is_discard_node`. |
| `can_win` | `(34,)` | `bool` | **Self only.** For each tile type, whether it completes the hand (the wait table). Opponents' rows exist in state but are hidden and deliberately not exposed. Stale after a pon/chi — see the note below. |
| `discard_can_win` | `(34, 34)` | `int8` | Per-discard wait table. Row `t` is "if I discard tile type `t`, which tiles then complete my hand". `-1` fills rows with no answer: not a discard node, or a tile the hand does not hold; elsewhere `0`/`1`. `discard_shanten` says **whether** a discard reaches tenpai, this says **what** the wait is — the difference between two tenpai discards. Derived at observe time, not stored in state. |
| `furiten` | `()` | `bool` | Furiten by own discard **or** by pass. |
| `is_discard_node` | `()` | `bool` | Whether a discard is legal, i.e. whether `discard_shanten` means "if I discard this" (3n+2 hand) rather than a floater map one discard further out (3n+1). |
| `melds` | `(3, 4, 4)` | `int8` | Melds per seat. Channels `[action, called_tile, src]`; `action == -1` is the mask. `src` is `(discarder - owner) mod 4` and `0` means a closed kan. |
| `action_history` | `(3, 200)` | `int8` | This round's actions in order, `[player, action, tsumogiri]`. Row 0 is a seat **relative** to the observer (0 == me). Discards store the tile, other actions the raw action id, told apart by the tsumogiri channel. Per-round buffer, cleared at every round boundary. |
| `scores` | `(4,)` | `int32` | Scores, seat-rotated. |
| `target` | `()` | `int8` | Tile the pending call/ron decision is about, `-1` if none. Red-aware for calls on a discard; a bare tile type for chankan. |
| `last_player` | `()` | `int8` | Relative seat of whoever acted last, `-1` if none. Only refers to the pending call when `target >= 0`; otherwise read it as "who moved last". |
| `tiles_seen` | `(34,)` | `int8` | Copies of each tile type already visible from this seat, `[0, 4]`: own concealed hand + every river + every meld + revealed dora indicators. Red fives fold into their type. Called tiles are counted once. |
| `ippatsu` | `(4,)` | `bool` | Seat-rotated. Not derivable from anything else here. |
| `riichi` | `(4,)` | `bool` | Seat-rotated. Public for every seat. |
| `is_hand_concealed` | `(4,)` | `bool` | Seat-rotated. Gates riichi legality and menzen tsumo. Deriving it from `melds` means reimplementing the rule that a closed kan keeps the hand concealed. |
| `wall_remaining` | `()` | `int32` | Tiles still drawable from the live wall, `[0, 70]`. Drives haitei, the exhaustive draw and the riichi precondition. |
| `round` | `()` | `int8` | Kyoku counter in `[0, round_limit]`. |
| `round_limit` | `()` | `int8` | Last kyoku index: 4 for `east`, 8 for `single`/`half`. With `round`, how much game is left. |
| `honba` | `()` | `int8` | Honba count. |
| `kyotaku` | `()` | `int8` | Riichi sticks on the table. |
| `prevalent_wind` | `()` | `int8` | Round wind, `round // 4`. Reaches 2 (West) on the last kyoku of a `half` game. |
| `seat_wind` | `()` | `int8` | Observer's seat wind `[0-3]`; 0 is the dealer. |
| `dora_indicators` | `(5,)` | `int8` | Indicators `[0-36]`, `-1` for unrevealed slots. |

`melds` is derivable from an intact `action_history`, but only by learning a parser —
linking a call to the discard it consumed, and a kan upgrade to the pon before it, are
non-adjacent lookups. It is provided decoded because it costs nothing (state reads, no
shanten evaluation) and removes that sub-problem.

There is no `river` key. Every discard is already in `action_history` (the tile in channel 1,
tsumogiri in channel 2) and every call in `melds`, so a river plane would ship the same
events twice.

**`can_win` vs `discard_can_win`.** They answer at complementary nodes. `can_win` is exact
at a 3n+1 response node (ron/pon/chi/pass), where it is the current wait; at a 3n+2 discard
node it is the wait of the hand **before** the draw, i.e. the wait you keep if you tsumogiri.
`discard_can_win` is the reverse: all `-1` at a response node, exact at a discard node.

One caveat on `can_win`: the env refreshes it on init, on discard and on the post-kan draw,
but **not** in `_pon` / `_chi`. Those rewrite the hand, so at a post-call discard node it can
claim a wait the player no longer has. `shanten_count` is recomputed on every observation;
trust it where the two disagree. Recomputing `can_win` there would not help — after a call the
concealed hand is 3n+2, where `Hand.can_ron` is structurally `False` for every tile.

## Privileged observation (`env.observe_privileged`)

`env.observe_privileged(state)` returns everything `observe()` does, plus the other players'
concealed hands. This is **hidden information**: use it for centralised critics, opponent
modelling, dataset analysis and debugging, never for a policy that will be evaluated or
deployed against real opponents. It always returns a dict, independent of `observe_type`.

| Key | Shape | Type | Meaning |
| :--- | :---: | :--- | :--- |
| `others_hand` | `(3, 34)` | `int8` | Per-type counts of each **other** player's concealed hand, `[0, 4]`. Melds excluded, as for `hand`. |
| `others_is_red` | `(3, 34)` | `bool` | Whether that player holds the red five of the type. **`red_mahjong` only.** |

Rows use the same seat-relative frame as `scores`: index 0 is the player to the observer's
right, 1 across, 2 left. The observer's own hand is not repeated — it is already `hand`.

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
