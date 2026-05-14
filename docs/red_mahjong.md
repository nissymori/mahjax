# Red Mahjong

## Usage

```py
import mahjax

env = mahjax.make("red_mahjong", observe_type="dict")
```

or load the environment class directly:

```py
from mahjax.red_mahjong.env import RedMahjong
from mahjax.red_mahjong.state import GameConfig

env = RedMahjong(
    round_mode="half",
    observe_type="dict",
    game_config=GameConfig(),
)
```

## Description

`red_mahjong` is the full red-five riichi mahjong environment in MahJax. It is the environment intended to track modern online riichi mahjong more closely.

This implementation is designed to follow **Tenhou** as closely as practical, and we validate it against downloaded Tenhou game logs. We do not describe the log-test details here, but this is the main correctness target for the environment.

## Rules

The environment implements 4-player riichi mahjong with red fives. By default, the rule configuration enables:

- red fives
- double ron
- special abortive draws
- pao

The detailed external rule reference is:

- [Tenhou rules (official English page)](https://tenhou.net/0/mj/mjlog/en/mjlog-en-rules.html)

### `GameConfig`

The rule behavior is controlled through `GameConfig`. Advanced users can change settings such as red-fives usage, double ron, pao, and special abortive draws.

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `use_red_fives` | `True` | Enable red 5m / 5p / 5s tiles. |
| `allow_double_ron` | `True` | Allow two players to win simultaneously on the same discard. |
| `enable_special_abortive_draw` | `True` | Enable 九種九牌 / 四風連打 / 四家立直 / 三家和 / 四開槓. |
| `enable_pao` | `True` | Enable 包 (responsibility payment) for 大三元 / 大四喜. |
| `allow_open_tanyao` | `True` | Allow tanyao with open melds (`喰いタン`). |
| `allow_kuikae` | `False` | Allow swap-calling (`喰い替え`). |
| `seed_wall_from_key` | `True` | Use the env's PRNG key to shuffle the wall. |
| `starting_points` | `250` | Starting score (hundreds of points). |
| `target_points` | `300` | Target score for ending sudden death overtime. |
| `honba_bonus` | `300` | Honba bonus payment. |
| `riichi_bet` | `1000` | Riichi stick value. |

## State

`red_mahjong` uses the same nested state layout as `no_red_mahjong`:

- top-level RL handles (`current_player`, `terminated`, `rewards`, …)
- `state.players` — per-player arrays (`PlayerStateArrays`)
- `state.round_state` — round-level arrays (`RoundState`)

The shared field list and the round-transition style (`auto` / `dummy_share`) are documented once in [API](api.md). `red_mahjong` adds the following **red-five-aware fields** on top of the shared `PlayerStateArrays`:

| Field | Shape | Type | Meaning |
| :--- | :---: | :--- | :--- |
| `hand_with_red` | `(4, 37)` | `int8` | Hand histogram in the 37-tile-type space (34 tile types + 3 red fives). |
| `hand_ids` | `(4, 14)` | `int16` | Individual tile ids in the hand (each id in `[0, 135]`); unused slots are sentinel. |
| `hand_counts` | `(4,)` | `int8` | Number of valid tiles in `hand_ids`. |
| `drawn_tile` | `(4,)` | `int16` | Most recent drawn tile id per player; sentinel if none. |
| `meld_tiles` | `(4, 4, 4)` | `int16` | Tile ids that make up each meld. |
| `meld_info` | `(4, 4, 3)` | `int8` | Auxiliary info per meld (target / src / action). |
| `discards` | `(4, 24)` | `int16` | Individual discarded tile ids per player. |
| `discard_info` | `(4, 24, 4)` | `int8` | Auxiliary info per discard (tsumogiri flag, etc.). |
| `riichi_step` | `(4,)` | `int8` | Step at which each player declared riichi (for ippatsu tracking). |
| `has_nagashi_mangan` | `(4,)` | `bool` | Whether each player is still eligible for 流し満貫. |

The shared `hand` array (tile-type histogram in `[0, 33]` space) is also still kept for backward compatibility with code paths that do not need red-five resolution.

## Specs

| Name | Value |
| :--- | :--- |
| Version | `beta` |
| Number of players | `4` |
| Number of actions | `87` |
| Observation types | `dict`, `2D` |
| Reward shape | `(4,)` |
| Reward semantics | score deltas in hundreds of points |

## Action

The action space (87 actions):

| Range | Meaning |
| :--- | :--- |
| `0-36` | Discard a tile, including red-five tile ids |
| `37-70` | Closed kan / added kan |
| `71` | `TSUMOGIRI` |
| `72` | `RIICHI` |
| `73` | `TSUMO` |
| `74` | `RON` |
| `75` | `PON` |
| `76` | `PON_RED` |
| `77` | `OPEN_KAN` |
| `78-83` | Chi variants including red-five-aware actions |
| `84` | `PASS` |
| `85` | `KYUUSHU` |
| `86` | `DUMMY` (only legal under `next_round_style="dummy_share"`) |

## Dict Observation

The current training-oriented observation is `dict`. It is the format we recommend if you need a structured observation today, though we still reserve the right to revise it in future releases.

The returned dictionary contains:

| Key | Shape | Meaning |
| :--- | :---: | :--- |
| `hand` | `(14,)` | Current player's hand as sorted 34-type tiles; unused slots are `-1`. |
| `last_draw` | `()` | Last drawn tile in `[0, 36]`; `-1` means there is no drawn tile to expose. |
| `action_history` | `(3, 200)` | Relative-view action history. |
| `shanten_count` | `()` | Current player's shanten number. |
| `furiten` | `()` | Whether the current player is in furiten. |
| `scores` | `(4,)` | Scores ordered from the current player's perspective. |
| `round` | `()` | Round index. |
| `honba` | `()` | Honba count. |
| `kyotaku` | `()` | Riichi stick count. |
| `prevalent_wind` | `()` | Prevailing wind. |
| `seat_wind` | `()` | Current player's seat wind. |
| `dora_indicators` | `(4,)` | Dora indicator tile **types** in `[0, 33]`; missing entries are `-1`. |

### Action History

`action_history` is stored as:

- Row `0`: acting player index, converted to the current player's relative view
- Row `1`: action payload
- Row `2`: tsumogiri flag

The semantics are:

- For discards, row `1` stores the **actual discarded tile**
- For non-discard actions, row `1` stores the raw action id
- Row `2` is `1` for tsumogiri, `0` for a non-tsumogiri discard, and `-1` for non-discard actions

For `red_mahjong`, discard tiles are in `[0, 36]` because red fives have dedicated tile ids in the action space, while raw action ids are in `[0, 86]`.

## 2D Observation

`observe_type="2D"` is available, but we do not consider its design finalized yet. If you need a representation whose semantics are less likely to move, prefer `dict`.

## Rewards

Rewards are 4-player score deltas, represented in hundreds of points.

This includes:

- ron and tsumo score transfers
- honba and kyotaku handling
- exhaustive draw payments
- pao when enabled in `GameConfig`
- illegal-action termination penalties

For how to consume these rewards in turn-based MARL training (per-player reward accumulator + GAE), see the [API → Using `auto` rewards in RL](api.md#using-auto-rewards-in-rl) section.

## Termination

- `round_mode="single"` terminates after the first round ends.
- `round_mode="east"` runs East-only progression with `round_limit=4`.
- `round_mode="half"` runs East-South progression with `round_limit=8`.

In multi-round modes, the next-round transition behavior is controlled by `next_round_style` (see [API](api.md#round-transition-style-next_round_style)).

`red_mahjong` is the env used by `mahjax_tenhou_test`, which validates round trajectories against real tenhou mjlogs using `next_round_style="dummy_share"`. The state-level equivalence between `auto` and `dummy_share` at round boundaries is asserted by the parity tests in `tests/red_mahjong/test_env.py`, so the same correctness target also covers `auto` mode end-to-end.
