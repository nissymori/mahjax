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

`red_mahjong` is the full red-five riichi mahjong environment in MahJax.
It is the environment intended to track modern online riichi mahjong more closely.

This implementation is designed to follow **Tenhou** as closely as practical, and we validate it against downloaded Tenhou game logs.
We do not describe the log-test details here, but this is the main correctness target for the environment.

## Rules

The environment implements 4-player riichi mahjong with red fives.
By default, the rule configuration enables:

- red fives
- double ron
- special abortive draws
- pao

The detailed external rule reference is:

- [Tenhou rules (official English page)](https://tenhou.net/0/mj/mjlog/en/mjlog-en-rules.html)

The rule behavior is controlled through `GameConfig`, so advanced users can change settings such as red-fives usage, double ron, pao, and special abortive draws.

## State Representation

`red_mahjong` uses the newer nested state layout.
Mahjong-specific fields are grouped into:

- `state.players`
- `state.round_state`

This differs from `no_red_mahjong`, which still uses the older flat underscore-prefixed fields.
The two environments therefore do **not** currently share the same internal state representation, though we may align them in the future.

## Specs

| Name | Value |
| :--- | :--- |
| Version | `beta` |
| Number of players | `4` |
| Number of actions | `87` |
| Observation types | `dict`, `2D` |
| Dict action history shape | `(3, 200)` |
| Reward shape | `(4,)` |
| Reward semantics | score deltas in hundreds of points |

## Dict Observation

The current training-oriented observation is `dict`.
It is the format we recommend if you need a structured observation today, though we still reserve the right to revise it in future releases.

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

`observe_type="2D"` is available, but we do not consider its design finalized yet.
If you need a representation whose semantics are less likely to move, prefer `dict`.

## Action

The action space is:

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
| `86` | `DUMMY` |

## Rewards

Rewards are 4-player score deltas, represented in hundreds of points.

This includes:

- ron and tsumo score transfers
- honba and kyotaku handling
- exhaustive draw payments
- pao when enabled in `GameConfig`
- illegal-action termination penalties

## Termination

- `round_mode="single"` terminates after the first round ends.
- `round_mode="east"` runs East-only progression with `round_limit=4`.
- `round_mode="half"` runs East-South progression with `round_limit=8`.
