# No-Red Mahjong

## Usage

```py
import mahjax

env = mahjax.make("no_red_mahjong", observe_type="dict")
```

or load the environment class directly:

```py
from mahjax.no_red_mahjong.env import NoRedMahjong

env = NoRedMahjong(round_mode="half", observe_type="dict")
```

## Description

`no_red_mahjong` is a fast 4-player riichi mahjong environment without red fives.
It is the lightweight environment used by the current offline RL and PPO examples.

Compared with `red_mahjong`, this environment intentionally omits some rules in order to keep the implementation simpler and faster.

## Rules

The environment is a simplified Japanese riichi mahjong variant without red fives.

- No red fives
- No special abortive draws (`特殊流局`)
- No pao
- No double ron

This environment exists primarily for speed and for simpler RL experiments.
It is covered by hand-written tests, and we do not expect major rule-breaking bugs, but some corner cases may still remain.
We expect this to improve over time.

For general riichi mahjong rules, see the [European Mahjong Association rulebook](https://mahjong-europe.org/portal/images/docs/Riichi-rules-2025-EN.pdf).

## State Representation

`no_red_mahjong` currently uses the older flat state layout.
Mahjong-specific fields live directly on the state object, such as `_hand`, `_river`, `_melds`, `_score`, and `_action_history`.

This differs from `red_mahjong`, which uses a newer nested representation with `players` and `round_state`.
These two environments therefore do **not** currently share the same internal state representation, though we may unify them in the future.

## Specs

| Name | Value |
| :--- | :--- |
| Version | `beta` |
| Number of players | `4` |
| Number of actions | `79` |
| Observation types | `dict`, `2D` |
| Dict action history shape | `(3, 200)` |
| Reward shape | `(4,)` |
| Reward semantics | score deltas in hundreds of points |

## Dict Observation

The current training examples use the `dict` observation.
It is the most stable observation format in this repository right now, but it may still change in future releases.

The returned dictionary contains:

| Key | Shape | Meaning |
| :--- | :---: | :--- |
| `hand` | `(14,)` | Current player's hand as sorted tile types in `[0, 33]`; unused slots are `-1`. |
| `last_draw` | `()` | Last drawn tile in `[0, 33]`; `-1` means there is no drawn tile to expose. |
| `action_history` | `(3, 200)` | Action history in the same style as `red_mahjong`. |
| `shanten_count` | `()` | Current player's shanten number. |
| `furiten` | `()` | Whether the current player is in furiten. |
| `scores` | `(4,)` | Scores ordered from the current player's perspective. |
| `round` | `()` | Round index used by the environment. |
| `honba` | `()` | Honba count. |
| `kyotaku` | `()` | Riichi stick count. |
| `prevalent_wind` | `()` | Current round wind information used by the environment. |
| `seat_wind` | `()` | Current player's seat wind information used by the environment. |
| `dora_indicators` | `(4,)` | Dora indicator tile types in `[0, 33]`; missing entries are `-1`. |

### Action History

`action_history` is stored as:

- Row `0`: acting player index, converted to the current player's relative view
- Row `1`: action payload
- Row `2`: tsumogiri flag

The semantics match `red_mahjong`:

- For discards, row `1` stores the **actual discarded tile**
- For non-discard actions, row `1` stores the raw action id
- Row `2` is `1` for tsumogiri, `0` for a non-tsumogiri discard, and `-1` for non-discard actions

For `no_red_mahjong`, discard tiles are in `[0, 33]` and raw action ids are in `[0, 78]`.

## 2D Observation

`observe_type="2D"` exists, but its design is not yet fixed.
You can use it for experiments, but we do not recommend treating it as a stable interface yet.

## Action

The action space is:

| Range | Meaning |
| :--- | :--- |
| `0-33` | Discard a tile type |
| `34-67` | Closed kan / added kan |
| `68` | `TSUMOGIRI` |
| `69` | `RIICHI` |
| `70` | `TSUMO` |
| `71` | `RON` |
| `72` | `PON` |
| `73` | `OPEN_KAN` |
| `74-76` | `CHI_L`, `CHI_M`, `CHI_R` |
| `77` | `PASS` |
| `78` | `DUMMY` |

## Rewards

Rewards are 4-player score deltas, represented in hundreds of points.

Examples:

- winning a hand gives positive reward to the winner and negative reward to the payer(s)
- exhaustive draw can produce tenpai / noten payments
- illegal actions end the game immediately with the standard illegal-action penalty

## Termination

- `round_mode="single"` terminates after the first round ends.
- `round_mode="east"` runs East-only progression with `round_limit=4`.
- `round_mode="half"` runs East-South progression with `round_limit=8`.
