# No-Red Mahjong

## Usage

```py
import mahjax

env = mahjax.make("no_red_4p", observe_type="dict")
```

`no_red_4p` is `red_4p` with the `NO_RED` rule preset, so this is the same as:

```py
from mahjax.mahjong_4p.config import NO_RED
from mahjax.mahjong_4p.env import Mahjong4p

env = Mahjong4p(round_mode="half", observe_type="dict", game_config=NO_RED)
```

## Rules

`NO_RED` turns off every optional rule of `red_4p`:

- No red fives: every five is dealt as a plain five.
- No double ron: the players who can ron a tile answer in turn order from the discarder, and the first RON ends the round, so one player wins it.
- No pao.
- No special abortive draws (`特殊流局`): no 九種九牌, 四風連打, 四家立直, 三家和 or 四開槓.

Everything else is `red_4p`'s: calls, riichi, furiten, kans and dora, scoring and round progression. See [Red Mahjong](red_mahjong.md) for the rules and for the `GameConfig` fields behind the preset.

## State, observation and actions

`no_red_4p` uses `red_4p`'s state layout, dict observation ([API](api.md#observation-envobserve)) and 87-wide action space. Under `NO_RED`:

- `is_red` is always `False`.
- The red-five calls (`PON_RED`, `CHI_L_RED`, `CHI_M_RED`, `CHI_R_RED`) and `KYUUSHU` are never legal.
