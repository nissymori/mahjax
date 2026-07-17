# Hong Kong Old Style Mahjong

MahJax exposes the Hong Kong environment as `hong_kong_mahjong`:

```python
import jax
import mahjax

env = mahjax.make("hong_kong_mahjong", round_mode="half")
state = env.init(jax.random.PRNGKey(0))
```

The default `HKOS_V1` configuration is an immutable `Rules` value. It uses 144
physical tiles (four copies of the 34 suited/honor types and eight unique bonus
tiles), 13-tile hands, a three-faan minimum, and a ten-faan cap. Seven pairs is
not a winning shape. Thirteen orphans and the other listed special hands are
enabled.

Flowers and seasons are removed from the hand immediately and replaced from the
wall. Kongs also receive a replacement tile. A flower encountered while drawing
a replacement is recorded and drawing continues until a standard tile is found.

## Faan Table

`HKOS_V1` uses this additive baseline:

| Pattern | Faan |
| --- | ---: |
| All chows with an unvalued pair and qualifying wait | 1 |
| All simples | 1 |
| Concealed hand | 1 |
| Self draw | 1 |
| Each dragon pung/kong | 1 |
| Seat wind pung/kong | 1 |
| Prevalent wind pung/kong | 1 |
| All pungs | 3 |
| Three concealed pungs | 3 |
| Three kongs | 3 |
| Half flush | 3 |
| Little three dragons | 5 |
| Full flush | 7 |
| All terminals and honors | 10 |
| Last tile, kong replacement, or robbing a kong | 1 each |

No flowers scores one faan. Each flower or season matching the player's seat
scores one faan. A complete set of four flowers or four seasons scores two faan
instead of its individual seat match, so all eight bonus tiles score four faan.

The following are automatic ten-faan limit hands: big three dragons, little four
winds, big four winds, nine gates, thirteen orphans, all terminals, all honors,
all green, four concealed pungs, four kongs, heavenly hand, and earthly hand.

## Payments

The `hk_old_style_v1` base payout table is:

| Faan | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10+ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Units | 8 | 16 | 24 | 32 | 48 | 64 | 96 | 128 |

On a discard win, only the discarder pays. On self draw, all three opponents
pay. Dealer payments and receipts are doubled. Every settlement is zero-sum in
`state.rewards` and is accumulated in `state.round_state.score`.

Only one player may win a discard. If more than one player has a legal win, the
nearest player following the discarder in turn order receives priority. A win
has priority over kong, pung, and chow claims.

The dealer repeats after a dealer win and after an exhausted-wall draw. A
non-dealer win advances the dealer. `round_mode="single"` always ends after one
hand, while `east` and `half` use four and eight non-repeat hands respectively.

## Actions and Observations

The environment retains the existing 79-action layout so agents can share
network heads with the riichi environments. Riichi and dummy actions are never
legal. Tile actions `0..33`, self-kongs `34..67`, tsumogiri, win, pung, open
kong, chow, and pass retain their existing numeric IDs.

`env.observe(state)` returns the current player's 34-count hand, eight flower
flags, melds, all rivers, relative scores, winds, and the number of tiles left
in the wall.

## Local Tests

On an Apple Silicon Mac, the CPU JAX wheels work without a separate XLA setup:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python -e . --group test --group lint
.venv/bin/pytest -q tests/hong_kong_mahjong
```

Run all regression tests with:

```bash
.venv/bin/pytest -q tests
```
