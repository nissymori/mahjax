# Mahjong Basics

This page is a short introduction to Japanese riichi mahjong for newcomers.
MahJax ships with two tile sets:

- `standard`: the original tile faces
- `bilingual`: an English-friendly tile set with the same game information

The first half of this page shows both side by side, so you can learn the game and the tile notation at the same time.

<style>
img[src*="assets/tiles/"] {
  background: #fff;
  border-radius: 4px;
  padding: 2px;
}
</style>

## 1. Reading the tiles

Riichi mahjong uses **34 tile types**.
There are **4 copies** of each type, so a full set has **136 tiles**.

### The three suits

The numbered suits are:

- **Characters (manzu)** (`m`)
- **Circles (pinzu)** (`p`)
- **Bamboos (souzu)** (`s`)

| Group | Standard tiles | Bilingual tiles |
| --- | --- | --- |
| Characters (manzu) | ![1m](assets/tiles/ja/1m.svg){ width="34" } ![2m](assets/tiles/ja/2m.svg){ width="34" } ![3m](assets/tiles/ja/3m.svg){ width="34" } ![4m](assets/tiles/ja/4m.svg){ width="34" } ![5m](assets/tiles/ja/5m.svg){ width="34" } ![6m](assets/tiles/ja/6m.svg){ width="34" } ![7m](assets/tiles/ja/7m.svg){ width="34" } ![8m](assets/tiles/ja/8m.svg){ width="34" } ![9m](assets/tiles/ja/9m.svg){ width="34" } | ![1m](assets/tiles/en/1m.svg){ width="34" } ![2m](assets/tiles/en/2m.svg){ width="34" } ![3m](assets/tiles/en/3m.svg){ width="34" } ![4m](assets/tiles/en/4m.svg){ width="34" } ![5m](assets/tiles/en/5m.svg){ width="34" } ![6m](assets/tiles/en/6m.svg){ width="34" } ![7m](assets/tiles/en/7m.svg){ width="34" } ![8m](assets/tiles/en/8m.svg){ width="34" } ![9m](assets/tiles/en/9m.svg){ width="34" } |
| Circles (pinzu) | ![1p](assets/tiles/ja/1p.svg){ width="34" } ![2p](assets/tiles/ja/2p.svg){ width="34" } ![3p](assets/tiles/ja/3p.svg){ width="34" } ![4p](assets/tiles/ja/4p.svg){ width="34" } ![5p](assets/tiles/ja/5p.svg){ width="34" } ![6p](assets/tiles/ja/6p.svg){ width="34" } ![7p](assets/tiles/ja/7p.svg){ width="34" } ![8p](assets/tiles/ja/8p.svg){ width="34" } ![9p](assets/tiles/ja/9p.svg){ width="34" } | ![1p](assets/tiles/en/1p.svg){ width="34" } ![2p](assets/tiles/en/2p.svg){ width="34" } ![3p](assets/tiles/en/3p.svg){ width="34" } ![4p](assets/tiles/en/4p.svg){ width="34" } ![5p](assets/tiles/en/5p.svg){ width="34" } ![6p](assets/tiles/en/6p.svg){ width="34" } ![7p](assets/tiles/en/7p.svg){ width="34" } ![8p](assets/tiles/en/8p.svg){ width="34" } ![9p](assets/tiles/en/9p.svg){ width="34" } |
| Bamboos (souzu) | ![1s](assets/tiles/ja/1s.svg){ width="34" } ![2s](assets/tiles/ja/2s.svg){ width="34" } ![3s](assets/tiles/ja/3s.svg){ width="34" } ![4s](assets/tiles/ja/4s.svg){ width="34" } ![5s](assets/tiles/ja/5s.svg){ width="34" } ![6s](assets/tiles/ja/6s.svg){ width="34" } ![7s](assets/tiles/ja/7s.svg){ width="34" } ![8s](assets/tiles/ja/8s.svg){ width="34" } ![9s](assets/tiles/ja/9s.svg){ width="34" } | ![1s](assets/tiles/en/1s.svg){ width="34" } ![2s](assets/tiles/en/2s.svg){ width="34" } ![3s](assets/tiles/en/3s.svg){ width="34" } ![4s](assets/tiles/en/4s.svg){ width="34" } ![5s](assets/tiles/en/5s.svg){ width="34" } ![6s](assets/tiles/en/6s.svg){ width="34" } ![7s](assets/tiles/en/7s.svg){ width="34" } ![8s](assets/tiles/en/8s.svg){ width="34" } ![9s](assets/tiles/en/9s.svg){ width="34" } |

### Honor tiles

Honor tiles are not numbered.
They are the four winds and the three dragons.

| Group | Standard tiles | Bilingual tiles |
| --- | --- | --- |
| Winds | ![east](assets/tiles/ja/east.svg){ width="34" } ![south](assets/tiles/ja/south.svg){ width="34" } ![west](assets/tiles/ja/west.svg){ width="34" } ![north](assets/tiles/ja/north.svg){ width="34" } | ![east](assets/tiles/en/east.svg){ width="34" } ![south](assets/tiles/en/south.svg){ width="34" } ![west](assets/tiles/en/west.svg){ width="34" } ![north](assets/tiles/en/north.svg){ width="34" } |
| Dragons | ![white](assets/tiles/ja/white.svg){ width="34" } ![green](assets/tiles/ja/gd.svg){ width="34" } ![red](assets/tiles/ja/rd.svg){ width="34" } | ![white](assets/tiles/en/white.svg){ width="34" } ![green](assets/tiles/en/gd.svg){ width="34" } ![red](assets/tiles/en/rd.svg){ width="34" } |

### Red fives

Some rulesets include one special red 5 in each suit.
MahJax can visualize those too.

| Group | Standard tiles | Bilingual tiles |
| --- | --- | --- |
| Red fives | ![5mr](assets/tiles/ja/5mr.svg){ width="34" } ![5pr](assets/tiles/ja/5pr.svg){ width="34" } ![5sr](assets/tiles/ja/5sr.svg){ width="34" } | ![5mr](assets/tiles/en/5mr.svg){ width="34" } ![5pr](assets/tiles/en/5pr.svg){ width="34" } ![5sr](assets/tiles/en/5sr.svg){ width="34" } |

## 2. Building combinations

Most winning hands in riichi mahjong are built from:

- **4 sets** (3 tiles)
- **1 pair** (2 tiles)

The three basic building blocks are:

| Unit | Standard tiles | Bilingual tiles | Meaning |
| --- | --- | --- | --- |
| Sequence | <nobr>![4m](assets/tiles/ja/4m.svg){ width="34" } ![5m](assets/tiles/ja/5m.svg){ width="34" } ![6m](assets/tiles/ja/6m.svg){ width="34" }</nobr> | <nobr>![4m](assets/tiles/en/4m.svg){ width="34" } ![5m](assets/tiles/en/5m.svg){ width="34" } ![6m](assets/tiles/en/6m.svg){ width="34" }</nobr> | Three consecutive suit tiles. Honors cannot make sequences. |
| Triplet | <nobr>![7p](assets/tiles/ja/7p.svg){ width="34" } ![7p](assets/tiles/ja/7p.svg){ width="34" } ![7p](assets/tiles/ja/7p.svg){ width="34" }</nobr> | <nobr>![7p](assets/tiles/en/7p.svg){ width="34" } ![7p](assets/tiles/en/7p.svg){ width="34" } ![7p](assets/tiles/en/7p.svg){ width="34" }</nobr> | Three identical tiles. |
| Pair | <nobr>![north](assets/tiles/ja/north.svg){ width="34" } ![north](assets/tiles/ja/north.svg){ width="34" }</nobr> | <nobr>![north](assets/tiles/en/north.svg){ width="34" } ![north](assets/tiles/en/north.svg){ width="34" }</nobr> | Two identical tiles. A regular winning hand needs exactly one pair. |

Here is a complete standard hand:

| Tile set | Hand |
| --- | --- |
| Std. | <nobr>![1m](assets/tiles/ja/1m.svg){ width="34" } ![2m](assets/tiles/ja/2m.svg){ width="34" } ![3m](assets/tiles/ja/3m.svg){ width="34" } ![4m](assets/tiles/ja/4m.svg){ width="34" } ![5m](assets/tiles/ja/5m.svg){ width="34" } ![6m](assets/tiles/ja/6m.svg){ width="34" } ![7p](assets/tiles/ja/7p.svg){ width="34" } ![8p](assets/tiles/ja/8p.svg){ width="34" } ![9p](assets/tiles/ja/9p.svg){ width="34" } ![7s](assets/tiles/ja/7s.svg){ width="34" } ![7s](assets/tiles/ja/7s.svg){ width="34" } ![7s](assets/tiles/ja/7s.svg){ width="34" } ![east](assets/tiles/ja/east.svg){ width="34" } ![east](assets/tiles/ja/east.svg){ width="34" }</nobr> |
| Biling. | <nobr>![1m](assets/tiles/en/1m.svg){ width="34" } ![2m](assets/tiles/en/2m.svg){ width="34" } ![3m](assets/tiles/en/3m.svg){ width="34" } ![4m](assets/tiles/en/4m.svg){ width="34" } ![5m](assets/tiles/en/5m.svg){ width="34" } ![6m](assets/tiles/en/6m.svg){ width="34" } ![7p](assets/tiles/en/7p.svg){ width="34" } ![8p](assets/tiles/en/8p.svg){ width="34" } ![9p](assets/tiles/en/9p.svg){ width="34" } ![7s](assets/tiles/en/7s.svg){ width="34" } ![7s](assets/tiles/en/7s.svg){ width="34" } ![7s](assets/tiles/en/7s.svg){ width="34" } ![east](assets/tiles/en/east.svg){ width="34" } ![east](assets/tiles/en/east.svg){ width="34" }</nobr> |

This hand is:

- `123m`
- `456m`
- `789p`
- `777s`
- `east east`

!!! note "Exceptions exist"
    The standard shape is `4 sets + 1 pair`, but famous exceptions such as **Seven Pairs** and **Thirteen Orphans** also exist.

### How a turn works

At a high level, every turn is simple:

1. Draw one tile.
2. Check whether the hand is now ready or complete.
3. Discard one tile.

Other players may claim the discard:

- **chi**: take the previous player's discard to make a sequence
- **pon**: take any player's discard to make a triplet
- **kan**: make a four-of-a-kind
- **ron**: win on another player's discard

If you complete your own hand on your draw, that win is **tsumo**.

## 3. Winning hands and yaku

In riichi mahjong, a complete shape is **not enough**.
To win, you normally need:

- a complete hand
- at least one **yaku** (a scoring pattern or winning condition)

For beginners, the three most useful yaku to learn first are:

- **riichi**
- **All simples (tanyao)**
- **Value honor triplet (yakuhai)**

From this section onward, the examples use the Bilingual tiles only.

### All simples (tanyao)

![2m](assets/tiles/en/2m.svg){ width="34" } ![3m](assets/tiles/en/3m.svg){ width="34" } ![4m](assets/tiles/en/4m.svg){ width="34" }
![3p](assets/tiles/en/3p.svg){ width="34" } ![4p](assets/tiles/en/4p.svg){ width="34" } ![5p](assets/tiles/en/5p.svg){ width="34" }
![4s](assets/tiles/en/4s.svg){ width="34" } ![5s](assets/tiles/en/5s.svg){ width="34" } ![6s](assets/tiles/en/6s.svg){ width="34" }
![6s](assets/tiles/en/6s.svg){ width="34" } ![7s](assets/tiles/en/7s.svg){ width="34" } ![8s](assets/tiles/en/8s.svg){ width="34" }
![8s](assets/tiles/en/8s.svg){ width="34" } ![8s](assets/tiles/en/8s.svg){ width="34" }

This hand contains only suit tiles from **2 to 8**.
No terminals (`1` or `9`) and no honors appear, so it qualifies for **tanyao**.

### Value honor triplet (yakuhai)

![rd](assets/tiles/en/rd.svg){ width="34" } ![rd](assets/tiles/en/rd.svg){ width="34" } ![rd](assets/tiles/en/rd.svg){ width="34" }
![2m](assets/tiles/en/2m.svg){ width="34" } ![3m](assets/tiles/en/3m.svg){ width="34" } ![4m](assets/tiles/en/4m.svg){ width="34" }
![4p](assets/tiles/en/4p.svg){ width="34" } ![5p](assets/tiles/en/5p.svg){ width="34" } ![6p](assets/tiles/en/6p.svg){ width="34" }
![5s](assets/tiles/en/5s.svg){ width="34" } ![6s](assets/tiles/en/6s.svg){ width="34" } ![7s](assets/tiles/en/7s.svg){ width="34" }
![9s](assets/tiles/en/9s.svg){ width="34" } ![9s](assets/tiles/en/9s.svg){ width="34" }

A triplet of dragons is a basic yaku.
This is one of the easiest yaku to recognize at a glance.

### Riichi: closed hand in ready state

Tenpai shape:

![1m](assets/tiles/en/1m.svg){ width="34" } ![2m](assets/tiles/en/2m.svg){ width="34" } ![3m](assets/tiles/en/3m.svg){ width="34" }
![4m](assets/tiles/en/4m.svg){ width="34" } ![5m](assets/tiles/en/5m.svg){ width="34" } ![6m](assets/tiles/en/6m.svg){ width="34" }
![2p](assets/tiles/en/2p.svg){ width="34" } ![3p](assets/tiles/en/3p.svg){ width="34" } 
![6s](assets/tiles/en/6s.svg){ width="34" } ![7s](assets/tiles/en/7s.svg){ width="34" } ![8s](assets/tiles/en/8s.svg){ width="34" }
![east](assets/tiles/en/east.svg){ width="34" } ![east](assets/tiles/en/east.svg){ width="34" }

Waiting tile:

![1p](assets/tiles/en/1p.svg){ width="34" } and ![4p](assets/tiles/en/4p.svg){ width="34" }

If this hand is **closed** and you declare **riichi** while waiting on `1p` or `4p`, then a later win on `1p` or `4p` is legal because riichi itself is the yaku.

### Two core ideas to remember

- A hand can look complete, but it still needs a yaku.
- Calling tiles makes some hands easier to finish, but it also removes some yaku. For example, **riichi** only works with a closed hand.

## 4. Where to learn the full rules

This page only covers the core ideas.
For the full rules, scoring details, and corner cases, the best next reference is:

- [European Mahjong Association: Riichi Rules 2025 (official PDF)](http://mahjong-europe.org/portal/images/docs/Riichi-rules-2025-EN.pdf)

If you want the MahJax-specific summary of supported rules, see [Rules](rule.md).

## 5. Where to play online

If you want to practice after reading the basics, these are good starting points:

- [Tenhou](https://tenhou.net/): long-running competitive platform. The interface is still Japanese-first, but it is one of the most established places to play riichi mahjong online.
- [Mahjong Soul on Steam](https://store.steampowered.com/app/2739990/Mahjong_Soul/): polished client, native English interface, and a built-in tutorial.
- [Riichi City on Steam](https://store.steampowered.com/app/1954420/RiichiCity/): another English-friendly client with ranked play.
- [Riichi.Wiki: Playing online](https://riichi.wiki/index.php?mobileaction=toggle_view_desktop&title=Playing_online): practical English notes comparing online clients, including Tenhou, Mahjong Soul, and Riichi City.
