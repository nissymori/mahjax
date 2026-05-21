# Supported Mahjong Games

MahJax currently implements 4-player Japanese Riichi Mahjong variants.
To get familiar with the basic rules, please refer to the official rulebook:
[European Riichi Mahjong Rules (2025, EN)](http://mahjong-europe.org/portal/images/docs/Riichi-rules-2025-EN.pdf).

While different variants require different strategies due to their specific rules, we believe the core skills they demand are the same:

- Efficient tile and hand evaluation (reasonable combination calculation)
- Risk management (defense vs. attack)
- Reading and inferring opponents’ hands

MahJax currently provides both **No-Red Mahjong** and **Red Mahjong** variants.

The “red” in red mahjong refers to **additional dora tiles assigned to 5 tiles** (usually 5m, 5p, 5s). This has a large impact on strategy:

- In **no red** mahjong, players must rely more on *yaku* (hand patterns) to increase score.
- In **red** mahjong, players prioritize **efficiency** to complete a winning hand quickly and must pay closer attention to **risk**, as the average hand value is higher and the chance of dealing into a high-value RON increases.

---

## No Red Mahjong

Although no red dora rules are not as popular in the gaming industry or in existing Mahjong AI research, they are widely played in Japan. Some professional leagues adopting this style include:

- [Japan Professional Mahjong League](https://www.ma-jan.or.jp/guide/game_rule.html)
- [Nihon Pro Mahjong](https://npm2001.com/about/rule/)
- [SAIKOUISEN](https://saikouisen.com/about/rules/)

These rules emphasize careful yaku construction and more traditional scoring without the volatility introduced by red dora.

---

## Red Mahjong

Red mahjong, especially the rule set used by [Tenhou](https://tenhou.net/), is currently the **most popular** variant both in AI research and in commercial online games.
MahJax includes a `red_mahjong` environment with red fives.

### Research Projects Using/Targeting Red Rules

- [Suphx](https://arxiv.org/abs/2003.13590): First top-player level mahjong agent trained by Deep RL.
- [Mjx](https://github.com/mjx-project/mjx): C++-based simulator for Tenhou rules.
- [Mortal](https://github.com/Equim-chan/Mortal): Rust-based simulator and agent training code. They also provide a [reviewer](https://mjai.ekyu.moe/) powered by their trained agent.
- [NAGA](https://naga.dmv.nico/naga_report/top/): Agent that reached 10-dan on Tenhou. They also provide a service to review play logs with the agent.

### Games Using/Supporting Red Rules

- [Tenhou](https://tenhou.net/)
- [Mahjong Soul](https://mahjongsoul.yo-star.com/) (There are minor differences.)

These environments typically feature fast-paced play and high average hand values, making them attractive both for players and for AI research.

---

## Future Pathways

Planned future extensions for MahJax include:

- Supporting additional **4-player Japanese Riichi Mahjong** rule sets
- Integrating **3-player Japanese Mahjong** rules

These expansions aim to make MahJax a more comprehensive platform for both Mahjong AI research and gameplay experimentation.
