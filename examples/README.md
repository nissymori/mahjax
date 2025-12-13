# Behavior Cloning and RL for Single-Round Mahjong

This repository provides a complete pipeline for training Mahjong agents in a **single-round** "no red dora" setting (the game ends after one round regardless of the outcome).

The code covers three main steps:
1. **Data Collection:** Collecting gameplay data using a rule-based agent ([`collect_offline_data.py`](collect_offline_data).
2. **Behavior Cloning (BC):** Training a baseline model using supervised learning ([`bc.py`](https://github.com/nissymori/mahjax/blob/main/examples/bc.py).
3. **Reinforcement Learning (RL):** Training a PPO agent with regularization towards the BC model ([`ppo_with_reg.py`](https://github.com/nissymori/mahjax/blob/main/examples/ppo_reg.py).

<div align="center">
<img src="https://github.com/nissymori/mahjax/blob/main/examples/assets/results.png" width="90%">
</div>

### Data Collection
We collect training data using a hand-crafted rule-based agent (designed to minimize shanten, etc.). Please refer to [`mahjax/no_red_mahjong/rule_based_players.py`](https://github.com/nissymori/mahjax/blob/main/mahjax/no_red_mahjong/rule_based_players.py) for implementation details.

In this example, we adopt a Transformer architecture that utilizes dictionary-based observations.
- **Network Architecture:** Refer to [`networks.py`](https://github.com/nissymori/mahjax/blob/main/examples/network.py) for details.
- **Observation Semantics:** Documentation is available [here](https://github.com/nissymori/mahjax/blob/ea6aa59521d1bbdae0876638af5a200b992c8bb7/mahjax/no_red_mahjong/env.py#L1810).
- **Note for CNN Users:** If you prefer to train CNN-based agents (common in the Mahjong AI community), please use [`observe_2D`](https://github.com/nissymori/mahjax/blob/ea6aa59521d1bbdae0876638af5a200b992c8bb7/mahjax/no_red_mahjong/env.py#L1871) instead.

### Behavior Cloning and RL with Regularization
As shown in the figure, our BC agent achieves reasonable accuracy (**TODO: Note on overfitting**).

For the RL phase, we use the pretrained BC agent as a regularizer to enhance sample efficiency. In addition to the standard PPO loss, we add a KL-divergence penalty that keeps the policy close to the BC agent during training.

While this may not be the definitive method for creating a superhuman agent, it effectively lowers the average rank against the baseline to below 2.5 (see figure). We provide this as a **minimal working example** for improving Mahjong agents.

### Usage

**Installation**
```bash
pip install -U pip && pip install -r requirements.txt
```

Data collection
```
python collect_offline_data.py dataset_path=XXX
```

Behavior cloning
```
python bc.py dataset_path=XXX save_model_path=YYY
```

RL
```
python ppo_with_reg.py seed=0 pretrained_model_path=YYY
```

### Reference

- [Yu+2025](https://arxiv.org/abs/2510.18183): We referenced this implementation for PPO components tailored to competitive multi-player imperfect-information games, specifically regarding GAE and KL regularization.
- [Abe+2023](https://arxiv.org/abs/2305.16610): Discusses regularization-based algorithms in imperfect-information games.


