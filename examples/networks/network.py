"""Actor-critic models over a token sequence.

This module contains NO mahjong. It takes a tokenizer class -- ``red_feature`` or
``no_red_feature``, which expose the same contract -- and reads everything env
specific off it:

    tokenizer_cls.D_MODEL         token width
    tokenizer_cls.NUM_ACTIONS     policy head width (87 red, 79 no-red)
    tokenizer_cls.CLS_POSITIONS   which sequence positions are readout heads

so the same model serves both envs.

``TransformerEncoder`` self-attends over the whole sequence. It reads the
per-segment CLS positions and CONCATENATES them: each segment's CLS answers a
different question (my hand's shape / who is open / what has been played / the
score situation), and concatenation keeps that separation into the head.
Mean-pooling them would re-mix exactly what the segmentation is for.
"""

from typing import Dict

import flax.linen as nn
import jax.numpy as jnp

try:
    from .transformer import TransformerBlock, orthogonal_init
except ImportError:
    from networks.transformer import TransformerBlock, orthogonal_init

NUM_HEADS = 4
MLP_DIM = 256
FINAL_MLP_DIM = 256
NUM_LAYERS = 4


class TransformerEncoder(nn.Module):
    tokenizer_cls: type

    @nn.compact
    def __call__(self, obs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        spec = self.tokenizer_cls
        tokens, mask = spec()(obs)
        x = tokens
        for _ in range(NUM_LAYERS):
            x = TransformerBlock(spec.D_MODEL, num_heads=NUM_HEADS, mlp_dim=MLP_DIM)(x, mask=mask)
        # TransformerBlock is PRE-norm, so its output is a raw residual stream whose
        # scale grows with depth. Without a final norm the head receives that
        # unnormalized stream, and the CLS concat multiplies the imbalance by four.
        x = nn.LayerNorm()(x)
        return jnp.concatenate([x[:, p] for p in spec.CLS_POSITIONS], axis=-1)


class ACNet(nn.Module):
    """Policy and value get INDEPENDENT encoders: they want different things out of
    the same tokens, and sharing a trunk couples their gradients."""

    tokenizer_cls: type

    def setup(self):
        self.policy_extractor = TransformerEncoder(tokenizer_cls=self.tokenizer_cls)
        self.critic_extractor = TransformerEncoder(tokenizer_cls=self.tokenizer_cls)
        self.policy_mlp = nn.Sequential([
            nn.Dense(FINAL_MLP_DIM, kernel_init=orthogonal_init()),
            nn.relu,
            nn.Dense(self.tokenizer_cls.NUM_ACTIONS, kernel_init=orthogonal_init(0.01)),
        ])
        self.value_critic_mlp = nn.Sequential([
            nn.Dense(FINAL_MLP_DIM, kernel_init=orthogonal_init()),
            nn.relu,
            nn.Dense(1, kernel_init=orthogonal_init()),
        ])

    def __call__(self, obs):
        return self.get_action_logits(obs), self.get_value(obs)

    def get_action_logits(self, obs):
        return self.policy_mlp(self.policy_extractor(obs))

    def get_value(self, obs):
        return self.value_critic_mlp(self.critic_extractor(obs)).squeeze(-1)


def make_acnet(tokenizer_cls: type) -> type:
    """Bind a tokenizer into a zero-arg constructible class.

    ``examples/common.py`` returns a CLASS that callers instantiate with no
    arguments (``NETWORK_CLS()``), so the binding has to live in the type.
    """
    # Bound under a DIFFERENT name than the dataclass field below: inside a class
    # body ``tokenizer_cls: type = tokenizer_cls`` does not close over the enclosing
    # function local -- the annotation target shadows it and the lookup fails.
    _tok = tokenizer_cls

    class BoundACNet(ACNet):
        tokenizer_cls: type = _tok

    BoundACNet.__name__ = f"{tokenizer_cls.__module__.split('.')[-1]}_ACNet"
    return BoundACNet
