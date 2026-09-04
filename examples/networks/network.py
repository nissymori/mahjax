"""Actor-critic models over a token sequence.

This module contains NO mahjong. It takes a tokenizer class -- ``red_feature`` or
``no_red_feature``, which expose the same contract -- and reads everything env
specific off it:

    tokenizer_cls.D_MODEL         token width
    tokenizer_cls.NUM_ACTIONS     policy head width (87 red, 79 no-red)
    tokenizer_cls.CLS_POSITIONS   which sequence positions are readout heads

so the same two models serve both envs.

Two sequence models:

``TransformerEncoder``
    Self-attention over the whole sequence. O(N^2 d), so it grows quadratically if
    the history segment is lengthened.

``PerceiverEncoder``
    A small array of learned latents cross-attends to the tokens, then self-attends
    among itself. O(N L d) -- LINEAR in sequence length -- so history can grow
    without the attention bill following. At N=255, L=32 that is ~7x fewer attention
    scores, and the gap widens with N.

The transformer reads the per-segment CLS positions and CONCATENATES them: each
segment's CLS answers a different question (my hand's shape / who is open / what has
been played / the score situation), and concatenation keeps that separation into the
head. Mean-pooling them would re-mix exactly what the segmentation is for.

The Perceiver cannot do that -- no CLS *position* survives into the latent array -- so
it pools the latents and projects to the same width. The CLS tokens are still real
inputs the latents attend to, but per-segment readout is genuinely transformer-only.
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

# Transformer
NUM_LAYERS = 4

# Perceiver
NUM_LATENTS = 64
NUM_PERCEIVER_BLOCKS = 4
# Re-attending to the inputs at every block costs another cross-attention each time
# but lets later blocks ask new questions of the raw tokens. With a single
# cross-attend the latents get one look and everything after is latent-only.
CROSS_ATTEND_EVERY_BLOCK = True


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention: latents (queries) attend to inputs (keys/values)."""

    features: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, latents, inputs, mask=None):
        q = nn.LayerNorm()(latents)
        kv = nn.LayerNorm()(inputs)
        if mask is not None and mask.ndim == 2:
            # (B, S) -> (B, 1, 1, S): every latent may attend to every valid input.
            mask = mask[:, None, None, :]
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, kernel_init=orthogonal_init(), deterministic=True
        )(q, kv, mask=mask)
        latents = latents + y
        y = nn.LayerNorm()(latents)
        y = nn.Dense(self.mlp_dim, kernel_init=orthogonal_init())(y)
        y = nn.relu(y)
        y = nn.Dense(self.features, kernel_init=orthogonal_init())(y)
        return latents + y


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


class PerceiverEncoder(nn.Module):
    tokenizer_cls: type

    @nn.compact
    def __call__(self, obs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        spec = self.tokenizer_cls
        tokens, mask = spec()(obs)
        B, D = tokens.shape[0], spec.D_MODEL

        latents = self.param("latents", orthogonal_init(), (NUM_LATENTS, D))
        z = jnp.broadcast_to(latents[None], (B, NUM_LATENTS, D))

        for i in range(NUM_PERCEIVER_BLOCKS):
            if i == 0 or CROSS_ATTEND_EVERY_BLOCK:
                z = CrossAttentionBlock(D, NUM_HEADS, MLP_DIM)(z, tokens, mask=mask)
            z = TransformerBlock(D, num_heads=NUM_HEADS, mlp_dim=MLP_DIM)(z)

        z = nn.LayerNorm()(z)  # same pre-norm trunk, same reason as above
        pooled = jnp.concatenate([z.mean(axis=1), z.max(axis=1)], axis=-1)
        # Match the transformer's readout width so the heads below are identical.
        return nn.Dense(D * len(spec.CLS_POSITIONS), kernel_init=orthogonal_init())(pooled)


class ACNet(nn.Module):
    """Policy and value get INDEPENDENT encoders: they want different things out of
    the same tokens, and sharing a trunk couples their gradients."""

    tokenizer_cls: type
    encoder_cls: type = TransformerEncoder

    def setup(self):
        self.policy_extractor = self.encoder_cls(tokenizer_cls=self.tokenizer_cls)
        self.critic_extractor = self.encoder_cls(tokenizer_cls=self.tokenizer_cls)
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


def make_acnet(tokenizer_cls: type, encoder: str = "transformer") -> type:
    """Bind a tokenizer and an encoder into a zero-arg constructible class.

    ``examples/common.py`` returns a CLASS that callers instantiate with no
    arguments (``NETWORK_CLS()``), so the binding has to live in the type.
    """
    encoders = {"transformer": TransformerEncoder, "perceiver": PerceiverEncoder}
    if encoder not in encoders:
        raise ValueError(f"encoder must be one of {sorted(encoders)}, got {encoder!r}")

    # Bound under DIFFERENT names than the dataclass fields below: inside a class
    # body ``tokenizer_cls: type = tokenizer_cls`` does not close over the enclosing
    # function local -- the annotation target shadows it and the lookup fails.
    _tok, _enc = tokenizer_cls, encoders[encoder]

    class BoundACNet(ACNet):
        tokenizer_cls: type = _tok
        encoder_cls: type = _enc

    BoundACNet.__name__ = f"{tokenizer_cls.__module__.split('.')[-1]}_{encoder}_ACNet"
    return BoundACNet
