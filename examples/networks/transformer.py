import jax
import jax.numpy as jnp
import flax.linen as nn

# Initialization function
def orthogonal_init(scale: float = jnp.sqrt(2.0)):
    return nn.initializers.orthogonal(scale)

class TransformerBlock(nn.Module):
    features: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x, mask=None):
        # Attention Block (Pre-Norm)
        y = nn.LayerNorm()(x)
        
        # Mask shape adjustment for MultiHeadDotProductAttention
        # (Batch, SeqLen) -> (Batch, 1, 1, SeqLen)
        if mask is not None and mask.ndim == 2:
            mask = mask[:, None, None, :]
        
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=orthogonal_init(),
            deterministic=True
        )(y, mask=mask)
        x = x + y

        # MLP Block (Pre-Norm)
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim, kernel_init=orthogonal_init())(y)
        y = nn.relu(y)
        y = nn.Dense(self.features, kernel_init=orthogonal_init())(y)
        x = x + y
        
        return x