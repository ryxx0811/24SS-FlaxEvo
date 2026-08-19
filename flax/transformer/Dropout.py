'''NNX dropout wrapper matching the original component API.'''

from flax import nnx
import jax.numpy as jnp


class Dropout(nnx.Module):
    def __init__(self, *, p_dropout: float, rngs: nnx.Rngs):
        self.p_dropout = float(p_dropout)
        self.dropout = nnx.Dropout(self.p_dropout, rngs=rngs)

    def __call__(self, x: jnp.ndarray, *, deterministic: bool | None = None) -> jnp.ndarray:
        return self.dropout(x, deterministic=deterministic)
