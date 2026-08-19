'''NNX layer-normalization component.'''

from flax import nnx
import jax.numpy as jnp


class Norm(nnx.Module):
    def __init__(self, embedding_size: int, *, rngs: nnx.Rngs, epsilon: float = 1e-5):
        self.embedding_size = int(embedding_size)
        self.layer_norm = nnx.LayerNorm(self.embedding_size, epsilon=epsilon, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.layer_norm(x)
