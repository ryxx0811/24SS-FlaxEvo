'''NNX token embedding component.'''

from flax import nnx
import jax.numpy as jnp


class Embedder(nnx.Module):
    def __init__(self, input_size: int, output_size: int, *, rngs: nnx.Rngs):
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.embed = nnx.Embed(self.input_size, self.output_size, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.embed(x)
        
