'''NNX dense projection component.'''

from flax import nnx
import jax.numpy as jnp


class Linear(nnx.Module):
    def __init__(self, *, input_size: int, output_size: int, rngs: nnx.Rngs):
        self.input_size, self.output_size = int(input_size), int(output_size)
        self.linear = nnx.Linear(self.input_size, self.output_size, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(x)
