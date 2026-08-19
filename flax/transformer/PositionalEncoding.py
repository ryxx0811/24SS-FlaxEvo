'''Parameter-free NNX sinusoidal positional encoding.'''

from flax import nnx
import jax.numpy as jnp


class PositionalEncoding(nnx.Module):
    def __init__(self, embedding_size: int, seq_length: int):
        self.embedding_size, self.seq_length = int(embedding_size), int(seq_length)
        position = jnp.arange(self.seq_length, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.embedding_size, 2, dtype=jnp.float32)
                           * (-jnp.log(10000.0) / self.embedding_size))
        encoding = jnp.zeros((self.seq_length, self.embedding_size), dtype=jnp.float32)
        encoding = encoding.at[:, 0::2].set(jnp.sin(position * div_term))
        encoding = encoding.at[:, 1::2].set(jnp.cos(position * div_term))
        self.positional_encoding = encoding[None, ...]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.shape[1] > self.seq_length:
            raise ValueError('input sequence exceeds PositionalEncoding.seq_length')
        return x + self.positional_encoding[:, :x.shape[1]].astype(x.dtype)
