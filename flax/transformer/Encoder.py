'''NNX Transformer encoder components.'''

from collections.abc import Sequence

from flax import nnx
import jax.numpy as jnp


class EncoderLayer(nnx.Module):
    def __init__(self, *, self_attention: nnx.Module, norm1: nnx.Module,
                 norm2: nnx.Module, feed_forward: nnx.Module,
                 dropout1: nnx.Module, dropout2: nnx.Module):
        self.attention = self_attention
        self.norm1 = norm1
        self.norm2 = norm2
        self.feed_forward = feed_forward
        self.dropout1 = dropout1
        self.dropout2 = dropout2

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        residual = x
        x = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask=mask)
        x = residual + self.dropout1(x)
        residual = x
        x = self.feed_forward(self.norm2(x))
        return residual + self.dropout2(x)


class Encoder(nnx.Module):
    def __init__(self, *, embedder: nnx.Module, positional_encoding: nnx.Module, layers: Sequence[nnx.Module | None], 
                 norm: nnx.Module, embedding_size: int):
        self.embedder = embedder
        self.positional_encoding = positional_encoding
        self.layers = list(layers)
        self.norm = norm
        self.embedding_size = int(embedding_size)

    def __call__(self, src: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
        x = self.embedder(src) * jnp.sqrt(jnp.asarray(self.embedding_size, dtype=jnp.float32))
        x = self.positional_encoding(x)
        for layer in self.layers:
            if layer is not None:
                x = layer(x, mask)
        return self.norm(x)
        
