'''NNX Transformer decoder components.'''

from collections.abc import Sequence

from flax import nnx
import jax.numpy as jnp


class DecoderLayer(nnx.Module):
    def __init__(self, *, norm1: nnx.Module, norm2: nnx.Module, norm3: nnx.Module,
                 dropout1: nnx.Module, dropout2: nnx.Module, dropout3: nnx.Module,
                 self_attention: nnx.Module, cross_attention: nnx.Module,
                 feed_forward: nnx.Module):
        
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1, self.norm2, self.norm3 = norm1, norm2, norm3
        self.dropout1, self.dropout2, self.dropout3 = dropout1, dropout2, dropout3
        self.feed_forward = feed_forward

    def __call__(self, x: jnp.ndarray, e_outputs: jnp.ndarray | None,
                 src_mask: jnp.ndarray | None, tgt_mask: jnp.ndarray | None = None) -> jnp.ndarray:
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout1(self.self_attention(q=x, k=x, v=x, mask=tgt_mask))
        if e_outputs is not None:
            residual = x
            x = residual + self.dropout2(
                self.cross_attention(self.norm2(x), e_outputs, e_outputs, mask=src_mask)
            )
        residual = x
        x = self.feed_forward(self.norm3(x))
        return residual + self.dropout3(x)


class Decoder(nnx.Module):
    def __init__(self, *, embedder: nnx.Module, positional_encoding: nnx.Module,
                 norm: nnx.Module, layers: Sequence[nnx.Module | None], embedding_size: int):
        self.embedder = embedder
        self.positional_encoding = positional_encoding
        self.layers = [layer for layer in layers if layer is not None]
        self.norm = norm
        self.embedding_size = int(embedding_size)

    def __call__(self, tgt: jnp.ndarray, e_outputs: jnp.ndarray | None,
                 src_mask: jnp.ndarray | None, tgt_mask: jnp.ndarray | None = None) -> jnp.ndarray:
        x = self.embedder(tgt) * jnp.sqrt(jnp.asarray(self.embedding_size, dtype=jnp.float32))
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, e_outputs, src_mask, tgt_mask)
        return self.norm(x)
