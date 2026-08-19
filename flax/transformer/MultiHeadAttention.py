'''NNX multi-head scaled dot-product attention.'''

import math

from flax import nnx
import jax.numpy as jnp


class MultiHeadAttention(nnx.Module):
    def __init__(self, *, embedding_size: int, num_heads: int, dropout: nnx.Module,
                 q: nnx.Module, k: nnx.Module, v: nnx.Module, out: nnx.Module):
        
        self.embedding_size, self.num_heads = int(embedding_size), int(num_heads)
        if self.embedding_size % self.num_heads:
            raise ValueError('embedding_size must be divisible by num_heads')
        
        self.head_dim = self.embedding_size // self.num_heads
        self.q, self.k, self.v, self.out, self.dropout = q, k, v, out, dropout

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
                 mask: jnp.ndarray | None = None) -> jnp.ndarray:
        
        batch_size, query_length = q.shape[:2]
        
        def split_heads(x: jnp.ndarray) -> jnp.ndarray:
            return x.reshape(batch_size, x.shape[1], self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k, v = split_heads(self.q(q)), split_heads(self.k(k)), split_heads(self.v(v))
        scores = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            if mask.ndim == 3:
                mask = mask[:, None, :, :]
            if mask.ndim != 4:
                raise ValueError('mask must have shape (B,Q,K) or (B,H,Q,K)')
            scores = jnp.where(mask.astype(bool), scores, jnp.finfo(scores.dtype).min)
        weights = self.dropout(nnx.softmax(scores, axis=-1))
        attended = jnp.matmul(weights, v)
        combined = attended.transpose(0, 2, 1, 3).reshape(batch_size, query_length, self.embedding_size)
        return self.out(combined)
