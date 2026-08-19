'''NNX position-wise Transformer feed-forward network.'''

from flax import nnx
import jax.numpy as jnp


class FeedForward(nnx.Module):
    def __init__(self, embedding_size: int, feed_forward_size: int,
                 dropout: nnx.Module | None, *, rngs: nnx.Rngs):
        self.embedding_size, self.feed_forward_size = int(embedding_size), int(feed_forward_size)
        self.linear1 = nnx.Linear(self.embedding_size, self.feed_forward_size, rngs=rngs)
        self.dropout = dropout
        self.linear2 = nnx.Linear(self.feed_forward_size, self.embedding_size, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # The source accepted dropout but did not apply it; preserve that behavior.
        return self.linear2(nnx.relu(self.linear1(x)))
