'''PyTorch-compatible adaptive average pooling for (batch, channels, length).'''

from flax import nnx
import jax.numpy as jnp


class AdaptiveAvgPool1d(nnx.Module):
    def __init__(self, *, output_size: int):
        self.output_size = int(output_size)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim != 3:
            raise ValueError('AdaptiveAvgPool1d expects (batch, channels, length)')
        if self.output_size < 1:
            raise ValueError('output_size must be positive')
        input_size = x.shape[-1]
        windows = []
        for index in range(self.output_size):
            start = index * input_size // self.output_size
            end = ((index + 1) * input_size + self.output_size - 1) // self.output_size
            windows.append(jnp.mean(x[..., start:end], axis=-1, keepdims=True))
        return jnp.concatenate(windows, axis=-1)
