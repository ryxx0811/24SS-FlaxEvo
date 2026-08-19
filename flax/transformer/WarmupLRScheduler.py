'''Pure JAX Transformer warmup schedule.'''

import jax.numpy as jnp


def lr_rate(step_num: int | jnp.ndarray, embedding_size: int, warmup_steps: int) -> jnp.ndarray:
    if warmup_steps <= 0:
        raise ValueError('warmup_steps must be positive')
    step = jnp.maximum(jnp.asarray(step_num, dtype=jnp.float32), 1.0)
    return embedding_size ** -0.5 * jnp.minimum(step ** -0.5, step * warmup_steps ** -1.5)
