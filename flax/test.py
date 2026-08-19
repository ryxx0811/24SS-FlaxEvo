import flax
from flax import nnx
import jax
import jax.numpy as jnp
import os

for filename in os.listdir('flax/data'):
    if filename.endswith('.py'):
        break
    if filename.endswith('.en'):
        file_path = os.path.join('flax/data', filename)

        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = file.read().splitlines()
