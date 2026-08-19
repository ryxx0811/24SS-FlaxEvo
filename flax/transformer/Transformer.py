from flax import nnx
import jax.numpy as jnp

class Transformer(nnx.Module):
    def __init__(self, encoder: nnx.Module, decoder: nnx.Module, out: nnx.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.out = out

    def __call__(self, src: jnp.ndarray, tgt: jnp.ndarray, src_mask: jnp.ndarray | None = None, tgt_mask: jnp.ndarray | None = None) -> jnp.ndarray:
        e_outputs = self.encoder(src, src_mask)
        d_outputs = self.decoder(tgt, e_outputs, src_mask, tgt_mask)
        out = self.out(d_outputs)
        return out