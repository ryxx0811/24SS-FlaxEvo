'''Flax NNX/JAX version of the original TranslationTrainer.'''

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from transformer.WarmupLRScheduler import lr_rate


def create_masks(src: jax.Array, tgt: jax.Array, pad_token: int) -> tuple[jax.Array, jax.Array]:
    src_mask = (src != pad_token)[:, None, :]
    target_length = tgt.shape[1]
    nopeak_mask = jnp.tril(jnp.ones((target_length, target_length), dtype=bool))
    tgt_mask = (tgt != pad_token)[:, None, :] & nopeak_mask[None, :, :]
    return src_mask, tgt_mask


class TranslationTrainer:
    def __init__(
        self,
        model: nnx.Module,
        embedding_size: int,
        warmup_steps: int,
        pad_token: int = 0,
        sos_token: int = 1,
    ) -> None:
        self.model = model
        self.pad_token = pad_token
        self.sos_token = sos_token

        schedule = lambda step: lr_rate(step, embedding_size, warmup_steps)
        optimizer = optax.adam(schedule, b1=0.9, b2=0.98, eps=1e-9)
        self.optimizer = nnx.Optimizer(model, optimizer, wrt=nnx.Param)

    def compute_loss(self, y_pred: jax.Array, y: jax.Array) -> jax.Array:
        return optax.softmax_cross_entropy_with_integer_labels(y_pred, y).sum()

    def _forward_loss(
        self,
        model: nnx.Module,
        src: jax.Array,
        tgt: jax.Array,
    ) -> jax.Array:
        decoder_padding = jnp.full((tgt.shape[0], 1), self.pad_token, dtype=tgt.dtype)
        tgt = jnp.concatenate((tgt, decoder_padding), axis=-1)

        trg_input = tgt[:, :-1]
        ys = tgt[:, 1:]
        src_mask, tgt_mask = create_masks(src, trg_input, self.pad_token)
        y_pred = model(src, trg_input, src_mask, tgt_mask)

        label_mask = ys != self.pad_token
        y_pred_masked = y_pred[label_mask]
        ys_masked = ys[label_mask]
        return self.compute_loss(y_pred_masked, ys_masked)

    def training_step(self, src: jax.Array, tgt: jax.Array) -> jax.Array:
       
        self.model.train()
        loss, grads = nnx.value_and_grad(self._forward_loss)(self.model, src, tgt)
        self.optimizer.update(grads)
        return loss

    def validation_step(self, src: jax.Array, tgt: jax.Array) -> jax.Array:
     
        self.model.eval()
        return self._forward_loss(self.model, src, tgt)

    def do_prediction(self, src: jax.Array, seq_length: int) -> jax.Array:
        batch_size = src.shape[0]
        outputs = jnp.full((batch_size, seq_length), self.pad_token, dtype=jnp.int32)
        outputs = outputs.at[:, 0].set(self.sos_token)

        self.model.eval()
        src_mask = (src != self.pad_token)[:, None, :]
        e_outputs = self.model.encoder(src, src_mask)

        for index in range(1, seq_length):
            _, tgt_mask = create_masks(src, outputs, self.pad_token)
            d_output = self.model.decoder(outputs, e_outputs, src_mask, tgt_mask)
            y_pred = self.model.out(d_output)
            next_token = jnp.argmax(y_pred[:, index - 1, :], axis=-1)
            outputs = outputs.at[:, index].set(next_token)

        return outputs
