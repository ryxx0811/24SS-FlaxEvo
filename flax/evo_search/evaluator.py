'''Build, train, and score a single architecture candidate.'''

from collections.abc import Iterator

import jax
import jax.numpy as jnp
from flax import nnx

from instance import TransformerConfig, build_transformer
from trainer import TranslationTrainer
from .candidate import Candidate
from .config import EvoConfig


def _batches(
    src: jax.Array,
    tgt: jax.Array,
    batch_size: int,
    *,
    key: jax.Array | None = None,
) -> Iterator[tuple[jax.Array, jax.Array]]:
    '''Yield full source/target batches, optionally in a shuffled order.'''
    full_batch_count = len(src) // batch_size
    if full_batch_count == 0:
        raise ValueError('dataset must contain at least one full batch')

    indices = jnp.arange(len(src)) if key is None else jax.random.permutation(key, len(src))
    for batch_index in range(full_batch_count):
        start = batch_index * batch_size
        batch_ids = indices[start:start + batch_size]
        yield src[batch_ids], tgt[batch_ids]


def _mean_validation_loss(
    trainer: TranslationTrainer,
    dval: tuple[jax.Array, jax.Array],
    batch_size: int,
) -> float:
    losses = [
        float(trainer.validation_step(src_batch, tgt_batch))
        for src_batch, tgt_batch in _batches(dval[0], dval[1], batch_size)
    ]
    return sum(losses) / len(losses)


def _weighted_fitness(losses: list[float], gamma: float) -> float:
    '''Return higher-is-better fitness from the original project's losses.'''
    weighted_loss = sum(
        gamma ** (len(losses) - index - 1) * loss
        for index, loss in enumerate(losses)
    )
    return -weighted_loss


def evaluate_candidate(
    config: TransformerConfig,
    dtrain: tuple[jax.Array, jax.Array],
    dval: tuple[jax.Array, jax.Array],
    *,
    vocab_size: int,
    max_len: int,
    evo_config: EvoConfig,
    seed: int,
    mutation_path: str | None = None,
) -> Candidate:
    '''Build, train, and validate one fresh Transformer architecture.'''
    evo_config.validate()
    config.validate()
    model = build_transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        max_len=max_len,
        config=config,
        rngs=nnx.Rngs(params=seed, dropout=seed + 1),
    )
    trainer = TranslationTrainer(
        model=model,
        embedding_size=config.embedding_size,
        warmup_steps=evo_config.warmup_steps,
    )

    key = jax.random.PRNGKey(seed + 2)
    train_losses: list[float] = []
    validation_losses: list[float] = []
    for _ in range(evo_config.candidate_epochs):
        key, train_key = jax.random.split(key)
        epoch_train_losses = [
            float(trainer.training_step(src_batch, tgt_batch))
            for src_batch, tgt_batch in _batches(
                dtrain[0], dtrain[1], evo_config.batch_size, key=train_key
            )
        ]
        train_losses.append(sum(epoch_train_losses) / len(epoch_train_losses))
        validation_losses.append(
            _mean_validation_loss(trainer, dval, evo_config.batch_size)
        )

    return Candidate(
        config=config,
        fitness=_weighted_fitness(validation_losses, evo_config.fitness_gamma),
        validation_loss=validation_losses[-1],
        train_losses=tuple(train_losses),
        validation_losses=tuple(validation_losses),
        seed=seed,
        mutation_path=mutation_path,
    )
