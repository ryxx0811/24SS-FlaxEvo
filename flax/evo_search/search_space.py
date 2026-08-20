'''Valid Transformer configurations and architecture mutations.'''

from dataclasses import replace
import random

from instance import TransformerConfig


_EMBEDDING_SIZES = (32, 64, 128)
_HEAD_COUNTS = (2, 4, 8)
_LAYER_COUNTS = (1, 2, 3)
_DROPOUT_RATES = (0.0, 0.1, 0.2)


def _valid_heads(embedding_size: int) -> tuple[int, ...]:
    return tuple(heads for heads in _HEAD_COUNTS if embedding_size % heads == 0)


def _different_choice(values: tuple, current: int | float, rng: random.Random):
    choices = tuple(value for value in values if value != current)
    return current if not choices else rng.choice(choices)


def random_transformer_config(rng: random.Random) -> TransformerConfig:
    '''Create one random, valid Transformer architecture.'''
    embedding_size = rng.choice(_EMBEDDING_SIZES)
    return TransformerConfig(
        embedding_size=embedding_size,
        feed_forward_size=rng.choice((embedding_size * 2, embedding_size * 4)),
        num_heads=rng.choice(_valid_heads(embedding_size)),
        num_encoder_layers=rng.choice(_LAYER_COUNTS),
        num_decoder_layers=rng.choice(_LAYER_COUNTS),
        dropout_rate=rng.choice(_DROPOUT_RATES),
    )


def mutate_config(
    parent_config: TransformerConfig,
    rng: random.Random,
    *,
    min_mutations: int = 1,
    max_mutations: int = 2,
) -> tuple[TransformerConfig, str]:
    '''Copy a parent configuration and mutate one or two architecture fields.'''
    if not 1 <= min_mutations <= max_mutations:
        raise ValueError('mutation counts must satisfy 1 <= min_mutations <= max_mutations')

    child_config = parent_config
    mutation_paths: list[str] = []
    fields = (
        'embedding_size', 'feed_forward_size', 'num_heads',
        'num_encoder_layers', 'num_decoder_layers', 'dropout_rate',
    )

    selected_fields = rng.sample(
        fields,
        k=rng.randint(min_mutations, max_mutations),
    )

    for field in selected_fields:

        if field == 'embedding_size':
            old_value = child_config.embedding_size
            new_value = _different_choice(_EMBEDDING_SIZES, old_value, rng)
            new_heads = child_config.num_heads
            if new_value % new_heads != 0:
                new_heads = rng.choice(_valid_heads(new_value))
            child_config = replace(
                child_config,
                embedding_size=new_value,
                feed_forward_size=new_value * 4,
                num_heads=new_heads,
            )
        elif field == 'feed_forward_size':
            old_value = child_config.feed_forward_size
            new_value = _different_choice(
                (child_config.embedding_size * 2, child_config.embedding_size * 4),
                old_value,
                rng,
            )
            child_config = replace(child_config, feed_forward_size=new_value)
        elif field == 'num_heads':
            old_value = child_config.num_heads
            new_value = _different_choice(_valid_heads(child_config.embedding_size), old_value, rng)
            child_config = replace(child_config, num_heads=new_value)
        elif field == 'num_encoder_layers':
            old_value = child_config.num_encoder_layers
            new_value = _different_choice(_LAYER_COUNTS, old_value, rng)
            child_config = replace(child_config, num_encoder_layers=new_value)
        elif field == 'num_decoder_layers':
            old_value = child_config.num_decoder_layers
            new_value = _different_choice(_LAYER_COUNTS, old_value, rng)
            child_config = replace(child_config, num_decoder_layers=new_value)
        else:
            old_value = child_config.dropout_rate
            new_value = _different_choice(_DROPOUT_RATES, old_value, rng)
            child_config = replace(child_config, dropout_rate=new_value)

        mutation_paths.append(f'{field}: {old_value} -> {new_value}')

    child_config.validate()
    return child_config, '; '.join(mutation_paths)
