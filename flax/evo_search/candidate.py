'''The result recorded for one evaluated architecture.'''

from dataclasses import dataclass

from instance import TransformerConfig


@dataclass
class Candidate:
    '''One evaluated Transformer architecture in the population.'''

    config: TransformerConfig
    fitness: float
    validation_loss: float
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    seed: int
    mutation_path: str | None = None
