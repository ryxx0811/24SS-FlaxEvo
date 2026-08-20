'''Settings for regularized evolutionary architecture search.'''

from dataclasses import dataclass


@dataclass(frozen=True)
class EvoConfig:
    '''Settings controlling one regularized-evolution search.'''

    population_size: int = 4
    sample_size: int = 3
    cycles: int = 20
    candidate_epochs: int = 3
    batch_size: int = 16
    warmup_steps: int = 100
    fitness_gamma: float = 0.9

    def validate(self) -> None:
        if self.population_size < 1:
            raise ValueError('population_size must be positive')
        if not 1 <= self.sample_size <= self.population_size:
            raise ValueError('sample_size must be between 1 and population_size')
        if self.cycles < 0:
            raise ValueError('cycles cannot be negative')
        if self.candidate_epochs < 1:
            raise ValueError('candidate_epochs must be positive')
        if self.batch_size < 1:
            raise ValueError('batch_size must be positive')
        if not 0.0 < self.fitness_gamma <= 1.0:
            raise ValueError('fitness_gamma must be in (0, 1]')
