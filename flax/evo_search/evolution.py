'''Regularized (aging) evolution search orchestration.

The architecture search space lives in ``search_space.py`` and training one
candidate lives in ``evaluator.py``. This file only controls population flow.
'''

from collections import deque
import random

import jax

from .candidate import Candidate
from .config import EvoConfig
from .evaluator import evaluate_candidate
from .search_space import mutate_config, random_transformer_config


def _candidate_summary(candidate: Candidate) -> str:
    '''Create one short line describing an evaluated candidate.'''
    return (
        f'validation_loss={candidate.validation_loss:.6f}, '
        f'fitness={candidate.fitness:.6f}, '
        f'config={candidate.config}'
    )


def _print_population(population: deque[Candidate]) -> None:
    '''Print active candidates from oldest to newest.'''
    print('Active population, oldest -> newest:')
    for index, candidate in enumerate(population, start=1):
        print(f'  {index}. {_candidate_summary(candidate)}')


def choose_parent(
    population: deque[Candidate],
    sample_size: int,
    rng: random.Random,
) -> Candidate:
    '''Select the highest-fitness candidate from a random tournament sample.'''
    if not 1 <= sample_size <= len(population):
        raise ValueError('sample_size must be between 1 and the population size')
    return max(rng.sample(list(population), sample_size), key=lambda candidate: candidate.fitness)


def regularized_evolution(
    dtrain: tuple[jax.Array, jax.Array],
    dval: tuple[jax.Array, jax.Array],
    *,
    vocab_size: int,
    max_len: int,
    evo_config: EvoConfig = EvoConfig(),
    seed: int = 0,
) -> list[Candidate]:
    '''Run original-style aging evolution and return every evaluated candidate.'''
    evo_config.validate()
    rng = random.Random(seed)
    population: deque[Candidate] = deque()
    history: list[Candidate] = []

    for candidate_index in range(evo_config.population_size):
        print(
            f'Initial candidate '
            f'{candidate_index + 1}/{evo_config.population_size}...'
        )
        candidate = evaluate_candidate(
            random_transformer_config(rng),
            dtrain,
            dval,
            vocab_size=vocab_size,
            max_len=max_len,
            evo_config=evo_config,
            seed=seed + candidate_index * 10,
            mutation_path='initial architecture',
        )
        population.append(candidate)
        history.append(candidate)
        print(f'  {_candidate_summary(candidate)}')

    for cycle in range(evo_config.cycles):
        print(f'\nEvolution cycle {cycle + 1}/{evo_config.cycles}')
        parent = choose_parent(population, evo_config.sample_size, rng)
        print(f'Selected parent: {_candidate_summary(parent)}')
        child_config, mutation_path = mutate_config(parent.config, rng)
        print(f'Mutation: {mutation_path}')
        child = evaluate_candidate(
            child_config,
            dtrain,
            dval,
            vocab_size=vocab_size,
            max_len=max_len,
            evo_config=evo_config,
            seed=seed + (evo_config.population_size + cycle) * 10,
            mutation_path=mutation_path,
        )
        population.append(child)
        history.append(child)
        print(f'Child result: {_candidate_summary(child)}')

        removed = population.popleft()
        print(f'Removed oldest: {_candidate_summary(removed)}')
        _print_population(population)
        print(f'Best so far: {_candidate_summary(best_candidate(history))}')

    return history


def best_candidate(history: list[Candidate]) -> Candidate:
    '''Return the candidate with the highest original-style fitness.'''
    if not history:
        raise ValueError('history cannot be empty')
    return max(history, key=lambda candidate: candidate.fitness)


__all__ = [
    'Candidate',
    'EvoConfig',
    'best_candidate',
    'choose_parent',
    'evaluate_candidate',
    'mutate_config',
    'random_transformer_config',
    'regularized_evolution',
]
