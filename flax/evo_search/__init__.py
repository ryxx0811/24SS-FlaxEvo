'''Evolutionary architecture search for the Flax Transformer.'''

from .candidate import Candidate
from .config import EvoConfig
from .evaluator import evaluate_candidate
from .evolution import best_candidate, choose_parent, regularized_evolution
from .search_space import mutate_config, random_transformer_config

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
