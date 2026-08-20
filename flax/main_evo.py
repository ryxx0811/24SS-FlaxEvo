'''Run evolutionary architecture search for the translation Transformer.'''

import json
from dataclasses import asdict
from pathlib import Path

import jax

from data.dataloader import Dataloader
from evo_search import Candidate, EvoConfig, best_candidate, regularized_evolution


def _candidate_to_dict(candidate: Candidate) -> dict:
    '''Convert a search result into a JSON-compatible dictionary.'''
    return asdict(candidate)


def _save_results(history: list[Candidate], best: Candidate) -> None:
    results_dir = Path('flax/evo_search/results')
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / 'evolution_history.json').write_text(
        json.dumps([_candidate_to_dict(candidate) for candidate in history], indent=2),
        encoding='utf-8',
    )
    (results_dir / 'best_candidate.json').write_text(
        json.dumps(_candidate_to_dict(best), indent=2),
        encoding='utf-8',
    )


def main() -> None:
    dataloader = Dataloader(dir='flax/data')
    dataloader.preparing_data()

    word2index, _ = dataloader.get_dict()
    vocab_size = len(word2index)
    dataloader.get_src()
    dataloader.get_tgt()
    dtrain, dval, _dtest = dataloader.split_data(key=jax.random.PRNGKey(0))

    # Start small to confirm the search works. Increase these values afterward.
    evo_config = EvoConfig(
        population_size=3,
        sample_size=2,
        cycles=5,
        candidate_epochs=3,
        batch_size=16,
        warmup_steps=100,
    )

    history = regularized_evolution(
        dtrain,
        dval,
        vocab_size=vocab_size,
        max_len=dataloader.max_len,
        evo_config=evo_config,
        seed=1,
    )
    best = best_candidate(history)
    _save_results(history, best)

    print('Evolution search finished.')
    print(f'Evaluated candidates: {len(history)}')
    print(f'Best validation loss: {best.validation_loss:.6f}')
    print(f'Best fitness: {best.fitness:.6f}')
    print(f'Best architecture: {best.config}')
    print(f'Mutation path: {best.mutation_path}')


if __name__ == '__main__':
    main()
