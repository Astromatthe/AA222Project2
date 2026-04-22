#
# File: hypersearch.py
#
# Searches a wide parameter space for each strategy/problem combination,
# ranks configs by feasibility rate over n_trials, saves best configs to JSON.

import json
import itertools
import os
import numpy as np
from tqdm import tqdm

from project2_py.helpers import Simple1, Simple2, Simple3
from project2_py.project2 import (
    penalty_gradient_descent,
    penalty_l_bfgs,
    augmented_lagrangian_gradient_descent,
    augmented_lagrangian_l_bfgs,
)

PROBLEMS = [Simple1, Simple2, Simple3]
N_TRIALS = 200      # seeds per config
PASS_THRESHOLD = 0.95

# ---- parameter grids ----

PENALTY_GRID = {
    'rho_init': [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 50.0],
    'rho_max':  [1e6],
    'inc':      [1.3, 1.5, 1.7, 2.0, 3.0, 5.0],
}

LINE_SEARCH_GRID = {
    'n_searches': [20],
    'step':       [0.5, 0.1, 0.01],
    'beta':       [0.5, 0.3],
    'sigma':      [1e-4],
}

LBFGS_GRID = {
    'm_max': [5, 10, 20],
}


def product_dicts(grid):
    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))


def run_strategy(strategy, config, ProblemClass, n_trials):
    feasible = 0
    for seed in range(n_trials):
        p = ProblemClass()
        np.random.seed(seed)
        x0 = p.x0()
        try:
            if strategy == 'quadratic_penalty_gradient_descent':
                x_hist, _, _, _ = penalty_gradient_descent(
                    p.f, p.g, p.c, x0, p.n, p.count, config, penalty_mode='quadratic')
            elif strategy == 'quadratic_penalty_l_bfgs':
                x_hist, _, _, _, _, _, _ = penalty_l_bfgs(
                    p.f, p.g, p.c, x0, p.n, p.count, config, penalty_mode='quadratic')
            elif strategy == 'absolute_penalty_gradient_descent':
                x_hist, _, _, _ = penalty_gradient_descent(
                    p.f, p.g, p.c, x0, p.n, p.count, config, penalty_mode='absolute')
            elif strategy == 'absolute_penalty_l_bfgs':
                x_hist, _, _, _, _, _, _ = penalty_l_bfgs(
                    p.f, p.g, p.c, x0, p.n, p.count, config, penalty_mode='absolute')
            elif strategy == 'augmented_lagrangian_gradient_descent':
                x_hist = augmented_lagrangian_gradient_descent(
                    p.f, p.g, p.c, x0, p.n, p.count, config)
            elif strategy == 'augmented_lagrangian_l_bfgs':
                x_hist = augmented_lagrangian_l_bfgs(
                    p.f, p.g, p.c, x0, p.n, p.count, config)
            else:
                return 0.0
            if p.count() > p.n:
                return 0.0  # discard: budget exceeded
            x_best = x_hist[-1]
            p._reset()
            if np.all(p.c(x_best) <= 0.0):
                feasible += 1
        except Exception:
            pass
    return feasible / n_trials


def build_configs(strategy):
    """Yield all configs for a given strategy."""
    for pen in product_dicts(PENALTY_GRID):
        for ls in product_dicts(LINE_SEARCH_GRID):
            if strategy == 'quadratic_penalty_gradient_descent':
                yield {'penalty': pen, 'line_search': ls}
            elif strategy == 'quadratic_penalty_l_bfgs':
                for lb in product_dicts(LBFGS_GRID):
                    yield {'penalty': pen, 'line_search': ls, 'l_bfgs': lb}
            elif strategy == 'absolute_penalty_gradient_descent':
                yield {'penalty': pen, 'line_search': ls}
            elif strategy == 'absolute_penalty_l_bfgs':
                for lb in product_dicts(LBFGS_GRID):
                    yield {'penalty': pen, 'line_search': ls, 'l_bfgs': lb}
            elif strategy == 'augmented_lagrangian_gradient_descent':
                yield {'penalty': pen, 'line_search': ls}
            elif strategy == 'augmented_lagrangian_l_bfgs':
                for lb in product_dicts(LBFGS_GRID):
                    yield {'penalty': pen, 'line_search': ls, 'l_bfgs': lb}
            else:
                return

def search(force=False):
    strategies = [
        'quadratic_penalty_gradient_descent',
        'quadratic_penalty_l_bfgs',
        'absolute_penalty_gradient_descent',
        'absolute_penalty_l_bfgs',
        'augmented_lagrangian_gradient_descent',
        'augmented_lagrangian_l_bfgs',
    ]

    os.makedirs('hypersearch_results', exist_ok=True)
    results = {}

    for ProblemClass in PROBLEMS:
        prob_name = ProblemClass()._prob
        results[prob_name] = {}
        print(f'\n=== {prob_name} ===')

        for strategy in strategies:
            individual_path = f'hypersearch_results/{prob_name}_{strategy}.json'
            # load existing result if it exists and --force not set, to avoid rerunning expensive configs
            if not force and os.path.exists(individual_path):
                with open(individual_path) as fh:
                    entry = json.load(fh)
                results[prob_name][strategy] = entry
                print(f'  strategy: {strategy} - skipped (result exists, use --force to rerun)')
                continue

            print(f'  strategy: {strategy}')
            configs = list(build_configs(strategy))
            print(f'  {len(configs)} configs to try')

            best_rate = -1.0
            best_config = None

            for config in tqdm(configs, desc=f'{prob_name}/{strategy}'):
                rate = run_strategy(strategy, config, ProblemClass, N_TRIALS)
                if rate > best_rate:
                    best_rate = rate
                    best_config = config

            entry = {
                'feasibility_rate': best_rate,
                'config': best_config,
                'passes': best_rate >= PASS_THRESHOLD,
            }
            results[prob_name][strategy] = entry
            print(f'  best rate: {best_rate:.3f} (pass={best_rate >= PASS_THRESHOLD})')

            with open(individual_path, 'w') as fh:
                json.dump(entry, fh, indent=2)
            print(f'  saved to {individual_path}')

    out_path = 'hypersearch_results/all_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')

    print('\n=== SUMMARY ===')
    for prob, strategies_res in results.items():
        for strategy, res in strategies_res.items():
            status = 'PASS' if res['passes'] else 'FAIL'
            print(f'{prob}/{strategy}: {res["feasibility_rate"]:.3f} [{status}]')

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Rerun all strategies even if result files already exist')
    args = parser.parse_args()
    search(force=args.force)
