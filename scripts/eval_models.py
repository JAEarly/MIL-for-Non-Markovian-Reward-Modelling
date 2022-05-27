import argparse

import latextable
import numpy as np
from texttable import Texttable
from tqdm import tqdm

import scripts
from model import synthetic_models, rl_models
from bonfire.train import get_default_save_path, DEFAULT_SEEDS
from bonfire.util import get_device
from interpretability.oracle_interpretability import OracleMILInterpretabilityStudy


device = get_device()


def run():
    dataset_name = parse_args()
    if dataset_name in scripts.SYNTHETIC_DATASET_NAMES:
        eval_all_models(dataset_name, synthetic_models.get_model_clzs())
    elif dataset_name in scripts.RL_DATASET_NAMES:
        eval_all_models(dataset_name, rl_models.get_model_clzs())


def parse_args():
    parser = argparse.ArgumentParser(description='MIL evaluation script for the oracle datasets.')
    scripts.add_dataset_parser_arg(parser, scripts.SYNTHETIC_DATASET_NAMES + scripts.RL_DATASET_NAMES)
    args = parser.parse_args()
    return args.dataset_name


def eval_all_models(dataset_name, model_clzs):
    csv_path = scripts.get_dataset_path_from_name(dataset_name)
    dataset_clz = scripts.get_trainer_clz(dataset_name).dataset_clz

    n_repeats = 10
    all_results = []
    best_results = []

    with tqdm(total=len(model_clzs) * n_repeats, desc='Evaluating models') as pbar:
        for model_clz in model_clzs:
            model_results = np.empty((n_repeats, 6))
            for i in range(n_repeats):
                model_path, _, _ = get_default_save_path(dataset_name, model_clz.__name__, repeat=i)
                seed = DEFAULT_SEEDS[i]
                repeat_results = eval_model(model_clz, model_path, dataset_clz, dataset_name, csv_path, seed)
                model_results[i] = [r.loss for r in repeat_results]
                pbar.update()

            avgs = list(np.mean(model_results, axis=0))
            sems = list(np.std(model_results, axis=0) / np.sqrt(n_repeats))

            print('\n{:} test return variation:\n{:}\n'.format(model_clz, model_results[:, -2]))

            all_results.append((model_clz.__name__, avgs, sems))
            best_idx = np.argmin(model_results[:, -1])
            best_results.append((model_clz.__name__, model_results[best_idx], best_idx))

    all_results = sorted(all_results, key=lambda x: x[1][-1])
    rows = [['Model', 'Train Return Loss', 'Train Reward Loss', 'Val Return Loss', 'Val Reward Loss',
             'Test Return Loss', 'Test Reward Loss']]
    for row in all_results:
        formatted_row = [row[0]] + ["{:.4f} +- {:.4f}".format(row[1][i], row[2][i]) for i in range(6)]
        rows.append(formatted_row)
    table = Texttable()
    table.set_cols_dtype(['t'] * 7)
    table.set_cols_align(['c'] * 7)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    print(latextable.draw_latex(table))

    best_results = sorted(best_results, key=lambda x: x[1][-1])
    rows = [['Model', 'Train Return Loss', 'Train Reward Loss', 'Val Return Loss', 'Val Reward Loss',
             'Test Return Loss', 'Test Reward Loss', 'Best idx']]
    for row in best_results:
        formatted_row = [row[0]] + ["{:.4f}".format(row[1][i]) for i in range(6)] + [row[2]]
        rows.append(formatted_row)
    table = Texttable()
    table.set_cols_dtype(['t'] * 8)
    table.set_cols_align(['c'] * 8)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    print(latextable.draw_latex(table))


def eval_model(model_clz, model_path, dataset_clz, dataset_name, csv_path, seed):
    dataset_params = {
        'csv_path': csv_path,
    }
    study = OracleMILInterpretabilityStudy(device, dataset_clz, model_clz, model_path,
                                           dataset_name, dataset_params, seed=seed)

    return study.run_evaluation(verbose=False)


if __name__ == "__main__":
    run()
