import argparse
import os
import pickle as pkl

import latextable
import numpy as np
from texttable import Texttable
from tqdm import tqdm

import dataset
import scripts
from interpretability.oracle_interpretability import OracleMILInterpretabilityStudy
from model import synthetic_models, rl_models, lunar_lander_models
from pytorch_mil.train import get_default_save_path, DEFAULT_SEEDS
from pytorch_mil.util import get_device

device = get_device()


def run():
    dataset_name = parse_args()
    if dataset_name in scripts.SYNTHETIC_DATASET_NAMES:
        run_for_all_models(dataset_name, synthetic_models.get_model_clzs())
    elif dataset_name in scripts.RL_DATASET_NAMES:
        run_for_all_models(dataset_name, rl_models.get_model_clzs())
    elif dataset_name in scripts.LL_DATASET_NAMES:
        run_for_all_models(dataset_name, lunar_lander_models.get_model_clzs(), reward_scale=1e-5)
    else:
        raise "Dataset name {:s} not recognised.".format(dataset_name)


def parse_args():
    parser = argparse.ArgumentParser(description='MIL evaluation script for the oracle datasets.')
    scripts.add_dataset_parser_arg(parser, scripts.SYNTHETIC_DATASET_NAMES
                                   + scripts.RL_DATASET_NAMES
                                   + scripts.LL_DATASET_NAMES)
    args = parser.parse_args()
    return args.dataset_name


def run_for_all_models(dataset_name, model_clzs, reward_scale=1, n_repeats=10):
    results_cache_path = "results/rl/{:s}_results.pkl".format(dataset_name)

    if os.path.exists(results_cache_path):
        print('Loading results from cache')
        with open(results_cache_path, 'rb') as f:
            all_results_raw = pkl.load(f)
    else:
        print('Results cache not found. Generating now.')
        all_results_raw = eval_all_models(dataset_name, model_clzs, n_repeats, reward_scale=reward_scale)
        with open(results_cache_path, 'wb+') as f:
            pkl.dump(all_results_raw, f)

    # Parse results
    avg_results = []
    best_by_return_results = []
    best_by_reward_results = []
    top_half_by_return_results = []
    top_half_by_reward_results = []
    for model_idx, model_clz in enumerate(model_clzs):
        model_name = model_clz.__name__
        n_top_half = n_repeats // 2

        # Compute average results for this model
        mean_model_results = list(np.nanmean(all_results_raw[model_idx, :, :], axis=0))
        sem_model_results = list(np.nanstd(all_results_raw[model_idx, :, :], axis=0) / np.sqrt(n_repeats))
        avg_results.append((model_name, mean_model_results, sem_model_results))

        # Compute top 50% results for this model
        model_idxs_by_return = np.argsort(all_results_raw[model_idx, :, -2])
        model_idxs_by_reward = np.argsort(all_results_raw[model_idx, :, -1])
        print("{:s} best idxs by return: {:} -> {:}".format(
            model_name, model_idxs_by_return,
            ['{:.3f}'.format(r) for r in all_results_raw[model_idx, model_idxs_by_return, -2]])
        )
        print("{:s} best idxs by reward: {:} -> {:}".format(
            model_name, model_idxs_by_reward,
            ['{:.3f}'.format(r) for r in all_results_raw[model_idx, model_idxs_by_reward, -1]])
        )
        print("{:s} best idxs overlap: {:}\n".format(model_name,
                                                    np.intersect1d(model_idxs_by_return[:5], model_idxs_by_reward[:5])))
        #  Top 50 by return
        top_half_by_return_model_results = all_results_raw[model_idx, model_idxs_by_return[:n_top_half], :]
        mean_top_half_by_return_results = list(np.nanmean(top_half_by_return_model_results, axis=0))
        sem_top_half_by_return_results = list(np.nanstd(top_half_by_return_model_results, axis=0) / np.sqrt(n_top_half))
        top_half_by_return_results.append((model_name, mean_top_half_by_return_results, sem_top_half_by_return_results))
        #  Top 50 by reward
        top_half_by_reward_model_results = all_results_raw[model_idx, model_idxs_by_reward[:n_top_half], :]
        mean_top_half_by_return_results = list(np.nanmean(top_half_by_reward_model_results, axis=0))
        sem_top_half_by_return_results = list(np.nanstd(top_half_by_reward_model_results, axis=0) / np.sqrt(n_top_half))
        top_half_by_reward_results.append((model_name, mean_top_half_by_return_results, sem_top_half_by_return_results))

        # Compute best results for this model
        best_return_idx = model_idxs_by_return[0]
        best_reward_idx = model_idxs_by_reward[0]
        best_by_return_results.append((model_name, all_results_raw[model_idx, best_return_idx, :], best_return_idx))
        best_by_reward_results.append((model_name, all_results_raw[model_idx, best_reward_idx, :], best_reward_idx))

    print('\nAll results')
    out_avg_results(avg_results, reward_scale=reward_scale)

    print('\nTop 50% by return results')
    out_avg_results(top_half_by_return_results, reward_scale=reward_scale)

    print('\nTop 50% by reward results')
    out_avg_results(top_half_by_reward_results, reward_scale=reward_scale)

    print('\nBest return results')
    out_best_results(best_by_return_results, reward_scale=reward_scale)

    print('\nBest reward results')
    out_best_results(best_by_reward_results, reward_scale=reward_scale)


def out_avg_results(avg_results, reward_scale=1):
    # avg_results = sorted(avg_results, key=lambda x: x[1][-1])
    rows = [['Model', 'Train Return Loss', 'Train Reward Loss', 'Val Return Loss', 'Val Reward Loss',
             'Test Return Loss', 'Test Reward Loss']]
    if reward_scale != 1:
        rows[0][2::2] = [h + " ({:})".format(reward_scale) for h in rows[0][2::2]]

    for row in avg_results:
        formatted_row = [row[0]] + ["{:.4f} +- {:.4f}".format(row[1][i], row[2][i]) for i in range(6)]
        rows.append(formatted_row)
    table = Texttable()
    table.set_cols_dtype(['t'] * 7)
    table.set_cols_align(['c'] * 7)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    # print(latextable.draw_latex(table))


def out_best_results(best_results, reward_scale=1):
    # best_results = sorted(best_results, key=lambda x: x[1][-1])
    rows = [['Model', 'Train Return Loss', 'Train Reward Loss', 'Val Return Loss', 'Val Reward Loss',
             'Test Return Loss', 'Test Reward Loss', 'Best idx']]
    if reward_scale != 1:
        rows[0][2::2] = [h + " ({:})".format(reward_scale) for h in rows[0][2::2]]

    for row in best_results:
        formatted_row = [row[0]] + ["{:.4f}".format(row[1][i]) for i in range(6)] + [row[2]]
        rows.append(formatted_row)
    table = Texttable()
    table.set_cols_dtype(['t'] * 8)
    table.set_cols_align(['c'] * 8)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    # print(latextable.draw_latex(table))


def eval_all_models(dataset_name, model_clzs, n_repeats, reward_scale=1):
    csv_path = dataset.get_dataset_path_from_name(dataset_name)
    dataset_clz = scripts.get_trainer_clz(dataset_name).dataset_clz
    all_results_raw = np.full((len(model_clzs), n_repeats, 6), np.nan)

    with tqdm(total=len(model_clzs) * n_repeats, desc='Evaluating models') as pbar:
        for model_clz_idx, model_clz in enumerate(model_clzs):
            model_results = np.full((n_repeats, 6), np.nan)
            for i in range(n_repeats):
                model_path, _, _ = get_default_save_path(dataset_name, model_clz.__name__, repeat=i)
                seed = DEFAULT_SEEDS[i]
                try:
                    repeat_results = eval_model(model_clz, model_path, dataset_clz, dataset_name, csv_path, seed)
                    model_results[i] = [r.loss for r in repeat_results]
                    # Scale rewards according to reward (i.e., if they're really small, make them reasonable).
                    model_results[i, 1::2] /= reward_scale
                except FileNotFoundError:
                    print('Could not find model {:s}. Skipping.'.format(model_path))
                pbar.update()
            all_results_raw[model_clz_idx] = model_results
    return all_results_raw


def eval_model(model_clz, model_path, dataset_clz, dataset_name, csv_path, seed):
    dataset_params = {
        'csv_path': csv_path,
    }
    study = OracleMILInterpretabilityStudy(device, dataset_clz, model_clz, model_path,
                                           dataset_name, dataset_params, seed=seed)

    return study.run_evaluation(verbose=False)


if __name__ == "__main__":
    run()
