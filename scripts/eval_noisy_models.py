import argparse
import os.path

import numpy as np
from texttable import Texttable
from tqdm import tqdm

import dataset
import scripts
from interpretability.oracle_interpretability import OracleMILInterpretabilityStudy
from model import rl_models
from pytorch_mil.train import get_default_save_path, DEFAULT_SEEDS
from pytorch_mil.util import get_device
from train_model_noisy import get_noisy_model_path

device = get_device()


def run():
    dataset_name = parse_args()
    eval_all_models(dataset_name)


def parse_args():
    parser = argparse.ArgumentParser(description='MIL evaluation script for the oracle datasets.')
    scripts.add_dataset_parser_arg(parser, scripts.RL_DATASET_NAMES)
    args = parser.parse_args()
    return args.dataset_name


def eval_all_models(dataset_name):
    model_clzs = rl_models.get_model_clzs()
    csv_path = dataset.get_dataset_path_from_name(dataset_name)
    dataset_clz = scripts.get_trainer_clz(dataset_name).dataset_clz

    n_repeats = 3
    noise_levels = [0.05, 0.15, 0.25, 0.35, 0.45]

    return_results = np.full((len(noise_levels), len(model_clzs), 2), np.nan)
    reward_results = np.full((len(noise_levels), len(model_clzs), 2), np.nan)
    with tqdm(total=len(noise_levels) * len(model_clzs) * n_repeats, desc='Evaluating models') as pbar:
        for model_idx, model_clz in enumerate(model_clzs):
            for noise_idx, noise_level in enumerate(noise_levels):
                model_return_results = np.full(n_repeats, np.nan)
                model_reward_results = np.full(n_repeats, np.nan)
                n_models_found = 0
                for repeat_idx in range(n_repeats):
                    if noise_level == 0:
                        model_path, _, _ = get_default_save_path(dataset_name, model_clz.__name__, repeat=repeat_idx)
                    else:
                        model_path, _ = get_noisy_model_path(dataset_name, model_clz.__name__, noise_level, repeat_idx)
                    if not os.path.exists(model_path):
                        print('Model not found for noise level {:.2f} repeat {:d}'.format(noise_level, repeat_idx))
                    else:
                        n_models_found += 1
                        seed = DEFAULT_SEEDS[repeat_idx]
                        repeat_results = eval_model(model_clz, model_path, dataset_clz, dataset_name, csv_path, seed)
                        model_return_results[repeat_idx] = repeat_results[4].loss
                        model_reward_results[repeat_idx] = repeat_results[5].loss
                    pbar.update()
                return_mean = np.nanmean(model_return_results)
                return_sem = np.nanstd(model_return_results, axis=0) / np.sqrt(n_models_found)
                reward_mean = np.nanmean(model_reward_results)
                reward_sem = np.nanstd(model_reward_results, axis=0) / np.sqrt(n_models_found)
                return_results[noise_idx, model_idx, :] = [return_mean, return_sem]
                reward_results[noise_idx, model_idx, :] = [reward_mean, reward_sem]

    print('-- Return Results --')
    output_results(noise_levels, model_clzs, return_results)
    print('-- Reward Results --')
    output_results(noise_levels, model_clzs, reward_results)


def output_results(noise_levels, model_clzs, results):
    rows = [['Noise Level', 'InstanceSpaceNN', 'EmbeddingSpaceLSTM', 'InstanceSpaceLSTM', 'CSCInstanceLSTM']]
    for noise_idx, noise_level in enumerate(noise_levels):
        row = [noise_level]
        for model_idx in range(len(model_clzs)):
            row.append("{:.4f} +- {:.4f}".format(results[noise_idx, model_idx, 0], results[noise_idx, model_idx, 1]))
        rows.append(row)
    table = Texttable()
    table.set_cols_dtype(['t'] * 5)
    table.set_cols_align(['c'] * 5)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())


def eval_model(model_clz, model_path, dataset_clz, dataset_name, csv_path, seed):
    dataset_params = {
        'csv_path': csv_path,
    }
    study = OracleMILInterpretabilityStudy(device, dataset_clz, model_clz, model_path,
                                           dataset_name, dataset_params, seed=seed)
    return study.run_evaluation(verbose=False)


if __name__ == "__main__":
    run()
