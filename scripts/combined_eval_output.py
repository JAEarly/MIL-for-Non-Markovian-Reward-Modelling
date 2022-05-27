import copy
import csv

import latextable
import numpy as np
from texttable import Texttable


def run_synthetic():
    results_files = {
        'Toggle Switch': "results/synthetic/toggle_switch_oracle_results.txt",
        'Push Switch': "results/synthetic/push_switch_oracle_results.txt",
        'Dial': "results/synthetic/dial_oracle_results.txt",
    }
    model_names = {
        'OracleInstanceSpaceNN': 'Instance Space NN',
        'OracleEmbeddingSpaceLSTM': 'Embedding Space LSTM',
        'OracleInstanceSpaceLSTM': 'Instance Space LSTM',
        'OracleCSCInstanceSpaceLSTM': 'CSC Instance Space LSTM',
    }
    run(results_files, model_names, list(results_files.keys()))


def run_rl():
    results_files = {
        'Timer Treasure': "results/rl/timer_treasure_results.txt",
        'Moving Treasure': "results/rl/moving_treasure_results.txt",
        'Key Treasure': "results/rl/key_treasure_results.txt",
        'Charger Treasure': "results/rl/charger_treasure_results.txt",
    }
    model_names = {
        'RLInstanceSpaceNN': 'Instance Space NN',
        'RLEmbeddingSpaceLSTM': 'Embedding Space LSTM',
        'RLInstanceSpaceLSTM': 'Instance Space LSTM',
        'RLCSCInstanceSpaceLSTM': 'CSC Instance Space LSTM',
    }
    run(results_files, model_names, list(results_files.keys()))


def run(results_files, model_names, datasets):
    parsed_results = {}
    for dataset_name, path in results_files.items():
        results = parse_result_file(path)
        parsed_results[dataset_name] = results
    output_return_results(parsed_results, model_names, datasets, "Synthetic dataset return results.")
    output_reward_results(parsed_results, model_names, datasets, "Synthetic dataset reward results.")


def output_return_results(parsed_results, model_names, datasets, caption):
    results_to_val_func = lambda r: r[-2]
    output_results(parsed_results, results_to_val_func, model_names, datasets, caption)


def output_reward_results(parsed_results, model_names, datasets, caption):
    results_to_val_func = lambda r: r[-1]
    output_results(parsed_results, results_to_val_func, model_names, datasets, caption)


def output_results(parsed_results, results_to_val_func, model_names, datasets, caption):
    header = ['Model'] + datasets + ['Overall']
    rows = [header]
    scores = np.zeros((len(model_names), len(datasets) + 1))
    for model_idx, model_name in enumerate(model_names.keys()):
        # Get nice model name
        row = [model_names[model_name]]
        for dataset_idx, dataset in enumerate(datasets):
            mean, std = results_to_val_func(parsed_results[dataset][model_name])
            row.append('{:.3f} +- {:.3f}'.format(mean, std))
            scores[model_idx, dataset_idx] = float('{:.3f}'.format(mean))
        scores[model_idx, -1] = scores[model_idx, :-1].mean()
        row.append('{:.3f}'.format(scores[model_idx, -1]))
        rows.append(row)

    latex_rows = copy.deepcopy(rows)

    row_idxs, col_idxs = np.where(scores == scores.min(axis=0))
    for row_idx, col_idx in zip(row_idxs, col_idxs):
        val = rows[row_idx + 1][col_idx + 1]
        rows[row_idx + 1][col_idx + 1] = "*" + val + "*"
        latex_rows[row_idx + 1][col_idx + 1] = "\\textbf{" + val + "}"

    for row_idx in range(len(model_names) + 1):
        for col_idx in range(len(header)):
            latex_rows[row_idx][col_idx] = latex_rows[row_idx][col_idx].replace("+-", "$\pm$")

    table = Texttable()
    table.set_cols_align(['c'] * len(header))
    table.set_max_width(0)
    table.add_rows(rows)
    print(table.draw())

    latex_table = Texttable()
    latex_table.set_cols_align(['l'] * len(header))
    latex_table.set_max_width(0)
    latex_table.add_rows(latex_rows)
    print(latextable.draw_latex(latex_table, use_booktabs=True, caption=caption))


def parse_result_file(path):
    results = {}
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='|')
        for idx in range(10):
            row = next(reader)
            if idx in [3, 5, 7, 9]:
                row = [s.strip() for s in row]
                model_name = row[1]
                parsed_row_results = []
                for i in range(6):
                    mean, std = [float(f) for f in row[i + 2].split(' +- ')]
                    parsed_row_results.append((mean, std))
                results[model_name] = parsed_row_results
    return results


if __name__ == "__main__":
    # run_synthetic()
    run_rl()
