import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

noise_levels = np.asarray([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
model_names = ['Instance Space NN', 'Embedding Space LSTM', 'Instance Space LSTM', 'CSC Instance Space LSTM']
datasets = {
    'Timer': "results/rl_noisy/timer_treasure_noisy_results.txt",
    'Moving': "results/rl_noisy/moving_treasure_noisy_results.txt",
    'Key': "results/rl_noisy/key_treasure_noisy_results.txt",
    'Charger': "results/rl_noisy/charger_treasure_noisy_results.txt",
}


def run():
    color_cycle = ['orange', 'purple', 'blue', 'green']

    fig, axes = plt.subplots(nrows=2, ncols=len(datasets), figsize=(9, 4))
    for dataset_idx, (dataset_name, dataset_path) in enumerate(datasets.items()):
        return_axis = axes[0][dataset_idx]
        reward_axis = axes[1][dataset_idx]
        if dataset_path is not None:
            return_results, reward_results = parse_results(dataset_path)
            for model_idx in range(len(model_names)):
                plot_results(return_axis, return_results[:, model_idx, 0],
                             return_results[:, model_idx, 1], color_cycle[model_idx])
                if model_idx != 1:
                    plot_results(reward_axis, reward_results[:, model_idx, 0],
                                 reward_results[:, model_idx, 1], color_cycle[model_idx])
        return_axis.set_title(dataset_name)
        return_axis.set_xlim(0, 0.5)
        return_axis.set_xticks([0, 0.25, 0.5])
        return_axis.set_xlabel("Noise")
        return_axis.set_ylim(0.0)

        reward_axis.set_xlim(0, 0.5)
        reward_axis.set_xticks([0, 0.25, 0.5])
        reward_axis.set_xlabel("Noise")
        reward_axis.set_ylim(0.0)

    axes[0][0].set_ylabel('Return\nMSE Loss')
    axes[1][0].set_ylabel('Reward\nMSE Loss')

    custom_lines = [Line2D([0], [0], color=color_cycle[i], linestyle='--', alpha=0.5) for i in range(len(model_names))]
    fig.legend(custom_lines, model_names, loc='lower center', ncol=len(model_names), prop={'size': 8})
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    plt.show()
    fig_path = "out/fig/robustness_study.png"
    fig.savefig(fig_path, format='png', dpi=300)
    fig_path = "out/fig/robustness_study.svg"
    fig.savefig(fig_path, format='svg', dpi=300)


def parse_results(dataset_path):
    return_results = parse_table(dataset_path, 4)
    reward_results = parse_table(dataset_path, 40)
    return return_results, reward_results


def parse_table(path, offset):
    results = np.full((len(noise_levels), len(model_names), 2), np.nan)
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter='|')
        row_idx = 0
        target_idxs = np.arange(len(noise_levels)) * 2 + offset
        for row in reader:
            if row_idx in target_idxs:
                row = [s.strip() for s in row]
                for model_idx in range(len(model_names)):
                    noise_level = float(row[1])
                    try:
                        noise_idx = np.where(noise_levels == noise_level)[0]
                        results[noise_idx, model_idx] = [float(f) for f in row[model_idx + 2].split(' +- ')]
                    except ValueError:
                        pass
            row_idx += 1
    return results


def plot_results(axis, avgs, sems, color):
    valid_idxs = np.where(~np.isnan(avgs))[0]
    axis.scatter(noise_levels[valid_idxs], avgs[valid_idxs],
                 marker='x', color=color, s=5)
    axis.plot(noise_levels[valid_idxs], avgs[valid_idxs],
              linestyle='--', alpha=0.5, color=color)
    axis.fill_between(noise_levels[valid_idxs], avgs[valid_idxs] - sems[valid_idxs],
                      avgs[valid_idxs] + sems[valid_idxs], alpha=0.1, color=color)


if __name__ == "__main__":
    run()
