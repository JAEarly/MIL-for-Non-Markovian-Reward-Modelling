import pickle as pkl

import numpy as np
import torch
# from texttable import Texttable
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 6})


def run():
    n_repeats = 10

    models = [
        ("LLEmbeddingSpaceLSTM", 1, 50, 0.11),
        ("LLInstanceSpaceLSTM", 2, 80, None),
        ("LLCSCInstanceSpaceLSTM", 3, 75, 0.02)
    ]

    fig, axes = plt.subplots(nrows=3, ncols=len(models), figsize=(6, 5.5))
    for model_name, model_idx, neg_pred_sig_mid, std_sig_mid in models:
        if model_idx != 3:
            continue

        test_return_losses, pred_stats = get_stats_for_model_clz(model_name, model_idx, n_repeats)

        print('Fitting neg pred sigmoid for {:s}'.format(model_name))
        neg_pred_x_sigmoid, neg_pred_y_sigmoid = fit_sigmoid([p[0] for p in pred_stats], test_return_losses,
                                                             sig_mid=neg_pred_sig_mid)
        print('Fitting neg pred vs std linear for {:s}'.format(model_name))
        neg_pred_vs_std_x_linear, neg_pred_vs_std_y_linear = fit_linear([p[0] for p in pred_stats],
                                                                        [p[1] for p in pred_stats])
        print('Fitting std sigmoid for {:s}'.format(model_name))
        if model_idx != 3:
            std_x_sigmoid, std_pred_y_sigmoid = fit_sigmoid([p[1] for p in pred_stats], test_return_losses,
                                                            sig_mid=std_sig_mid, sig_k=1000)
        else:
            # Remove outliers
            xs = [p[1] for p in pred_stats]
            ys = test_return_losses
            true_xs = []
            true_ys = []
            for x, y in zip(xs, ys):
                if x > 0.015 and y > 1.5 or x < 0.015 and y < 1.5:
                    true_xs.append(x)
                    true_ys.append(y)
            std_x_sigmoid, std_pred_y_sigmoid = fit_sigmoid(true_xs, true_ys, sig_mid=std_sig_mid, sig_k=1000)

        axes[model_idx - 1][0].scatter([p[0] for p in pred_stats], test_return_losses, marker='x')
        x_lims = axes[model_idx - 1][0].get_xlim()
        axes[model_idx - 1][0].plot(neg_pred_x_sigmoid, neg_pred_y_sigmoid,
                                    linewidth='1', linestyle='--', alpha=0.5)
        axes[model_idx - 1][0].set_xlim(*x_lims)
        axes[model_idx - 1][0].set_xlabel('Reward Prediction Negative %')
        axes[model_idx - 1][0].set_ylabel('Test Return Loss')
        axes[model_idx - 1][0].set_ylim(0, 3)

        axes[model_idx - 1][1].scatter([p[0] for p in pred_stats], [p[1] for p in pred_stats], marker='x')
        x_lims = axes[model_idx - 1][1].get_xlim()
        y_lims = axes[model_idx - 1][1].get_ylim()
        axes[model_idx - 1][1].plot(neg_pred_vs_std_x_linear, neg_pred_vs_std_y_linear,
                                    linewidth='1', linestyle='--', alpha=0.5)
        axes[model_idx - 1][1].set_xlim(*x_lims)
        axes[model_idx - 1][1].set_ylim(*y_lims)
        axes[model_idx - 1][1].set_xlabel('Reward Prediction Negative %')
        axes[model_idx - 1][1].set_ylabel('Reward Prediction Std')
        axes[model_idx - 1][1].set_title(model_name[2:])

        axes[model_idx - 1][2].scatter([p[1] for p in pred_stats], test_return_losses, marker='x')
        x_lims = axes[model_idx - 1][2].get_xlim()
        axes[model_idx - 1][2].plot(std_x_sigmoid, std_pred_y_sigmoid,
                                    linewidth='1', linestyle='--', alpha=0.5)
        axes[model_idx - 1][2].set_xlim(*x_lims)
        axes[model_idx - 1][2].set_xlabel('Reward Prediction Std')
        axes[model_idx - 1][2].set_ylabel('Test Return Loss')
        axes[model_idx - 1][2].set_ylim(0, 3)

        # axes[model_idx][2].legend(loc='best')

        # Output figure
        # for idx in range(5):
        #     axis = axes[idx // 3][idx % 3]
        #     xs = [float(p[idx + 2]) for p in pred_stats]
        #     ys = [float(p[1]) for p in pred_stats]
        #     axis.scatter(xs, ys, marker='x')
        #     axis.set_xlabel(header[idx + 2])
        #     axis.set_ylabel("Test Return Loss")
        #     # axis.set_xlim(x_lims[idx][0], x_lims[idx][1])
        #     # if reward:
        #     #     axis.set_ylim(0, 70)
        #     # else:
        #     #     axis.set_ylim(0, 70)
        # axes[1][2].scatter([0], [0], marker='x', label=model_name[2:])
        # axes[1][2].legend(loc='center')
        # axes[1][2].axis('off')

    plt.tight_layout()
    plt.show()

    # fig, axis = plt.subplots(nrows=1, ncols=1)
    # for model_name, model_idx in models:
    #     pred_stats = get_stats_for_model_clz(model_name, model_idx, n_repeats, reward=reward)
    #     axis.scatter([float(p[5]) for p in pred_stats], [float(p[6]) for p in pred_stats], marker='x')
    # axis.set_xlabel(header[5])
    # axis.set_ylabel(header[6])
    # plt.tight_layout()
    # plt.show()


def get_stats_for_model_clz(model_name, model_idx, n_repeats):
    with open("results/rl/lunar_lander_results.pkl", 'rb') as f:
        all_results_raw = pkl.load(f)

    test_return_losses = all_results_raw[model_idx, :, -2]

    pred_stats = []
    for i in range(n_repeats):
        plotting_data = load_plotting_data(model_name, i)
        preds = plotting_data['reward_preds']
        repeat_pred_stats = get_pred_stats(preds)
        pred_stats.append(repeat_pred_stats)

    return test_return_losses, pred_stats


def get_pred_stats(preds, eta=0):
    pred_lt_eta_prop = len(torch.where(preds < -eta)[0]) / len(preds) * 100
    pred_std = preds.std().item()
    return pred_lt_eta_prop, pred_std


def load_plotting_data(model_name, repeat_num):
    data_path = "out/fig/lunar_lander/{:s}/interpretability_global_{:d}.data".format(model_name, repeat_num)
    with open(data_path, 'rb') as f:
        plotting_data = pkl.load(f)
    return plotting_data


def fit_sigmoid(x_data, y_data, sig_mid=None, sig_k=1):

    def sigmoid(x, l, x0, k, b):
        y = l / (1 + np.exp(-k * (x - x0))) + b
        return y

    p0 = [max(y_data), sig_mid if sig_mid else np.median(x_data), sig_k, min(y_data)]
    popt, pcov = curve_fit(sigmoid, x_data, y_data, p0, method='dogbox', maxfev=100000)  #lm trf

    x_sigmoid = np.linspace(min(x_data) * 0.8, max(x_data) * 1.2, 1000)
    y_sigmoid = sigmoid(x_sigmoid, *popt)

    return x_sigmoid, y_sigmoid


def fit_linear(x_data, y_data):
    theta = np.polyfit(x_data, y_data, 1)
    x_linear = np.linspace(min(x_data) * 0.8, max(x_data) * 1.2, 1000)
    y_linear = theta[1] + theta[0] * x_linear
    return x_linear, y_linear


if __name__ == "__main__":
    run()
