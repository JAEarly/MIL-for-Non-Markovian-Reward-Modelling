import argparse

import matplotlib as mpl
from matplotlib import pyplot as plt

import dataset
from interpretability.rl import charger_treasure_interpretability, key_treasure_interpretability, \
    timer_treasure_interpretability, moving_treasure_interpretability
from model.rl_models import RLEmbeddingSpaceLSTM, RLInstanceSpaceLSTM, RLCSCInstanceSpaceLSTM
from pytorch_mil.train import get_default_save_path
from pytorch_mil.util import get_device
from scripts.plots.plot_interpretability import get_best_repeat_num

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='MIL global interpretability plotting script.')
    parser.add_argument('mode', choices=['state', 'main', 'pos_x', 'pos_y'], help='The type of plot to produce.')
    args = parser.parse_args()
    return args.mode


def run_interpretability_study():
    print('Running global plotting script')
    mode = parse_args()

    models = [
        ('EmbeddingSpaceLSTM', RLEmbeddingSpaceLSTM),
        ('InstanceSpaceLSTM', RLInstanceSpaceLSTM),
        ('CSCInstanceSpaceLSTM', RLCSCInstanceSpaceLSTM),
    ]
    dataset_names = ['timer_treasure', 'moving_treasure', 'key_treasure', 'charger_treasure']

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(ncols=4, nrows=4, width_ratios=[1, 1, 1, 1], height_ratios=[3, 3, 3, 0.3])
    csc_axes = []
    csc_plotters = []
    for model_idx, model_data in enumerate(models):
        model_name, model_clz = model_data
        for dataset_idx, dataset_name in enumerate(dataset_names):
            axis = fig.add_subplot(gs[model_idx, dataset_idx])

            csv_path = dataset.get_dataset_path_from_name(dataset_name)
            plotter_clz = get_plotter_clz(dataset_name)

            repeat_idx = get_best_repeat_num(dataset_name, model_name)

            print('  Dataset: {:s}'.format(dataset_name))
            print('   Path: {:s}'.format(csv_path))
            print('  Model: {:s}'.format(model_name))
            print('   Clz: {:}'.format(model_clz))
            print('  Repeat: {:}'.format(repeat_idx))

            model_path, _, _ = get_default_save_path(dataset_name, model_clz.__name__, repeat=repeat_idx)
            plotter = plotter_clz(device, model_clz, model_path, csv_path, repeat_idx)
            plotting_data = plotter.get_all_plotting_data()

            print('Adding to plot')
            if mode == 'state':
                plot_content_state(axis, plotter, plotting_data)
            elif mode == 'main':
                plot_content_main(axis, dataset_name, plotter, plotting_data)
            elif mode == 'pos_x':
                plot_content_pos_x(axis, plotter, plotting_data)
            elif mode == 'pos_y':
                plot_content_pos_y(axis, plotter, plotting_data)

            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.set_title("")

            if model_idx == 0:
                axis.set_title(dataset_name)
            if dataset_idx == 0:
                axis.set_ylabel(model_name)
            if model_idx == 2:
                csc_axes.append(axis)
                csc_plotters.append(plotter)
            print('Done')

    print('Adding legends')
    for i in range(4):
        csc_axis = csc_axes[i]
        leg_axis = fig.add_subplot(gs[3, i])
        if mode == 'state':
            plot_legend_state(leg_axis, csc_axis)
        elif mode == 'main':
            plot_legend_main(leg_axis, csc_axis, dataset_names[i], csc_plotters[i])
        elif mode == 'pos_x':
            plot_legend_pos_x(leg_axis)
        elif mode == 'pos_y':
            plot_legend_pos_y(leg_axis)
    plt.tight_layout()

    fig_path = "out/fig/interpretability_comparison_{:s}.png".format(mode)
    print('Saving to {:s}'.format(fig_path))
    fig.savefig(fig_path, format='png', dpi=300)
    print('Showing')
    plt.show()


def plot_content_state(axis, plotter, plotting_data, sample_rate=0.01):
    plotter.plot_hidden_states_vs_state(axis, plotting_data, add_legend=False, sample_rate=sample_rate)


def plot_legend_state(leg_axis, csc_axis):
    handles, labels = csc_axis.get_legend_handles_labels()
    legend = leg_axis.legend(handles, labels, loc='center', handletextpad=0.05, fontsize=8, ncol=2,
                             columnspacing=0.3, prop={'size': 7})
    for h in legend.legendHandles:
        h.set_alpha(1)
        h.set_sizes([40])
    leg_axis.set_axis_off()


def plot_content_main(axis, dataset_name, plotter, plotting_data, sample_rate=0.01):
    if dataset_name == 'timer_treasure':
        plotter.plot_hidden_states_vs_time(axis, plotting_data, add_cbar=False, sample_rate=sample_rate)
    elif dataset_name == 'moving_treasure':
        plotter.plot_hidden_states_vs_time(axis, plotting_data, add_cbar=False, sample_rate=sample_rate)
    elif dataset_name == 'key_treasure':
        plotter.plot_hidden_states_vs_key(axis, plotting_data, add_legend=False, sample_rate=sample_rate)
    elif dataset_name == 'charger_treasure':
        plotter.plot_hidden_states_vs_charge(axis, plotting_data, add_cbar=False, sample_rate=sample_rate)


def plot_legend_main(leg_axis, csc_axis, dataset_name, csc_plotter):
    if dataset_name == 'timer_treasure':
        plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=csc_plotter.time_color_norm),
                     cax=leg_axis, orientation='horizontal', label='Time')
    elif dataset_name == 'moving_treasure':
        plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=csc_plotter.time_color_norm),
                     cax=leg_axis, orientation='horizontal', label='Time')
    elif dataset_name == 'key_treasure':
        handles, labels = csc_axis.get_legend_handles_labels()
        legend = leg_axis.legend(handles, labels, loc='center', handletextpad=0.05, fontsize=8, ncol=1,
                                 columnspacing=0.3, prop={'size': 7})
        for h in legend.legendHandles:
            h.set_alpha(1)
        leg_axis.set_axis_off()
    elif dataset_name == 'charger_treasure':
        plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=csc_plotter.charge_color_norm),
                     cax=leg_axis, orientation='horizontal', label='Charge')


def plot_content_pos_x(axis, plotter, plotting_data, sample_rate=0.01):
    plotter.plot_hidden_states_vs_pos_x(axis, plotting_data, add_cbar=False, sample_rate=sample_rate)


def plot_legend_pos_x(leg_axis):
    plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=mpl.colors.Normalize(vmin=0, vmax=1)),
                 cax=leg_axis, orientation='horizontal', label='Pos X')


def plot_content_pos_y(axis, plotter, plotting_data, sample_rate=0.01):
    plotter.plot_hidden_states_vs_pos_y(axis, plotting_data, add_cbar=False, sample_rate=sample_rate)


def plot_legend_pos_y(leg_axis):
    plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=mpl.colors.Normalize(vmin=0, vmax=1)),
                 cax=leg_axis, orientation='horizontal', label='Pos Y')


def get_plotter_clz(dataset_name):
    if dataset_name == 'timer_treasure':
        return timer_treasure_interpretability.get_plotter_clz("global")
    elif dataset_name == 'moving_treasure':
        return moving_treasure_interpretability.get_plotter_clz("global")
    elif dataset_name == 'key_treasure':
        return key_treasure_interpretability.get_plotter_clz("global")
    elif dataset_name == 'charger_treasure':
        return charger_treasure_interpretability.get_plotter_clz("global")
    raise ValueError('No oracle study registered for dataset {:s}'.format(dataset_name))


if __name__ == "__main__":
    run_interpretability_study()
