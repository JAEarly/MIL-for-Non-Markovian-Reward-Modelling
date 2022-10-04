import matplotlib as mpl
from matplotlib import pyplot as plt

import dataset
from interpretability.rl import charger_treasure_interpretability, key_treasure_interpretability, \
    timer_treasure_interpretability, moving_treasure_interpretability, lunar_lander_interpretability
from model.lunar_lander_models import LLEmbeddingSpaceLSTM, LLInstanceSpaceLSTM, LLCSCInstanceSpaceLSTM
from model.rl_models import RLEmbeddingSpaceLSTM, RLInstanceSpaceLSTM, RLCSCInstanceSpaceLSTM
from pytorch_mil.train import get_default_save_path
from pytorch_mil.util import get_device
from scripts.plots.plot_interpretability import get_best_repeat_num

mpl.rcParams.update({"svg.fonttype": "none"})

device = get_device()


def run_interpretability_study():
    print('Running global plotting script')

    models = [
        ('EmbeddingSpaceLSTM', RLEmbeddingSpaceLSTM),
        ('InstanceSpaceLSTM', RLInstanceSpaceLSTM),
        ('CSCInstanceSpaceLSTM', RLCSCInstanceSpaceLSTM),
    ]
    ll_models = [
        ('EmbeddingSpaceLSTM', LLEmbeddingSpaceLSTM),
        ('InstanceSpaceLSTM', LLInstanceSpaceLSTM),
        ('CSCInstanceSpaceLSTM', LLCSCInstanceSpaceLSTM),
    ]
    dataset_names = ['timer_treasure', 'moving_treasure', 'key_treasure', 'charger_treasure', 'lunar_lander']

    fig = plt.figure(figsize=(5, 9))
    gs = fig.add_gridspec(ncols=4, nrows=10, width_ratios=[3, 3, 3, 2.5], height_ratios=[1] * 10)
    csc_state_axes = []
    csc_plotters = []
    for model_idx, model_data in enumerate(models):
        model_name, model_clz = model_data
        for dataset_idx, dataset_name in enumerate(dataset_names):
            if dataset_name == 'lunar_lander':
                model_name, model_clz = ll_models[model_idx]

            state_axis = fig.add_subplot(gs[dataset_idx * 2, model_idx])
            main_axis = fig.add_subplot(gs[dataset_idx * 2 + 1, model_idx])

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
            plot_content_state(state_axis, plotter, plotting_data, add_inset_axis=dataset_idx == 0 and model_idx == 2)
            plot_content_main(main_axis, dataset_name, plotter, plotting_data)

            for axis in [state_axis, main_axis]:
                axis.get_xaxis().set_ticks([])
                axis.get_yaxis().set_ticks([])
                axis.set_xlabel("")
                axis.set_ylabel("")
                axis.set_title("")

            if model_idx == 2:
                csc_state_axes.append(state_axis)
                csc_plotters.append(plotter)
            print('Done')

    print('Adding legends')
    for i in range(5):
        csc_state_axis = csc_state_axes[i]
        state_leg_axis = fig.add_subplot(gs[i * 2, 3])
        main_leg_axis = fig.add_subplot(gs[i * 2 + 1, 3])
        plot_legend_state(state_leg_axis, csc_state_axis)
        plot_legend_main(main_leg_axis, dataset_names[i], csc_plotters[i])
    plt.subplots_adjust(left=0.055, right=0.98, top=0.935, bottom=0.02, wspace=0.05, hspace=0.05)

    font_size = 8

    plt.figtext(0.18, 0.96, "Embedding\nSpace LSTM", va="center", ha="center", size=font_size)
    plt.figtext(0.43, 0.96, "Instance\nSpace LSTM", va="center", ha="center", size=font_size)
    plt.figtext(0.69, 0.96, "CSC Instance\nSpace LSTM", va="center", ha="center", size=font_size)

    plt.figtext(0.03, 0.84, "Timer", va="center", ha="center", size=font_size, rotation=90)
    plt.figtext(0.03, 0.66, "Moving", va="center", ha="center", size=font_size, rotation=90)
    plt.figtext(0.03, 0.48, "Key", va="center", ha="center", size=font_size, rotation=90)
    plt.figtext(0.03, 0.29, "Charger", va="center", ha="center", size=font_size, rotation=90)
    plt.figtext(0.03, 0.11, "Lunar Lander", va="center", ha="center", size=font_size, rotation=90)

    fig_path = "out/fig/interpretability_comparison_combined.png"
    print('Saving to {:s}'.format(fig_path))
    fig.savefig(fig_path, format='png', dpi=600)
    print('Showing')
    plt.show()


def plot_content_state(axis, plotter, plotting_data, sample_rate=0.01, add_inset_axis=False):
    plotter.plot_hidden_states_vs_state(axis, plotting_data, add_legend=False, sample_rate=sample_rate,
                                        add_inset_axis=add_inset_axis)


def plot_legend_state(leg_axis, csc_axis):
    handles, labels = csc_axis.get_legend_handles_labels()
    legend = leg_axis.legend(handles, labels, loc='center', handletextpad=0.05, fontsize=5, ncol=1,
                             columnspacing=0.3, prop={'size': 7})
    for h in legend.legendHandles:
        h.set_alpha(1)
        h.set_sizes([20])
    leg_axis.set_axis_off()


def plot_content_main(axis, dataset_name, plotter, plotting_data, sample_rate=0.01):
    if dataset_name == 'timer_treasure':
        plotter.plot_hidden_states_vs_time(axis, plotting_data, add_cbar=False, sample_rate=sample_rate)
    elif dataset_name == 'moving_treasure':
        plotter.plot_hidden_states_vs_time(axis, plotting_data, add_cbar=False, sample_rate=sample_rate)
    elif dataset_name == 'key_treasure':
        plotter.plot_hidden_states_vs_pos_x(axis, plotting_data, add_cbar=False, sample_rate=sample_rate)
    elif dataset_name == 'charger_treasure':
        plotter.plot_hidden_states_vs_charge(axis, plotting_data, add_cbar=False, sample_rate=sample_rate)
    elif dataset_name == 'lunar_lander':
        plotter.plot_hidden_states_vs_time_split(axis, plotting_data, add_cbar=False, sample_rate=sample_rate)


def plot_legend_main(leg_axis, dataset_name, csc_plotter):
    leg_axis.set_axis_off()
    cbar = None
    if dataset_name == 'timer_treasure':
        cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=csc_plotter.time_color_norm),
                     ax=leg_axis, location='left', label='Time', fraction=0.9)
    elif dataset_name == 'moving_treasure':
        cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=csc_plotter.time_color_norm),
                     ax=leg_axis, location='left', label='Time', fraction=0.9)
    elif dataset_name == 'key_treasure':
        cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=csc_plotter.x_color_norm),
                     ax=leg_axis, location='left', label='X Position', fraction=0.9)
    elif dataset_name == 'charger_treasure':
        cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=csc_plotter.charge_color_norm),
                     ax=leg_axis, location='left', label='Charge', fraction=0.9)
    elif dataset_name == 'lunar_lander':
        time_on_pad_color_norm = mpl.colors.BoundaryNorm([0, 47, 48, 49, 50, 51, 52, 53, 300],
                                                         mpl.colormaps['cividis'].N)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=time_on_pad_color_norm),
                     ax=leg_axis, location='right', label='Time on Pad', fraction=0.9)
    cbar.ax.tick_params(labelsize=7)


def get_plotter_clz(dataset_name):
    if dataset_name == 'timer_treasure':
        return timer_treasure_interpretability.get_plotter_clz("global")
    elif dataset_name == 'moving_treasure':
        return moving_treasure_interpretability.get_plotter_clz("global")
    elif dataset_name == 'key_treasure':
        return key_treasure_interpretability.get_plotter_clz("global")
    elif dataset_name == 'charger_treasure':
        return charger_treasure_interpretability.get_plotter_clz("global")
    elif dataset_name == 'lunar_lander':
        return lunar_lander_interpretability.get_plotter_clz("global")
    raise ValueError('No oracle study registered for dataset {:s}'.format(dataset_name))


if __name__ == "__main__":
    run_interpretability_study()
