import matplotlib as mpl
from matplotlib import pyplot as plt

import dataset
from interpretability.rl.lunar_lander_interpretability import LunarLanderGlobalPlotter
from model.lunar_lander_models import LLEmbeddingSpaceLSTM, LLInstanceSpaceLSTM, LLCSCInstanceSpaceLSTM
from pytorch_mil.train import get_default_save_path
from pytorch_mil.util import get_device
from scripts.plots.plot_interpretability import get_best_repeat_num

mpl.rcParams.update({"svg.fonttype": "none"})

device = get_device()


def run_interpretability_study():
    print('Running global plotting script')

    models = [
        ('EmbeddingSpaceLSTM', LLEmbeddingSpaceLSTM),
        ('InstanceSpaceLSTM', LLInstanceSpaceLSTM),
        ('CSCInstanceSpaceLSTM', LLCSCInstanceSpaceLSTM),
    ]
    model_names_nice = ['Embedding Space LSTM', 'Instance Space LSTM', 'CSC Instance Space LSTM']
    dataset_name = 'lunar_lander'

    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(ncols=4, nrows=2, width_ratios=[1, 1, 1, 0.1], height_ratios=[1, 1])
    csc_state_axes = []
    csc_plotters = []

    plotter_clz = LunarLanderGlobalPlotter

    for model_idx, model_data in enumerate(models):
        model_name, model_clz = model_data

        state_axis = fig.add_subplot(gs[0, model_idx])
        time_axis = fig.add_subplot(gs[1, model_idx])

        csv_path = dataset.get_dataset_path_from_name(dataset_name)
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
        plot_content_state(state_axis, plotter, plotting_data, add_inset_axis=False)
        plot_content_time(time_axis, dataset_name, plotter, plotting_data)

        for axis in [state_axis, time_axis]:
            axis.get_xaxis().set_ticks([])
            axis.get_yaxis().set_ticks([])
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.set_title("")

        time_axis.set_xlim(*state_axis.get_xlim())
        time_axis.set_ylim(*state_axis.get_ylim())

        state_axis.set_title(model_names_nice[model_idx], fontsize=12)
        if model_idx == 2:
            # Add legends
            state_leg_axis = fig.add_subplot(gs[0, 3])
            plot_legend_state(state_leg_axis, state_axis)
            time_leg_axis = fig.add_subplot(gs[1, 3])
            plot_legend_time(time_leg_axis)

            # Add inset axis for csc state plot
            zoomed_axis = state_axis.inset_axes([0.2, 0.6, 0.6, 0.35])
            plot_content_state(zoomed_axis, plotter, plotting_data, add_inset_axis=False)
            x1, x2, y1, y2 = 0.925, 1.005, 0.07, 0.21
            zoomed_axis.set_xlim(x1, x2)
            zoomed_axis.set_ylim(y1, y2)
            zoomed_axis.set_title('')
            zoomed_axis.set_xlabel('')
            zoomed_axis.set_ylabel('')
            zoomed_axis.get_xaxis().set_ticks([])
            zoomed_axis.get_yaxis().set_ticks([])
            state_axis.indicate_inset_zoom(zoomed_axis, edgecolor="k")

        print('Done')

    # print('Adding legends')
    # for i in range(4):
    #     csc_state_axis = csc_state_axes[i]
    #     state_leg_axis = fig.add_subplot(gs[3, i * 2])
    #     main_leg_axis = fig.add_subplot(gs[3, i * 2 + 1])
    #     plot_legend_state(state_leg_axis, csc_state_axis)
    #     plot_legend_main(main_leg_axis, dataset_names[i], csc_plotters[i])
    # plt.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.05, wspace=0.1, hspace=0.1)
    #
    # plt.figtext(0.15, 0.98, "Timer Treasure", va="center", ha="center", size=10)
    # plt.figtext(0.39, 0.98, "Moving Treasure", va="center", ha="center", size=10)
    # plt.figtext(0.62, 0.98, "Key Treasure", va="center", ha="center", size=10)
    # plt.figtext(0.87, 0.98, "Charger Treasure", va="center", ha="center", size=10)
    #
    # fig_path = "out/fig/interpretability_comparison_combined_lunar_lander.png"
    # print('Saving to {:s}'.format(fig_path))
    # fig.savefig(fig_path, format='png', dpi=300)
    # print('Showing')
    plt.show()


def plot_content_state(axis, plotter, plotting_data, sample_rate=0.01, add_inset_axis=False):
    plotter.plot_hidden_states_vs_state(axis, plotting_data, add_legend=False, sample_rate=sample_rate,
                                        add_inset_axis=add_inset_axis)


def plot_legend_state(leg_axis, csc_axis):
    handles, labels = csc_axis.get_legend_handles_labels()
    legend = leg_axis.legend(handles, labels, loc='center', handletextpad=0.05, fontsize=11, ncol=1,
                             columnspacing=0.3, prop={'size': 9})
    for h in legend.legendHandles:
        h.set_alpha(1)
        h.set_sizes([20])
    leg_axis.set_axis_off()


def plot_content_time(axis, dataset_name, plotter, plotting_data, sample_rate=0.01):
    plotter.plot_hidden_states_vs_time_split(axis, plotting_data, add_cbar=False, sample_rate=sample_rate)


def plot_legend_time(leg_axis):
    leg_axis.set_axis_off()
    # if dataset_name == 'timer_treasure':
    #     plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=csc_plotter.time_color_norm),
    #                  ax=leg_axis, location='bottom', label='Time', fraction=0.9)
    # elif dataset_name == 'moving_treasure':
    #     plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=csc_plotter.time_color_norm),
    #                  ax=leg_axis, location='bottom', label='Time', fraction=0.9)
    # elif dataset_name == 'key_treasure':
    #     plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=csc_plotter.x_color_norm),
    #                  ax=leg_axis, location='bottom', label='X Position', fraction=0.9)
    # elif dataset_name == 'charger_treasure':
    #     plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=csc_plotter.charge_color_norm),
    #                  ax=leg_axis, location='bottom', label='Charge', fraction=0.9)

    time_on_pad_color_norm = mpl.colors.BoundaryNorm([0, 47, 48, 49, 50, 51, 52, 53,
                                                      LunarLanderGlobalPlotter.time_on_pad_max],
                                                     mpl.colormaps['cividis'].N)
    plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=time_on_pad_color_norm),
                 ax=leg_axis, label='Time on Pad', fraction=0.9)


if __name__ == "__main__":
    run_interpretability_study()
