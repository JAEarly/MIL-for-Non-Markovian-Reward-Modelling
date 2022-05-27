import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from overrides import overrides
from tqdm import tqdm

from dataset.rl.chargertreasure_dataset import ChargerTreasureDataset
from interpretability.oracle_interpretability import RLGlobalStudyPlotter, OracleMILInterpretabilityStudy, HeatmapMethod
from rl_training.maps import maps
from rl_training import probing
import matplotlib.gridspec as gridspec


def get_plotter_clz(mode):
    if mode == 'global':
        return ChargerTreasureGlobalPlotter
    elif mode == 'probe':
        return ChargerTreasureProbePlotter
    elif mode == 'animate':
        return ChargerTreasureAnimationPlotter
    raise ValueError('No plotter found for mode {:s}'.format(mode))


class ChargerTreasureGlobalPlotter(RLGlobalStudyPlotter):

    def __init__(self, device, model_clz, model_path, csv_path, repeat_num, seed=5):
        dataset_params = {
            'csv_path': csv_path,
        }
        study = OracleMILInterpretabilityStudy(device, ChargerTreasureDataset, model_clz, model_path,
                                               "charger_treasure", dataset_params, seed=seed)
        super().__init__(study, repeat_num)
        self.episode_length = 100
        self.reward_color_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self.charge_color_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self.x_color_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self.y_color_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self.hidden_state_color_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self.reward_heatmaps = None

    @overrides
    def generate_plotting_data(self, model, dataset):
        plotting_data = super().generate_plotting_data(model, dataset)
        plotting_data['true_pos_x'] = torch.cat([torch.as_tensor(bm['true_pos_x']) for bm in dataset.bags_metadata])
        plotting_data['true_pos_y'] = torch.cat([torch.as_tensor(bm['true_pos_y']) for bm in dataset.bags_metadata])
        plotting_data['charge'] = torch.cat([torch.as_tensor(bm['charge']) for bm in dataset.bags_metadata])
        plotting_data['in_charge'] = torch.cat([torch.as_tensor(bm['in_charge']) for bm in dataset.bags_metadata])
        plotting_data['in_treasure'] = torch.cat([torch.as_tensor(bm['in_treasure']) for bm in dataset.bags_metadata])
        return plotting_data

    def plot(self, plotting_data):
        fig, axes = plt.subplots(nrows=2, ncols=12, figsize=(14, 7))
        self.plot_return_overall(axes[0][0], plotting_data)
        self.plot_reward_overall(axes[0][1], plotting_data)
        self.plot_reward_vs_pos_xy(fig, axes[0][2], plotting_data, charge_value=0.1)
        self.plot_reward_vs_pos_xy(fig, axes[0][3], plotting_data, charge_value=0.7)
        self.plot_hidden_states_vs_state(axes[1][0], plotting_data)
        self.plot_hidden_states_vs_charge(axes[1][1], plotting_data)
        self.plot_hidden_states_vs_pos_x(axes[1][2], plotting_data)
        self.plot_hidden_states_vs_pos_y(axes[1][3], plotting_data)
        plt.tight_layout()
        return fig

    def plot_return_overall(self, axis, plotting_data):
        return_preds = plotting_data['return_preds']
        return_targets = plotting_data['return_targets']
        axis.scatter(return_targets, return_preds, alpha=0.01, marker='.')
        axis.set_xlabel('Target')
        axis.set_ylabel('Predicted')
        axis.set_title('Return Overall')

    def plot_reward_overall(self, axis, plotting_data):
        reward_preds = plotting_data['reward_preds']
        reward_targets = plotting_data['reward_targets']
        axis.scatter(reward_targets, reward_preds, alpha=0.01, marker='.')
        axis.set_xlabel('Target')
        axis.set_ylabel('Predicted')
        axis.set_title('Reward Overall')

    def plot_hidden_states_vs_state(self, axis, plotting_data, add_legend=True, sample_rate=0.01, add_starts=True,
                                    alpha=1.0, add_inset_axis=False):
        hidden_states = plotting_data['hidden_states']
        start_hidden_states = plotting_data['start_hidden_states']

        # Get different state idxs
        in_charge_idxs = torch.where(plotting_data['in_charge'] == 1)[0]
        not_in_charge_idxs = torch.where(plotting_data['in_charge'] == 0)[0]
        in_treasure_idxs = torch.where(plotting_data['in_treasure'] == 1)[0]
        not_in_treasure_idxs = torch.where(plotting_data['in_treasure'] == 0)[0]

        # Combine not in charge and not in treasure to produce normal
        normal_idxs = np.intersect1d(not_in_charge_idxs, not_in_treasure_idxs)

        # Sample list
        in_charge_idxs = self.sample_list(in_charge_idxs, sample_rate)
        in_treasure_idxs = self.sample_list(in_treasure_idxs, sample_rate)
        normal_idxs = self.sample_list(normal_idxs, sample_rate)

        axis.scatter(hidden_states[normal_idxs, 0], hidden_states[normal_idxs, 1], label='Normal',
                     marker='.', s=1, alpha=alpha)
        axis.scatter(hidden_states[in_charge_idxs, 0], hidden_states[in_charge_idxs, 1], label='Charging',
                     marker='.', s=1, alpha=alpha)
        axis.scatter(hidden_states[in_treasure_idxs, 0], hidden_states[in_treasure_idxs, 1], label='In Treasure',
                     marker='.', s=1, alpha=alpha)
        if add_starts:
            axis.scatter(start_hidden_states[:, 0], start_hidden_states[:, 1], marker='^', s=30, c='c', edgecolor='k')

        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs State')
        if add_legend:
            legend = axis.legend(loc='best', handletextpad=0.1)
            for h in legend.legendHandles:
                h.set_sizes([40])
                h.set_alpha(1)

    def plot_hidden_states_vs_charge(self, axis, plotting_data, add_cbar=True, sample_rate=1.0):
        hidden_states = plotting_data['hidden_states']
        start_hidden_states = plotting_data['start_hidden_states']
        charge = plotting_data['charge']
        random_idxs = self.get_random_idxs(len(hidden_states), sample_rate)
        axis.scatter(hidden_states[random_idxs, 0], hidden_states[random_idxs, 1], marker='.', s=1,
                     c=charge[random_idxs], cmap=mpl.colormaps['cividis'], norm=self.charge_color_norm)
        axis.scatter(start_hidden_states[:, 0], start_hidden_states[:, 1], marker='^', s=30, c='c', edgecolor='k')
        if add_cbar:
            plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=self.charge_color_norm), ax=axis)
        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs Charge')

    def plot_hidden_states_vs_pos_x(self, axis, plotting_data, sample_rate=0.1, add_cbar=True):
        hidden_states = plotting_data['hidden_states']
        xs = plotting_data['true_pos_x']
        random_idxs = self.get_random_idxs(len(hidden_states), sample_rate)
        axis.scatter(hidden_states[random_idxs, 0], hidden_states[random_idxs, 1], marker='.', alpha=0.1,
                     c=xs[random_idxs], cmap=mpl.colormaps['cividis'], norm=self.x_color_norm)
        if add_cbar:
            plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=self.x_color_norm), ax=axis)
        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs Pos X')

    def plot_hidden_states_vs_pos_y(self, axis, plotting_data, sample_rate=0.1, add_cbar=True):
        hidden_states = plotting_data['hidden_states']
        ys = plotting_data['true_pos_y']
        random_idxs = self.get_random_idxs(len(hidden_states), sample_rate)
        axis.scatter(hidden_states[random_idxs, 0], hidden_states[random_idxs, 1], marker='.', alpha=0.1,
                     c=ys[random_idxs], cmap=mpl.colormaps['cividis'], norm=self.y_color_norm)
        if add_cbar:
            plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=self.y_color_norm), ax=axis)
        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs Pos Y')

    def plot_reward_vs_pos_xy(self, fig, axis, plotting_data, charge_value, add_cbar=True, lw=3):
        # Generate all heatmaps if not already done (required for back filling values)
        if self.reward_heatmaps is None:
            self.reward_heatmaps = self._generate_reward_heatmap_values(plotting_data)

        # Get heatmap values for this charge value
        xx, yy, all_zs = self.reward_heatmaps
        charge_idx = int(charge_value * 50)
        zs = all_zs[:, :, charge_idx]

        # Heatmap plotting
        heatmap_method = HeatmapMethod.PCOLORMESH
        cmap = mpl.colormaps['cividis']
        self.plot_heatmap(axis, xx, yy, zs, cmap, self.reward_color_norm, heatmap_method)

        # Add map features
        self.add_bounding_box(axis, maps['chargertreasure']['boxes']['treasure']['coords'], 'g', lw=lw, alpha=1)
        self.add_bounding_box(axis, maps['chargertreasure']['boxes']['spawn_left']['coords'], 'm', lw=lw, alpha=1)
        self.add_bounding_box(axis, maps['chargertreasure']['boxes']['spawn_right']['coords'], 'm', lw=lw, alpha=1)
        self.add_bounding_box(axis, maps['chargertreasure']['boxes']['charge_zone']['coords'], 'gray', lw=lw,  alpha=1)

        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel('X Position')
        axis.set_ylabel('Y Position')
        axis.set_title('Reward vs Position\n(Charge = {:.2f})'.format(charge_value))

        if add_cbar:
            fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=self.reward_color_norm), ax=axis)

    def _generate_reward_heatmap_values(self, plotting_data):
        heatmap_grid_size = 50
        n_charge_values = 50

        # Get plotting data
        all_cs = plotting_data['charge']
        all_rs = plotting_data['reward_preds']
        all_xs = plotting_data['true_pos_x']
        all_ys = plotting_data['true_pos_y']

        # All heatmap values across all frames
        all_zs = torch.full((heatmap_grid_size, heatmap_grid_size, n_charge_values + 1), torch.nan)

        # Get initial heatmap values
        init_idxs = np.where(all_cs == 0)
        xx, yy, zs = self.generate_heatmap_values(all_xs[init_idxs], all_ys[init_idxs], all_rs[init_idxs],
                                                  heatmap_grid_size, prev_values=None)
        all_zs[:, :, 0] = zs

        # Iterate through the remainder of the episode, using the previous frame values to fill in missing gaps
        for idx in tqdm(range(n_charge_values), desc='Generating heatmap values', leave=False):
            charge = (idx+1) / n_charge_values
            frame_idxs = np.where(all_cs == charge)
            _, _, new_zs = self.generate_heatmap_values(all_xs[frame_idxs], all_ys[frame_idxs], all_rs[frame_idxs],
                                                        heatmap_grid_size, prev_values=all_zs[:, :, idx])
            all_zs[:, :, idx+1] = new_zs

        return xx, yy, all_zs


class ChargerTreasureProbePlotter(ChargerTreasureGlobalPlotter):

    @overrides
    def plot(self, plotting_data):
        probe_names = ["Optimal", "Under charged", "Over charged", "Challenging"]
        probes = [probing.probe("charger_treasure", name) for name in probe_names]
        probe_plotting_data = self.generate_probe_plotting_data(probes)

        hidden_state_alpha = 0.2

        def hide_axis(_axis):
            _axis.set_title('')
            _axis.set_xlabel('')
            _axis.set_ylabel('')
            _axis.get_xaxis().set_ticks([])
            _axis.get_yaxis().set_ticks([])

        charge_values = [0.0, 0.04, 0.2, 0.5, 0.75, 1.0]

        fig = plt.figure(figsize=(5.5, 3.9))
        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.1, 6])
        charge_gs = gs[0].subgridspec(nrows=1, ncols=len(charge_values) + 1,
                                      width_ratios=[1] * len(charge_values) + [0.5])
        probe_gs = gs[1].subgridspec(nrows=2, ncols=len(probes), hspace=0.1)

        for charge_idx, charge_value in enumerate(charge_values):
            axis = fig.add_subplot(charge_gs[charge_idx])
            charge = charge_values[charge_idx]
            # This is expensive to run - comment it out if you don't actually need to see the heatmaps
            self.plot_reward_vs_pos_xy(fig, axis, plotting_data, charge_value=charge, add_cbar=False, lw=1)
            hide_axis(axis)
            if charge_idx == 0:
                axis.set_xlabel("Charge: {:.2f}".format(charge), fontsize=7)
            else:
                axis.set_xlabel("{:.2f}".format(charge), fontsize=7)
            axis.set_aspect('equal')

        for probe_idx, probe in enumerate(probes):
            assert len(probe) == 100
            trajectory_axis = fig.add_subplot(probe_gs[0, probe_idx])
            hidden_state_axis = fig.add_subplot(probe_gs[1, probe_idx])
            # reward_axis = fig.add_subplot(probe_gs[2, probe_idx])

            self.plot_probe_trajectory(trajectory_axis, probe)
            self.plot_hidden_states_vs_state(hidden_state_axis, plotting_data, add_legend=False, add_starts=False,
                                             alpha=hidden_state_alpha)
            self.plot_probe_hidden_states(hidden_state_axis, probe_plotting_data[probe_idx])
            # self.plot_probe_rewards(reward_axis, probe_plotting_data[probe_idx], add_legend=probe_idx == 3)

            hide_axis(trajectory_axis)
            hide_axis(hidden_state_axis)

            trajectory_axis.set_title('{:s}\n(oracle return = {:.1f})'.format(
                probe_names[probe_idx], probe_plotting_data[probe_idx]['true_returns'][-1]),
                fontsize=7
            )

            if probe_idx == 0:
                trajectory_axis.set_ylabel('Trajectory', fontsize=10)
                hidden_state_axis.set_ylabel('Hidden State', fontsize=10)
                # reward_axis.set_ylabel('Reward', fontsize=10, labelpad=-6)

                # Add hidden state legend
                handles, labels = hidden_state_axis.get_legend_handles_labels()
                legend = hidden_state_axis.legend(handles=handles[:3], labels=labels[:3], loc='lower left', ncol=1,
                                                  handletextpad=0.01, fontsize=6)
                for h in legend.legendHandles:
                    h.set_sizes([20])
                    h.set_alpha(1)

            # else:
            #     reward_axis.set_title('')

            # Inset axis for adversarial probe hidden state plot
            if probe_idx == 3:
                zoomed_axis = hidden_state_axis.inset_axes([0.02, 0.02, 0.65, 0.39])
                self.plot_hidden_states_vs_state(zoomed_axis, plotting_data, add_legend=False, add_starts=False,
                                                 alpha=hidden_state_alpha)
                self.plot_probe_hidden_states(zoomed_axis, probe_plotting_data[probe_idx])
                x1, x2, y1, y2 = 0.09, 0.37, 0.77, 1.04
                zoomed_axis.set_xlim(x1, x2)
                zoomed_axis.set_ylim(y1, y2)
                hide_axis(zoomed_axis)
                hidden_state_axis.indicate_inset_zoom(zoomed_axis, edgecolor="k")

            # Add probe n markers
            s = 0
            n_added = 0
            for probe_action in probing.probe_actions['charger_treasure'][probe_names[probe_idx]][1:]:
                d, n = probe_action
                if d == 'N':
                    x, y = probe[s]
                    # The first one should be to the bottom right, and the second should be to the top left
                    self.plot_probe_n_marker(trajectory_axis, x, y, n + 1,
                                             0.01 if n_added == 0 else -0.02,
                                             -0.09 if n_added == 0 else 0.07)
                    n_added += 1
                s += n

        # Add reward colour bar
        cb_axis = fig.add_subplot(charge_gs[-1])
        cb = plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=self.reward_color_norm),
                          ax=cb_axis, location='right', fraction=1.0)
        cb.ax.tick_params(labelsize=6)
        cb.set_ticks([0, 1])
        cb.set_label(label='Reward', size=7)
        cb_axis.set_axis_off()

        plt.tight_layout()
        return fig

    @overrides
    def generate_probe_plotting_data(self, probes):
        all_probe_data = super().generate_probe_plotting_data(probes)

        chg_bounds = maps['chargertreasure']['boxes']['charge_zone']['coords']
        tre_bounds = maps['chargertreasure']['boxes']['treasure']['coords']
        chg_x0, chg_y0, chg_x1, chg_y1 = chg_bounds[0][0], chg_bounds[0][1], chg_bounds[1][0], chg_bounds[1][1]
        tre_x0, tre_y0, tre_x1, tre_y1 = tre_bounds[0][0], tre_bounds[0][1], tre_bounds[1][0], tre_bounds[1][1]

        for probe_idx, probe in enumerate(probes):
            probe_plotting_data = all_probe_data[probe_idx]
            returns = torch.zeros(len(probe))
            rewards = torch.zeros(len(probe))
            charges = torch.zeros(len(probe))
            r = 0
            c = 0
            for idx, instance in enumerate(probe):
                x, y = instance
                if chg_x0 <= x <= chg_x1 and chg_y0 <= y <= chg_y1:
                    if c < 1:
                        c += 0.02
                if tre_x0 <= x <= tre_x1 and tre_y0 <= y <= tre_y1:
                    r += c
                    rewards[idx] = c
                returns[idx] = r
                charges[idx] = c
            probe_plotting_data['true_returns'] = returns
            probe_plotting_data['true_rewards'] = rewards
            probe_plotting_data['charges'] = charges

        return all_probe_data

    def plot_probe_trajectory(self, axis, probe):
        marker_size = 15

        # Add map features
        self.add_bounding_box(axis, maps['chargertreasure']['boxes']['treasure']['coords'], 'orange', lw=1, alpha=1)
        self.add_bounding_box(axis, maps['chargertreasure']['boxes']['spawn_left']['coords'], 'purple', lw=1, alpha=1)
        self.add_bounding_box(axis, maps['chargertreasure']['boxes']['spawn_right']['coords'], 'purple', lw=1, alpha=1)
        self.add_bounding_box(axis, maps['chargertreasure']['boxes']['charge_zone']['coords'], 'gray', lw=1, alpha=1)

        axis.plot(probe[:, 0], probe[:, 1], linestyle='--', color='k', zorder=1, alpha=0.5, lw=0.5)
        # Start
        axis.scatter(probe[0, 0], probe[0, 1], marker='^', color='c',
                     s=marker_size, zorder=2, label='Start')
        # Middle
        axis.scatter(probe[1:-1, 0], probe[1:-1, 1], marker='.', color='b', s=marker_size, zorder=2)
        # End
        axis.scatter(probe[-1, 0], probe[-1, 1], marker='o', color='r',
                     s=marker_size, zorder=2, label='End')
        # axis.set_xlabel('Pos X')
        # axis.set_ylabel('Pos Y')

        axis.set_title('Trajectory')

    def plot_probe_n_marker(self, axis, x, y, num, pad_x, pad_y):
        x += pad_x
        y += pad_y
        axis.text(x, y, "x{:d}".format(num), c='k', fontsize=6, va='center', ha='center')

    def plot_probe_hidden_states(self, axis, probe_plotting_data):
        hidden_states = probe_plotting_data['hidden_states']
        marker_size = 5
        axis.plot(hidden_states[:, 0], hidden_states[:, 1], linestyle='--', color='k', zorder=1, lw=0.5, alpha=0.5)
        # Start
        axis.scatter(hidden_states[0, 0], hidden_states[0, 1], marker='^', color='c',
                     s=marker_size, zorder=3, label='Start')
        # Middle
        axis.scatter(hidden_states[1:-1, 0], hidden_states[1:-1, 1], marker='.', color='b', s=marker_size, zorder=2)
        # End
        axis.scatter(hidden_states[-1, 0], hidden_states[-1, 1], marker='o', color='r',
                     s=marker_size, zorder=3, label='End')
        axis.set_xlabel('')
        axis.set_ylabel('')
        axis.set_title('Hidden State')

    def plot_probe_charges(self, axis, probe_plotting_data):
        charges = probe_plotting_data['charges']
        axis.plot(range(len(charges)), charges)
        # axis.set_xlabel('Timestep')
        # axis.set_ylabel('Charge')
        axis.set_title('Charge')
        axis.set_ylim(-0.1, 1.1)

    def plot_probe_returns(self, axis, probe_plotting_data):
        pred_returns = probe_plotting_data['pred_returns']
        true_returns = probe_plotting_data['true_returns']
        axis.plot(range(len(pred_returns)), pred_returns, label='Pred', alpha=0.5, c='green')
        axis.plot(range(len(true_returns)), true_returns, label='True', alpha=0.5, c='black')
        axis.set_xlabel('Timestep', labelpad=-8, fontsize=8)
        axis.set_ylabel('Return', labelpad=-11, fontsize=8)
        axis.set_title('Return')
        axis.set_ylim(top=53)
        axis.set_xticks([0, 100])
        axis.set_yticks([0, 50])

        axis.tick_params(axis='both', labelsize=8, pad=1)

        # axis.legend(loc='upper left')
        # axis.set_ylim(-0.1, 2.3)

    def plot_probe_rewards(self, axis, probe_plotting_data, add_legend=True):
        pred_rewards = probe_plotting_data['pred_rewards']
        true_rewards = probe_plotting_data['true_rewards']
        axis.plot(range(len(pred_rewards)), pred_rewards, label='Predicted', alpha=0.5, c='green')
        axis.plot(range(len(pred_rewards)), true_rewards, label='True', alpha=0.5, c='black')
        axis.set_xlabel('Timestep', labelpad=-8, fontsize=8)
        # axis.set_ylabel('Reward', labelpad=-7, fontsize=8)
        # axis.set_title('Reward')
        if add_legend:
            axis.legend(loc='upper right', ncol=1, handletextpad=0.5, fontsize=6, columnspacing=0.3)
        axis.set_ylim(-0.1, 1.1)
        axis.set_xticks([0, 100])
        axis.set_yticks([0.0, 1.0])
        axis.tick_params(axis='both', labelsize=8, pad=1)

    @overrides
    def get_save_path(self):
        return "{:s}/interpretability_probe_{:d}.png".format(self.get_and_make_fig_dir(), self.repeat_num)


class ChargerTreasureAnimationPlotter(ChargerTreasureGlobalPlotter):

    @overrides
    def run_plotting(self, show_plots=True):
        # Get plotting data
        plotting_data = self.get_all_plotting_data()

        # Plot
        print('Plotting figure')
        self.plot(plotting_data)

    @overrides
    def plot(self, plotting_data):
        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        self.plot_animated_reward_vs_pos_xy(fig, axis, plotting_data)
        return fig

    def plot_animated_reward_vs_pos_xy(self, fig, axis, plotting_data):
        # Set animation values
        heatmap_grid_size = 50
        heatmap_method = HeatmapMethod.PCOLORMESH
        n_charge_values = 50
        cmap = mpl.colormaps['cividis']

        # Get plotting data
        all_charge = plotting_data['charge']
        all_reward_preds = plotting_data['reward_preds']
        all_xs = plotting_data['true_pos_x']
        all_ys = plotting_data['true_pos_y']

        # Heatmap is shared across animations
        heatmap = None

        # Initial frame function
        def init_func():
            global heatmap
            init_idxs = np.where(all_charge == 0)
            heatmap = self.generate_and_plot_heatmap(axis, all_xs[init_idxs], all_ys[init_idxs], all_reward_preds[init_idxs],
                                                     cmap, self.reward_color_norm, heatmap_grid_size, heatmap_method)
            # Add map features
            self.add_bounding_box(axis, maps['chargertreasure']['boxes']['treasure']['coords'], 'orange', lw=3, alpha=1)
            self.add_bounding_box(axis, maps['chargertreasure']['boxes']['spawn_left']['coords'], 'purple', lw=3,
                                  alpha=1)
            self.add_bounding_box(axis, maps['chargertreasure']['boxes']['spawn_right']['coords'], 'purple', lw=3,
                                  alpha=1)
            self.add_bounding_box(axis, maps['chargertreasure']['boxes']['charge_zone']['coords'], 'gray', lw=3,
                                  alpha=1)
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.set_xlabel('X Position')
            axis.set_ylabel('Y Position')
            axis.set_title('Reward vs Position\n(Charge = 0.00)')
            fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=self.reward_color_norm), ax=axis)

        def update(frame):
            global heatmap
            charge = frame / 50
            frame_idxs = np.where(all_charge == charge)
            xs = all_xs[frame_idxs]
            ys = all_ys[frame_idxs]
            rps = all_reward_preds[frame_idxs]
            self.animate_heatmap(heatmap, xs, ys, rps, heatmap_grid_size, heatmap_method)
            axis.set_title('Reward vs Position\n(Charge = {:.2f})'.format(charge))
            if frame == n_charge_values:
                plt.close(fig)

        anim = animation.FuncAnimation(fig, update, init_func=init_func, interval=2,
                                       frames=n_charge_values + 1, repeat=False)
        with tqdm(total=n_charge_values + 1, desc='Creating animation') as pbar:
            anim.save(self.get_save_path(),
                      writer=animation.PillowWriter(fps=10),
                      progress_callback=lambda i, n: pbar.update(1))

    @overrides
    def get_save_path(self):
        return "{:s}/interpretability_animation_{:d}.gif".format(self.get_and_make_fig_dir(), self.repeat_num)
