import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from overrides import overrides
from tqdm import tqdm

from dataset.rl.timertreasure_dataset import TimerTreasureDataset
from interpretability.oracle_interpretability import RLGlobalStudyPlotter, OracleMILInterpretabilityStudy, HeatmapMethod
from rl_training import probing
from rl_training.maps import maps


def get_plotter_clz(mode):
    if mode == 'global':
        return TimerTreasureGlobalPlotter
    elif mode == 'probe':
        return TimerTreasureProbePlotter
    raise ValueError('No plotter found for mode {:s}'.format(mode))


class TimerTreasureGlobalPlotter(RLGlobalStudyPlotter):

    def __init__(self, device, model_clz, model_path, csv_path, repeat_num, seed=5):
        dataset_params = {
            'csv_path': csv_path,
        }
        study = OracleMILInterpretabilityStudy(device, TimerTreasureDataset, model_clz, model_path,
                                               "timer_treasure", dataset_params, seed=seed)
        super().__init__(study, repeat_num)
        self.episode_length = 100
        self.reward_color_norm = mpl.colors.Normalize(vmin=-1, vmax=1)
        self.time_color_norm = mpl.colors.Normalize(vmin=0, vmax=100)
        self.x_color_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self.y_color_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self.reward_heatmaps = None

    @overrides
    def generate_plotting_data(self, model, dataset):
        plotting_data = super().generate_plotting_data(model, dataset)
        plotting_data['true_pos_x'] = torch.cat([torch.as_tensor(bm['true_pos_x']) for bm in dataset.bags_metadata])
        plotting_data['true_pos_y'] = torch.cat([torch.as_tensor(bm['true_pos_y']) for bm in dataset.bags_metadata])
        plotting_data['in_treasure'] = torch.cat([torch.as_tensor(bm['in_treasure']) for bm in dataset.bags_metadata])
        return plotting_data

    def plot(self, plotting_data):
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(14, 7))
        self.plot_return_overall(axes[0][0], plotting_data)
        self.plot_reward_overall(axes[0][1], plotting_data)
        self.plot_reward_vs_pos_xy(axes[0][2], plotting_data, before_timer=True)
        self.plot_reward_vs_pos_xy(axes[0][3], plotting_data, before_timer=False)
        self.plot_hidden_states_vs_state(axes[1][0], plotting_data, add_legend=True)
        self.plot_hidden_states_vs_time(axes[1][1], plotting_data)
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

    def plot_hidden_states_vs_state(self, axis, plotting_data, add_legend=True, add_inset_axis=False, sample_rate=0.1,
                                    add_starts=True, alpha=1.0):
        hidden_states = plotting_data['hidden_states']
        start_hidden_states = plotting_data['start_hidden_states']

        # Get timer and treasure states
        time_idxs = np.arange(len(hidden_states)) % self.episode_length
        before_idxs = np.where(time_idxs < 50)
        after_idxs = np.where(time_idxs >= 50)
        no_treasure_idxs = torch.where(plotting_data['in_treasure'] == 0)[0]
        treasure_idxs = torch.where(plotting_data['in_treasure'] == 1)[0]

        # Intersect timer and treasure states
        before_no_treasure_idxs = np.intersect1d(before_idxs, no_treasure_idxs)
        after_no_treasure_idxs = np.intersect1d(after_idxs, no_treasure_idxs)
        before_treasure_idxs = np.intersect1d(before_idxs, treasure_idxs)
        after_treasure_idxs = np.intersect1d(after_idxs, treasure_idxs)

        # Filter to reduced number of idxs to save on plotting time
        before_no_treasure_idxs = self.sample_list(before_no_treasure_idxs, sample_rate)
        after_no_treasure_idxs = self.sample_list(after_no_treasure_idxs, sample_rate)
        before_treasure_idxs = self.sample_list(before_treasure_idxs, sample_rate)
        after_treasure_idxs = self.sample_list(after_treasure_idxs, sample_rate)

        axis.scatter(hidden_states[before_no_treasure_idxs, 0], hidden_states[before_no_treasure_idxs, 1],
                     marker='.', s=1, label='t <= 50, Out', alpha=alpha)
        axis.scatter(hidden_states[after_no_treasure_idxs, 0], hidden_states[after_no_treasure_idxs, 1],
                     marker='.', s=1, label='t > 50, Out', alpha=alpha)
        axis.scatter(hidden_states[before_treasure_idxs, 0], hidden_states[before_treasure_idxs, 1],
                     marker='.', s=1, label='t <= 50, In', alpha=alpha)
        axis.scatter(hidden_states[after_treasure_idxs, 0], hidden_states[after_treasure_idxs, 1],
                     marker='.', s=1, label='t > 50, In', alpha=alpha)
        if add_starts:
            axis.scatter(start_hidden_states[:, 0], start_hidden_states[:, 1], marker='^', s=30, c='c', edgecolor='k')

        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs State')
        if add_legend:
            # axis.set_ylim(*self.expand_axis(*axis.get_ylim(), 0.2, fix_max=True))
            handles, labels = axis.get_legend_handles_labels()
            order = [0, 2, 1, 3]
            legend = axis.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower left',
                                 handletextpad=0.05, fontsize=8, ncol=1, columnspacing=0.3, prop={'size': 7})
            for h in legend.legendHandles:
                h.set_sizes([40])
                h.set_alpha(1)

        if add_inset_axis:
            zoomed_axis = axis.inset_axes([0.02, 0.02, 0.55, 0.55])
            self.plot_hidden_states_vs_state(zoomed_axis, plotting_data, add_legend=False, sample_rate=sample_rate)
            x1, x2, y1, y2 = -0.25, 0.123, -0.21, 0.133
            zoomed_axis.set_xlim(x1, x2)
            zoomed_axis.set_ylim(y1, y2)
            zoomed_axis.set_title('')
            zoomed_axis.set_xlabel('')
            zoomed_axis.set_ylabel('')
            zoomed_axis.get_xaxis().set_ticks([])
            zoomed_axis.get_yaxis().set_ticks([])
            axis.indicate_inset_zoom(zoomed_axis, edgecolor="k")
            pass

    def plot_hidden_states_vs_time(self, axis, plotting_data, add_cbar=True, sample_rate=1.0):
        hidden_states = plotting_data['hidden_states']
        start_hidden_states = plotting_data['start_hidden_states']
        random_idxs = self.get_random_idxs(len(hidden_states), sample_rate)
        times = np.arange(len(hidden_states)) % self.episode_length
        axis.scatter(hidden_states[random_idxs, 0], hidden_states[random_idxs, 1], marker='.', s=1,
                     c=times[random_idxs], cmap=mpl.colormaps['cividis'], norm=self.time_color_norm)
        axis.scatter(start_hidden_states[:, 0], start_hidden_states[:, 1], marker='^', s=30, c='c', edgecolor='k')
        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs Time')
        if add_cbar:
            plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=self.time_color_norm), ax=axis)

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

    def plot_reward_vs_pos_xy(self, axis, plotting_data, time, add_cbar=True, lw=3):
        # Generate all heatmaps if not already done (required for back filling values)
        if self.reward_heatmaps is None:
            self.reward_heatmaps = self._generate_reward_heatmap_values(plotting_data)

        # Get relevant data
        xx, yy, all_zs = self.reward_heatmaps
        zs = all_zs[:, :, time]

        # Heatmap plotting
        heatmap_method = HeatmapMethod.PCOLORMESH
        cmap = mpl.colormaps['cividis']
        cmap.set_bad(color='w', alpha=1.)
        self.plot_heatmap(axis, xx, yy, zs, cmap, self.reward_color_norm, heatmap_method)

        # Add map features
        self.add_bounding_box(axis, maps['timertreasure']['boxes']['treasure']['coords'], 'green', alpha=1, lw=lw)
        self.add_bounding_box(axis, maps['timertreasure']['boxes']['spawn_left']['coords'], 'purple', alpha=1, lw=lw)
        self.add_bounding_box(axis, maps['timertreasure']['boxes']['spawn_right']['coords'], 'purple', alpha=1, lw=lw)

        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel('X Position')
        axis.set_ylabel('Y Position')
        title = 'Reward vs Position '
        axis.set_title('Reward vs Position\n(Time = {:.2f})'.format(time/self.episode_length))

        if add_cbar:
            plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=self.reward_color_norm), ax=axis)

    def _generate_reward_heatmap_values(self, plotting_data):
        heatmap_grid_size = 50

        # Get plotting data
        reward_preds = plotting_data['reward_preds']
        xs = plotting_data['true_pos_x']
        ys = plotting_data['true_pos_y']

        # All heatmap values across all frames
        all_zs = torch.full((heatmap_grid_size, heatmap_grid_size, self.episode_length + 1), torch.nan)

        def get_idxs_for_time(t):
            return np.arange(len(reward_preds) // self.episode_length) * self.episode_length + t

        # Get initial heatmap values
        init_idxs = get_idxs_for_time(0)
        xx, yy, zs = self.generate_heatmap_values(xs[init_idxs], ys[init_idxs], reward_preds[init_idxs],
                                                  heatmap_grid_size, prev_values=None)
        all_zs[:, :, 0] = zs

        # Iterate through the remainder of the episode, using the previous frame values to fill in missing gaps
        for idx in tqdm(range(self.episode_length - 1), desc='Generating heatmap values', leave=False):
            frame_idxs = get_idxs_for_time(idx + 1)
            _, _, new_zs = self.generate_heatmap_values(xs[frame_idxs], ys[frame_idxs], reward_preds[frame_idxs],
                                                        heatmap_grid_size, prev_values=all_zs[:, :, idx])
            all_zs[:, :, idx+1] = new_zs

        return xx, yy, all_zs


class TimerTreasureProbePlotter(TimerTreasureGlobalPlotter):

    @overrides
    def plot(self, plotting_data):
        probe_names = ["Optimal", "Too Late", "Too Early", "Challenging"]
        probes = [probing.probe("timer_treasure", name) for name in probe_names]
        probe_plotting_data = self.generate_probe_plotting_data(probes)

        hidden_state_alpha = 0.05

        def hide_axis(_axis):
            _axis.set_title('')
            _axis.set_xlabel('')
            _axis.set_ylabel('')
            _axis.get_xaxis().set_ticks([])
            _axis.get_yaxis().set_ticks([])

        time_values = [0.05, 0.49, 0.5, 0.51, 1.0]

        fig = plt.figure(figsize=(5.5, 3.7))
        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.1, 4])
        time_gs = gs[0].subgridspec(nrows=1, ncols=len(time_values) + 1,
                                      width_ratios=[1] * len(time_values) + [0.5])
        probe_gs = gs[1].subgridspec(nrows=2, ncols=len(probes), hspace=0.1)

        for time_idx, time_f in enumerate(time_values):
            axis = fig.add_subplot(time_gs[time_idx])
            time = int(self.episode_length * time_f)
            if time == 100:
                time = 99
            # This is expensive to run - comment it out if you don't actually need to see the heatmaps
            self.plot_reward_vs_pos_xy(axis, plotting_data, time, add_cbar=False, lw=1)
            hide_axis(axis)
            axis.set_xlabel("t = {:.2f}".format(time_f), fontsize=7)
            axis.set_aspect('equal')

        for probe_idx, probe in enumerate(probes):
            assert len(probe) == 100
            trajectory_axis = fig.add_subplot(probe_gs[0, probe_idx])
            hidden_state_axis = fig.add_subplot(probe_gs[1, probe_idx])

            self.plot_probe_trajectory(trajectory_axis, probe)
            self.plot_hidden_states_vs_state(hidden_state_axis, plotting_data, add_legend=False, add_starts=False,
                                             alpha=hidden_state_alpha)
            self.plot_probe_hidden_states(hidden_state_axis, probe_plotting_data[probe_idx])

            hide_axis(trajectory_axis)
            hide_axis(hidden_state_axis)

            trajectory_axis.set_title('{:s}\n(oracle return = {:d})'.format(
                probe_names[probe_idx], int(probe_plotting_data[probe_idx]['true_returns'][-1])),
                fontsize=8
            )

            if probe_idx == 0:
                trajectory_axis.set_ylabel('Environment State\nTrajectory', fontsize=8)
                hidden_state_axis.set_ylabel('Hidden State\nTrajectory', fontsize=8)

                # Add hidden state legend
                handles, labels = hidden_state_axis.get_legend_handles_labels()
                order = [0, 2, 1, 3]
                legend = hidden_state_axis.legend(handles=[handles[i] for i in order],
                                                  labels=[labels[i] for i in order], loc='lower left', ncol=1,
                                                  handletextpad=0.01, fontsize=5)
                for h in legend.legendHandles:
                    h.set_sizes([20])
                    h.set_alpha(1)

            # Inset axis for adversarial probe hidden state plot
            if probe_idx != 0:
                zoomed_axis = hidden_state_axis.inset_axes([0.02, 0.02, 0.55, 0.55])
                self.plot_hidden_states_vs_state(zoomed_axis, plotting_data, add_legend=False, add_starts=False,
                                                 alpha=hidden_state_alpha)
                self.plot_probe_hidden_states(zoomed_axis, probe_plotting_data[probe_idx])
                x1, x2, y1, y2 = -0.36, 0.16, -0.18, 0.14
                zoomed_axis.set_xlim(x1, x2)
                zoomed_axis.set_ylim(y1, y2)
                hide_axis(zoomed_axis)
                hidden_state_axis.indicate_inset_zoom(zoomed_axis, edgecolor="k")

            # Add probe n markers
            s = 0
            n_added = 0
            for probe_action in probing.probe_actions['timer_treasure'][probe_names[probe_idx]][1:]:
                if len(probe_action) == 2:
                    d, n = probe_action
                else:
                    d, n, _ = probe_action
                if d == 'N':
                    x, y = probe[s]
                    if probe_idx in [0, 2]:
                        self.plot_probe_n_marker(trajectory_axis, x, y, n + 1,
                                                 0.11 if n_added == 0 else 0.05,
                                                 0.0 if n_added == 0 else 0.07)
                    elif probe_idx == 1:
                        self.plot_probe_n_marker(trajectory_axis, x, y, n + 1,
                                                 -0.11 if n_added == 0 else -0.05,
                                                 0.0 if n_added == 0 else 0.07)
                    else:
                        self.plot_probe_n_marker(trajectory_axis, x, y, n + 1,
                                                 -0.11 if n_added == 0 else 0,
                                                 0.0 if n_added == 0 else -0.13)
                    n_added += 1
                s += n

        # Add reward colour bar
        cb_axis = fig.add_subplot(time_gs[-1])
        cb = plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=self.reward_color_norm),
                          ax=cb_axis, location='right', fraction=1.0)
        cb.ax.tick_params(labelsize=6)
        cb.set_ticks([-1, 0, 1])
        cb.set_label(label='Reward', size=7)
        cb_axis.set_axis_off()

        plt.tight_layout()
        return fig

    @overrides
    def generate_probe_plotting_data(self, probes):
        all_probe_data = super().generate_probe_plotting_data(probes)
        tre_bounds = maps['timertreasure']['boxes']['treasure']['coords']
        tre_x0, tre_y0, tre_x1, tre_y1 = tre_bounds[0][0], tre_bounds[0][1], tre_bounds[1][0], tre_bounds[1][1]
        for probe_idx, probe in enumerate(probes):
            probe_plotting_data = all_probe_data[probe_idx]
            returns = torch.zeros(len(probe))
            rewards = torch.zeros(len(probe))
            r = 0
            for idx, instance in enumerate(probe):
                x, y = instance
                # Update goal position
                if tre_x0 <= x <= tre_x1 and tre_y0 <= y <= tre_y1:
                    if idx < 50:
                        r -= 1
                        rewards[idx] = -1
                    else:
                        r += 1
                        rewards[idx] = 1
                returns[idx] = r
            probe_plotting_data['true_returns'] = returns
            probe_plotting_data['true_rewards'] = rewards
        return all_probe_data

    def plot_probe_trajectory(self, axis, probe):
        marker_size = 6

        # Add map features
        self.add_bounding_box(axis, maps['timertreasure']['boxes']['treasure']['coords'], 'green', lw=1, alpha=1)
        self.add_bounding_box(axis, maps['timertreasure']['boxes']['spawn_left']['coords'], 'purple', lw=1, alpha=1)
        self.add_bounding_box(axis, maps['timertreasure']['boxes']['spawn_right']['coords'], 'purple', lw=1, alpha=1)

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

        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.01, 1.01)

    def plot_probe_n_marker(self, axis, x, y, num, pad_x, pad_y):
        x += pad_x
        y += pad_y
        axis.text(x, y, 'x{:d}'.format(num), c='k', fontsize=6, va='center', ha='center')

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
