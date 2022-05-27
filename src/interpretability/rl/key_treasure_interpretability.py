import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from overrides import overrides
from tqdm import tqdm

from dataset.rl.keytreasure_dataset import KeyTreasureDataset
from interpretability.oracle_interpretability import RLGlobalStudyPlotter, OracleMILInterpretabilityStudy, HeatmapMethod
from rl_training import probing
from rl_training.maps import maps


def get_plotter_clz(mode):
    if mode == 'global':
        return KeyTreasureGlobalPlotter
    elif mode == 'probe':
        return KeyTreasureProbePlotter
    raise ValueError('No plotter found for mode {:s}'.format(mode))


class KeyTreasureGlobalPlotter(RLGlobalStudyPlotter):

    def __init__(self, device, model_clz, model_path, csv_path, repeat_num, seed=5):
        dataset_params = {
            'csv_path': csv_path,
        }
        study = OracleMILInterpretabilityStudy(device, KeyTreasureDataset, model_clz, model_path,
                                               "key_treasure", dataset_params, seed=seed)
        super().__init__(study, repeat_num)
        self.episode_length = 100
        self.reward_color_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self.x_color_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self.y_color_norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self.key_reward_heatmaps = None
        self.no_key_reward_heatmaps = None

    @overrides
    def generate_plotting_data(self, model, dataset):
        plotting_data = super().generate_plotting_data(model, dataset)
        plotting_data['true_pos_x'] = torch.cat([torch.as_tensor(bm['true_pos_x']) for bm in dataset.bags_metadata])
        plotting_data['true_pos_y'] = torch.cat([torch.as_tensor(bm['true_pos_y']) for bm in dataset.bags_metadata])
        plotting_data['has_key'] = torch.cat([torch.as_tensor(bm['has_key']) for bm in dataset.bags_metadata])
        plotting_data['in_treasure'] = torch.cat([torch.as_tensor(bm['in_treasure']) for bm in dataset.bags_metadata])
        return plotting_data

    def plot(self, plotting_data):
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(11, 5))
        self.plot_return_overall(axes[0][0], plotting_data)
        self.plot_reward_overall(axes[0][1], plotting_data)
        self.plot_reward_vs_in_treasure(axes[0][2], plotting_data)
        self.plot_reward_vs_pos_xy(fig, axes[0][3], plotting_data, False)
        self.plot_hidden_states_vs_state(axes[1][0], plotting_data)
        self.plot_hidden_states_vs_pos_x(axes[1][1], plotting_data)
        self.plot_hidden_states_vs_pos_y(axes[1][2], plotting_data)
        self.plot_reward_vs_pos_xy(fig, axes[1][3], plotting_data, True)
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

        # Hist
        # target_0_idxs = torch.where(reward_targets == 0)[0]
        # target_1_idxs = torch.where(reward_targets == 1)[0]
        # axis.hist(np.asarray(reward_preds[target_0_idxs]), bins=50, alpha=0.5, label='Target = 0', log=True)
        # axis.hist(np.asarray(reward_preds[target_1_idxs]), bins=50, alpha=0.5, label='Target = 1', log=True)
        # axis.set_xlabel('Predicted Reward')
        # axis.set_ylabel('Frequency')
        # axis.legend(loc='best')

        # Scatter
        axis.scatter(reward_targets, reward_preds, alpha=0.01, marker='.')
        axis.set_xlabel('Target')
        axis.set_ylabel('Predicted')

        axis.set_title('Reward Overall')

    def plot_reward_vs_in_treasure(self, axis, plotting_data):
        reward_preds = plotting_data['reward_preds']
        in_treasure = plotting_data['in_treasure']

        # Hist
        # outside_treasure_idxs = torch.where(in_treasure == 0)[0]
        # inside_treasure_idxs = torch.where(in_treasure == 1)[0]
        # axis.hist(np.asarray(reward_preds[outside_treasure_idxs]), bins=50, alpha=0.5, label='Out', log=True)
        # axis.hist(np.asarray(reward_preds[inside_treasure_idxs]), bins=50, alpha=0.5, label='In', log=True)
        # axis.set_xlabel('Predicted Reward')
        # axis.set_ylabel('Frequency')
        # axis.legend(loc='best')

        axis.scatter(in_treasure, reward_preds, alpha=0.01, marker='.')
        axis.set_xlabel('Target')
        axis.set_ylabel('Predicted')

        axis.set_title('Reward vs Treasure')

    def plot_hidden_states_vs_state(self, axis, plotting_data, add_legend=True, sample_rate=0.01, add_inset_axis=False,
                                    add_starts=True, alpha=1.0):
        hidden_states = plotting_data['hidden_states']
        start_hidden_states = plotting_data['start_hidden_states']

        # Get key and treasure states
        no_key_idxs = torch.where(plotting_data['has_key'] == 0)[0]
        key_idxs = torch.where(plotting_data['has_key'] == 1)[0]
        no_treasure_idxs = torch.where(plotting_data['in_treasure'] == 0)[0]
        treasure_idxs = torch.where(plotting_data['in_treasure'] == 1)[0]

        # Intersect key and treasure states
        no_key_no_treasure_idxs = np.intersect1d(no_key_idxs, no_treasure_idxs)
        key_no_treasure_idxs = np.intersect1d(key_idxs, no_treasure_idxs)
        no_key_treasure_idxs = np.intersect1d(no_key_idxs, treasure_idxs)
        key_treasure_idxs = np.intersect1d(key_idxs, treasure_idxs)

        # Filter to reduced number of idxs to save on plotting time
        no_key_no_treasure_idxs = self.sample_list(no_key_no_treasure_idxs, sample_rate)
        key_no_treasure_idxs = self.sample_list(key_no_treasure_idxs, sample_rate)
        no_key_treasure_idxs = self.sample_list(no_key_treasure_idxs, sample_rate)
        key_treasure_idxs = self.sample_list(key_treasure_idxs, sample_rate)

        # Actually plot
        axis.scatter(hidden_states[no_key_no_treasure_idxs, 0], hidden_states[no_key_no_treasure_idxs, 1],
                     marker='.', label='No Key, Out', s=1, alpha=alpha)
        axis.scatter(hidden_states[key_no_treasure_idxs, 0], hidden_states[key_no_treasure_idxs, 1],
                     marker='.', label='Key, Out', s=1, alpha=alpha)
        axis.scatter(hidden_states[no_key_treasure_idxs, 0], hidden_states[no_key_treasure_idxs, 1],
                     marker='.', label='No Key, In', s=1, alpha=alpha)
        axis.scatter(hidden_states[key_treasure_idxs, 0], hidden_states[key_treasure_idxs, 1],
                     marker='.', label='Key, In', s=1, alpha=alpha)
        if add_starts:
            axis.scatter(start_hidden_states[:, 0], start_hidden_states[:, 1], marker='^', s=30, c='c', edgecolor='k')

        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs State')

        if add_legend:
            axis.set_ylim(*self.expand_axis(*axis.get_ylim(), 0.4, fix_max=True))
            handles, labels = axis.get_legend_handles_labels()
            order = [0, 2, 1, 3]
            legend = axis.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='lower center',
                                 handletextpad=0.05, fontsize=8, ncol=2, columnspacing=0.3, prop={'size': 7})
            for h in legend.legendHandles:
                h.set_alpha(1)

    def plot_hidden_states_vs_key(self, axis, plotting_data, add_legend=True, sample_rate=0.01):
        hidden_states = plotting_data['hidden_states']

        # Get key and treasure states
        no_key_idxs = torch.where(plotting_data['has_key'] == 0)[0]
        key_idxs = torch.where(plotting_data['has_key'] == 1)[0]

        # Filter to reduced number of idxs to save on plotting time
        no_key_idxs = self.sample_list(no_key_idxs, sample_rate)
        key_idxs = self.sample_list(key_idxs, sample_rate)

        # Actually plot
        axis.scatter(hidden_states[no_key_idxs, 0], hidden_states[no_key_idxs, 1],
                     marker='.', label='Key Not Collected', color=mpl.colormaps['cividis'](0.1), s=1)
        axis.scatter(hidden_states[key_idxs, 0], hidden_states[key_idxs, 1],
                     marker='.', label='Key Collected', color=mpl.colormaps['cividis'](0.9), s=1)
        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs State')

        if add_legend:
            axis.set_ylim(*self.expand_axis(*axis.get_ylim(), 0.4, fix_max=True))
            handles, labels = axis.get_legend_handles_labels()
            legend = axis.legend(handles, labels, loc='lower center',
                                 handletextpad=0.05, fontsize=8, ncol=2, columnspacing=0.3, prop={'size': 7})
            for h in legend.legendHandles:
                h.set_alpha(1)

    def plot_hidden_states_vs_pos_x(self, axis, plotting_data, sample_rate=0.1, add_cbar=True):
        hidden_states = plotting_data['hidden_states']
        start_hidden_states = plotting_data['start_hidden_states']
        xs = plotting_data['true_pos_x']
        random_idxs = self.get_random_idxs(len(hidden_states), sample_rate)
        axis.scatter(hidden_states[random_idxs, 0], hidden_states[random_idxs, 1], marker='.', s=1,
                     c=xs[random_idxs], cmap=mpl.colormaps['cividis'], norm=self.x_color_norm)
        axis.scatter(start_hidden_states[:, 0], start_hidden_states[:, 1], marker='^', s=30, c='c', edgecolor='k')
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

    def plot_reward_vs_pos_xy(self, axis, plotting_data, with_key, time, add_cbar=True, lw=3):
        # Generate all heatmaps if not already done (required for back filling values)
        if self.key_reward_heatmaps is None:
            self.key_reward_heatmaps = self._generate_reward_heatmap_values(plotting_data, True)
        if self.no_key_reward_heatmaps is None:
            self.no_key_reward_heatmaps = self._generate_reward_heatmap_values(plotting_data, False)

        # Get relevant data
        xx, yy, all_zs = self.key_reward_heatmaps if with_key else self.no_key_reward_heatmaps
        zs = all_zs[:, :, time]

        # Heatmap plotting
        heatmap_method = HeatmapMethod.PCOLORMESH
        cmap = mpl.colormaps['cividis']
        self.plot_heatmap(axis, xx, yy, zs, cmap, self.reward_color_norm, heatmap_method)

        # Add map features
        self.add_bounding_box(axis, maps['keytreasure_A']['boxes']['key']['coords'], 'orange', lw=lw)
        self.add_bounding_box(axis, maps['keytreasure_A']['boxes']['treasure']['coords'], 'green', lw=lw)
        self.add_bounding_box(axis, maps['keytreasure_A']['boxes']['spawn_left']['coords'], 'purple', lw=lw)
        self.add_bounding_box(axis, maps['keytreasure_A']['boxes']['spawn_right']['coords'], 'purple', lw=lw)

        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel('X Position')
        axis.set_ylabel('Y Position')
        axis.set_title('Reward vs Position\n(Key = {:})'.format(with_key))

        if add_cbar:
            plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=self.reward_color_norm), ax=axis)

    def _generate_reward_heatmap_values(self, plotting_data, with_key):
        heatmap_grid_size = 50

        # Get plotting data
        has_key = plotting_data['has_key']
        reward_preds = plotting_data['reward_preds']
        xs = plotting_data['true_pos_x']
        ys = plotting_data['true_pos_y']

        # All heatmap values across all frames
        all_zs = torch.full((heatmap_grid_size, heatmap_grid_size, self.episode_length + 1), torch.nan)

        def get_idxs_for_time(t):
            return np.arange(len(xs) // self.episode_length) * self.episode_length + t

        # Get initial heatmap values
        init_idxs = get_idxs_for_time(0)
        key_idxs = torch.where(has_key == 1)[0] if with_key else torch.where(has_key == 0)[0]
        init_idxs = np.intersect1d(init_idxs, key_idxs)
        xx, yy, zs = self.generate_heatmap_values(xs[init_idxs], ys[init_idxs], reward_preds[init_idxs],
                                                  heatmap_grid_size, prev_values=None)
        all_zs[:, :, 0] = zs

        # Iterate through the remainder of the episode, using the previous frame values to fill in missing gaps
        for idx in tqdm(range(self.episode_length - 1), desc='Generating heatmap values, key = {:}'.format(with_key),
                        leave=False):
            frame_idxs = get_idxs_for_time(idx + 1)
            frame_idxs = np.intersect1d(frame_idxs, key_idxs)
            _, _, new_zs = self.generate_heatmap_values(xs[frame_idxs], ys[frame_idxs], reward_preds[frame_idxs],
                                                        heatmap_grid_size, prev_values=all_zs[:, :, idx])
            all_zs[:, :, idx+1] = new_zs
        return xx, yy, all_zs


class KeyTreasureProbePlotter(KeyTreasureGlobalPlotter):

    @overrides
    def plot(self, plotting_data):
        probe_names = ["Optimal Left", "Optimal Right", "Sub Optimal", "Failure Case"]
        probes = [probing.probe("key_treasure", name) for name in probe_names]
        probe_plotting_data = self.generate_probe_plotting_data(probes)

        hidden_state_alpha = 0.2

        def hide_axis(_axis):
            _axis.set_title('')
            _axis.set_xlabel('')
            _axis.set_ylabel('')
            _axis.get_xaxis().set_ticks([])
            _axis.get_yaxis().set_ticks([])

        time_values = [0.07, 0.3, 0.07, 0.3, 1.0]
        key_values = [False, False, True, True, True]

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
            with_key = key_values[time_idx]
            self.plot_reward_vs_pos_xy(axis, plotting_data, with_key, time, add_cbar=False, lw=1)
            hide_axis(axis)
            label = "t = {:.2f}\n".format(time_f)
            label += 'Key' if with_key else 'No Key'
            axis.set_xlabel(label, fontsize=7)
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
                                                  labels=[labels[i] for i in order], loc='upper left', ncol=1,
                                                  handletextpad=0.01, fontsize=6)
                for h in legend.legendHandles:
                    h.set_sizes([20])
                    h.set_alpha(1)

            # Add probe n markers
            s = 0
            for probe_action in probing.probe_actions['key_treasure'][probe_names[probe_idx]][1:]:
                if len(probe_action) == 2:
                    d, n = probe_action
                else:
                    d, n, _ = probe_action
                if d == 'N':
                    x, y = probe[s]
                    if probe_idx == 0:
                        self.plot_probe_n_marker(trajectory_axis, x, y, n + 1, 0.08, 0.03)
                    else:
                        self.plot_probe_n_marker(trajectory_axis, x, y, n + 1, -0.08, 0.03)
                s += n

        # Add reward colour bar
        cb_axis = fig.add_subplot(time_gs[-1])
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
        key_bounds = maps['keytreasure_A']['boxes']['key']['coords']
        tre_bounds = maps['keytreasure_A']['boxes']['treasure']['coords']
        key_x0, key_y0, key_x1, key_y1 = key_bounds[0][0], key_bounds[0][1], key_bounds[1][0], key_bounds[1][1]
        tre_x0, tre_y0, tre_x1, tre_y1 = tre_bounds[0][0], tre_bounds[0][1], tre_bounds[1][0], tre_bounds[1][1]
        for probe_idx, probe in enumerate(probes):
            probe_plotting_data = all_probe_data[probe_idx]
            returns = torch.zeros(len(probe))
            rewards = torch.zeros(len(probe))
            r = 0
            found_key = False
            for idx, instance in enumerate(probe):
                x, y = instance
                if not found_key and key_x0 <= x <= key_x1 and key_y0 <= y <= key_y1:
                    found_key = True
                if found_key and tre_x0 <= x <= tre_x1 and tre_y0 <= y <= tre_y1:
                    r += 1
                    returns[idx] = r
                    rewards[idx] = 1
            probe_plotting_data['true_returns'] = returns
            probe_plotting_data['true_rewards'] = rewards
        return all_probe_data

    def plot_probe_n_marker(self, axis, x, y, num, pad_x, pad_y):
        x += pad_x
        y += pad_y
        axis.text(x, y, "x{:d}".format(num), c='k', fontsize=5, va='center', ha='center')

    def plot_probe_trajectory(self, axis, probe):
        marker_size = 15

        # Add map features
        self.add_bounding_box(axis, maps['keytreasure_A']['boxes']['key']['coords'], 'orange')
        self.add_bounding_box(axis, maps['keytreasure_A']['boxes']['treasure']['coords'], 'green')
        self.add_bounding_box(axis, maps['keytreasure_A']['boxes']['spawn_left']['coords'], 'purple')
        self.add_bounding_box(axis, maps['keytreasure_A']['boxes']['spawn_right']['coords'], 'purple')

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

    def plot_probe_hidden_states(self, axis, probe_plotting_data):
        hidden_states = probe_plotting_data['hidden_states']
        marker_size = 5
        axis.plot(hidden_states[:, 0], hidden_states[:, 1], linestyle='--', color='k', zorder=1, alpha=0.5, lw=0.5)
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

    def plot_probe_returns(self, axis, probe_plotting_data):
        pred_returns = probe_plotting_data['pred_returns']
        true_returns = probe_plotting_data['true_returns']
        axis.plot(range(len(pred_returns)), pred_returns, label='Pred')
        axis.plot(range(len(pred_returns)), true_returns, label='True')
        # axis.set_xlabel('Timestep')
        # axis.set_ylabel('Return')
        axis.set_title('Return')
        # axis.legend(loc='upper left')
        # axis.set_ylim(-0.1, 3.3)

    def plot_probe_rewards(self, axis, probe_plotting_data):
        pred_rewards = probe_plotting_data['pred_rewards']
        true_rewards = probe_plotting_data['true_rewards']
        axis.plot(range(len(pred_rewards)), pred_rewards, label='Pred')
        axis.plot(range(len(pred_rewards)), true_rewards, label='True')
        # axis.set_xlabel('Timestep')
        # axis.set_ylabel('Reward')
        axis.set_title('Reward')
        # axis.legend(loc='upper left')
        axis.set_ylim(-0.1, 1.1)

    @overrides
    def get_save_path(self):
        return "{:s}/interpretability_probe_{:d}.png".format(self.get_and_make_fig_dir(), self.repeat_num)
