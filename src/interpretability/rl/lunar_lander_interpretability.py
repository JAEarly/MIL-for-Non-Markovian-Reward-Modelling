import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from overrides import overrides
from tqdm import tqdm

from dataset.rl.lunar_lander_dataset import LunarLanderDataset
from interpretability.oracle_interpretability import RLGlobalStudyPlotter, OracleMILInterpretabilityStudy, HeatmapMethod
from rl_training.maps import maps


def get_plotter_clz(mode):
    if mode == 'global':
        return LunarLanderGlobalPlotter
    elif mode == 'local':
        return LunarLanderLocalPlotter
    elif mode == 'probe':
        return LunarLanderProbePlotter
    raise ValueError('No plotter found for mode {:s}'.format(mode))


class LunarLanderGlobalPlotter(RLGlobalStudyPlotter):

    time_on_pad_max = 300

    def __init__(self, device, model_clz, model_path, csv_path, repeat_num, seed=5):
        dataset_params = {
            'csv_path': csv_path,
        }
        study = OracleMILInterpretabilityStudy(device, LunarLanderDataset, model_clz, model_path,
                                               "lunar_lander", dataset_params, seed=seed)
        super().__init__(study, repeat_num, add_start_hidden_states=True)

    @overrides
    def generate_plotting_data(self, model, dataset):
        plotting_data = super().generate_plotting_data(model, dataset)
        plotting_data['true_pos_x'] = torch.cat([torch.as_tensor(bm['true_pos_x']) for bm in dataset.bags_metadata])
        plotting_data['true_pos_y'] = torch.cat([torch.as_tensor(bm['true_pos_y']) for bm in dataset.bags_metadata])
        plotting_data['on_pad'] = torch.cat([torch.as_tensor(bm['on_pad']) for bm in dataset.bags_metadata])
        plotting_data['time_on_pad'] = torch.cat([torch.as_tensor(bm['time_on_pad']) for bm in dataset.bags_metadata])
        plotting_data['in_hover'] = torch.cat([torch.as_tensor(bm['in_hover']) for bm in dataset.bags_metadata])
        plotting_data['bag_clzs'] = torch.as_tensor([torch.as_tensor(bm['bag_clz']) for bm in dataset.bags_metadata])
        return plotting_data

    @overrides
    def generate_start_state_plotting_data(self, dataset, model):
        # Get hidden states for start positions (normalise then pass through model)
        #  Start positions generated from gym env
        start_positions = torch.as_tensor([
            [0.0034350394, 1.4155055, 0.3479195, 0.20378537, -0.003973585, -0.078808926, 0.0, 0.0],
            [-0.0033464432, 1.4007066, -0.33898574, -0.4539428, 0.003884598, 0.076785274, 0.0, 0.0],
            [-0.0057038306, 1.4208806, -0.57775474, 0.44267595, 0.0066161295, 0.13087003, 0.0, 0.0],
        ])
        start_positions_norm = torch.zeros_like(start_positions)
        for idx, start_position in enumerate(start_positions):
            s_norm = (start_position - dataset.instance_min) / (dataset.instance_max - dataset.instance_min) - 0.5
            start_positions_norm[idx, :] = s_norm
        start_hidden_states = model.get_hidden_states(start_positions_norm).detach().cpu()
        return start_hidden_states

    def plot(self, plotting_data):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 5.5))
        self.plot_return_overall(axes[0][0], plotting_data)
        self.plot_reward_overall(axes[0][1], plotting_data)
        self.plot_return_loss_vs_bag_size(axes[0][2], plotting_data)
        # self.plot_reward_loss_vs_bag_size(axes[0][3], plotting_data)
        self.plot_return_loss_vs_bag_clz(axes[0][3], plotting_data)
        self.plot_hidden_states_vs_pos_x(axes[0][4], plotting_data)
        self.plot_hidden_states_vs_on_pad(axes[1][0], plotting_data)
        self.plot_hidden_states_vs_in_hover(axes[1][1], plotting_data)
        self.plot_hidden_states_vs_state(axes[1][2], plotting_data)
        self.plot_hidden_states_vs_time_split(axes[1][3], plotting_data)
        self.plot_hidden_states_vs_pos_y(axes[1][4], plotting_data)
        print('Plotting done')
        plt.tight_layout()
        return fig

    def plot_return_overall(self, axis, plotting_data, sample_rate=0.05):
        print('Plotting return overall')
        return_preds = plotting_data['return_preds']
        return_targets = plotting_data['return_targets']
        all_idxs = list(range(len(return_preds)))
        sampled_idxs = self.sample_list(all_idxs, sample_rate)
        return_preds = return_preds[sampled_idxs]
        return_targets = return_targets[sampled_idxs]
        axis.scatter(return_targets, return_preds, alpha=0.01, marker='.')
        axis.set_xlabel('Target')
        axis.set_ylabel('Predicted')
        axis.set_title('Return Overall')

    def plot_reward_overall(self, axis, plotting_data, sample_rate=0.05):
        print('Plotting reward overall')
        reward_preds = plotting_data['reward_preds']
        reward_targets = plotting_data['reward_targets']
        all_idxs = list(range(len(reward_preds)))
        sampled_idxs = self.sample_list(all_idxs, sample_rate)
        reward_preds = reward_preds[sampled_idxs]
        reward_targets = reward_targets[sampled_idxs]
        axis.scatter(reward_targets, reward_preds, alpha=0.01, marker='.')
        axis.set_xlabel('Target')
        axis.set_ylabel('Predicted')
        axis.set_title('Reward Overall')

    def plot_return_loss_vs_bag_size(self, axis, plotting_data):
        print('Plotting return loss vs bag size')
        # Get all required data
        return_preds = plotting_data['return_preds']
        return_targets = plotting_data['return_targets']
        bag_sizes = plotting_data['bag_sizes']

        # Get return pred and target for each bag, then compute loss
        bag_idxs = torch.cumsum(bag_sizes, 0) - 1
        bag_return_preds = return_preds[bag_idxs]
        bag_return_targets = return_targets[bag_idxs]
        bag_return_losses = torch.nn.MSELoss(reduction='none')(bag_return_preds, bag_return_targets)

        # Plot
        axis.scatter(bag_sizes, bag_return_losses, alpha=0.1, marker='.')
        axis.set_xlabel('Bag Size')
        axis.set_ylabel('Return Loss')
        axis.set_title('Return Loss vs Bag Size')

    def plot_reward_loss_vs_bag_size(self, axis, plotting_data, sample_rate=0.05):
        print('Plotting reward loss vs bag size')
        # Get all required data
        reward_preds = plotting_data['reward_preds']
        reward_targets = plotting_data['reward_targets']
        bag_sizes = plotting_data['bag_sizes']

        # Compute reward losses
        reward_losses = torch.nn.MSELoss(reduction='none')(reward_preds, reward_targets)

        # Expand bag sizes tensor to match the dimensionality of reward losses
        expanded_bag_sizes = torch.zeros_like(reward_losses)
        b = 0
        for bag_size in bag_sizes:
            expanded_bag_sizes[b:b+bag_size] = bag_size
            b += bag_size

        # Sample random idxs
        all_idxs = list(range(len(reward_preds)))
        sampled_idxs = self.sample_list(all_idxs, sample_rate)
        expanded_bag_sizes = expanded_bag_sizes[sampled_idxs]
        reward_losses = reward_losses[sampled_idxs]

        # Plot
        axis.scatter(expanded_bag_sizes, reward_losses, alpha=0.1, marker='.')
        axis.set_xlabel('Bag Size')
        axis.set_ylabel('Reward Loss')
        axis.set_title('Reward Loss vs Bag Size')

    def plot_return_loss_vs_bag_clz(self, axis, plotting_data):
        print('Plotting return loss vs bag class')

        # Get all required data
        return_preds = plotting_data['return_preds']
        return_targets = plotting_data['return_targets']
        bag_sizes = plotting_data['bag_sizes']
        bag_clzs = plotting_data['bag_clzs']

        # Get return pred and target for each bag, then compute loss
        bag_idxs = torch.cumsum(bag_sizes, 0) - 1
        bag_return_preds = return_preds[bag_idxs]
        bag_return_targets = return_targets[bag_idxs]
        bag_return_losses = torch.nn.MSELoss(reduction='none')(bag_return_preds, bag_return_targets)

        # Group losses by class
        bag_loss_dict = {}
        for clz_idx in range(9):
            bag_loss_dict[clz_idx] = []
        for bag_clz, bag_loss in zip(bag_clzs, bag_return_losses):
            bag_loss_dict[bag_clz.item()].append(bag_loss)

        # Calculate per class loss
        all_clz_losses = {}
        for bag_clz, clz_losses in bag_loss_dict.items():
            all_clz_losses[bag_clz] = np.mean(clz_losses)

        axis.bar(all_clz_losses.keys(), all_clz_losses.values())
        axis.set_xlabel('Bag class')
        axis.set_ylabel('Avg return loss')
        axis.set_xticks(ticks=list(range(9)))
        axis.set_title('Return Loss vs Bag Class')

    def plot_hidden_states_vs_on_pad(self, axis, plotting_data, sample_rate=0.05, add_legend=True, alpha=0.1):
        print('Plotting hidden states vs on pad')

        # Load plotting data
        hidden_states = plotting_data['hidden_states']
        on_pad = plotting_data['on_pad']
        time_on_pad = plotting_data['time_on_pad']

        # Get idxs for on/off
        off_pad_idxs = torch.where(on_pad == 0)[0]
        on_pad_idxs = torch.where(on_pad == 1)[0]

        # Get idxs for before/after t=50
        t_lt_50_idxs = torch.where(time_on_pad < 50)[0]
        t_gt_50_idxs = torch.where(time_on_pad >= 50)[0]

        # Combine hover_idxs with time idxs
        off_pad_lt_idxs = np.intersect1d(off_pad_idxs, t_lt_50_idxs)
        on_pad_lt_idxs = np.intersect1d(on_pad_idxs, t_lt_50_idxs)
        off_pad_gt_idxs = np.intersect1d(off_pad_idxs, t_gt_50_idxs)
        on_pad_gt_idxs = np.intersect1d(on_pad_idxs, t_gt_50_idxs)

        # Sample lists
        off_pad_lt_idxs = self.sample_list(off_pad_lt_idxs, sample_rate)
        on_pad_lt_idxs = self.sample_list(on_pad_lt_idxs, sample_rate)
        off_pad_gt_idxs = self.sample_list(off_pad_gt_idxs, sample_rate)
        on_pad_gt_idxs = self.sample_list(on_pad_gt_idxs, sample_rate)

        # Plot
        axis.scatter(hidden_states[off_pad_lt_idxs, 0], hidden_states[off_pad_lt_idxs, 1],
                     label='Off Pad; t < 50', marker='.', s=1, alpha=alpha)
        axis.scatter(hidden_states[on_pad_lt_idxs, 0], hidden_states[on_pad_lt_idxs, 1],
                     label='On Pad; t < 50', marker='.', s=1, alpha=alpha)
        axis.scatter(hidden_states[off_pad_gt_idxs, 0], hidden_states[off_pad_gt_idxs, 1],
                     label='Off Pad; t >= 50', marker='.', s=1, alpha=alpha)
        axis.scatter(hidden_states[on_pad_gt_idxs, 0], hidden_states[on_pad_gt_idxs, 1],
                     label='On Pad; t >= 50', marker='.', s=1, alpha=alpha)
        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs On/Off Pad')
        if add_legend:
            legend = axis.legend(loc='best', handletextpad=0.1, fontsize=8)
            for h in legend.legendHandles:
                h.set_sizes([40])
                h.set_alpha(1)

    def plot_hidden_states_vs_time_on_pad(self, axis, plotting_data, add_cbar=True, sample_rate=0.05, alpha=0.1):
        print('Plotting hidden states vs split time')

        hidden_states = plotting_data['hidden_states']
        time_on_pad = plotting_data['time_on_pad']

        # Filter for only t <= 50
        time_gte_47_idxs = torch.where(time_on_pad >= 47)[0]
        time_lte_53_idxs = torch.where(time_on_pad <= 53)[0]
        time_idxs = np.intersect1d(time_gte_47_idxs, time_lte_53_idxs)
        hidden_states = hidden_states[time_idxs]
        time_on_pad = time_on_pad[time_idxs]

        print('Time on Pad Split Min/Max:', time_on_pad.min(), time_on_pad.max())
        time_on_pad_color_norm = mpl.colors.Normalize(vmin=time_on_pad.min().item(), vmax=time_on_pad.max().item())

        random_idxs = self.get_random_idxs(len(hidden_states), sample_rate)
        axis.scatter(hidden_states[random_idxs, 0], hidden_states[random_idxs, 1], marker='.', s=1, alpha=alpha,
                     c=time_on_pad[random_idxs], cmap=mpl.colormaps['cividis'], norm=time_on_pad_color_norm)
        if add_cbar:
            plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=time_on_pad_color_norm), ax=axis)
        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs Time on Pad')

    def plot_hidden_states_vs_time_split(self, axis, plotting_data, sample_rate=0.05, add_legend=True, add_cbar=True,
                                         alpha=0.1, add_starts=True):
        print('Plotting hidden states vs split time')

        hidden_states = plotting_data['hidden_states']
        time_on_pad = plotting_data['time_on_pad']
        start_hidden_states = plotting_data['start_hidden_states']

        # Get all time on pads between 47 and 53
        time_gte_47_idxs = torch.where(time_on_pad >= 47)[0]
        time_lte_53_idxs = torch.where(time_on_pad <= 53)[0]
        time_middle_idxs = self.sample_list(np.intersect1d(time_gte_47_idxs, time_lte_53_idxs), 0.1)

        # Sample from other extremes
        time_lt_47_idxs = self.sample_list(torch.where(time_on_pad < 47)[0], sample_rate)
        time_gt_53_idxs = torch.where(time_on_pad > 53)[0]
        time_lt_max_idxs = torch.where(time_on_pad < self.time_on_pad_max)[0]
        time_gt_53_lt_max_idxs = np.intersect1d(time_gt_53_idxs, time_lt_max_idxs)
        time_gt_53_lt_max_idxs = self.sample_list(time_gt_53_lt_max_idxs, sample_rate)

        # Combine
        time_idxs = np.concatenate([time_middle_idxs, time_lt_47_idxs, time_gt_53_lt_max_idxs])
        hidden_states = hidden_states[time_idxs]
        time_on_pad = time_on_pad[time_idxs]

        print('Time on Pad Split Min/Max:', time_on_pad.min(), time_on_pad.max())
        time_on_pad_color_norm = mpl.colors.BoundaryNorm([0, 47, 48, 49, 50, 51, 52, 53, self.time_on_pad_max],
                                                         mpl.colormaps['cividis'].N)
        # time_on_pad_color_norm = mpl.colors.Normalize(vmin=time_on_pad.min().item(), vmax=time_on_pad.max().item())

        # random_idxs = self.get_random_idxs(len(hidden_states), sample_rate)
        axis.scatter(hidden_states[:, 0], hidden_states[:, 1], marker='.', s=1, alpha=alpha,
                     c=time_on_pad, cmap=mpl.colormaps['cividis'], norm=time_on_pad_color_norm)
        if add_cbar:
            plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=time_on_pad_color_norm), ax=axis)
        if add_starts:
            # Only plot the first start state for simplicity
            axis.scatter(start_hidden_states[0, 0], start_hidden_states[0, 1], marker='^', s=30, c='c', edgecolor='k')
        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs Time on Pad')

    def plot_hidden_states_vs_pos_x(self, axis, plotting_data, sample_rate=0.05, add_cbar=True):
        print('Plotting hidden states vs pos x')
        hidden_states = plotting_data['hidden_states']
        xs = plotting_data['true_pos_x']
        print('Pos X Min/Max:', xs.min(), xs.max())
        x_color_norm = mpl.colors.Normalize(vmin=xs.min(), vmax=xs.max())
        random_idxs = self.get_random_idxs(len(hidden_states), sample_rate)
        axis.scatter(hidden_states[random_idxs, 0], hidden_states[random_idxs, 1], marker='.', alpha=0.1, s=1,
                     c=xs[random_idxs], cmap=mpl.colormaps['cividis'], norm=x_color_norm)
        if add_cbar:
            plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=x_color_norm), ax=axis)
        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs Pos X')

    def plot_hidden_states_vs_pos_y(self, axis, plotting_data, sample_rate=0.05, add_cbar=True):
        print('Plotting hidden states vs pos y')
        hidden_states = plotting_data['hidden_states']
        ys = plotting_data['true_pos_y']
        print('Pos Y Min/Max:', ys.min(), ys.max())
        y_color_norm = mpl.colors.Normalize(vmin=ys.min(), vmax=ys.max())
        random_idxs = self.get_random_idxs(len(hidden_states), sample_rate)
        axis.scatter(hidden_states[random_idxs, 0], hidden_states[random_idxs, 1], marker='.', alpha=0.1, s=1,
                     c=ys[random_idxs], cmap=mpl.colormaps['cividis'], norm=y_color_norm)
        if add_cbar:
            plt.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=y_color_norm), ax=axis)
        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs Pos Y')

    def plot_hidden_states_vs_in_hover(self, axis, plotting_data, sample_rate=0.05, add_legend=True, alpha=0.1):
        print('Plotting hidden states vs in hover')

        # Load plotting data
        hidden_states = plotting_data['hidden_states']
        in_hover = plotting_data['in_hover']
        time_on_pad = plotting_data['time_on_pad']

        # Get idxs for on/off
        out_hover_idxs = torch.where(in_hover == 0)[0]
        in_hover_idxs = torch.where(in_hover == 1)[0]

        # Get idxs for before/after t=50
        t_lt_50_idxs = torch.where(time_on_pad < 50)[0]
        t_gt_50_idxs = torch.where(time_on_pad >= 50)[0]

        # Combine hover_idxs with time idxs
        out_hover_lt_idxs = np.intersect1d(out_hover_idxs, t_lt_50_idxs)
        in_hover_lt_idxs = np.intersect1d(in_hover_idxs, t_lt_50_idxs)
        out_hover_gt_idxs = np.intersect1d(out_hover_idxs, t_gt_50_idxs)
        in_hover_gt_idxs = np.intersect1d(in_hover_idxs, t_gt_50_idxs)

        # Sample lists
        out_lt_hover_idxs = self.sample_list(out_hover_lt_idxs, sample_rate)
        in_lt_hover_idxs = self.sample_list(in_hover_lt_idxs, sample_rate)
        out_gt_hover_idxs = self.sample_list(out_hover_gt_idxs, sample_rate)
        in_gt_hover_idxs = self.sample_list(in_hover_gt_idxs, sample_rate)

        # Plot
        axis.scatter(hidden_states[out_lt_hover_idxs, 0], hidden_states[out_lt_hover_idxs, 1],
                     label='Out Hover; t < 50', marker='.', s=1, alpha=alpha)
        axis.scatter(hidden_states[in_lt_hover_idxs, 0], hidden_states[in_lt_hover_idxs, 1],
                     label='In Hover; t < 50', marker='.', s=1, alpha=alpha)
        axis.scatter(hidden_states[out_gt_hover_idxs, 0], hidden_states[out_gt_hover_idxs, 1],
                     label='Out Hover; t >= 50', marker='.', s=1, alpha=alpha)
        axis.scatter(hidden_states[in_gt_hover_idxs, 0], hidden_states[in_gt_hover_idxs, 1],
                     label='In Hover; t >= 50', marker='.', s=1, alpha=alpha)
        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs In/Out Hover')
        if add_legend:
            legend = axis.legend(loc='best', handletextpad=0.1, fontsize=8)
            for h in legend.legendHandles:
                h.set_sizes([40])
                h.set_alpha(1)

    def plot_hidden_states_vs_state(self, axis, plotting_data, sample_rate=0.05, add_legend=True, alpha=0.1,
                                    add_inset_axis=False, add_starts=True):
        print('Plotting hidden states vs state')

        # Load plotting data
        hidden_states = plotting_data['hidden_states']
        on_pad = plotting_data['on_pad']
        in_hover = plotting_data['in_hover']
        start_hidden_states = plotting_data['start_hidden_states']

        # Get idxs for on/off pad
        off_pad_idxs = torch.where(on_pad == 0)[0]
        on_pad_idxs = torch.where(on_pad == 1)[0]

        # Get idxs for in/out hover
        out_hover_idxs = torch.where(in_hover == 0)[0]
        in_hover_idxs = torch.where(in_hover == 1)[0]

        # Combine out hover idxs with off pad idxs to get normal idxs
        normal_idxs = np.intersect1d(out_hover_idxs, off_pad_idxs)

        # Sample lists
        normal_idxs = self.sample_list(normal_idxs, 0.01)
        on_pad_idxs = self.sample_list(on_pad_idxs, 0.1)
        in_hover_idxs = self.sample_list(in_hover_idxs, 0.02)

        # Plot
        axis.scatter(hidden_states[normal_idxs, 0], hidden_states[normal_idxs, 1],
                     label='Normal', marker='.', s=1, alpha=alpha)
        axis.scatter(hidden_states[on_pad_idxs, 0], hidden_states[on_pad_idxs, 1],
                     label='On Pad', marker='.', s=1, alpha=alpha)
        axis.scatter(hidden_states[in_hover_idxs, 0], hidden_states[in_hover_idxs, 1],
                     label='In Hover', marker='.', s=1, alpha=alpha)
        if add_starts:
            # Only plot the first start state for simplicity
            axis.scatter(start_hidden_states[0, 0], start_hidden_states[0, 1], marker='^', s=30, c='c', edgecolor='k')

        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State vs State')
        if add_legend:
            legend = axis.legend(loc='best', handletextpad=0.1, fontsize=8)
            for h in legend.legendHandles:
                h.set_sizes([40])
                h.set_alpha(1)


class LunarLanderLocalPlotter(LunarLanderGlobalPlotter):

    @overrides
    def plot(self, plotting_data):

        for s_idx, e_idx in self.bag_idx_range():
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 5))
            self.plot_probe_rewards(axes[0], plotting_data, s_idx, e_idx)
            self.plot_probe_returns(axes[1], plotting_data, s_idx, e_idx)
            plt.tight_layout()
            plt.show()

        exit(0)

    def bag_idx_range(self):
        dataset = self.study.load_experiment_complete_dataset()
        i = 0
        for bag in dataset.bags:
            s_idx = i
            e_idx = i + len(bag)
            i = e_idx
            yield s_idx, e_idx

    def plot_probe_rewards(self, axis, plotting_data, s_idx, e_idx):
        reward_preds = plotting_data['reward_preds'][s_idx:e_idx]
        reward_targets = plotting_data['reward_targets'][s_idx:e_idx]
        print('Sum pred reward', sum(reward_preds))
        print('Sum targ reward', sum(reward_targets))
        axis.plot(range(len(reward_preds)), reward_preds, label='Pred', alpha=0.5, c='green')
        axis.plot(range(len(reward_targets)), reward_targets, label='True', alpha=0.5, c='black')
        axis.set_xlabel('Timestep')
        axis.set_ylabel('Reward')
        axis.set_title('Reward')

    def plot_probe_returns(self, axis, plotting_data, s_idx, e_idx):
        return_preds = plotting_data['return_preds'][s_idx:e_idx]
        return_targets = plotting_data['return_targets'][s_idx:e_idx]
        print('Pred return', return_preds[-1])
        print('Targ return', return_targets[-1])
        axis.plot(range(len(return_preds)), return_preds, label='Pred', alpha=0.5, c='green')
        axis.plot(range(len(return_targets)), return_targets, label='True', alpha=0.5, c='black')
        axis.set_xlabel('Timestep')
        axis.set_ylabel('Return')
        axis.set_title('Return')


class LunarLanderProbePlotter(LunarLanderGlobalPlotter):

    def __init__(self, device, model_clz, model_path, csv_path, repeat_num):
        super().__init__(device, model_clz, model_path, csv_path, repeat_num)
        self.pad_times = ["0", "1-48", "49", "50", ">50"]
        self.reward_color_norm = mpl.colors.Normalize(vmin=-0.5, vmax=2)
        self.reward_heatmaps = None
        self.cmap = self._create_heatmap()

    @overrides
    def plot(self, plotting_data):
        probe_names = ["Correct", "No Hover", "Too Short", "Never Landed"]
        probe_clzs = [8, 6, 1, 0]
        probe_idxs = [723, 723, 3, 0]

        probe_data = self._generate_probes(probe_clzs, plotting_data, probe_idxs)
        probe_xys, probe_hidden_states, probe_return_preds = probe_data
        hidden_state_alpha = 0.2

        def config_axis(_axis):
            _axis.set_title('')
            _axis.set_xlabel('')
            _axis.set_ylabel('')
            _axis.get_xaxis().set_ticks([])
            _axis.get_yaxis().set_ticks([])

        fig = plt.figure(figsize=(5.5, 3.9))
        gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.1, 6])
        time_on_pad_gs = gs[0].subgridspec(nrows=1, ncols=len(self.pad_times) + 1,
                                           width_ratios=[1] * len(self.pad_times) + [0.5])
        probe_gs = gs[1].subgridspec(nrows=2, ncols=len(probe_xys), hspace=0.1)

        # Add reward vs pos x y on the top
        for pad_time_idx, time_on_pad in enumerate(self.pad_times):
            axis = fig.add_subplot(time_on_pad_gs[pad_time_idx])
            # This is expensive to run - comment it out if you don't actually need to see the heatmaps
            self.plot_reward_vs_pos_xy(fig, axis, plotting_data, pad_time_idx, self.cmap, add_cbar=False, lw=0.5)
            config_axis(axis)
            axis.set_aspect('equal')
            if pad_time_idx == 0:
                axis.set_xlabel("Time on Pad: {:s}".format(time_on_pad), fontsize=7)
            else:
                axis.set_xlabel("{:s}".format(time_on_pad), fontsize=7)

        # Plot probe info
        for probe_idx, probe_clz in enumerate(probe_clzs):
            trajectory_axis = fig.add_subplot(probe_gs[0, probe_idx])
            hidden_state_axis = fig.add_subplot(probe_gs[1, probe_idx])
            # reward_axis = fig.add_subplot(probe_gs[2, probe_idx])

            self.plot_probe_trajectory(trajectory_axis, probe_xys[probe_clz])
            trajectory_axis.set_aspect('equal')
            self.plot_hidden_states_vs_state(hidden_state_axis, plotting_data, add_legend=False, add_starts=False,
                                             alpha=hidden_state_alpha)
            self.plot_probe_hidden_states(hidden_state_axis, probe_hidden_states[probe_clz])

            config_axis(trajectory_axis)
            config_axis(hidden_state_axis)

            trajectory_axis.set_title('{:s}\n(oracle return = {:.1f})'.format(
                probe_names[probe_idx], probe_return_preds[probe_clz]),
                fontsize=7
            )

            if probe_idx in [0, 1]:
                if probe_idx == 0:
                    trajectory_axis.set_ylabel('Trajectory', fontsize=10)
                    hidden_state_axis.set_ylabel('Hidden State', fontsize=10)
                zoomed_axis = hidden_state_axis.inset_axes([0.15, 0.62, 0.66, 0.35])
                self.plot_hidden_states_vs_state(zoomed_axis, plotting_data, add_legend=False, add_starts=False,
                                                 alpha=hidden_state_alpha)
                self.plot_probe_hidden_states(zoomed_axis, probe_hidden_states[probe_clz])
                x1, x2, y1, y2 = -1.01, -0.91, 0.01, 0.21
                zoomed_axis.set_xlim(x1, x2)
                zoomed_axis.set_ylim(y1, y2)
                config_axis(zoomed_axis)
                hidden_state_axis.indicate_inset_zoom(zoomed_axis, edgecolor="k")

        # Add reward colour bar
        cb_axis = fig.add_subplot(time_on_pad_gs[-1])
        cb = plt.colorbar(mpl.cm.ScalarMappable(cmap=self.cmap, norm=self.reward_color_norm), ax=cb_axis,
                          location='right', fraction=1.0)
        cb.ax.tick_params(labelsize=6)
        cb.set_ticks([-0.5, 0, 1, 2])
        cb.set_label(label='Reward', size=7)
        cb_axis.set_axis_off()

        plt.tight_layout()
        return fig

    def plot_reward_vs_pos_xy(self, fig, axis, plotting_data, pad_time_idx, cmap, add_cbar=True, lw=3):
        # Generate all heatmaps if not already done (required for back filling values)
        if self.reward_heatmaps is None:
            self.reward_heatmaps = self._generate_reward_heatmap_values(plotting_data)

        # Get heatmap values for this charge value
        xx, yy, all_zs = self.reward_heatmaps
        zs = all_zs[:, :, pad_time_idx]

        # Heatmap plotting
        heatmap_method = HeatmapMethod.PCOLORMESH
        self.plot_heatmap(axis, xx, yy, zs, cmap, self.reward_color_norm, heatmap_method)

        # Add map features
        self._plot_map_features(axis, lw)

        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel('X Position')
        axis.set_ylabel('Y Position')
        axis.set_title('Reward vs Position\n(Time On Pad = {:s})'.format(self.pad_times[pad_time_idx]))

        # if add_cbar:
        #     fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=self.reward_color_norm), ax=axis)

    def plot_probe_trajectory(self, axis, probe_xy, lw=1):
        marker_size = 15
        # Add map features
        self._plot_map_features(axis, lw)
        # Line
        axis.plot(probe_xy[:, 0], probe_xy[:, 1], linestyle='--', color='k', zorder=1, alpha=0.5, lw=0.5)
        # Middle markers
        axis.scatter(probe_xy[1:-1, 0], probe_xy[1:-1, 1], marker='.', norm=mpl.colors.Normalize(vmin=0, vmax=500),
                     c=list(range(len(probe_xy) - 2)), s=marker_size, zorder=2, cmap='cool')
        # Start marker
        axis.scatter(probe_xy[0, 0], probe_xy[0, 1], marker='^', color='c',
                     s=marker_size, zorder=2, label='Start')
        # End marker
        axis.scatter(probe_xy[-1, 0], probe_xy[-1, 1], marker='o', color='r',
                     s=marker_size, zorder=2, label='End')
        # axis.set_xlabel('Pos X')
        # axis.set_ylabel('Pos Y')
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_title('Trajectory')

    def plot_probe_hidden_states(self, axis, probe_hs):
        marker_size = 5
        axis.plot(probe_hs[:, 0], probe_hs[:, 1], linestyle='--', color='k', zorder=1, lw=0.5, alpha=0.5)
        # Start
        axis.scatter(probe_hs[0, 0], probe_hs[0, 1], marker='^', color='c',
                     s=marker_size, zorder=3, label='Start')
        # Middle
        axis.scatter(probe_hs[1:-1, 0], probe_hs[1:-1, 1], marker='.', color='b', s=marker_size, zorder=2)
        # End
        axis.scatter(probe_hs[-1, 0], probe_hs[-1, 1], marker='o', color='r',
                     s=marker_size, zorder=3, label='End')
        # axis.set_xlabel('')
        # axis.set_ylabel('')
        # axis.set_title('Hidden State')

    @overrides
    def get_save_path(self):
        return "{:s}/interpretability_probe_{:d}.png".format(self.get_and_make_fig_dir(), self.repeat_num)

    def _generate_reward_heatmap_values(self, plotting_data):
        heatmap_grid_size = 50

        # Get plotting data
        all_time_on_pad = plotting_data['time_on_pad']
        all_rs = plotting_data['reward_preds']
        all_xs = plotting_data['true_pos_x']
        all_ys = plotting_data['true_pos_y']

        # Normalise
        #   0 to 1 for x and y
        #   Mul by 100 to get original reward
        all_xs = (all_xs - all_xs.min()) / (all_xs.max() - all_xs.min())
        all_ys = (all_ys - all_ys.min()) / (all_ys.max() - all_ys.min())
        all_rs *= 100

        # All heatmap values across all frames
        all_zs = torch.full((heatmap_grid_size, heatmap_grid_size, len(self.pad_times)), torch.nan)

        xx = None
        yy = None

        # Iterate through the remainder of the episode, using the previous frame values to fill in missing gaps
        for idx in tqdm(range(len(self.pad_times)), desc='Generating heatmap values', leave=False):
            time_on_pad = self.pad_times[idx]

            if time_on_pad in ["0", "49", "50"]:
                frame_idxs = torch.where(all_time_on_pad == int(time_on_pad))[0]
            elif time_on_pad == '1-48':
                frame_idxs_gt_0 = torch.where(all_time_on_pad > 0)[0]
                frame_idxs_lt_50 = torch.where(all_time_on_pad < 50)[0]
                frame_idxs = np.intersect1d(frame_idxs_gt_0, frame_idxs_lt_50)
            elif time_on_pad == '>50':
                frame_idxs = torch.where(all_time_on_pad > 50)[0]
            else:
                raise NotImplementedError("No frame conversion found for time on pad: {:}".format(time_on_pad))

            if idx == 0:
                xx, yy, zs = self.generate_heatmap_values(all_xs[frame_idxs], all_ys[frame_idxs], all_rs[frame_idxs],
                                                          heatmap_grid_size, prev_values=None)
            else:
                _, _, zs = self.generate_heatmap_values(all_xs[frame_idxs], all_ys[frame_idxs], all_rs[frame_idxs],
                                                        heatmap_grid_size, prev_values=all_zs[:, :, idx - 1])
            all_zs[:, :, idx] = zs

        return xx, yy, all_zs

    @staticmethod
    def _create_heatmap():
        cmap_main = mpl.colormaps['cividis']
        r, g, b, _ = cmap_main(0)
        cmap_neg = mpl.colors.LinearSegmentedColormap("", {
            'red': [(0.0, 0.0, 1.0),
                    (1.0, r, 1.0)],
            'green': [(0.0, 0.0, g),
                      (1.0, g, 1.0)],
            'blue': [(0.0, 0.0, b),
                     (1.0, b, 1.0)]
        })
        main_colours = cmap_main(np.linspace(0, 1, 256))
        neg_colours = cmap_neg(np.linspace(0, 1, 64))
        new_colours = np.concatenate((neg_colours, main_colours))
        cmap = mpl.colors.ListedColormap(new_colours)
        return cmap

    def _generate_probes(self, probe_clzs, plotting_data, probe_idxs):
        probe_clzs = probe_clzs[:]
        probe_counts = dict(zip(probe_clzs, [0, 0, 0, 0]))
        probe_idxs = dict(zip(probe_clzs, probe_idxs))

        probe_xys = dict(zip(probe_clzs, [None, None, None, None]))
        probe_hidden_states = dict(zip(probe_clzs, [None, None, None, None]))
        probe_return_targets = dict(zip(probe_clzs, [None, None, None, None]))

        xs = plotting_data['true_pos_x']
        ys = plotting_data['true_pos_y']
        hidden_states = plotting_data['hidden_states']
        return_targets = plotting_data['return_targets']

        xs = (xs - xs.min()) / (xs.max() - xs.min())
        ys = (ys - ys.min()) / (ys.max() - ys.min())

        i = 0
        for bag_clz, bag_size in zip(plotting_data['bag_clzs'], plotting_data['bag_sizes']):
            bag_clz = int(bag_clz)
            s_idx = i
            e_idx = i + bag_size
            i = e_idx

            if bag_clz in probe_clzs:
                if probe_counts[bag_clz] == probe_idxs[bag_clz]:
                    bag_xs = xs[s_idx:e_idx]
                    bag_ys = ys[s_idx:e_idx]
                    bag_hs = hidden_states[s_idx:e_idx]
                    probe_xys[bag_clz] = torch.stack([bag_xs, bag_ys], dim=1)
                    probe_hidden_states[bag_clz] = bag_hs
                    probe_return_targets[bag_clz] = return_targets[e_idx - 1]
                    probe_clzs.remove(bag_clz)
                    if len(probe_clzs) == 0:
                        break
                probe_counts[bag_clz] += 1

        #     if bag_clz == 8 and return_targets[e_idx - 1] > 7:
        #         print(probe_counts[8])
        # exit(0)

        return probe_xys, probe_hidden_states, probe_return_targets

    def _plot_map_features(self, axis, lw):
        self.add_bounding_box(axis, [[0.4, 0.18], [0.6, 0.25]], 'orange', lw=lw, alpha=1)
        self.add_bounding_box(axis, [[0.25, 0.5], [0.75, 0.7]], 'green', lw=lw, alpha=1)
        self.add_bounding_box(axis, [[0.45, 0.72], [0.55, 0.82]], 'm', lw=lw, alpha=1)
