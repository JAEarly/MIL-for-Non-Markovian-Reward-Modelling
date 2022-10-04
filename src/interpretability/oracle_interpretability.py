import os
import pickle as pkl
from abc import ABC, abstractmethod

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from enum import Enum


from pytorch_mil.train.metrics import MinimiseRegressionMetric


class HeatmapMethod(Enum):
    PCOLORMESH = 1
    CONTOURF = 2
    IMG = 3


class OracleMILInterpretabilityStudy(ABC):

    def __init__(self, device, dataset_clz, model_clz, model_path, dataset_name, dataset_params, seed=5):
        self.device = device
        self.dataset_clz = dataset_clz
        self.model_clz = model_clz
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.dataset_params = dataset_params
        self.seed = seed

    def run_evaluation(self, verbose=False):
        model, split_dataset = self.load_experiment_model(), self.load_experiment_split_dataset()
        train_dataset, val_dataset, test_dataset = split_dataset

        # Calculate performance
        train_return_result, train_reward_result = \
            self.calculate_model_instance_prediction_performance(model, train_dataset)
        if verbose:
            print('-- Train Performance --')
            print('MSE Return (Bag) Loss: {:.3f}'.format(train_return_result.loss))
            print('MSE Reward (Ins) Loss: {:.3f}'.format(train_reward_result.loss))
        val_return_result, val_reward_result = \
            self.calculate_model_instance_prediction_performance(model, val_dataset)
        if verbose:
            print('-- Val Performance --')
            print('MSE Return (Bag) Loss: {:.3f}'.format(val_return_result.loss))
            print('MSE Reward (Ins) Loss: {:.3f}'.format(val_reward_result.loss))
        test_return_result, test_reward_result = \
            self.calculate_model_instance_prediction_performance(model, test_dataset)
        if verbose:
            print('-- Test Performance --')
            print('MSE Return (Bag) Loss: {:.3f}'.format(test_return_result.loss))
            print('MSE Reward (Ins) Loss: {:.3f}'.format(test_reward_result.loss))
        return train_return_result, train_reward_result, val_return_result, val_reward_result, \
            test_return_result, test_reward_result

    def load_experiment_complete_dataset(self):
        if self.dataset_params is not None:
            dataset = self.dataset_clz.create_complete_dataset(**self.dataset_params)
        else:
            dataset = self.dataset_clz.create_complete_dataset()
        return dataset

    def load_experiment_split_dataset(self):
        if self.dataset_params is not None:
            split_dataset = self.dataset_clz.create_datasets(**self.dataset_params, seed=self.seed)
        else:
            split_dataset = self.dataset_clz.create_datasets(seed=self.seed)
        return split_dataset

    def load_experiment_model(self):
        return self.model_clz.load_model(self.device, self.model_path, self.dataset_clz.d_in,
                                         self.dataset_clz.n_expected_dims)

    def calculate_model_instance_prediction_performance(self, model, dataset):
        # Calculate MSE losses
        criterion = lambda outputs, targets: nn.MSELoss()(outputs.squeeze(), targets.squeeze())
        all_return_preds, all_reward_preds = self.get_model_output_for_dataset(model, dataset)

        # Get the labels that we're actually trying to predict over
        labels = list(range(model.n_classes))

        # Calculate return result
        final_return_preds = torch.as_tensor([p[-1] for p in all_return_preds])
        return_result = MinimiseRegressionMetric.calculate_metric(final_return_preds, dataset.targets,
                                                                  criterion, labels)

        # Calculate reward result
        flat_reward_preds = torch.cat(all_reward_preds)
        flat_reward_targets = torch.cat(dataset.instance_targets)
        reward_result = MinimiseRegressionMetric.calculate_metric(flat_reward_preds, flat_reward_targets,
                                                                  criterion, labels)

        return return_result, reward_result

    def get_model_output_for_dataset(self, model, dataset):
        all_return_preds = []
        all_reward_preds = []
        for bag in dataset.bags:
            return_preds, reward_preds = self.get_model_output_for_bag(model, bag)
            all_return_preds.append(return_preds)
            all_reward_preds.append(reward_preds)
        return all_return_preds, all_reward_preds

    @staticmethod
    def get_model_output_for_bag(model, bag):
        with torch.no_grad():
            return_preds, reward_preds = model.forward_returns_and_rewards(bag)
            return_preds = return_preds.cpu()
            reward_preds = reward_preds.cpu()
            return return_preds, reward_preds


class GlobalStudyPlotter(ABC):

    def __init__(self, study, repeat_num, add_start_hidden_states=True):
        self.study = study
        self.repeat_num = repeat_num
        self.add_start_hidden_states = add_start_hidden_states

    @abstractmethod
    def plot(self, plotting_data):
        pass

    def run_plotting(self, show_plots=True):
        # Get plotting data
        plotting_data = self.get_all_plotting_data()

        # Plot
        print('Plotting figure')
        fig = self.plot(plotting_data)

        # Save
        print('Saving')
        fig_path = self.get_save_path()
        fig.savefig(fig_path, format='png', dpi=600)
        fig.savefig(fig_path.replace(".png", ".svg"), format='svg', dpi=600)

        # Show
        if show_plots:
            plt.show()

    def run_animation(self):
        # Get plotting data
        plotting_data = self.get_all_plotting_data()
        # Plot
        self.plot(plotting_data)

    def get_all_plotting_data(self):
        # Generate plotting data if it doesn't exist
        data_path = "{:s}/interpretability_global_{:d}.data".format(self.get_and_make_fig_dir(), self.repeat_num)
        if not os.path.exists(data_path):
            print('Existing plotting data not found. Will generate now.')
            # Load model and dataset
            print('  Loading model and dataset')
            model, dataset = self.study.load_experiment_model(), self.study.load_experiment_complete_dataset()
            # Generate plotting data
            print('  Generating plotting data')
            plotting_data = self.generate_plotting_data(model, dataset)
            print('  Saving data')
            with open(data_path, 'wb+') as f:
                pkl.dump(plotting_data, f)
            print('  Done')
        # Load from file
        print('Loading plotting data')
        with open(data_path, 'rb') as f:
            plotting_data = pkl.load(f)
        return plotting_data

    def generate_plotting_data(self, model, dataset):
        all_return_targets = torch.cat([torch.cumsum(its, 0) for its in dataset.instance_targets])
        all_reward_targets = torch.cat(dataset.instance_targets)
        all_return_preds, all_reward_preds = self.study.get_model_output_for_dataset(model, dataset)
        all_return_preds = torch.cat(all_return_preds)
        all_reward_preds = torch.cat(all_reward_preds)
        all_hidden_states = torch.cat([model.get_hidden_states(bag).detach().cpu() for bag in dataset.bags])
        bag_sizes = torch.asarray([len(bag) for bag in dataset.bags])
        plotting_data = {
            'return_preds': all_return_preds,
            'return_targets': all_return_targets,
            'reward_preds': all_reward_preds,
            'reward_targets': all_reward_targets,
            'hidden_states': all_hidden_states,
            'bag_sizes': bag_sizes,
        }

        if self.add_start_hidden_states:
            plotting_data['start_hidden_states'] = self.generate_start_state_plotting_data(dataset, model)

        return plotting_data

    def generate_start_state_plotting_data(self, dataset, model):
        # Get hidden states for start positions (normalise then pass through model)
        start_position_a = (torch.as_tensor([[0.15, 0.5]]) - dataset.instance_mean) / dataset.instance_std
        start_position_b = (torch.as_tensor([[0.85, 0.5]]) - dataset.instance_mean) / dataset.instance_std
        start_hidden_state_a = model.get_hidden_states(start_position_a).detach().cpu()
        start_hidden_state_b = model.get_hidden_states(start_position_b).detach().cpu()
        start_hidden_states = torch.stack([start_hidden_state_a, start_hidden_state_b])
        return start_hidden_states

    def get_and_make_fig_dir(self):
        fig_dir = "out/fig/{:s}/{:s}".format(self.study.dataset_name, self.study.model_clz.__name__)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        return fig_dir

    def get_save_path(self):
        return "{:s}/interpretability_global_{:d}.png".format(self.get_and_make_fig_dir(), self.repeat_num)


class RLGlobalStudyPlotter(GlobalStudyPlotter, ABC):

    def generate_and_plot_heatmap(self, axis, xs, ys, reward_preds, cmap, norm, grid_size, method, prev_values=None):
        xx, yy, zs = self.generate_heatmap_values(xs, ys, reward_preds, grid_size, prev_values=prev_values)
        if method == HeatmapMethod.PCOLORMESH:
            return axis.pcolormesh(xx, yy, zs, cmap=cmap, norm=norm)
        elif method == HeatmapMethod.CONTOURF:
            return axis.contourf(xx, yy, zs, 10, cmap=cmap, norm=norm)
        elif method == HeatmapMethod.IMG:
            return axis.imshow(zs, extent=[0, 1, 0, 1], origin='lower', cmap=cmap, norm=norm,
                               interpolation='bicubic', interpolation_stage='rgba')
        raise ValueError('Invalid heatmap method: {:}'.format(method))

    def plot_heatmap(self, axis, xx, yy, zs, cmap, norm, method):
        if method == HeatmapMethod.PCOLORMESH:
            return axis.pcolormesh(xx, yy, zs, cmap=cmap, norm=norm)
        elif method == HeatmapMethod.CONTOURF:
            return axis.contourf(xx, yy, zs, 10, cmap=cmap, norm=norm)
        elif method == HeatmapMethod.IMG:
            return axis.imshow(zs, extent=[0, 1, 0, 1], origin='lower', cmap=cmap, norm=norm,
                               interpolation='bicubic', interpolation_stage='rgba')
        raise ValueError('Invalid heatmap method: {:}'.format(method))

    def animate_heatmap(self, heatmap, xs, ys, reward_preds, grid_size, method):
        if method == HeatmapMethod.PCOLORMESH:
            prev_values = torch.as_tensor(heatmap.get_array().reshape(grid_size, grid_size).filled(np.nan))
            _, _, new_zs = self.generate_heatmap_values(xs, ys, reward_preds, grid_size, prev_values=prev_values)
            heatmap.set_array(new_zs)
        elif method == HeatmapMethod.CONTOURF:
            raise NotImplementedError
        elif method == HeatmapMethod.IMG:
            prev_values = torch.as_tensor(heatmap.get_array().filled(np.nan))
            _, _, new_zs = self.generate_heatmap_values(xs, ys, reward_preds, grid_size, prev_values=prev_values)
            heatmap.set_array(new_zs)
        else:
            raise ValueError('Invalid heatmap method: {:}'.format(method))

    def generate_heatmap_values(self, true_xs, true_ys, reward_preds, grid_size, prev_values=None):
        # Size of each grid cell
        grid_step = 1/grid_size
        # Offset to align grid with other stuff in plot
        grid_offset = grid_step/2
        xs = np.linspace(grid_offset, 1-grid_offset, grid_size)
        ys = np.linspace(grid_offset, 1-grid_offset, grid_size)
        # Grid of x and y positions (used to plot)
        xx, yy = np.meshgrid(xs, ys)

        # Used to track the current sum of rewards in each cell, as well as the number of points in each cell
        z_sums = torch.zeros((grid_size, grid_size))
        z_counts = torch.zeros((grid_size, grid_size))

        # Iterate through each position
        for idx in range(len(reward_preds)):
            # Get positions and rewards for this idx
            x = true_xs[idx].item()
            y = true_ys[idx].item()
            rp = reward_preds[idx]

            # Convert to grid position
            grid_x = int(x // grid_step)
            grid_y = int(y // grid_step)

            # Update sums and counts
            z_sums[grid_y, grid_x] += rp
            z_counts[grid_y, grid_x] += 1

        # Compute average value at each grid position
        zs = z_sums / z_counts

        if prev_values is not None:
            if type(prev_values) is int:
                prev_values = torch.ones((grid_size, grid_size)) * prev_values
            zs = torch.where(torch.isnan(zs), prev_values, zs)

        return xx, yy, zs

    def generate_probe_plotting_data(self, probes):
        model, dataset = self.study.load_experiment_model(), self.study.load_experiment_complete_dataset()
        all_probe_data = []
        for probe in probes:
            bag = torch.as_tensor(probe, dtype=torch.float32)
            bag = (bag - dataset.instance_mean) / dataset.instance_std
            hidden_states = model.get_hidden_states(bag)
            pred_returns, pred_rewards = model.forward_returns_and_rewards(bag)
            probe_data = {
                'hidden_states': hidden_states.detach().cpu(),
                'pred_returns': pred_returns.detach().cpu(),
                'pred_rewards': pred_rewards.detach().cpu(),
            }
            all_probe_data.append(probe_data)
        return all_probe_data

    @staticmethod
    def expand_axis(min_, max_, scale, fix_min=False, fix_max=False):
        d = max_ - min_
        s = scale * d
        new_min = min_ if fix_min else min_ - s
        new_max = max_ if fix_max else max_ + s
        return new_min, new_max

    @staticmethod
    def add_bounding_box(axis, bounds, color, lw=1, alpha=0.5):
        x0, y0, x1, y1 = bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]
        rect = mpl.patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                     linewidth=lw, edgecolor=color, facecolor='none', alpha=alpha)
        axis.add_patch(rect)
        return rect

    @staticmethod
    def get_random_idxs(n, sample_rate):
        mask = np.zeros(n)
        mask[:int(n * sample_rate)] = 1
        np.random.shuffle(mask)
        return np.where(mask == 1)[0]

    @staticmethod
    def sample_list(idxs, sample_rate):
        n_to_sample = int(len(idxs) * sample_rate)
        return np.random.choice(idxs, size=n_to_sample, replace=False)
