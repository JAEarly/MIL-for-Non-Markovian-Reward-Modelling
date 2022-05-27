import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from dataset.synthetic.dial_dataset import DialOracleDataset
from interpretability.oracle_interpretability import OracleMILInterpretabilityStudy


class DialOracleInterpretabilityStudy(OracleMILInterpretabilityStudy):

    dataset_clz = DialOracleDataset

    def __init__(self, device, model_clz, model_path, dataset_name, csv_path, seed=5):
        dataset_params = {
            'csv_path': csv_path,
        }
        super().__init__(device, model_clz, model_path, dataset_name, dataset_params, seed=seed)
        self.min_dial_value = None
        self.max_dial_value = None
        self.min_hidden_state_0 = None
        self.max_hidden_state_0 = None
        self.min_hidden_state_1 = None
        self.max_hidden_state_1 = None
        self.color_norm = None

    def get_additional_plotting_data_overall(self, model, dataset):
        all_dial_value = []
        all_hidden_states = []
        for idx in range(len(dataset)):
            bag, _, instance_targets = dataset[idx]
            hidden_states = model.get_hidden_states(bag).detach().cpu()
            all_hidden_states.append(hidden_states)
            bag_metadata = dataset.bags_metadata[idx]
            all_dial_value.append(torch.as_tensor(bag_metadata['dial_value']))
        all_hidden_states = torch.cat(all_hidden_states)
        all_dial_value = torch.cat(all_dial_value)
        additional_plotting_data = {
            'dial_value': all_dial_value,
            'hidden_states': all_hidden_states,
        }
        return additional_plotting_data

    def get_additional_plotting_data_single(self, model, dataset, idx):
        bag, _, instance_targets = dataset[idx]
        bag_metadata = dataset.bags_metadata[idx]
        hidden_states = model.get_hidden_states(bag).detach().cpu()
        dial_value = torch.as_tensor(bag_metadata['dial_value'])
        additional_plotting_data = {
            'hidden_states': hidden_states,
            'dial_value': dial_value,
        }
        return additional_plotting_data

    def set_additional_plot_bounds(self, all_plotting_data):
        self.min_dial_value = min(all_plotting_data['dial_value'])
        self.max_dial_value = max(all_plotting_data['dial_value'])
        self.min_hidden_state_0 = min([s[0] for s in all_plotting_data['hidden_states']])
        self.max_hidden_state_0 = max([s[0] for s in all_plotting_data['hidden_states']])
        self.min_hidden_state_1 = min([s[1] for s in all_plotting_data['hidden_states']])
        self.max_hidden_state_1 = max([s[1] for s in all_plotting_data['hidden_states']])
        self.color_norm = mpl.colors.Normalize(vmin=self.min_dial_value, vmax=self.max_dial_value)
        self.min_dial_value, self.max_dial_value = self.expand_min_max(self.min_dial_value, self.max_dial_value)
        self.min_hidden_state_0, self.max_hidden_state_0 = self.expand_min_max(self.min_hidden_state_0,
                                                                               self.max_hidden_state_0)
        self.min_hidden_state_1, self.max_hidden_state_1 = self.expand_min_max(self.min_hidden_state_1,
                                                                               self.max_hidden_state_1)

    def plot_overall_output(self, name, plotting_data):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(13, 4))
        self.plot_return_overall(axes[0], plotting_data)
        self.plot_reward_overall(axes[1], plotting_data)
        self.plot_reward_vs_dial(axes[2], plotting_data)
        self.plot_hidden_states(fig, axes[3], plotting_data)
        fig.suptitle("{:s} Summary".format(self.dataset_name, name))
        return fig

    def plot_single_output(self, name, plotting_data):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(13, 4))
        self.plot_return_preds_vs_targets(axes[0], plotting_data)
        self.plot_reward_preds_vs_targets(axes[1], plotting_data)
        self.plot_reward_vs_dial(axes[2], plotting_data)
        self.plot_hidden_states(fig, axes[3], plotting_data)
        return_preds = plotting_data['return_preds']
        final_return_target = plotting_data['return_targets'][-1]
        reward_preds = plotting_data['reward_preds']
        reward_targets = plotting_data['reward_targets']
        return_loss = nn.MSELoss()(return_preds[-1], final_return_target)
        reward_loss = nn.MSELoss()(reward_preds, reward_targets)
        fig.suptitle("{:s}: {:s} \n".format(self.dataset_name, name) +
                     "Target: {:.3f} Prediction: {:.3f}\n".format(final_return_target, return_preds[-1]) +
                     "Return Loss: {:.3f} Reward Loss: {:.3f}".format(return_loss, reward_loss)
                     )
        return fig

    def plot_return_overall(self, axis, plotting_data):
        return_preds = plotting_data['return_preds']
        return_targets = plotting_data['return_targets']
        axis.scatter(return_targets, return_preds, alpha=0.5, marker='.')
        axis.set_xlim(self.min_return, self.max_return)
        axis.set_ylim(self.min_return, self.max_return)
        axis.set_xlabel('Target')
        axis.set_ylabel('Predicted')
        axis.set_title('Return Overall')

    def plot_reward_overall(self, axis, plotting_data):
        reward_preds = plotting_data['reward_preds']
        reward_targets = plotting_data['reward_targets']
        axis.scatter(reward_preds, reward_targets, alpha=0.5, marker='.')
        axis.set_xlim(self.min_reward, self.max_reward)
        axis.set_ylim(self.min_reward, self.max_reward)
        axis.set_xlabel('Target')
        axis.set_ylabel('Predicted')
        axis.set_title('Reward Overall')

    def plot_return_preds_vs_targets(self, axis, plotting_data):
        return_preds = plotting_data['return_preds']
        return_targets = plotting_data['return_targets']
        xs = range(len(return_preds))
        axis.plot(xs, return_preds, label='Predicted')
        axis.plot(xs, return_targets, label='Target')
        axis.set_xlim(0, len(xs) - 1)
        axis.set_ylim(self.min_return, self.max_return)
        axis.set_xlabel('Time')
        axis.set_ylabel('Return')
        axis.legend(loc='upper left')
        axis.set_title('Return vs Time')

    def plot_reward_preds_vs_targets(self, axis, plotting_data):
        reward_preds = plotting_data['reward_preds']
        reward_targets = plotting_data['reward_targets']
        dial_value = plotting_data['dial_value']
        xs = range(len(reward_preds))
        axis.plot(xs, reward_preds, label='Predicted', alpha=0.5)  # drawstyle="steps-pre"
        axis.plot(xs, reward_targets, label='Target', alpha=0.5)
        axis.plot(xs, dial_value, label='Dial Value', alpha=0.2, linestyle='--')
        axis.set_xlim(0, len(xs) - 1)
        axis.set_ylim(self.min_reward, self.max_reward + 0.6)
        axis.set_xlabel('Time')
        axis.set_ylabel('Reward')
        axis.legend(loc='upper left')
        axis.set_title('Reward vs Time')

    def plot_reward_vs_dial(self, axis, plotting_data):
        dial_values = np.asarray(plotting_data['dial_value'])
        reward_preds = np.asarray(plotting_data['reward_preds'])

        # This is the basic approach
        axis.scatter(dial_values, reward_preds, marker='.', alpha=0.5)

        # This works quite well
        # axis.hist2d(dial_values, reward_preds, bins=100, norm=mpl.colors.LogNorm(), cmap='hot')

        # This looks okay - bit small
        # sns.kdeplot(x=dial_values, y=reward_preds, shade=True, ax=axis, gridsize=50, hue_norm=mpl.colors.LogNorm(),
        #             thresh=0.01)

        min_ = min(self.min_dial_value, self.min_reward_pred)
        max_ = max(self.max_dial_value, self.max_reward_pred)
        # axis.plot([min_, max_], [min_, max_], color='k', linestyle='--')

        # corr, _ = pearsonr(dial_values, reward_preds)

        axis.set_xlim(min_, max_)
        axis.set_ylim(min_, max_)
        axis.set_xlabel('Dial Value')
        axis.set_ylabel('Predicted Reward')
        axis.set_title('Reward vs Dial')
        # axis.text(0.98, 0.02, 'r = {:.3f}'.format(corr), horizontalalignment='right', verticalalignment='bottom',
        #           transform=axis.transAxes)

    def plot_hidden_states(self, fig, axis, plotting_data, add_colorbar=True):
        dial_values = plotting_data['dial_value']

        hidden_states = plotting_data['hidden_states']
        axis.scatter([s[0] for s in hidden_states], [s[1] for s in hidden_states], marker='.',
                     c=dial_values, cmap=mpl.colormaps['cividis'], norm=self.color_norm)
        axis.set_xlim(self.min_hidden_state_0, self.max_hidden_state_0)
        axis.set_ylim(self.min_hidden_state_1, self.max_hidden_state_1)
        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State')
        if add_colorbar:
            fig.colorbar(mpl.cm.ScalarMappable(cmap=mpl.colormaps['cividis'], norm=self.color_norm), ax=axis)
