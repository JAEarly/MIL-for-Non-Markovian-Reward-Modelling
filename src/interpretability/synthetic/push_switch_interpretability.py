import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from dataset.synthetic.push_switch_dataset import PushSwitchOracleDataset
from interpretability.oracle_interpretability import OracleMILInterpretabilityStudy


class PushSwitchOracleInterpretabilityStudy(OracleMILInterpretabilityStudy):

    dataset_clz = PushSwitchOracleDataset

    def __init__(self, device, model_clz, model_path, dataset_name, csv_path, seed=5):
        dataset_params = {
            'csv_path': csv_path,
        }
        super().__init__(device, model_clz, model_path, dataset_name, dataset_params, seed=seed)
        self.min_hidden_state_0 = None
        self.max_hidden_state_0 = None
        self.min_hidden_state_1 = None
        self.max_hidden_state_1 = None

    def get_additional_plotting_data_overall(self, model, dataset):
        all_switch_state = []
        all_hidden_states = []
        all_jc_0_to_1s = []
        all_jc_1_to_0s = []
        for idx in range(len(dataset)):
            bag, _, instance_targets = dataset[idx]
            hidden_states = model.get_hidden_states(bag).detach().cpu()
            all_hidden_states.append(hidden_states)
            bag_metadata = dataset.bags_metadata[idx]
            all_switch_state.append(torch.as_tensor(bag_metadata['switch_state']))
            all_jc_0_to_1s.append(torch.as_tensor(bag_metadata['just_changed_0_to_1']))
            all_jc_1_to_0s.append(torch.as_tensor(bag_metadata['just_changed_1_to_0']))
        all_hidden_states = torch.cat(all_hidden_states)
        all_switch_state = torch.cat(all_switch_state)
        all_jc_0_to_1s = torch.cat(all_jc_0_to_1s)
        all_jc_1_to_0s = torch.cat(all_jc_1_to_0s)
        additional_plotting_data = {
            'switch_state': all_switch_state,
            'hidden_states': all_hidden_states,
            'jc_0_to_1': all_jc_0_to_1s,
            'jc_1_to_0': all_jc_1_to_0s,
        }
        return additional_plotting_data

    def get_additional_plotting_data_single(self, model, dataset, idx):
        bag, _, instance_targets = dataset[idx]
        bag_metadata = dataset.bags_metadata[idx]
        hidden_states = torch.as_tensor(model.get_hidden_states(bag).detach().cpu())
        switch_state = torch.as_tensor(bag_metadata['switch_state'])
        jc_0_to_1 = torch.as_tensor(bag_metadata['just_changed_0_to_1'])
        jc_1_to_0 = torch.as_tensor(bag_metadata['just_changed_1_to_0'])
        additional_plotting_data = {
            'hidden_states': hidden_states,
            'switch_state': switch_state,
            'jc_0_to_1': jc_0_to_1,
            'jc_1_to_0': jc_1_to_0,
        }
        return additional_plotting_data

    def set_additional_plot_bounds(self, all_plotting_data):
        self.min_hidden_state_0 = min([s[0] for s in all_plotting_data['hidden_states']])
        self.max_hidden_state_0 = max([s[0] for s in all_plotting_data['hidden_states']])
        self.min_hidden_state_1 = min([s[1] for s in all_plotting_data['hidden_states']])
        self.max_hidden_state_1 = max([s[1] for s in all_plotting_data['hidden_states']])
        self.min_hidden_state_0, self.max_hidden_state_0 = self.expand_min_max(self.min_hidden_state_0,
                                                                               self.max_hidden_state_0)
        self.min_hidden_state_1, self.max_hidden_state_1 = self.expand_min_max(self.min_hidden_state_1,
                                                                               self.max_hidden_state_1)

    def plot_overall_output(self, name, plotting_data):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(13, 4))
        self.plot_return_overall(axes[0], plotting_data)
        self.plot_reward_overall(axes[1], plotting_data)
        self.plot_reward_vs_trigger(axes[2], plotting_data)
        self.plot_hidden_states(axes[3], plotting_data)
        fig.suptitle("{:s} Summary".format(self.dataset_name, name))
        return fig

    def plot_single_output(self, name, plotting_data):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(13, 4))
        self.plot_return_preds_vs_targets(axes[0], plotting_data)
        self.plot_reward_preds_vs_targets(axes[1], plotting_data)
        self.plot_reward_vs_trigger(axes[2], plotting_data)
        self.plot_hidden_states(axes[3], plotting_data)
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
        axis.scatter(return_targets, return_preds, alpha=0.5, marker='x')
        axis.set_xlim(self.min_return, self.max_return)
        axis.set_ylim(self.min_return, self.max_return)
        axis.set_xlabel('Target')
        axis.set_ylabel('Predicted')
        axis.set_title('Return Overall')

    def plot_reward_overall(self, axis, plotting_data):
        reward_preds = plotting_data['reward_preds']
        reward_targets = plotting_data['reward_targets']
        axis.scatter(reward_targets, reward_preds, alpha=0.5, marker='x')
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
        switch_state = plotting_data['switch_state']
        xs = range(len(reward_preds))
        axis.plot(xs, reward_preds, label='Predicted', drawstyle="steps-pre", alpha=0.5)
        axis.plot(xs, reward_targets, label='Target', drawstyle="steps-pre", alpha=0.5)
        axis.plot(xs, switch_state, label='Switch', drawstyle="steps-pre", alpha=0.2, linestyle='--')
        axis.set_xlim(0, len(xs) - 1)
        axis.set_ylim(self.min_reward, self.max_reward)
        axis.set_xlabel('Time')
        axis.set_ylabel('Reward')
        axis.legend(loc='upper left')
        axis.set_title('Reward vs Time')

    def plot_reward_vs_trigger(self, axis, plotting_data):
        reward_preds = plotting_data['reward_preds']
        switch_state = plotting_data['switch_state']
        switch_off = [i for i, s in enumerate(switch_state) if s == 0]
        switch_on = [i for i, s in enumerate(switch_state) if s == 1]
        axis.hist(np.asarray(reward_preds[switch_off]), bins=50, alpha=0.5)
        axis.hist(np.asarray(reward_preds[switch_on]), bins=50, alpha=0.5)
        axis.set_xlim(self.min_reward, self.max_reward)
        axis.set_xlabel('Reward')
        axis.set_ylabel('Density')
        axis.set_title('Reward vs Switch')

    def plot_hidden_states(self, axis, plotting_data):
        hidden_states = plotting_data['hidden_states']
        switch_state = plotting_data['switch_state']

        switch_on = [i for i, s in enumerate(switch_state) if s == 1]
        switch_off = [i for i, s in enumerate(switch_state) if s == 0]
        jc_0_to_1 = [i for i, s in enumerate(plotting_data['jc_0_to_1']) if s == 1]
        jc_1_to_0 = [i for i, s in enumerate(plotting_data['jc_1_to_0']) if s == 1]

        sample_rate = 100
        off_not_jc = [s for s in switch_off if s not in jc_1_to_0][::sample_rate]
        off_jc = [s for s in switch_off if s in jc_1_to_0][::sample_rate]
        on_not_jc = [s for s in switch_on if s not in jc_0_to_1][::sample_rate]
        on_jc = [s for s in switch_on if s in jc_0_to_1][::sample_rate]

        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        axis.scatter(hidden_states[off_not_jc, 0], hidden_states[off_not_jc, 1],
                     marker='.', alpha=0.5, color=color_cycle[0])
        axis.scatter(hidden_states[off_jc, 0], hidden_states[off_jc, 1],
                     marker='.', alpha=0.5, color=color_cycle[2])
        axis.scatter(hidden_states[on_not_jc, 0], hidden_states[on_not_jc, 1],
                     marker='.', alpha=0.5, color=color_cycle[1])
        axis.scatter(hidden_states[on_jc, 0], hidden_states[on_jc, 1],
                     marker='.', alpha=0.5, color=color_cycle[3])

        axis.set_xlim(self.min_hidden_state_0, self.max_hidden_state_0)
        axis.set_ylim(self.min_hidden_state_1, self.max_hidden_state_1)
        axis.set_xlabel('Hidden State 0')
        axis.set_ylabel('Hidden State 1')
        axis.set_title('Hidden State')
