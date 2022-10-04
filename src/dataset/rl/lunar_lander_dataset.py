import csv

import numpy as np
import torch
from matplotlib import pyplot as plt
from overrides import overrides
from torch import nn
from tqdm import tqdm

import dataset
from dataset.oracle_dataset import OracleDataset
# from pytorch_mil.train.metrics import MinimiseRegressionMetric


class LunarLanderDataset(OracleDataset):

    d_in = 8
    n_expected_dims = 2  # n_instances * d_bag
    n_classes = 1

    @classmethod
    def load_data(cls, csv_path=None):
        # Parse data from csv file and group by bag
        bag_dict = {}
        with open(csv_path) as f:
            r = csv.reader(f)
            next(r)  # Skip header
            for row in r:
                bag_num = int(row[0])
                if bag_num not in bag_dict:
                    bag_dict[bag_num] = []
                fv = torch.as_tensor([float(f) for f in row[1:9]])
                reward = float(row[9])
                bag_dict[bag_num].append((fv, reward))

        # Convert episode dictionary into MIL format
        bags = []
        targets = []
        instance_targets = []
        for bag_num in range(len(bag_dict)):
            bag_data = bag_dict[bag_num]
            # Create bag by stacking env feature vectors
            instances = [d[0] for d in bag_data]
            bag = torch.stack(instances)

            # Get rewards for each time step; overall target is sum of rewards
            rewards = torch.as_tensor([d[1] for d in bag_data]).float()
            target = rewards.sum()

            # Add to data lists
            bags.append(bag)
            targets.append(target)
            instance_targets.append(rewards)

        bags_metadata = cls.load_metadata(bags, csv_path)
        return bags, targets, instance_targets, bags_metadata

    @classmethod
    @overrides
    def normalise(cls, bags, targets, normalise_inputs=True, normalise_targets=False):

        def _normalise(features, fmin, fmax):
            return (features - fmin) / (fmax - fmin) - 0.5

        def _standardise(features, mean, std):
            return (features - mean) / std

        if normalise_inputs:
            all_instances = torch.cat(bags)
            cls.instance_mean = torch.mean(all_instances, dim=0)
            cls.instance_std = torch.std(all_instances, dim=0)
            cls.instance_min = torch.min(all_instances, dim=0)[0]
            cls.instance_max = torch.max(all_instances, dim=0)[0]

            # print(instance_min)
            # print(instance_max)

            norm_bags = []
            for bag in bags:
                norm_bag = torch.zeros_like(bag)
                for feature_idx in range(8):
                    features = bag[:, feature_idx]
                    if feature_idx in [6, 7]:
                        norm_bag[:, feature_idx] = features - 0.5
                    elif feature_idx in [0, 1, 2, 3, 4, 5]:
                        norm_features = _normalise(features,
                                                   cls.instance_min[feature_idx],
                                                   cls.instance_max[feature_idx])
                        norm_bag[:, feature_idx] = norm_features

                norm_bags.append(norm_bag)

            # all_norm_instances = torch.cat(norm_bags)
            # fig, axes = plt.subplots(nrows=2, ncols=8, figsize=(12, 5))
            # for idx in range(8):
            #     axes[0][idx].hist(all_instances[:, idx].numpy())
            #     axes[1][idx].hist(all_norm_instances[:, idx].numpy())
            # plt.tight_layout()
            # plt.show()
            # exit(0)

        else:
            norm_bags = bags

        if normalise_targets:
            targets = torch.as_tensor(targets)
            # target_mean = torch.mean(targets)
            # target_std = torch.std(targets)
            # norm_targets = (targets - target_mean) / target_std
            norm_targets = torch.log(1 + targets)

            # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 5))
            # axes[0].hist(targets.numpy(), bins=50)
            # axes[1].hist(norm_targets.numpy(), bins=50)
            # plt.tight_layout()
            # plt.show()
            #
            # exit(0)
        else:
            norm_targets = targets

        return norm_bags, norm_targets

    @classmethod
    def generate_metadata(cls, bags):
        print('Dataset metadata not found, generating now...')
        bags_metadata = []
        for bag_idx, bag in tqdm(enumerate(bags), desc='Generating metadata', leave=False, total=len(bags)):
            bag_metadata, _ = cls.generate_bag_metadata(bag)
            bags_metadata.append(bag_metadata)
        return bags_metadata

    @classmethod
    def generate_bag_metadata(cls, bag):
        land_pad_width = 0.2
        in_hover = torch.zeros(len(bag))
        on_pad = torch.zeros(len(bag))
        time_on_pad = torch.zeros(len(bag))
        take_off = torch.zeros(len(bag))
        cur_time_on_pad = 0
        landed = False  # Is the lander currently considered landed (has been on pad and not taken off again)

        # Iterate through bag
        for idx, instance in enumerate(bag):
            # Parse instance
            x, y, _, _, _, _, l_contact, r_contact = instance

            # Determine if in hover zone for this instance
            if (-0.5 <= x <= 0.5) and (0.75 <= y <= 1.25):
                in_hover[idx] = 1

            # Determine if on pad for this instance
            if -land_pad_width <= x <= land_pad_width and l_contact > 0.5 and r_contact > 0.5:
                on_pad[idx] = 1
                cur_time_on_pad += 1
                landed = True

            # Determine if taken off after landing
            if landed and y > 0.2 and l_contact < 0.5 and r_contact < 0.5:
                take_off[idx] = 1
                landed = False

            # Log time on pad up to this point
            time_on_pad[idx] = cur_time_on_pad

        # Work out bag class
        total_time_on_pad = time_on_pad.max()
        num_take_offs = take_off.sum()
        total_time_in_hover_after_t_1 = in_hover[torch.where(time_on_pad >= 1)].sum()
        total_time_in_hover_after_t_50 = in_hover[torch.where(time_on_pad >= 50)].sum()

        if total_time_on_pad == 0:
            # 0 - Pad never landed on
            assert 1 not in on_pad
            assert 1 not in take_off
            bag_clz = 0
        else:
            if total_time_on_pad < 50:
                if num_take_offs == 0:
                    # 1 - Pad landed on; num steps on pad < 50; no take off
                    bag_clz = 1
                else:
                    if total_time_in_hover_after_t_1 == 0:
                        # 2 - Pad landed on; num steps on pad < 50; one or more take offs; in hover = 0
                        bag_clz = 2
                    elif total_time_in_hover_after_t_1 <= 20:
                        # 3 - Pad landed on; num steps on pad < 50; one or more take offs; 0 < in hover <= 20
                        bag_clz = 3
                    else:
                        # 4 - Pad landed on; num steps on pad < 50; one or more take offs; in hover > 20
                        bag_clz = 4

            else:
                if num_take_offs == 0:
                    # 5 - Pad landed on; num steps on pad >= 50; no take off
                    bag_clz = 5
                else:
                    if total_time_in_hover_after_t_50 == 0:
                        # 6 - Pad landed on; num steps on pad >= 50; one or more take off; in hover = 0
                        bag_clz = 6
                    elif total_time_in_hover_after_t_50 <= 20:
                        # 7 - Pad landed on; num steps on pad >= 50; one or more take off; 0 < in hover <= 20
                        bag_clz = 7
                    else:
                        # 8 - Pad landed on; num steps on pad >= 50; one or more take off; in hover > 20
                        bag_clz = 8

        # Wrap all required info up in dict
        bag_metadata = {
            'true_pos_x': bag[:, 0],
            'true_pos_y': bag[:, 1],
            'on_pad': on_pad,
            'time_on_pad': time_on_pad,
            'in_hover': in_hover,
            'bag_clz': bag_clz,
        }
        return bag_metadata, bag_clz

    @classmethod
    def plot_bag_clz_hist(cls, bags_metadata):
        bag_clz_dist = {}
        for clz_idx in range(9):
            bag_clz_dist[clz_idx] = 0

        for bm in bags_metadata:
            bag_clz_dist[bm['bag_clz']] += 1

        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
        bars = axis.bar(range(9), [bag_clz_dist[idx] for idx in range(9)])
        axis.bar_label(bars)
        axis.set_xlabel('Bag class')
        axis.set_ylabel('Num bags')
        axis.set_xticks(ticks=list(range(9)))

        clz_strings = [
            '0 - Pad never landed on',
            '1 - Pad landed on; num steps on pad < 50; no take off',
            '2 - Pad landed on; num steps on pad < 50; one or more take offs; in hover = 0',
            '3 - Pad landed on; num steps on pad < 50; one or more take offs; 0 < in hover <= 20',
            '4 - Pad landed on; num steps on pad < 50; one or more take offs; in hover > 20',
            '5 - Pad landed on; num steps on pad >= 50; no take off',
            '6 - Pad landed on; num steps on pad >= 50; one or more take off; in hover = 0',
            '7 - Pad landed on; num steps on pad >= 50; one or more take off; 0 < in hover <= 20',
            '8 - Pad landed on; num steps on pad >= 50; one or more take off; in hover > 20',
        ]
        handles = [plt.Rectangle((0, 0), 0.1, 0.1,  fill=False, edgecolor='none', visible=False)
                   for _ in range(len(clz_strings))]

        axis.legend(handles, clz_strings, loc='best', prop={'size': 7})
        plt.show()

    @classmethod
    def show_return_hist(cls):
        csv_path = dataset.get_dataset_path_from_name("lunar_lander")
        _, targets, instance_targets, _ = cls.load_data(csv_path)
        targets = torch.stack(targets).numpy()
        instance_targets = torch.cat(instance_targets).numpy()

        print('{:d} bags'.format(len(targets)))
        print('{:d} instances'.format(len(instance_targets)))

        print('Return min:', min(targets))
        print('Return max:', max(targets))

        print('Reward min:', min(instance_targets))
        print('Reward max:', max(instance_targets))

        return_counts, return_bins = np.histogram(targets, bins=30)
        reward_counts, reward_bins = np.histogram(instance_targets, bins=30)

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        axes[0].hist(return_bins[:-1], return_bins, weights=return_counts)
        axes[0].set_xlabel('Return')
        axes[0].set_ylabel('Freq')
        axes[1].hist(reward_bins[:-1], reward_bins, weights=reward_counts)
        axes[1].set_xlabel('Reward')
        axes[1].set_ylabel('Freq')
        plt.show()

    @classmethod
    def baseline_performance(cls):
        print('Calculating baseline performance (avg reward prediction)')

        # Get targets and instance targets
        csv_path = dataset.get_dataset_path_from_name("lunar_lander")
        train, val, test = cls.create_datasets(csv_path=csv_path)
        all_instance_targets = torch.cat(train.instance_targets)

        # Calculate mean instance target and from this the mean target
        mean_instance_target = torch.mean(all_instance_targets)

        def _performance(dataset):
            # Create mean predictions of same dimensionality as true predictions
            bag_sizes = torch.as_tensor([len(bag) for bag in dataset.bags])
            targets = dataset.targets
            instance_targets = torch.cat(dataset.instance_targets)
            return_preds = mean_instance_target * bag_sizes
            reward_preds = mean_instance_target.repeat(len(instance_targets))

            # Calculate return and reward metrics
            criterion = lambda ps, ts: nn.MSELoss()(ps.squeeze(), ts.squeeze())
            labels = list(range(cls.n_classes))
            return_result = MinimiseRegressionMetric.calculate_metric(return_preds, targets, criterion, labels)
            reward_result = MinimiseRegressionMetric.calculate_metric(reward_preds, instance_targets, criterion, labels)
            print('Return MSE: {:.3f}'.format(return_result.loss))
            print('Reward MSE: {:.3f} (1e-5)'.format(reward_result.loss / 1e-5))

        print('Train')
        _performance(train)
        print('Val')
        _performance(val)
        print('Test')
        _performance(test)


if __name__ == "__main__":
    _csv_path = dataset.get_dataset_path_from_name("lunar_lander")
    _, _, _, _bags_metadata = LunarLanderDataset.load_data(_csv_path)
    LunarLanderDataset.plot_bag_clz_hist(_bags_metadata)
    # LunarLanderDataset.show_return_hist()
    # LunarLanderDataset.baseline_performance()
