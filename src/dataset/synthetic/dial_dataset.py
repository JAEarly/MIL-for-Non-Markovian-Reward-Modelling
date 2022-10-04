import csv

import numpy as np
import torch

from dataset.oracle_dataset import OracleDataset


class DialOracleDataset(OracleDataset):

    d_in = 2  # dial movement, input_value
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
                fv = torch.as_tensor([float(f) for f in row[1:3]])
                reward = float(row[3])
                bag_dict[bag_num].append((fv, reward))

        # Convert episode dictionary into MIL format
        bags = []
        targets = []
        instance_targets = []
        bags_metadata = []
        for bag_num in range(len(bag_dict)):
            bag_data = bag_dict[bag_num]
            # Create bag by stacking env feature vectors
            instances = [d[0] for d in bag_data]
            bag = torch.stack(instances)

            # Get rewards for each time step; overall target is sum of rewards
            rewards = torch.as_tensor([d[1] for d in bag_data]).float()
            target = rewards.sum()

            # Collect bag meta data
            dial_value = np.cumsum([fv[0] for fv in instances])
            bag_metadata = {
                'dial_value': dial_value,  # value on dial (cumulative sum of dial movements)
            }

            # Add to data lists
            bags.append(bag)
            targets.append(target)
            instance_targets.append(rewards)
            bags_metadata.append(bag_metadata)

        return bags, targets, instance_targets, bags_metadata
