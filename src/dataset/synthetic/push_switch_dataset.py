import csv

import numpy as np
import torch

from dataset.oracle_dataset import OracleDataset


class PushSwitchOracleDataset(OracleDataset):

    d_in = 2  # push_button, input_value
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

            switch_state = np.zeros((len(instances)))
            t = 0
            for i in range(len(switch_state)):
                if instances[i][0] == 1:
                    t = 1 - t
                switch_state[i] = t
            just_changed_0_to_1 = torch.zeros((len(instances)), dtype=torch.long)
            just_changed_1_to_0 = torch.zeros((len(instances)), dtype=torch.long)
            for idx, s in enumerate(switch_state):
                if idx == 0:
                    if s == 1:
                        just_changed_0_to_1[0] = 1
                else:
                    if s == 1 and switch_state[idx - 1] == 0:
                        just_changed_0_to_1[idx] = 1
                    if s == 0 and switch_state[idx - 1] == 1:
                        just_changed_1_to_0[idx] = 1

            # Collect bag meta data
            bag_metadata = {
                'switch_state': switch_state,  # internal switch state
                'just_changed_0_to_1': just_changed_0_to_1,
                'just_changed_1_to_0': just_changed_1_to_0
            }

            # Add to data lists
            bags.append(bag)
            targets.append(target)
            instance_targets.append(rewards)
            bags_metadata.append(bag_metadata)

        return bags, targets, instance_targets, bags_metadata
