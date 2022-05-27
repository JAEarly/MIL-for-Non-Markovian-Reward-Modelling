import csv
from abc import ABC

import numpy as np
import torch

from dataset.oracle_dataset import OracleDataset


class RLDataset(OracleDataset, ABC):

    d_in = 2  # pos_x, pos_y
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

        # Load metadata
        bags_metadata = cls.load_metadata(bags, csv_path)
        return bags, targets, instance_targets, bags_metadata

    def permute_targets(self, noise_level, seed, verbose=True):
        permuted_targets = self._run_label_permutation(self.targets, noise_level, seed, verbose=verbose)
        self.targets = torch.as_tensor(permuted_targets, dtype=torch.float32)

    @staticmethod
    def _run_label_permutation(original_targets, noise_level, seed, verbose=True):
        np.random.seed(seed)

        assert 0 <= noise_level <= 1
        n_to_permute = int(len(original_targets) * noise_level)

        if verbose:
            print('Permuting labels')
            print('  Noise level: {:.3f}'.format(noise_level))
            print('  Seed: {:d}'.format(seed))
            print('  Number to permute: {:d} ({:.2f}%)'.format(n_to_permute, n_to_permute/len(original_targets) * 100))

        # Permute everything
        # Convert original targets to np array just in case it isn't already
        original_targets = np.asarray(original_targets, dtype=float)

        attempt = 1
        max_attempts = 1000
        best_permutation = None
        best_error = None
        while True:
            if verbose:
                print('Permutation attempt {:d}'.format(attempt))
            permuted_targets = RLDataset._run_label_permutation_attempt(original_targets)
            matching_indices = np.where(original_targets == permuted_targets)[0]
            if len(matching_indices) == 0:
                if verbose:
                    print(' Found!')
                best_permutation = permuted_targets
                break
            else:
                attempt += 1
                if best_error is None or len(matching_indices) < best_error:
                    best_error = len(matching_indices)
                    best_permutation = permuted_targets
                if attempt == max_attempts:
                    if verbose:
                        print('Reached max number of attempts to find a valid permutation')
                    break

        matching_indices = np.where(original_targets == best_permutation)[0]
        if verbose:
            print('Best permutation found had {:d} matches ({:.2f}%)'.format(len(matching_indices),
                                                                             len(matching_indices)/len(original_targets)))

        # Verify both lists actually contain the same targets (sorted lists should match)
        sorted_original_targets = sorted(original_targets)
        sorted_permuted_targets = sorted(best_permutation)
        if sorted_original_targets != sorted_permuted_targets:
            raise ValueError('Indices contain different targets!')

        # Mask random labels to permute based on noise level to create final targets
        mask = np.zeros_like(original_targets)
        mask[:n_to_permute] = 1
        np.random.shuffle(mask)
        final_targets = np.where(mask == 1, best_permutation, original_targets)

        if verbose:
            print('Label permutation completed. Target = {:.4f}; Actual = {:.4f}'.format(
                noise_level, 1 - len(np.where(original_targets == final_targets)[0])/len(original_targets)
            ))

        return final_targets

    @staticmethod
    def _run_label_permutation_attempt(original_targets):
        # Create empty array of permuted targets
        permuted_targets = np.full_like(original_targets, np.nan)
        # Create shuffled order of original targets
        shuffled_targets = np.copy(original_targets)
        np.random.shuffle(shuffled_targets)
        for t_idx, t in enumerate(original_targets):
            # Find permutation for this target
            for s_idx, s in enumerate(shuffled_targets):
                if s != t:
                    permuted_targets[t_idx] = s
                    shuffled_targets = np.delete(shuffled_targets, s_idx)
                    break
            # If we can't find a non-matching target, choose one at random
            else:
                s_idx = np.random.choice(range(len(shuffled_targets)), 1)[0]
                permuted_targets[t_idx] = shuffled_targets[s_idx]
                shuffled_targets = np.delete(shuffled_targets, s_idx)
        return permuted_targets
