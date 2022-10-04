import os
import pickle as pkl
from abc import ABC, abstractmethod

import torch
from sklearn.model_selection import train_test_split

from pytorch_mil.data.mil_dataset import MilDataset


class OracleDataset(MilDataset, ABC):

    @classmethod
    def create_datasets(cls, seed=12, **kwargs):
        bags, targets, instance_targets, bags_metadata = cls.load_data(**kwargs)
        bags, targets = cls.normalise(bags, targets, normalise_targets=False)

        # Train/Test Split
        # TODO probably should wrap this up in a helper method
        splits = train_test_split(bags, targets, instance_targets, bags_metadata, train_size=0.8, random_state=seed)
        train_bags, train_targets, train_instance_targets, train_bags_metadata = [splits[i] for i in [0, 2, 4, 6]]
        test_bags, test_targets, test_instance_targets, test_bags_metadata = [splits[i] for i in [1, 3, 5, 7]]

        # Val/Test Split
        splits = train_test_split(test_bags, test_targets, test_instance_targets, test_bags_metadata,
                                  train_size=0.5, random_state=seed)
        val_bags, val_targets, val_instance_targets, val_bags_metadata = [splits[i] for i in [0, 2, 4, 6]]
        test_bags, test_targets, test_instance_targets, test_bags_metadata = [splits[i] for i in [1, 3, 5, 7]]

        # Actually make datasets
        train_dataset = cls(train_bags, train_targets, train_instance_targets, train_bags_metadata)
        val_dataset = cls(val_bags, val_targets, val_instance_targets, val_bags_metadata)
        test_dataset = cls(test_bags, test_targets, test_instance_targets, test_bags_metadata)
        return train_dataset, val_dataset, test_dataset

    @classmethod
    def create_complete_dataset(cls, **kwargs):
        bags, targets, instance_targets, bags_metadata = cls.load_data(**kwargs)
        bags, targets = cls.normalise(bags, targets)
        return cls(bags, targets, instance_targets, bags_metadata)

    @classmethod
    def load_metadata(cls, bags, csv_path):
        # Generate metadata from bags if it doesn't exist
        metadata_path = csv_path + ".metadata"
        if not os.path.exists(metadata_path):
            bags_metadata = cls.generate_metadata(bags)
            # Save to file for future use
            with open(metadata_path, 'wb+') as f:
                pkl.dump(bags_metadata, f)
            return bags_metadata
        with open(metadata_path, 'rb') as f:
            return pkl.load(f)

    @classmethod
    @abstractmethod
    def load_data(cls, **kwargs):
        pass

    @classmethod
    @abstractmethod
    def generate_metadata(cls, bags):
        pass

    @classmethod
    def normalise(cls, bags, targets, normalise_targets=False):
        all_instances = torch.cat(bags)
        cls.instance_mean = torch.mean(all_instances, dim=0)
        cls.instance_std = torch.std(all_instances, dim=0)
        norm_bags = []
        for bag in bags:
            norm_bag = (bag - cls.instance_mean) / cls.instance_std
            norm_bags.append(norm_bag)
        if normalise_targets:
            targets = torch.as_tensor(targets)
            target_mean = torch.mean(targets)
            target_std = torch.std(targets)
            norm_targets = (targets - target_mean) / target_std
        else:
            norm_targets = targets
        return norm_bags, norm_targets
