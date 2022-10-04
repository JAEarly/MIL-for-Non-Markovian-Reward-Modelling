import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AbstractOracle(ABC):
    """
    Abstract hidden-state oracle class for generating 
    temporally-dependent outputs given a sequence of instances.
    """
    def __init__(self):
        self.reset()
    
    def __call__(self, instance):
        """
        Call the oracle. First updates the state, then returns a scalar value.
        """
        assert type(instance) == np.ndarray and instance.shape == self.input_shape
        self.update_internal_state(instance)
        assert type(self.internal_state) == np.ndarray and self.internal_state.shape == self.internal_state_shape
        return self.calculate_reward(instance)

    @classmethod
    @property
    @abstractmethod
    def name(cls):
        pass

    @classmethod
    @property
    @abstractmethod
    def input_shape(cls):
        pass

    @classmethod
    @property
    @abstractmethod
    def internal_state_shape(cls):
        pass

    @classmethod
    @property
    @abstractmethod
    def input_names(cls):
        pass

    @abstractmethod
    def init_internal_state(self):
        pass

    @abstractmethod
    def update_internal_state(self, instance):
        pass

    @abstractmethod
    def calculate_reward(self, instance):
        pass

    @classmethod
    @abstractmethod
    def create_bags(cls, num_bags, min_bag_size, max_bag_size, seed, **kwargs):
        pass

    @classmethod
    def generate_dataset(cls, num_bags, min_bag_size, max_bag_size, seed, save_path=None, **kwargs):
        # Create oracle and bags
        print('Generating dataset')
        oracle = cls()
        print('Creating bags')
        bags = cls.create_bags(num_bags, min_bag_size, max_bag_size, seed, **kwargs)

        # Calculate rewards and aggregate all bag data
        print('Calculating rewards')
        data = []
        for bag_idx, bag in enumerate(bags):
            # Reset oracle each time we generate values for a bag
            oracle.reset()
            # Generate values for each instance in the bag
            instance_values = np.array([oracle(instance) for instance in bag])
            if "label_scale_factor" in kwargs:
                # Multiply by scale factor if applicable
                instance_values *= kwargs["label_scale_factor"]
            # Create data entries for this bag by hstacking columns (bag_idx, bag content, instance values)
            data.append(np.hstack((np.full((len(bag), 1), bag_idx), bag, instance_values.reshape(-1, 1))))

        # Create dataframe
        print('Creating dataframe')
        df = pd.DataFrame(np.vstack(data), columns=["bag_number"] + cls.input_names + ["output_value"])
        df = df.convert_dtypes()

        # Save dataframe to file
        if save_path is None:
            save_path = oracle.get_default_save_path()
        print('Saving to dataframe to {:s}'.format(save_path))
        save_dir = save_path[:save_path.rindex("/")]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        df.to_csv(save_path, index=False)
        print('Done!')

    def get_default_save_path(cls):
        return "data/oracle/{:s}/{:s}.csv".format(cls.name, cls.name)

    def reset(self):
        self.internal_state = self.init_internal_state()

    def env_mod_callback(self, env):
        pass
