import numpy as np

from oracles._abstract import AbstractOracle


class ToggleSwitchOracle(AbstractOracle):
    """
    Simple data generating process where the first input represents the position of a toggle switch (0 or 1) and the
     second input is a random value.
    The reward is the toggle switch position multiplied by the second input value.
    This is a "null" example where the internal state (toggle switch position) is entirely captured in the observed
     state.

    Input: toggle_position (0 or 1), value [0, 1]
    Internal state: toggle_position (0 or 1)
    Reward: toggle_position * input[1]
    """

    name = "toggle_switch"
    input_shape = (2,)
    input_names = ["toggle_position", "input_value"]
    internal_state_shape = (1,)

    def init_internal_state(self):
        return np.zeros(self.internal_state_shape, dtype=int)

    def update_internal_state(self, instance):
        self.internal_state[0] = instance[0]

    def calculate_reward(self, instance):
        return self.internal_state[0] * instance[1]

    @staticmethod
    def create_bags(num_bags, min_bag_size, max_bag_size, seed, **kwargs):
        toggle_proba = kwargs['toggle_proba']
        rng = np.random.default_rng(seed)
        bags = []
        position = 0
        for _ in range(num_bags):
            bag_size = rng.integers(min_bag_size, max_bag_size, endpoint=True)
            bag = np.zeros((bag_size, 2))
            for i in range(bag_size):
                if rng.random() < toggle_proba:
                    position = 1 - position
                bag[i, 0] = 1 - position
            bag[:, 1] = rng.random(bag_size)
            bags.append(bag)
        return bags
