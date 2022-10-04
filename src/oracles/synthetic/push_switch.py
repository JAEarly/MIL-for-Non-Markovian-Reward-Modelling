import numpy as np

from oracles._abstract import AbstractOracle


class PushSwitchOracle(AbstractOracle):
    """
    Data generating process where the first input represents the pressing of a push button that toggles an internal
     state and the second input is a random value.
    The reward is the current internal state multiplied by the second input value.
    In this example, pushes can be observed but the internal state is unknown from the observed state alone.

    Input: push_button (0 or 1), value [0, 1]
    Internal state: switch_state (0 or 1)
    Reward: switch_state * input[1]
    """

    name = "push_switch"
    input_shape = (2,)
    input_names = ["push_button", "input_value"]
    internal_state_shape = (1,)

    def init_internal_state(self):
        return np.zeros(self.internal_state_shape, dtype=int)

    def update_internal_state(self, instance):
        if instance[0] == 1:
            self.internal_state[0] = 1 - self.internal_state[0]

    def calculate_reward(self, instance):
        return self.internal_state[0] * instance[1]

    @staticmethod
    def create_bags(num_bags, min_bag_size, max_bag_size, seed, **kwargs):
        push_proba = kwargs['push_proba']
        rng = np.random.default_rng(seed)
        bags = []
        for _ in range(num_bags):
            bag_size = rng.integers(min_bag_size, max_bag_size, endpoint=True)
            bag = np.zeros((bag_size, 2))
            for i in range(bag_size):
                if rng.random() < push_proba:
                    bag[i, 0] = 1
            bag[:, 1] = rng.random(bag_size)
            bags.append(bag)
        return bags
