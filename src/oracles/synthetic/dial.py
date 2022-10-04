import numpy as np

from oracles._abstract import AbstractOracle


class DialOracle(AbstractOracle):
    """
    Data generating process where the first input represents the change in value of a dial. The value of the dial is
     tracked by an internal state. The second input is a random value.
    The reward is the current value on the dial plus by the second input value.
    This is a generalisation of the push switch oracle to be continuous rather than binary.

    Input: dial_movement [-0.5, 0.5], value [-0.5, 0.5]
    Internal state: dial_value
    Reward: dial_value + input[1]
    """

    name = "dial"
    input_shape = (2,)
    input_names = ["dial_movement", "input_value"]
    internal_state_shape = (1,)

    def init_internal_state(self):
        return np.zeros(self.internal_state_shape, dtype=float)

    def update_internal_state(self, instance):
        # Move dial by given amount
        self.internal_state[0] += instance[0]

    def calculate_reward(self, instance):
        # Reward = dial value + new value in
        return self.internal_state[0] + instance[1]

    @staticmethod
    def create_bags(num_bags, min_bag_size, max_bag_size, seed, **kwargs):
        dial_move_proba = kwargs['dial_move_proba']
        rng = np.random.default_rng(seed)
        bags = []
        for _ in range(num_bags):
            bag_size = rng.integers(min_bag_size, max_bag_size, endpoint=True)
            bag = np.zeros((bag_size, 2))
            for i in range(bag_size):
                # First see if we want to move the dial
                if rng.random() < dial_move_proba:
                    # Then we move it between -0.5 and 0.5
                    bag[i, 0] = rng.random() - 0.5
                else:
                    bag[i, 0] = 0
            bag[:, 1] = rng.random(bag_size) - 0.5
            bags.append(bag)
        return bags
