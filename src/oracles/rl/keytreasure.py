import numpy as np
from lib.env import HoloNav

from oracles._abstract import AbstractOracle
from rl_training.maps import maps
from rl_training.wrappers import OracleWrapper


class KeyTreasureOracle(AbstractOracle):
    """
    Data generating process where the 2D input represents the x,y position of an agent
    in a bounded 2D environment containing two 'boxes' of interest: a treasure chest and a key.
    To obtain positive reward, the agent must first enter the key box, then the treasure box.

    Input: x_position [0, 1], y_position [0, 1]
    Internal state: has_key (0 or 1)
    Reward: has_key * at_treasure
    """

    name = "keytreasure"
    input_shape = (2,)
    input_names = ["x_position", "y_position"]
    internal_state_shape = (1,)

    key = maps["keytreasure_A"]["boxes"]["key"]["coords"]
    treasure = maps["keytreasure_A"]["boxes"]["treasure"]["coords"]

    def init_internal_state(self):
        return np.zeros(self.internal_state_shape, dtype=int)

    def update_internal_state(self, instance):
        at_key = self.key[0][0] <= instance[0] <= self.key[1][0] and self.key[0][1] <= instance[1] <= self.key[1][1]
        if at_key: self.internal_state[0] = 1

    def calculate_reward(self, instance):
        at_treasure = self.treasure[0][0] <= instance[0] <= self.treasure[1][0] and self.treasure[0][1] <= instance[1] <= self.treasure[1][1]
        return float(self.internal_state[0] * at_treasure)

    def env_mod_callback(self, env):
        if not(env.map["boxes"]["treasure"]["active"]) and self.internal_state[0] > 0.5: env._set_activation("treasure", 1)

    @classmethod
    def create_bags(cls, num_bags, min_bag_size, max_bag_size, seed, **kwargs):
        """
        Generate episodes using a uniform random policy, but post-filter
        so that we end up with a specified ratio of outcome types, where
            type 0 = failure
            type 1 = key found
            type 2 = treasure found
        """
        r = np.array(kwargs["outcome_ratios"])
        num_bags_per_outcome = num_bags * r / r.sum()
        assert (num_bags_per_outcome % 1 == 0).all(), "Must have integer num_bags_per_outcome"
        num_bags_per_outcome = np.round(num_bags_per_outcome).astype(int)
        env = OracleWrapper(HoloNav(
            map=maps[f"keytreasure_{kwargs['map_layout']}"],
            # continuous=False, action_noise=("gaussian", 0.2),
            render_mode="human" if kwargs["render"] else False
            ),
            oracle=cls(), augment_state=False)
        env.seed(seed)
        bags = [[],[],[]]; n = 0
        while n < num_bags:
            bag = np.zeros((max_bag_size, cls.input_shape[0])) # NOTE: min_bag_size unused
            bag[0] = env.reset()
            if kwargs["render"]: env.render()
            outcome_type = 0
            for t in range(1, max_bag_size):
                bag[t], reward, _, _ = env.step(env.action_space.sample()) # Uniform random action selection
                if reward > 0:
                    outcome_type = 2 # Positive reward means treasure reached
                elif outcome_type == 0 and env.oracle.internal_state[0] == 1:
                    outcome_type = 1 # Internal state 1 means key found
                if kwargs["render"]: env.render()
            if len(bags[outcome_type]) < num_bags_per_outcome[outcome_type]:
                bags[outcome_type].append(bag); n += 1
                print(", ".join([f"{o}: {len(x)}/{n}" for o, x, n in 
                    zip(["failure", "key found", "treasure found"], bags, num_bags_per_outcome)]))
        return [bag for b in bags for bag in b]
