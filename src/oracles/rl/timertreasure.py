import numpy as np
from lib.env import HoloNav
import time
import matplotlib.pyplot as plt

from oracles._abstract import AbstractOracle
from rl_training.maps import maps
from rl_training.wrappers import OracleWrapper


class TimerTreasureOracle(AbstractOracle):
    """
    Data generating process where the 2D input represents the x,y position of an agent
    in a bounded 2D environment containing a treasure chest which is "poisoned" for
    t <= wear_off_time, giving -1 reward per timestep. For t > wear_off_time the poison
    wears off, and the treasure chest gives +1 reward per timestep.

    Input: x_position [0, 1], y_position [0, 1]
    Internal state: time [0, max_bag_size]
    Reward: (-1 if time <= flip_time else +1) * at_treasure
    """

    name = "timertreasure"
    input_shape = (2,)
    input_names = ["x_position", "y_position"]
    internal_state_shape = (1,)

    treasure = maps["timertreasure"]["boxes"]["treasure"]["coords"]
    wear_off_time = 50

    def init_internal_state(self):
        return np.zeros(self.internal_state_shape, dtype=int)

    def update_internal_state(self, instance):
        self.internal_state[0] += 1

    def calculate_reward(self, instance):
        at_treasure = self.treasure[0][0] <= instance[0] <= self.treasure[1][0] and self.treasure[0][1] <= instance[1] <= self.treasure[1][1]
        if at_treasure:
            if self.internal_state[0] > self.wear_off_time: return 1.
            else: return -1.
        return 0.

    def env_mod_callback(self, env):
        if self.internal_state[0] == self.wear_off_time: env._set_activation("treasure", 1)

    @classmethod
    def create_bags(cls, num_bags, min_bag_size, max_bag_size, seed, **kwargs):
        """
        Generate episodes using a uniform random policy, but post-filter so that we end up
        with a high-entropy joint distribution in the outcome space (num_treasure, num_poison).
        This is achieved by limiting the number of episodes at each point in the space to
        max_num_per_outcome. Increasing this value speeds up data generation, but also
        increases the proportion of easily-achieved (i.e. poor) outcomes.
        """
        env = OracleWrapper(HoloNav(
            map=maps["timertreasure"],
            # continuous=False, action_noise=("gaussian", 0.2),
            render_mode="human" if kwargs["render"] else False
            ),
            oracle=cls(), augment_state=False)
        env.seed(seed)
        bags = []; n = 0; n_per_outcome = np.zeros((max_bag_size, max_bag_size), dtype=int)
        t_start = time.time()
        while n < num_bags:
            bag = np.zeros((max_bag_size, cls.input_shape[0])) # NOTE: min_bag_size unused
            bag[0] = env.reset()
            if kwargs["render"]: env.render()
            num_treasure, num_poison = 0, 0
            for t in range(1, max_bag_size):
                bag[t], reward, _, _ = env.step(env.action_space.sample()) # Uniform random action selection
                if reward > 0: num_treasure += 1 # Positive reward means treasure
                elif reward < 0: num_poison += 1 # Negative reward means poison
                if kwargs["render"]: env.render()
            if n_per_outcome[num_treasure, num_poison] < kwargs["max_num_per_outcome"]:
                bags.append(bag); n_per_outcome[num_treasure, num_poison] += 1; n += 1
                t = time.time() - t_start
                print(f"bags: {n} / {num_bags}, {t}")
                if "wandb" in kwargs: kwargs["wandb"].log({"num_bags": n})
                if kwargs["plot_outcomes"]: plt.scatter(t, n, c="k", s=3)
        if kwargs["plot_outcomes"]:
            plt.figure()
            plt.imshow(n_per_outcome)
            plt.xlabel("num_poison"); plt.ylabel("num_treasure"); plt.colorbar()
            plt.show()
        return bags
