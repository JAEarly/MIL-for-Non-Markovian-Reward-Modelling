import numpy as np
from lib.env import HoloNav
import time
import matplotlib.pyplot as plt

from oracles._abstract import AbstractOracle
from rl_training.maps import maps
from rl_training.wrappers import OracleWrapper


class MovingTreasureOracle(AbstractOracle):
    """
    Data generating process where the 2D input represents the x,y position of an agent
    in a bounded 2D environment containing a treasure chest which gives +1 reward per timestep.
    The treasure chest moves left and right in cycles over time.

    Input: x_position [0, 1], y_position [0, 1]
    Internal state: treasure_x_position [0, 1-treasure_width], treasure_x_velocity [-treasure_speed, treasure_speed]
    Reward: 1 * at_treasure
    """

    name = "movingtreasure"
    input_shape = (2,)
    input_names = ["x_position", "y_position"]
    internal_state_shape = (2,)

    treasure = maps["movingtreasure"]["boxes"]["treasure"]["coords"]
    treasure_speed = 0.02
    treasure_width = treasure[1][0] - treasure[0][0]

    def init_internal_state(self):
        return np.array([0.5-(self.treasure_width/2), -self.treasure_speed])

    def update_internal_state(self, instance):
        self.internal_state[0] += self.internal_state[1]
        if not (1e-4 < self.internal_state[0] < (1-(self.treasure_width+1e-4))):
            self.internal_state[1] *= -1

    def calculate_reward(self, instance):
        mn, mx = self.internal_state[0], self.internal_state[0] + self.treasure_width
        at_treasure = mn <= instance[0] <= mx and self.treasure[0][1] <= instance[1] <= self.treasure[1][1]
        return 1. if at_treasure else 0.

    def env_mod_callback(self, env):
        el = env.map_elements["treasure"]
        if el is not None: el.set_xy((self.internal_state[0], self.treasure[0][1]))

    @classmethod
    def create_bags(cls, num_bags, min_bag_size, max_bag_size, seed, **kwargs):
        """
        Generate episodes using a uniform random policy, but post-filter so that we end up
        with a high-entropy distribution of num_treasure counts. This is achieved by
        limiting the number of episodes with each possible count to max_num_per_outcome.
        Increasing this value speeds up data generation, but also increases the proportion
        of easily-achieved (i.e. poor) outcomes.
        """
        env = OracleWrapper(HoloNav(
            map=maps["movingtreasure"],
            # continuous=False, action_noise=("gaussian", 0.2),
            render_mode="human" if kwargs["render"] else False
            ),
            oracle=cls(), augment_state=False)
        env.seed(seed)
        bags = []; n = 0; n_per_outcome = np.zeros(max_bag_size, dtype=int)
        t_start = time.time()
        while n < num_bags:
            bag = np.zeros((max_bag_size, cls.input_shape[0])) # NOTE: min_bag_size unused
            bag[0] = env.reset()
            if kwargs["render"]: env.render()
            num_treasure = 0
            for t in range(1, max_bag_size):
                bag[t], reward, _, _ = env.step(env.action_space.sample()) # Uniform random action selection
                if reward > 0: num_treasure += 1 # Positive reward means treasure
                if kwargs["render"]: env.render()
            if n_per_outcome[num_treasure] < kwargs["max_num_per_outcome"]:
                bags.append(bag); n_per_outcome[num_treasure] += 1; n += 1
                t = time.time() - t_start
                print(f"bags: {n} / {num_bags}, {t}")
                if "wandb" in kwargs: kwargs["wandb"].log({"num_bags": n})
                if kwargs["plot_outcomes"]: plt.scatter(t, n, c="k", s=3)
        if kwargs["plot_outcomes"]:
            plt.figure()
            plt.imshow(n_per_outcome.reshape(1,-1))
            plt.axis("auto"); plt.xlabel("num_treasure"); plt.colorbar()
            plt.show()
        return bags
