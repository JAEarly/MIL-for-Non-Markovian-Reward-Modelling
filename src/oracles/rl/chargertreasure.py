import numpy as np
from lib.env import HoloNav
import time
import matplotlib.pyplot as plt

from oracles._abstract import AbstractOracle
from rl_training.maps import maps
from rl_training.wrappers import OracleWrapper


class ChargerTreasureOracle(AbstractOracle):
    """
    Data generating process where the 2D input represents the x,y position of an agent
    in a bounded 2D environment containing two 'boxes' of interest: a treasure chest and a charging zone.
    When the agent enters the treasure box, the reward given per timestep is proportional to the total
    number of timesteps it has previously spent inside the charging zone (0 if never visited, capped at 1).

    Input: x_position [0, 1], y_position [0, 1]
    Internal state: charge_level [0, 1]
    Reward: charge_level * at_treasure
    """

    name = "chargertreasure"
    input_shape = (2,)
    input_names = ["x_position", "y_position"]
    internal_state_shape = (1,)

    charge_zone = maps["chargertreasure"]["boxes"]["charge_zone"]["coords"]
    treasure = maps["chargertreasure"]["boxes"]["treasure"]["coords"]
    charge_rate = 0.02

    num_charge_bins = 20 # Used in create_bags only

    def init_internal_state(self):
        return np.zeros(self.internal_state_shape, dtype=float)

    def update_internal_state(self, instance):
        at_charge_zone = self.charge_zone[0][0] <= instance[0] <= self.charge_zone[1][0] \
                         and self.charge_zone[0][1] <= instance[1] <= self.charge_zone[1][1]
        if at_charge_zone: self.internal_state[0] = min(self.internal_state[0] + self.charge_rate, 1.)

    def _at_treasure(self, instance):
        return self.treasure[0][0] <= instance[0] <= self.treasure[1][0] and self.treasure[0][1] <= instance[1] <= self.treasure[1][1]

    def calculate_reward(self, instance):
        return float(self.internal_state[0] * self._at_treasure(instance))

    def env_mod_callback(self, env):
        el = env.map_elements["charge_bar"]
        if el is not None: el.set_width(self.internal_state[0])

    @classmethod
    def create_bags(cls, num_bags, min_bag_size, max_bag_size, seed, **kwargs):
        """
        Generate episodes using a uniform random policy, but post-filter so that we end up
        with a high-entropy joint distribution in the outcome space (num_treasure, charge_bin),
        where charge_bin is a binned representation of the *mean* charge level when in the treasure.
        This is achieved by limiting the number of episodes at each point in the space to
        max_num_per_outcome. Increasing this value speeds up data generation, but also
        increases the proportion of easily-achieved (i.e. poor) outcomes.
        """
        env = OracleWrapper(HoloNav(
            map=maps["chargertreasure"],
            # continuous=False, action_noise=("gaussian", 0.2),
            render_mode="human" if kwargs["render"] else False
            ),
            oracle=cls(), augment_state=False)
        env.seed(seed)
        bags = []; n = 0; n_per_outcome = np.zeros((max_bag_size, cls.num_charge_bins), dtype=int)
        t_start = time.time()
        while n < num_bags:
            bag = np.zeros((max_bag_size, cls.input_shape[0])) # NOTE: min_bag_size unused
            bag[0] = env.reset()
            if kwargs["render"]: env.render()
            num_treasure, retrn = 0, 0.
            for t in range(1, max_bag_size):
                bag[t], reward, _, _ = env.step(env.action_space.sample()) # Uniform random action selection
                if env.oracle._at_treasure(bag[t-1]):
                    num_treasure += 1; retrn += reward
                if kwargs["render"]: env.render()
            charge_bin = int(round((cls.num_charge_bins - 1) * (retrn / num_treasure))) if num_treasure > 0 else 0
            if n_per_outcome[num_treasure, charge_bin] < kwargs["max_num_per_outcome"]:
                bags.append(bag); n_per_outcome[num_treasure, charge_bin] += 1; n += 1
                t = time.time() - t_start
                print(f"bags: {n} / {num_bags}, {t}")
                if "wandb" in kwargs: kwargs["wandb"].log({"num_bags": n})
                if kwargs["plot_outcomes"]: plt.scatter(t, n, c="k", s=3)
        if kwargs["plot_outcomes"]:
            plt.figure()
            plt.imshow(n_per_outcome)
            plt.axis("auto"); plt.xlabel("charge_bin"); plt.ylabel("num_treasure"); plt.colorbar()
            plt.show()
        return bags
