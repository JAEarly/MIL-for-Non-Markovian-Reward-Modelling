from oracles.rl.lunar_lander import LunarLanderOracle
from oracles.rl.lunar_lander_timer import LunarLanderTimerOracle

class OracleObserver:
    """
    For evaluating rewards from an oracle during agent learning,
    *without* actually sending these rewards to the agent.
    """
    def __init__(self, oracle):
        assert len(oracle.input_shape) == 1
        self.oracle, self.P = oracle, {}
        self.run_names, self.ep_reward_sum = [], 0.,
        # self.ep_num_neg, self.ep_num_pos = 0, 0
        self.num_in_hover_zone = 0
        self.num_on_pad = 0

    def per_timestep(self, ep, t, state, action, next_state, reward_ext, done, info, extra):
        reward = self.oracle(state[:self.oracle.input_shape[0]]) # NOTE: Only use environment state
        self.ep_reward_sum += reward
        # if reward < 0: self.ep_num_neg += 1
        # elif reward > 0: self.ep_num_pos += 1

        # =============================================================
        if isinstance(self.oracle, LunarLanderOracle):
            if self.oracle.internal_state[0] >= self.oracle.land_duration \
                and self.oracle._in_hover_zone(state):
                self.num_in_hover_zone += 1
        elif isinstance(self.oracle, LunarLanderTimerOracle):
            if self.oracle.internal_state[0] > self.oracle.flip_time \
                and self.oracle._on_pad(state):
                self.num_on_pad += 1
        # =============================================================

    def per_episode(self, ep):
        logs = {"reward_sum_oracle": self.ep_reward_sum,
                "h_final": self.oracle.internal_state,
                # "num_neg": self.ep_num_neg, "num_pos": self.ep_num_pos
                "num_in_hover_zone": self.num_in_hover_zone,
                "num_on_pad": self.num_on_pad
                }
        self.oracle.reset()
        self.ep_reward_sum, self.ep_num_neg, self.ep_num_pos = 0., 0, 0
        self.num_in_hover_zone = 0
        self.num_on_pad = 0
        return logs
