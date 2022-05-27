class OracleObserver:
    """
    For evaluating rewards from an oracle during agent learning,
    *without* actually sending these rewards to the agent.
    """
    def __init__(self, oracle):
        self.oracle, self.P = oracle, {}
        self.run_names, self.ep_reward_sum, self.ep_num_neg, self.ep_num_pos = [], 0., 0, 0

        # self.rewards_0 = []
        # self.rewards_1 = []

    def per_timestep(self, ep, t, state, action, next_state, reward_ext, done, info, extra):
        reward = self.oracle(state[:2]) # NOTE: Hard-coded

        # if reward < 1e-4: self.rewards_0.append(reward_ext)
        # elif abs(reward - 1.) < 1e-4: self.rewards_1.append(reward_ext)

        self.ep_reward_sum += reward
        if reward < 0: self.ep_num_neg += 1
        elif reward > 0: self.ep_num_pos += 1

    def per_episode(self, ep):
        self.oracle.reset()
        logs = {"reward_sum_oracle": self.ep_reward_sum, "num_neg": self.ep_num_neg, "num_pos": self.ep_num_pos}
        self.ep_reward_sum, self.ep_num_neg, self.ep_num_pos = 0., 0, 0
        return logs
