from ._generic import Agent
from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory
from ..common.exploration import EpsilonGreedy

import torch
import torch.nn.functional as F


class DqnAgent(Agent):
    """
    Deep Q Network (DQN). From:
        Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness, Marc G. Bellemare, Alex Graves et al.
        "Human-level Control through Deep Reinforcement Learning."
        Nature 518, no. 7540 (2015): 529-533.
    # TODO: Integrate further improvements in Rainbow: https://arxiv.org/abs/1710.02298.
    """
    def __init__(self, env, hyperparameters):
        Agent.__init__(self, env, hyperparameters)
        # Create Q network.
        if len(self.env.observation_space.shape) > 1: net_preset, net_code = "CartPoleQ_Pixels", None
        else: net_preset, net_code = None, self.P["net_Q"]
        output_size = self.env.action_space.n * (1 if self.P["reward_components"] is None else self.P["reward_components"])
        self.Q = SequentialNetwork(code=net_code, preset=net_preset, input_space=[self.env.observation_space], output_size=output_size,
                                   normaliser=self.P["input_normaliser"], lr=self.P["lr_Q"], clip_grads=True, device=self.device)
        self.Q_target = SequentialNetwork(code=net_code, preset=net_preset, input_space=[self.env.observation_space], output_size=output_size,
                                          normaliser=self.P["input_normaliser"], eval_only=True, device=self.device)
        self.Q_target.load_state_dict(self.Q.state_dict()) # Clone.
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"])
        # Initialise epsilon-greedy exploration.
        self.exploration = EpsilonGreedy(self.P["epsilon_start"], self.P["epsilon_end"], self.P["epsilon_decay"])
        # Tracking variables.
        if self.P["target_update"][0] == "hard": self.updates_since_target_clone = 0
        else: assert self.P["target_update"][0] == "soft"
        self.ep_losses = []

    def act(self, state, explore=True, do_extra=False):
        """Epsilon-greedy action selection."""
        with torch.no_grad():
            Q = self.Q(state).squeeze()
            # If using reward decomposition, need to take sum.
            if self.P["reward_components"] is not None: Q = Q.reshape(self.env.action_space.n, -1).sum(axis=1)
            return self.exploration(Q, explore, do_extra)

    def update_on_batch(self):
        """Use a random batch from the replay memory to update the Q network parameters."""
        states, actions, rewards, nonterminal_mask, nonterminal_next_states = self.memory.sample(self.P["batch_size"])
        if states is None: return
        # Use target network to compute Q_target(s', a') for each nonterminal next state.
        next_Q_values = torch.zeros((self.P["batch_size"], 1 if self.P["reward_components"] is None else self.P["reward_components"]), device=self.device)
        Q_t_n = self.Q_target(nonterminal_next_states).detach()
        if self.P["reward_components"] is None: 
            # Compute Q(s, a) by running each s through self.Q, then selecting the corresponding column.
            Q_values = self.Q(states).gather(1, actions.reshape(-1,1))
            # In double DQN, a' is the Q-maximising action for self.Q. This decorrelation reduces overestimation bias.
            # In regular DQN, a' is the Q-maximising action for self.Q_target.
            nonterminal_next_actions = (self.Q(nonterminal_next_states) if self.P["double"] else Q_t_n).argmax(1).detach()
            Q_t_n = Q_t_n.unsqueeze(-1)
            rewards = rewards.unsqueeze(-1)
        else: 
            # Equivalent of above for decomposed reward.
            Q_values = self.Q(states).reshape(self.P["batch_size"], self.env.action_space.n, self.P["reward_components"])[torch.arange(self.P["batch_size"]), actions, :]
            Q_t_n = Q_t_n.reshape(Q_t_n.shape[0], self.env.action_space.n, self.P["reward_components"])
            Q_for_a_n = self.Q(nonterminal_next_states).reshape(*Q_t_n.shape) if self.P["double"] else Q_t_n
            nonterminal_next_actions = Q_for_a_n.sum(axis=2).argmax(1).detach()
        next_Q_values[nonterminal_mask] = Q_t_n[torch.arange(Q_t_n.shape[0]), nonterminal_next_actions, :]
        # Compute target = reward + discounted Q_target(s', a').
        Q_targets = rewards + (self.P["gamma"] * next_Q_values)
        # Update value in the direction of TD error using Huber loss.
        loss = F.smooth_l1_loss(Q_values, Q_targets)
        self.Q.optimise(loss)
        if self.P["target_update"][0] == "hard":
            # Perform periodic hard update on target.
            self.updates_since_target_clone += 1
            if self.updates_since_target_clone >= self.P["target_update"][1]:
                self.Q_target.load_state_dict(self.Q.state_dict())
                self.updates_since_target_clone = 0
        elif self.P["target_update"][0] == "soft": self.Q_target.polyak(self.Q, tau=self.P["target_update"][1])
        else: raise NotImplementedError()
        return loss.item()

    def per_timestep(self, state, action, reward, next_state, done):
        """Operations to perform on each timestep during training."""
        self.memory.add(state, action, reward, next_state, done)
        loss = self.update_on_batch()
        if loss: self.ep_losses.append(loss)
        self.exploration.decay()

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_loss = sum(self.ep_losses) / len(self.ep_losses)
        else: mean_loss = 0.
        del self.ep_losses[:]
        return {"value_loss": mean_loss, "epsilon": self.exploration.epsilon}
