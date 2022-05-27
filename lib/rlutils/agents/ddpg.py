from ._generic import Agent
from ..common.networks import SequentialNetwork
from ..common.memory import ReplayMemory
from ..common.exploration import OUNoise, UniformNoise
from ..common.utils import col_concat

import numpy as np
import torch
import torch.nn.functional as F


class DdpgAgent(Agent):
    """
    Deep deterministic policy gradient with optional TD3 extensions.
    """
    def __init__(self, env, hyperparameters):
        Agent.__init__(self, env, hyperparameters)
        # Create pi and Q networks.
        self.pi = SequentialNetwork(code=self.P["net_pi"], input_space=[self.env.observation_space], output_size=self.env.action_space.shape[0],
                                    normaliser=self.P["input_normaliser"], lr=self.P["lr_pi"], device=self.device)
        self.pi_target = SequentialNetwork(code=self.P["net_pi"], input_space=[self.env.observation_space], output_size=self.env.action_space.shape[0],
                                           normaliser=self.P["input_normaliser"], eval_only=True, device=self.device)
        self.pi_target.load_state_dict(self.pi.state_dict()) # Clone.
        self.Q, self.Q_target = [], []
        for _ in range(2 if self.P["td3"] else 1): # For TD3 we have two Q networks, each with their corresponding targets.
            # Action is an *input* to the Q network here.
            Q = SequentialNetwork(code=self.P["net_Q"], input_space=[self.env.observation_space, self.env.action_space], output_size=1,
                                  normaliser=self.P["input_normaliser"], lr=self.P["lr_Q"], clip_grads=True, device=self.device)
            Q_target = SequentialNetwork(code=self.P["net_Q"], input_space=[self.env.observation_space, self.env.action_space], output_size=1,
                                         normaliser=self.P["input_normaliser"], eval_only=True, device=self.device)
            Q_target.load_state_dict(Q.state_dict()) # Clone.
            self.Q.append(Q); self.Q_target.append(Q_target)
        # Create replay memory.
        self.memory = ReplayMemory(self.P["replay_capacity"])
        # Create noise process for exploration.
        if self.P["noise_params"][0] == "ou": self.exploration = OUNoise(self.env.action_space, *self.P["noise_params"][1:], device=self.device)
        elif self.P["noise_params"][0] == "un": self.exploration = UniformNoise(self.env.action_space, *self.P["noise_params"][1:], device=self.device)
        else: raise Exception()
        # Tracking variables.
        self.total_ep = 0 # Used for noise decay.
        self.total_t = 0 # Used for policy update frequency for TD3.
        self.ep_losses = []
    
    def act(self, state, explore=True, do_extra=False):
        """Deterministic action selection plus additive noise."""
        with torch.no_grad():
            action_greedy = self.pi(state)
            action = self.exploration(action_greedy) if explore else action_greedy
            action = self._action_scale(action) # NOTE: Apply action scaling *after* adding noise.
            if do_extra:
                sa = col_concat(state, action)
                sa_greedy = col_concat(state, action_greedy) if explore else sa
                extra = {"action_greedy": action_greedy}
                for i, Q in zip(["", "2"], self.Q):
                    extra[f"Q{i}"] = Q(sa).item(); extra[f"Q{i}_greedy"] = Q(sa_greedy).item()
            else: extra = {}
            return action.cpu().numpy()[0], extra

    def update_on_batch(self, states=None, actions=None, Q_targets=None):
        """Use a random batch from the replay memory to update the pi and Q network parameters.
        NOTE: If the STEVE algorithm is wrapped around DDPG, states, actions and Q_targets will be given."""
        if states is None:         
            states, actions, rewards, nonterminal_mask, nonterminal_next_states = self.memory.sample(self.P["batch_size"])
            if states is None: return 
            # Select a' using the target pi network.
            nonterminal_next_actions = self._action_scale(self.pi_target(nonterminal_next_states))
            if self.P["td3"]:
                # For TD3 we add clipped noise to a' to reduce overfitting.
                noise = (torch.randn_like(nonterminal_next_actions) * self.P["td3_noise_std"]
                        ).clamp(-self.P["td3_noise_clip"], self.P["td3_noise_clip"])
                nonterminal_next_actions = (nonterminal_next_actions + noise).clamp(-1, 1)
            # Use target Q networks to compute Q_target(s', a') for each nonterminal next state and take the minimum value. This is the "clipped double Q trick".
            next_Q_values = torch.zeros(states.shape[0], device=self.device)
            next_Q_values[nonterminal_mask] = torch.min(*(Q_target(col_concat(nonterminal_next_states, nonterminal_next_actions)) for Q_target in self.Q_target)).squeeze()       
            # Compute target = reward + discounted Q_target(s', a').
            Q_targets = (rewards + (self.P["gamma"] * next_Q_values)).detach()
        value_loss_sum = 0.
        for Q in self.Q:    
            # Update value in the direction of TD error using Huber loss. 
            value_loss = F.smooth_l1_loss(Q(col_concat(states, actions)).squeeze(), Q_targets)
            Q.optimise(value_loss)
            value_loss_sum += value_loss.item()
        policy_loss = np.nan
        if (not self.P["td3"]) or (self.total_t % self.P["td3_policy_freq"] == 0): # For TD3, only update policy and targets every N timesteps.
            # Update policy in the direction of increasing value according to self.Q (the policy gradient).
            # This means that the policy function approximates the argmax() operation used in Q learning for discrete action spaces.
            policy_loss = -self.Q[0](col_concat(states, self._action_scale(self.pi(states)))).mean() # NOTE: Using first Q network only.
            self.pi.optimise(policy_loss)
            policy_loss = policy_loss.item()
        # Perform soft (Polyak) updates on targets.
        for net, target in zip([self.pi]+self.Q, [self.pi_target]+self.Q_target): target.polyak(net, tau=self.P["tau"])
        return policy_loss, value_loss_sum

    def per_timestep(self, state, action, reward, next_state, done, suppress_update=False):
        """Operations to perform on each timestep during training."""
        self.memory.add(state, action, reward, next_state, done)               
        self.total_t += 1
        if not suppress_update:
            losses = self.update_on_batch()
            if losses: self.ep_losses.append(losses)

    def per_episode(self):
        """Operations to perform on each episode end during training."""
        if self.ep_losses: mean_policy_loss, mean_value_loss = np.nanmean(self.ep_losses, axis=0)
        else: mean_policy_loss, mean_value_loss = 0., 0.
        del self.ep_losses[:]; self.total_ep += 1
        self.exploration.decay(self.total_ep)
        return {"policy_loss": mean_policy_loss, "value_loss": mean_value_loss, "sigma": self.exploration.sigma}
