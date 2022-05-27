import numpy as np
import torch
import torch.nn.functional as F 

from .utils import reparameterise


def squashed_gaussian(pi): 
    """
    For continuous spaces. Interpret pi as the mean and log standard deviation of a Gaussian,
    then generate an action by sampling from that distribution and applying tanh squashing.
    """
    gaussian = reparameterise(pi)
    action_unsquashed = gaussian.rsample() # rsample() required to allow differentiation.
    action = torch.tanh(action_unsquashed)
    # Compute log_prob from Gaussian, then apply correction for tanh squashing.
    log_prob = gaussian.log_prob(action_unsquashed).sum(axis=-1)
    log_prob -= (2 * (np.log(2) - action_unsquashed - F.softplus(-2 * action_unsquashed))).sum(axis=-1)
    return action, log_prob


class EpsilonGreedy:
    def __init__(self, epsilon_start, epsilon_end, decay_period): 
        """
        Epsilon-greedy exploration for discrete spaces. 
        """
        self.epsilon         = epsilon_start
        self.epsilon_end     = epsilon_end
        self.decay_increment = (epsilon_start - epsilon_end) / decay_period

    def __call__(self, Q, explore, do_extra):
        action_probs = torch.ones_like(Q) * explore * self.epsilon / Q.shape[0]
        action_probs[Q.argmax()] += (1-action_probs.sum())
        dist = torch.distributions.Categorical(action_probs) 
        action = dist.sample().item()
        if do_extra: extra = {"pi": action_probs.cpu().numpy(), "Q": Q.cpu().numpy()}
        else: extra = {}
        return action, extra

    def decay(self): 
        self.epsilon = max(self.epsilon - self.decay_increment, self.epsilon_end) # Decay linearly as per DQN Nature paper.


class OUNoise:
    def __init__(self, action_space, mu, theta, sigma_start, sigma_end, decay_period, device):
        """
        Time-correlated noise for continuous spaces using the Ornstein-Ulhenbeck process.
        Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
        """
        self.mu           = mu
        self.theta        = theta
        self.sigma        = sigma_start 
        self.sigma_start  = sigma_start
        self.sigma_end    = sigma_end
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = -1 # action_space.low # NOTE: Just use [-1,1] as apply agent._action_scale later
        self.high         = 1 # action_space.high
        self.device       = device
        self.reset()
        
    def __call__(self, action):
        self.evolve_state()
        return torch.clamp(action + self.state, self.low, self.high)
    
    def reset(self):
        self.state = torch.ones(self.action_dim, device=self.device) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(self.action_dim, device=self.device)
        self.state = x + dx

    def decay(self, k):
        self.sigma = self.sigma_start - (self.sigma_start - self.sigma_end) * min(1.0, k / self.decay_period)


class UniformNoise:
    def __init__(self, action_space, sigma_start, sigma_end, decay_period, device):
        """
        Weighted averaging with random uniform noise. Use sigma as parameter for consistency with above.
        """
        self.action_space = action_space
        self.sigma        = sigma_start
        self.sigma_start  = sigma_start
        self.sigma_end    = sigma_end
        self.decay_period = decay_period
        self.device       = device

    def __call__(self, action):
        action_rand = torch.tensor(self.action_space.sample(), device=self.device).unsqueeze(0)
        return (action * (1-self.sigma)) + (action_rand * self.sigma)    

    def decay(self, k):
        self.sigma = self.sigma_start - (self.sigma_start - self.sigma_end) * min(1.0, k / self.decay_period)