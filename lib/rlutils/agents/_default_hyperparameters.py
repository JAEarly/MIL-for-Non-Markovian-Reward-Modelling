"""
NOTE: Agent names must be lowercase.
"""

default_hyperparameters = {

  "ddpg": {
    "net_pi": [(None, 256), "R", (256, 256), "R", (256, None), "T"], # Tanh policy (bounded in [-1,1]).
    "net_Q": [(None, 256), "R", (256, 256), "R", (256, 1)],
    "input_normaliser": None, # Set to "box_bounds" to pre-normalise network inputs.
    "replay_capacity": 50000, # Size of replay memory (starts overwriting when full).
    "batch_size": 128, # Size of batches to sample from replay memory during learning.
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "tau": 0.005, # Parameter for Polyak averaging of target network parameters.
    "noise_params": ("ou", 0., 0.15, 0.3, 0.3, 1000), # mu, theta, sigma_start, sigma_end, decay period (episodes).
    # "noise_params": ("un", 1, 0, 1000), # sigma_start, sigma_end, decay_period (episodes).
    "td3": False, # Whether or not to enable the TD3 enhancements. 
    # --- If TD3 enabled ---
    "td3_noise_std": 0.2,
    "td3_noise_clip": 0.5,
    "td3_policy_freq": 2
  },

  "dqn": {
    "net_Q": [(None, 256), "R", (256, 128), "R", (128, 64), "R", (64, None)], # From https://github.com/transedward/pytorch-dqn/blob/master/dqn_model.py.
    "input_normaliser": None, # Set to "box_bounds" to pre-normalise network inputs.
    "replay_capacity": 10000, # Size of replay memory (starts overwriting when full).
    "batch_size": 128, # Size of batches to sample from replay memory during learning.
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "epsilon_start": 0.9,
    "epsilon_end": 0.05,
    "epsilon_decay": 500000, # Decay period (timesteps).
    "target_update": ("soft", 0.005), # Either ("hard", decay_period) or ("soft", tau).
    # "target_update": ("hard", 10000),
    "double": True, # Whether to enable double DQN variant to reduce overestimation bias.
    "reward_components": None # For reward decomposition (set to None to disable).
  },

  "sac": {
    "net_pi": [(None, 256), "R", (256, 256), "R", (256, None)],
    "net_Q": [(None, 256), "R", (256, 256), "R", (256, None)],
    "input_normaliser": None, # Set to "box_bounds" to pre-normalise network inputs.
    "replay_capacity": 10000, # Size of replay memory (starts overwriting when full).
    "batch_size": 256, # Size of batches to sample from replay memory during learning.
    "lr_pi": 1e-4, # Learning rate for policy.
    "lr_Q": 1e-3, # Learning rate for state-action value function.
    "gamma": 0.99, # Discount factor.
    "alpha": 0.2, # Weighting for entropy regularisation term.
    "tau": 0.005, # Parameter for Polyak averaging of target network parameters.
    "update_freq": 1, # Number of timesteps between updates.
  }

}
