import torch
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box


class Agent:
    """
    Base agent class. All other agents inherit from this.
    """
    def __init__(self, env, hyperparameters):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.P = hyperparameters 
        if type(self.env.action_space) == Box: 
            self.continuous_actions = True
            # Parameters for action scaling.
            self.act_k = torch.tensor((self.env.action_space.high - self.env.action_space.low) / 2., device=self.device)
            self.act_b = torch.tensor((self.env.action_space.high + self.env.action_space.low) / 2., device=self.device)
        elif type(self.env.action_space) == Discrete: 
            self.continuous_actions = False
        else: 
            raise Exception("Incompatible action space.")

    def __str__(self):
        P = "\n".join([f"| - {k} = {v}" for k, v in self.P.items()])
        return f"\n| {self.__class__.__name__} in {self.env} with hyperparameters:\n{P}\n"

    def per_timestep(self, state, action, reward, next_state, done): pass

    def per_episode(self): return {}

    def _action_scale(self, actions):
        """Rescale continuous actions from [-1,1] to action space extents."""
        return (self.act_k * actions) + self.act_b

    def save(self, path, clear_memory=True):
        # Remove env for saving; stops pickle from throwing an error.
        env = self.env; self.env = None 
        if clear_memory: 
            try: memory = self.memory; self.memory = None
            except: pass # If no memory.
        torch.save(self, f"{path}.agent")
        self.env = env
        if clear_memory: 
            try: self.memory = memory
            except: pass
