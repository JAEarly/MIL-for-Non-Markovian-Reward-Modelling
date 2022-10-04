from abc import ABC, abstractmethod
from gym import Wrapper
from gym.spaces.box import Box
from numpy import float32, concatenate
from torch import tensor


class NonMarkovRewardWrapper(Wrapper, ABC):
    """
    Combine an OpenAI Gym environment with a state machine whose state evolves as a function of the
    environment state, and whose real-valued outputs are used as a non-Markovian reward function.
    """
    def __init__(self, env, internal_state_shape, augment_state=True):
        super().__init__(env)
        if augment_state: # Define augmented observation space
            assert isinstance(self.env.observation_space, Box)
            assert len(internal_state_shape) == 1; n = internal_state_shape[0]
            low =  concatenate((self.env.observation_space.low,  [-float32("inf") for _ in range(n)]))
            high = concatenate((self.env.observation_space.high, [ float32("inf") for _ in range(n)]))
            self.observation_space = Box(low, high)
        self.augment_state = augment_state

    def reset(self):
        self.init_internal_state()
        self.env_state = self.env.reset()
        self.env_mod_callback()
        return self.augment(self.env_state)

    def step(self, action):
        # NOTE: Uses *current* env_state rather than next, and does not use action
        reward = self.advance_state_and_reward(self.env_state)
        self.env_state, _, done, info = self.env.step(action)
        self.env_mod_callback()
        return self.augment(self.env_state), reward, done, info

    def augment(self, env_state):
        if self.augment_state: return concatenate([env_state, self.get_internal_state()])
        else: return env_state

    def env_mod_callback(self): pass

    @abstractmethod
    def init_internal_state(cls): pass

    @abstractmethod
    def get_internal_state(cls): pass

    @abstractmethod
    def advance_state_and_reward(cls): pass


class OracleWrapper(NonMarkovRewardWrapper):
    def __init__(self, env, oracle, augment_state=True):
        super().__init__(env, oracle.internal_state_shape, augment_state)
        self.oracle = oracle

    def init_internal_state(self):
        self.oracle.reset()

    def get_internal_state(self):
        return self.oracle.internal_state

    def advance_state_and_reward(self, env_state):
        return self.oracle(env_state)

    def env_mod_callback(self):
        return self.oracle.env_mod_callback(self.env)


class LstmWrapper(NonMarkovRewardWrapper):
    def __init__(self, env, model, norm_shift, norm_scale, reward_scale=1., augment_state=True):
        internal_state_shape = model.aggregator.lstm_block.init_hidden.squeeze().shape
        super().__init__(env, internal_state_shape, augment_state)
        self.model = model
        self.norm_shift = norm_shift
        self.norm_scale = norm_scale
        self.reward_scale = reward_scale
        self.init_hidden = model.aggregator.lstm_block.init_hidden
        self.init_cell = model.aggregator.lstm_block.init_cell

    def init_internal_state(self):
        self.hidden_state = self.init_hidden
        self.cell_state = self.init_cell
        # NOTE: Tracking of running return required for RLEmbeddingSpaceLSTM only
        self.return_ = tensor([0], device=self.model.device).float()

    def get_internal_state(self):
        return self.hidden_state.detach().cpu().squeeze().numpy()

    def advance_state_and_reward(self, env_state):
        reward, self.hidden_state, self.cell_state = self.model.partial_forward(
            self.normalise(env_state), self.hidden_state, self.cell_state, self.return_)
        self.return_ += reward.squeeze()
        return reward.item() * self.reward_scale

    def normalise(self, env_state):
        return (tensor(env_state, device=self.model.device).float() - self.norm_shift) / self.norm_scale
