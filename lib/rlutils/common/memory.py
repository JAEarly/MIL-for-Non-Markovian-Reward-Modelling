from math import e
import torch
import numpy as np
from collections import namedtuple
import random

# Structure of a memory element.
element = namedtuple('element', ('state', 'action', 'next_state', 'done', 'reward'))
element_no_reward = namedtuple('element', ('state', 'action', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity, reward=None, relabel_mode="eager"):
        """
        If have intrinsic reward (self.reward is not None), have two options for keeping memory samples up-to-date with the latest reward:
        - Eager: Compute reward on *adding* an element to memory, and use the relabel() function to update entire memory when changes made.
        - Lazy: Don't store reward in the memory itself, and compute reward on *sampling* a batch from the memory.
        Which one will be more efficient depends on the memory capacity, batch size, batch sampling frequency and reward update frequency.
        """
        self.capacity = int(capacity)
        self.reward = reward
        self.lazy_reward = self.reward is not None and relabel_mode == "lazy"
        self.element = element_no_reward if self.lazy_reward else element
        self.clear()

    def __len__(self): return len(self.memory) # Length is length of memory list.

    def clear(self): self.memory = []; self.position = 0

    def add(self, state, action, reward, next_state, done):
        """Save a transition."""
        if type(action) == int: action_dtype = torch.int64
        elif type(action) == np.ndarray: action_dtype = torch.float
        action = torch.tensor(action, device=state.device, dtype=action_dtype).unsqueeze(0)
        done = torch.tensor(done, device=state.device, dtype=torch.bool).unsqueeze(0)
        el = [state, action, next_state, done]
        # Save reward if applicable.
        if not self.lazy_reward: 
            if self.reward is not None: 
                with torch.no_grad(): el.append(self.reward(state, action, next_state)) # Eagerly compute intrinsic reward.           
            else: el.append(torch.tensor(reward, device=state.device, dtype=torch.float).unsqueeze(0))
        # Extend memory if capacity not yet reached.
        if len(self) < self.capacity: self.memory.append(None)
        # Overwrite current entry at this position.
        self.memory[self.position] = el # self.element(*el)
        # Increment position, cycling back to the beginning if needed.
        self.position = (self.position + 1) % self.capacity

    def relabel(self):
        l = len(self)
        if l > 0:
            print("Relabelling memory with latest reward...")
            _, _, rewards, _, _ = self.sample(l, mode="relabel", keep_terminal_next=True) # NOTE: Slightly odd usage here.
            for el, r in zip(self.memory, rewards): el[4] = r.unsqueeze(0) # NOTE: reward must be at index 4 of the namedtuple spec.
            print("Done.")

    def sample(self, batch_size, mode="uniform", range=None, keep_terminal_next=False):
        """Retrieve a random sample of transitions and refactor.
        See https://stackoverflow.com/a/19343/3343043."""
        if len(self) < batch_size: return None, None, None, None, None
        if mode == "relabel": batch = self._all()
        elif mode == "latest": batch = self._latest(batch_size)
        elif mode == "uniform": batch = self._uniform(batch_size)
        elif mode == "prioritised": batch = self._prioritised(batch_size)
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        next_states = torch.cat(batch.next_state)
        if self.lazy_reward or mode == "relabel":
            # If have intrinsic reward function and relabel_mode=="lazy", use it to lazily compute rewards at the point of sampling.
            with torch.no_grad(): rewards = self.reward(states, actions, next_states)
        else: rewards = torch.cat(batch.reward) 
        if keep_terminal_next: nonterminal_mask = None
        else: 
            nonterminal_mask = ~torch.cat(batch.done)
            next_states = next_states[nonterminal_mask]
        return states, actions, rewards, nonterminal_mask, next_states

    def _all(self): return self.element(*zip(*self.memory))

    def _latest(self, batch_size): return self.element(*zip(*self.memory[-batch_size:]))

    def _uniform(self, batch_size): return self.element(*zip(*random.sample(self.memory, batch_size)))

    def _prioritised(self, batch_size): raise NotImplementedError()


# TODO: Unify.
class PpoMemory:
    def __init__(self): self.clear()

    def clear(self):
        self.state = []
        self.action = []
        self.log_prob = []
        self.reward = []
        self.done = []