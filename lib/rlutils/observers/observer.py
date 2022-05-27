import numpy as np
import pandas as pd


PROTECTED_DIM_NAMES = {"step", "ep", "time", "reward", "pi", "Q", "V"}
INFO_TO_IGNORE = {"TimeLimit.truncated"}

class Observer:
    """
    Class for collecting observational data from an agent and its environment during deployment.
    """
    def __init__(self, P, state_dims, action_dims, do_next_state=False, do_info=False, do_extra=False, save_path="observations"):
        self.P, self.do_next_state, self.do_info, self.do_extra, self.save_path = P, do_next_state, do_info, do_extra, save_path
        # If state_dims or action_dims are integers, use default names.
        if type(state_dims) is int:  state_dims = [f"s_{i}" for i in range(state_dims)] if state_dims > 1 else ["s"]
        if type(action_dims) is int: action_dims = [f"a_{i}" for i in range(action_dims)] if action_dims > 1 else ["a"]
        self.num_actions = len(action_dims)
        # Check for protected dim_names.
        illegal = PROTECTED_DIM_NAMES & set(state_dims + action_dims)
        if illegal: raise ValueError(f"Dimension names {illegal} already in use.")
        # List the dimensions in the dataset to be constructed.
        self.dim_names = ["ep", "time"] + state_dims + action_dims + ([f"n_{d}" for d in state_dims] if self.do_next_state else [])
        # Initialise empty dataset.
        self.run_names, self.data, self.first = [], [], True

    def per_timestep(self, ep, t, state, action, next_state, reward, done, info, extra):
        """Make an observation of a single timestep."""
        if self.first: extra_dim_names = []
        # Basics: state, action, next_state, reward.
        try: action = action.item() # If action is 1D, just extract its item().
        except: pass    
        observation = [ep, t] \
                    + list(state) \
                    + list([action] if self.num_actions == 1 else list(action)) \
                    + (list(next_state) if self.do_next_state else []) # Already in NumPy format.
        if type(reward) == np.ndarray:
            observation += list(reward)
            if self.first: extra_dim_names += [f"reward_{i}" for i in range(len(reward))]   
        else:
            observation += [reward]
            if self.first: extra_dim_names.append("reward")            
        # Dictionaries containing additional information produced by agent and environment.
        for k,v in {**(info if self.do_info else {}), **(extra if self.do_extra else {})}.items():
            if k in INFO_TO_IGNORE: continue
            if type(v) == np.ndarray: shp = v.shape; v = list(v.flatten())
            else: shp = False; v = [v]
            observation += v
            if self.first: 
                if shp: 
                    if len(shp) > 1 and shp[1] > 1: extra_dim_names += [f"{k}_{i}_{j}" for i in range(shp[0]) for j in range(shp[1])]
                    else: extra_dim_names += [f"{k}_{i}" for i in range(shp[0])]
                else: extra_dim_names.append(k)
        self.data.append(observation)
        # Add extra dim names.
        if self.first:
            illegal = set(self.dim_names) & set(extra_dim_names)
            if illegal: raise ValueError(f"dim_names {illegal} already in use.")
            self.dim_names += extra_dim_names
            self.first = False

    def per_episode(self, ep): 
        # Periodically save out.
        if self.P["save_freq"] > 0 and (ep+1) % self.P["save_freq"] == 0: self.save()
        return {}

    def add_future(self, dims, gamma, mode="sum", new_dims=None):
        """
        Add dimensions to the dataset corresponding to the discounted sum of existing ones.
        """
        self.data = np.array(self.data)
        data_time = self.data[:,self.dim_names.index("time")]
        data_dims = self.data[:,[self.dim_names.index(d) for d in dims]]
        data_new_dims = np.zeros_like(data_dims)
        terminal = True
        for i, (t, x) in enumerate(reversed(list(zip(data_time, data_dims)))):
            if terminal: data_new_dims[i] = x
            else: data_new_dims[i] = x + (gamma * data_new_dims[i-1])
            if t == 0: terminal = True
            else: terminal = False
        self.data = np.hstack((self.data, np.flip(data_new_dims, axis=0)))
        if not new_dims: new_dims = [f"future_{mode}_{d}" for d in dims]
        self.dim_names += new_dims

    def add_derivatives(self, dims, new_dims=None): 
        """
        Add dimensions to the dataset corresponding to the change in existing dimensions
        between successive timesteps.
        """
        self.data = np.array(self.data)
        data_time = self.data[:,self.dim_names.index("time")]
        data_dims = self.data[:,[self.dim_names.index(d) for d in dims]]
        data_new_dims = np.zeros_like(data_dims)
        for i in range(len(self.data)-1):
            # NOTE: Just repeat last derivatives for terminal.
            if data_time[i+1] == 0: data_new_dims[i] = data_new_dims[i-1]             
            else: data_new_dims[i] = data_dims[i+1] - data_dims[i]
        self.data = np.hstack((self.data, data_new_dims))
        if not new_dims: new_dims = [f"d_{d}" for d in dims]
        self.dim_names += new_dims

    def add_custom_dims(self, func, new_dims): 
        self.data = np.array(self.data)
        data_new_dims = np.apply_along_axis(func, 1, self.data)
        assert data_new_dims.shape[1] == len(new_dims)
        self.data = np.hstack((self.data, data_new_dims))
        self.dim_names += new_dims

    def dataframe(self):
        df = pd.DataFrame(self.data, columns=self.dim_names)
        df.index.name = "step"
        return df

    def save(self): 
        self.dataframe().to_csv(f"{self.save_path}/{self.run_names[-1]}.csv")