from ..agents._default_hyperparameters import default_hyperparameters

from torch import device, load as torch_load
from torch.cuda import is_available


def make(agent, env, hyperparameters=dict()):
    """
    Make an instance of an agent class to train/deploy in env, overwriting default hyperparameters with those provided.
    """
    agent = agent.lower()
    # Special treatment for TD3 (a variant of DDPG).
    if agent == "td3": hyperparameters["td3"] = True; agent = "ddpg"
    assert agent in default_hyperparameters, "Agent type not recognised."
    # Overwrite default hyperparameters where applicable.
    P = default_hyperparameters[agent]
    for k, v in hyperparameters.items(): P[k] = v
    # Load agent class.
    if   agent == "ddpg":               from ..agents.ddpg import DdpgAgent as agent_class
    elif agent == "dqn":                from ..agents.dqn import DqnAgent as agent_class
    elif agent == "sac":                from ..agents.sac import SacAgent as agent_class
    return agent_class(env, P)

def load(path, env): 
    _device = device("cuda" if is_available() else "cpu")
    agent = torch_load(path, map_location=_device)
    agent.device = _device
    agent.env = env
    return agent