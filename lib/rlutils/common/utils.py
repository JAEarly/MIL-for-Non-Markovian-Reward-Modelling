import torch


def col_concat(x, y):
    """Concatenate x and y along the final (column) dimension."""
    return torch.cat([x, y], dim=-1)

def one_hot(idx, len, device):
    """Create a tensor of length len which is one-hot at idx."""
    assert type(idx) == int and 0 <= idx < len
    return torch.tensor([[1 if i == idx else 0 for i in range(len)]], device=device, dtype=torch.float)

def reparameterise(x, clamp=("hard", -20, 2), params=False):
    """
    The reparameterisation trick. 
    Construct a Gaussian from x, taken to parameterise the mean and log standard deviation.
    """
    mean, log_std = torch.split(x, int(x.shape[-1]/2), dim=-1)
    # Bounding log_std helps to regulate its behaviour outside the training data (see PETS paper Appendix A.1).
    if clamp[0] == "hard": # This is used by default for the SAC policy.
        log_std = torch.clamp(log_std, clamp[1], clamp[2])
    elif clamp[0] == "soft": # This is used by default for the PETS model.
        log_std = clamp[1] + torch.nn.functional.softplus(log_std - clamp[1])
        log_std = clamp[2] - torch.nn.functional.softplus(clamp[2] - log_std)
    return (mean, log_std) if params else torch.distributions.Normal(mean, torch.exp(log_std))

def truncated_normal(tensor, mean, std, a, b):
    """
    Sample from a truncated normal distribution.
    Adapted from torch.nn.init._no_grad_trunc_normal_.
    """
    def norm_cdf(x): return (1. + torch.erf(x / 2.**.5)) / 2.
    with torch.no_grad():
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        # Uniformly fill tensor with values in [0, 1], then transform to [2l-1, 2u-1]
        tensor.uniform_()
        tensor = 2 * (l + tensor * (u - l)) - 1
        # Use inverse cdf transform for normal distribution to get truncated standard normal
        tensor.erfinv_()
        # Transform to proper mean, std
        tensor.mul_(std * (2.**.5))
        tensor.add_(mean)
        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def edit_output_size(net, mode:str, A:int, x:int=None, frac:list=None, x0:int=None, x1:int=None):
    """
    Edit the output size of a network by splitting or merging.
    TODO: Complete an arbitrary number of splits and merges with a single call.
    """
    with torch.no_grad():
        Am, n = net.layers[-1].weight.shape; m = int(Am / A)
        if mode == "split": assert 0 <= x < m; assert len(frac) > 1; assert torch.isclose(sum(frac), 1); m_new = m+len(frac)-1
        elif mode == "merge": assert 0 <= x0 < m-1; assert x0 < x1 < m; m_new = m-(x1-x0) 

        d = net.state_dict()
        weights_key, bias_key = list(d.keys())[-2:]
        weights = d[weights_key].reshape(A, m, n)
        bias = d[bias_key].reshape(A, m)
        net.layers[-1] = torch.nn.Linear(n, A*m_new)
        if mode == "split":
            weights_x, bias_x, x0, x1 = weights[:,x,:].reshape(A, 1, n), bias[:,x].reshape(A, 1), x, x
            if False:
                insert_w = torch.cat(tuple(torch.Tensor(((weights_x * f)+torch.normal(0, 1e-3, size=weights_x.shape))) for f in frac), dim=1)
                insert_b = torch.cat(tuple(torch.Tensor(((bias_x * f)+torch.normal(0, 1e-3, size=bias_x.shape))) for f in frac), dim=1)
            else:
                insert_w = torch.cat((weights_x, torch.normal(0, 1/n, size=weights_x.shape)), dim=1)
                insert_b = torch.cat((bias_x, torch.normal(0, 1/n, size=bias_x.shape)), dim=1)

        elif mode == "merge":
            insert_w = weights[:,x0:x1+1,:].sum(axis=1).reshape(A, 1, n)
            insert_b = bias[:,x0:x1+1].sum(axis=1).reshape(A, 1)
        d[weights_key] = torch.cat((weights[:,:x0,:], insert_w, weights[:,x1+1:,:]), dim=1).reshape(A*m_new, n)
        d[bias_key] = torch.cat((bias[:,:x0], insert_b, bias[:,x1+1:]), dim=1).reshape(A*m_new)
        net.load_state_dict(d)
        
        # ==============================================================
        
        # weights = net.layers[-1].weight.data.reshape(A, m, n)
        # bias = net.layers[-1].bias.data.reshape(A, m)
        # net.layers[-1] = torch.nn.Linear(n, A*m_new)
        # if mode == "split":
        #     weights_x, bias_x, x0, x1 = weights[:,x,:].reshape(A, 1, n), bias[:,x].reshape(A, 1), x, x
        #     insert_w = torch.cat(tuple(weights_x * f for f in frac), dim=1).detach()
        #     insert_b = torch.cat(tuple(bias_x * f for f in frac), dim=1).detach()
        # elif mode == "merge":
        #     insert_w = weights[:,x0:x1+1,:].sum(axis=1).reshape(A, 1, n)
        #     insert_b = bias[:,x0:x1+1].sum(axis=1).reshape(A, 1)
        # net.layers[-1].weight = torch.nn.Parameter(torch.cat((weights[:,:x0,:], insert_w, weights[:,x1+1:,:]), dim=1).reshape(A*m_new, n))
        # net.layers[-1].bias = torch.nn.Parameter(torch.cat((bias[:,:x0], insert_b, bias[:,x1+1:]), dim=1).reshape(A*m_new))np