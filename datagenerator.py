import torch
import numpy as np


def Generator(f, dt, n_data, t_span=None):
    """ Generator function that will return a matrix of data D according to function f(x)
    If t_span is not specified, matrix will be n x 2. Otherwise will be n x int(t_span/dt)
    x shall have the form: (t=0)
        [ x0[t] x0[t+dt]]
    x = [ x1[t] x1[t+dt]]
        [ ...           ]
        [ xn[t] xn[t+dt]]
    """
    m = int(t_span/dt) if t_span else 2
    np.random.seed(42)
    x = torch.tensor(np.random.uniform(low=-50., high=50., size=(n_data, m))).float()

    # overwrite random columns and step them forward from the first column
    for i in range(1, m):
        x[:, i] = f(x[:, 0].clone().detach(), dt*i)
    return x


def Dynamics(x, dt):
    """ function defining f(x) = 0.5x**2 -> minimize energy with dx/dt = -x
    If there is a time span greater than dt, the generator function handles the change by modifying dt passed here
    """
    return x*np.exp(-dt)
