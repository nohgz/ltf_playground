"""
## fourvectors

A few functions for dealing with four vectors in relativity.
"""
import numpy as np
from numpy.linalg import norm


### Define some helpful equations
def gamma_3v(v):
    """Returns the Lorentz γ(gamma) factor from a three-velocity."""
    return 1/np.sqrt(1 - norm(v)**2/9E16)

def gamma_4v(eta):
    """Returns the Lorentz γ(gamma) factor from a four-velocity."""
    return eta[0] / 3E8

def lorentz_matrix_x_3v(v):
    """Returns the **Lorentz transformation matrix** given a three-velocity only
    in the x-hat direction."""
    beta = norm(v) / 3E8
    gamma = gamma_3v(norm(v))
    return np.array([
        [gamma, -gamma*beta, 0, 0],
        [-gamma*beta, gamma, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

def lorentz_matrix_x_4v(eta):
    """Returns the **Lorentz transformation matrix** given a four-velocity only
    in the x-hat direction."""
    return lorentz_matrix_x_3v(to_three_velocity(eta))

### velocity stuff
def to_three_velocity(eta):
    """Returns a three velocity from a four velocity."""
    return 3E8 * np.array([eta[1:]]) / eta[0]

def to_four_velocity(v):
    """Returns a four velocity from a three velocity."""
    return gamma_3v(v) * np.array([3E8, *v])
