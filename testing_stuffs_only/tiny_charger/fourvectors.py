"""
## fourvectors

A few functions for dealing with four vectors in special relativity.
"""
import numpy as np
from numpy.linalg import norm


### Define some helpful equations
def gamma_3v(v):
    # ensure we're not going above c
    check_vel_3v(v)

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

def lorentz_matrix_z_3v(v):
    """Returns the **Lorentz transformation matrix** given a three-velocity only
    in the z-hat direction."""
    beta = norm(v) / 3E8
    gamma = gamma_3v(norm(v))
    return np.array([
        [gamma, 0, 0, -gamma*beta],
        [ 0, 1, 0, 0],
        [ 0, 0, 1, 0],
        [-gamma*beta, 0, 0, gamma]
    ])

def lorentz_matrix_z_4v(eta):
    """Returns the **Lorentz transformation matrix** given a four-velocity only
    in the z-hat direction."""
    return lorentz_matrix_z_3v(to_three_velocity(eta))

def check_vel_3v(vel_3v):
    """Checks velocity to ensure that we're not going above c."""
    if norm(vel_3v) > 3E8:
        raise ValueError("Velocity is above the speed of light (c)!")

### velocity stuff
def to_three_velocity(eta):
    """Returns a three velocity from a four velocity."""
    return 3E8 * np.array([eta[1:]]) / eta[0]

def to_four_velocity(v):
    """Returns a four velocity from a three velocity."""
    return gamma_3v(v) * np.array([3E8, *v])

### position stuff
def to_three_position(a):
    """Returns a three-position and time from a four-position."""
    return np.array(a[1:]), a[0] / 3E8

def to_four_position(x, t):
    """Returns a four-position from a three-position"""
    return np.array([3E8 * t, *x])


def fieldTransform(efld, bfld, v):
    """Returns the electric and magnetic fields in a comoving
    frame moving in the *z* direction."""
    efld_bar = np.array([
        gamma_3v(v) * (efld[0] - v * bfld[1]), # x
        gamma_3v(v) * (efld[1] + v * bfld[0]), # y
        efld[2]                                # z
        ])

    bfld_bar = np.array([
        gamma_3v(v) * (bfld[0] + v * efld[1]/9E16),
        gamma_3v(v) * (bfld[1] - v * efld[0]/9E16),
        bfld[2]
    ])

    return efld_bar, bfld_bar

def inverseFieldTransform(efld_bar, bfld_bar, v):
    """Returns the electric and magnetic fields in the **lab**
       frame from a comoving frame moving in the *z direction"""
    efld = np.array([
        gamma_3v(v) * (efld_bar[0] + v * bfld_bar[1]), # x
        gamma_3v(v) * (efld_bar[1] - v * bfld_bar[0]), # y
        efld_bar[2]                                    # z
        ])

    bfld = np.array([
        gamma_3v(v) * (bfld_bar[0] - v * efld_bar[1]/9E16),
        gamma_3v(v) * (bfld_bar[1] + v * efld_bar[0]/9E16),
        bfld_bar[2]
    ])

    return efld, bfld