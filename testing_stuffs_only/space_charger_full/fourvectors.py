"""
## fourvectors

A few functions for dealing with four vectors in special relativity.
"""
import numpy as np
from numpy.linalg import norm


def check_vector_length(vec, expected_length, vec_name="vector"):
    """Raises an error if the vector does not have the expected length."""
    if len(vec) != expected_length:
        raise ValueError(
            f"{vec_name} must have length {expected_length}, got {len(vec)}."
        )

### Define some helpful equations
def gamma_3v(vel_3v):
    """Returns the Lorentz γ(gamma) factor from a three-velocity."""

    check_vector_length(vel_3v, 3, "Three-velocity")

    # ensure we're not going above c
    check_vel_3v(vel_3v)

    return 1/np.sqrt(1 - norm(vel_3v)**2/9E16)


def gamma_4v(vel_4v):
    """Returns the Lorentz γ(gamma) factor from a four-velocity."""

    check_vector_length(vel_4v, 4, "Four-velocity")

    #FIXME: i dont think this is right lets stay away from it
    return vel_4v[0] / 3E8

def lorentz_matrix_z_3v(vel_3v):
    """Returns the **Lorentz transformation matrix** given a three-velocity only
    in the z-hat direction."""
    check_vector_length(vel_3v, 3, "Three-velocity")

    beta = norm(vel_3v) / 3E8
    gamma = gamma_3v(vel_3v)
    return np.array([
        [gamma, 0, 0, -gamma*beta],
        [ 0, 1, 0, 0],
        [ 0, 0, 1, 0],
        [-gamma*beta, 0, 0, gamma]
    ])

def lorentz_matrix_z_4v(vel_4v):
    """Returns the **Lorentz transformation matrix** given a four-velocity only
    in the z-hat direction."""
    check_vector_length(vel_4v, 4, "Four-velocity")

    # print("I'm being used!")
    # print(vel_4v)
    return lorentz_matrix_z_3v(to_three_velocity(vel_4v))

def check_vel_3v(vel_3v):
    """Checks velocity to ensure that we're not going above c."""
    check_vector_length(vel_3v, 3, "Three-velocity")

    if norm(vel_3v) > 3E8:
        print(vel_3v)
        raise ValueError("Velocity is above the speed of light (c)!")

### velocity stuff
def to_three_velocity(vel_4v):

    check_vector_length(vel_4v, 4, "Four-velocity")

    three_v = np.array(3E8 * vel_4v[1:] / vel_4v[0])
    """Returns a three velocity from a four velocity."""
    return three_v

def to_four_velocity(vel_3v):

    check_vector_length(vel_3v, 3, "Three-velocity")

    four_v = np.zeros(4)
    four_v[0] = 3E8
    four_v[1:] = vel_3v
    """Returns a four velocity from a three velocity."""
    return four_v

### position stuff
def to_three_position(a):
    check_vector_length(a, 4, "Four-position")
    """Returns a three-position and time from a four-position."""
    return np.array(a[1:]), a[0] / 3E8

def to_four_position(x, t):
    check_vector_length(x, 3, "Three-position")
    """Returns a four-position from a three-position"""
    return np.array([3E8 * t, *x])


def fieldTransform(efld, bfld, vel_3v):
    """Returns the electric and magnetic fields in a comoving
    frame moving in the *z* direction."""
    check_vector_length(vel_3v, 3, "Three-velocity")
    efld_bar = np.array([
        gamma_3v(vel_3v) * (efld[0] - norm(vel_3v) * bfld[1]), # x
        gamma_3v(vel_3v) * (efld[1] + norm(vel_3v) * bfld[0]), # y
        efld[2]                                                # z
        ])

    bfld_bar = np.array([
        gamma_3v(vel_3v) * (bfld[0] + norm(vel_3v) * efld[1]/9E16),
        gamma_3v(vel_3v) * (bfld[1] - norm(vel_3v) * efld[0]/9E16),
        bfld[2]
    ])

    return efld_bar, bfld_bar

def inverseFieldTransform(efld_bar, bfld_bar, vel_3v):
    """Returns the electric and magnetic fields in the **lab**
       frame from a comoving frame moving in the *z direction"""
    check_vector_length(vel_3v, 3, "Three-velocity")
    efld = np.array([
        gamma_3v(vel_3v) * (efld_bar[0] + norm(vel_3v) * bfld_bar[1]), # x
        gamma_3v(vel_3v) * (efld_bar[1] - norm(vel_3v) * bfld_bar[0]), # y
        efld_bar[2]                                                    # z
        ])

    bfld = np.array([
        gamma_3v(vel_3v) * (bfld_bar[0] - norm(vel_3v) * efld_bar[1]/9E16),
        gamma_3v(vel_3v) * (bfld_bar[1] + norm(vel_3v) * efld_bar[0]/9E16),
        bfld_bar[2]
    ])

    return efld, bfld