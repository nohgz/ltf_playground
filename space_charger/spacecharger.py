"""
## SpaceCharger
A little library to generate the transverse & longitudinal space charge fields
from a given particle bunch. Contains functionality to lorentz boost properly.
"""

from enum import Enum
import numpy as np
from numpy.linalg import norm
from typing import Callable, Optional
from numpy.typing import NDArray
from numba import jit, prange

### Constants
c = 3E8
EPSILON_0 = 8.85E-12
COULOMB_K = 1 / (4 * np.pi * EPSILON_0)

### Create enums of particle properties, could be useful for multi-species experiments
class Proton(Enum):
    NAME   = "PROTON"
    MASS   = 1.672E-27 #kg
    CHARGE = 1.602E-19 #C

class Electron(Enum):
    NAME   = "ELECTRON"
    MASS   = 9.109E-21 #kg
    CHARGE =-1.602E-19 #C

class Reference(Enum):
    """A particle that is simply here to be a reference point."""
    NAME   = "REFERENCE PARTICLE"
    MASS   = 0        #kg
    CHARGE = 0        #C

### BEGIN FOUR VECTORS SECTION

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

### END FOUR VECTORS SECTION


### Create the particle class, this holds all of the nice info about each particle
class Particle():
    def __init__(
            self,
            species,
            pos0_3v = None,  # a three-position
            pos0_4v = None,  # a four-position
            v0_3v  = None,   # a three-velocity
            v0_4v  = None,   # a four-velocity
            frame  = "LAB"   # defaults to lab frame
        ):

        ### HANDLE INITIAL VELOCITIES

        # if no initial three-velocity specified, assume 0
        if v0_3v is None or v0_4v is None:
            v0_4v = np.array([0, 0, 0, 0])

        # if there is a 3v provided, convert it
        if v0_3v is not None:
            # otherwise set the four velocity to match it
            v0_4v = to_four_velocity(v0_3v)

        ### HANDLE INITIAL POSITIONS

        # if no position specified, assume origin
        if pos0_3v is None or pos0_4v is None:
            pos0_4v = np.array([0, 0, 0, 0])

        # if there is a pos 3v provided, convert it
        if pos0_3v is not None:
            pos0_4v = to_four_position(pos0_3v, 0)

        # Get the parameters from the species of particle
        self.name = species.NAME.value
        self.mass = species.MASS.value
        self.charge = species.CHARGE.value

        self.vel_4v = v0_4v    # four velocity
        self.pos_4v = pos0_4v  # four position
        self.frame  = frame

    ### Lorentz Boost Functionality
    def lorentz_boost_to(self, other):
        """Lorentz boost to another particle's frame"""
        # if the frame is lab, then we're boosting to another particle
        if self.frame == "LAB":
            self.frame = other.name
        else:
            self.frame = other.frame

        # first lorentz boost the velocity
        self.lorentz_boost_from_4v(other.vel_4v)

        # then shift it
        if self.name == "REFERENCE PARTICLE":
            # here, the particle is the reference, and becomes the new reference frame
            self.pos_4v = self.get_separation(other)
        else:
            # otherwise lorentz the separation vector given the velocity of the other
            # (likely the reference)
            self.pos_4v = np.linalg.matmul(
                    lorentz_matrix_z_4v(other.vel_4v), self.get_separation(other)
                )

    def boost_as_reference(self):
        self.lorentz_boost_to(self)

    def lorentz_boost_from_4v(self, to_vel_4v):
        """Lorentz boost given a four-velocity"""
        lorentz_matrix = lorentz_matrix_z_4v(to_vel_4v)
        self.vel_4v = np.linalg.matmul(lorentz_matrix, self.vel_4v)

    def lorentz_boost_from_3v(self, to_vel_3v):
        """Lorentz boost given a three-velocity"""
        self.lorentz_boost_from_4v(to_four_velocity(to_vel_3v))

    def get_separation(self, other):
        """"get the separation vector between the two positions"""
        return self.pos_4v - other.pos_4v

    ### Getters
    def get_3v(self):
        return to_three_velocity(self.vel_4v)

    def get_3p(self):
        return to_three_position(self.pos_4v)
    
    
### QUADRATURE FIELD CALCULATIONS
@jit(nopython=True)
def _jit_linTransform(t:float, a:float, b:float ) -> float:
    """Linear transform from [-1,1] to [a,b]"""
    return 0.5 * ((b-a) * t + (a + b))

@jit(nopython=True, parallel=True)
def _jit_scField(
    field_pt: NDArray[np.float64],
    bunch_rad: float,
    bunch_len: float,
    roots: NDArray[np.float64],
    weights: NDArray[np.float64],
    mesh: NDArray[np.float64],
    rho_line: NDArray[np.float64],
    n: int = 3) -> NDArray[np.float64]:

    # TODO: mayhaps get rid of a_s & a_f for speeeed? is an unnecessary assign
    # but might just get patched away in the JIT optimizations
    a_s, b_s = 0, bunch_rad
    a_z, b_z = -bunch_len / 2, bunch_len / 2
    a_phi, b_phi = 0, 2*np.pi

    result = np.zeros(3, dtype=np.float64)

    for i in prange(n):
        # get the root from -1->1 to a_s->b_s
        s = _jit_linTransform(roots[i], a_s, b_s)
        ws = weights[i]
        for j in range(n):
            z = _jit_linTransform(roots[j], a_z, b_z)
            wz = weights[j]

            idz = int((z - mesh[0]) / (mesh[1] - mesh[0]))
            # rho = 0 if s > bunch_rad else rho_line[idz]
            rho = 1 / (2 * np.pi * bunch_rad**2 * bunch_len)

            for k in range(n):
                phi = _jit_linTransform(roots[k], a_phi, b_phi)
                wf = weights[k]

                # NOTE: optimization- get rid of norm for my own thing
                # may be necessary because numba might throw a fit

                separation = np.array([
                    field_pt[0] - s * np.cos(phi), # x
                    field_pt[1] - s * np.sin(phi), # y
                    field_pt[2] - z])            # z
                integrand = s * rho * separation / np.linalg.norm(separation)**3

                # multiply the fxn by the weights and vol element
                result += ws * wz * wf * integrand

    # the magic number is 1/8 * 1/ 4 pi epsilon 0
    return 0.125 * COULOMB_K * (b_s - a_s) * (b_z - a_z) * (b_phi - a_phi) * result


def call_jit_scField(
    field_pt: NDArray[np.float64],
    bunch_rad: float,
    bunch_len: float,
    rho: MultiGaussFit,
    n: int = 3) -> NDArray[np.float64]:
    """Computes the Electric Field at carteisan point (field_pt) away from a cylinder of
    radius (bunch_rad) and length (bunch_len). This is for a 1-D bunch along z and
    the optimizations here reflect that. Assumes regular grid spacing."""

    mesh, rho_line = rho.mesh, rho.fitted_line
    roots, weights = np.polynomial.legendre.leggauss(n)
    return _jit_scField(
        field_pt=field_pt,
        bunch_rad=bunch_rad,
        bunch_len=bunch_len,
        roots=roots,
        weights=weights,
        mesh=mesh,
        rho_line=rho_line,
        n=n)

        
