from numba import jit, prange
import numpy as np
from numpy.typing import NDArray
import multigauss

c = 3E8
EPSILON_0 = 8.85E-12
COULOMB_K = 1 / (4 * np.pi * EPSILON_0)

@jit(nopython=True)
def _jit_linTransform(t:float, a:float, b:float ) -> float:
    """Linear transform from [-1,1] to [a,b]"""
    return 0.5 * ((b-a) * t + (a + b))

@jit(nopython=True)
def _jit_flat(
    z:float,
    sigma_z:float) -> float:
    Rise = (_jit_heaviside(z+3*sigma_z,0) - _jit_heaviside(2.0*sigma_z+z,1))*np.sin(abs(np.pi/2*(z+3*sigma_z)/(sigma_z)))
    Flat = (_jit_heaviside(z+2.0*sigma_z,1) - _jit_heaviside(z-2.0*sigma_z,1))
    Fall = (_jit_heaviside(z-2.0*sigma_z,1) - _jit_heaviside(z-3*sigma_z,0))*np.sin(abs(np.pi/2*(z-3*sigma_z)/(sigma_z)))

    return (Rise + Flat + Fall)

@jit(nopython=True)
def _jit_heaviside(x1, x2):
    """ vectorized implementation of the heaviside function """
    if np.isnan(x1):
        return np.nan
    elif x1 == 0:
        return x2
    elif x1 < 0:
        return 0.0
    else:
        return 1.0


@jit(nopython=True, parallel=True)
def _jit_quad_scField(
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

            # idz = int((z - mesh[0]) / (mesh[1] - mesh[0]))
            # rho = 0 if s > bunch_rad else rho_line[idz]
            # rho = 1 / (2 * np.pi * bunch_rad**2 * bunch_len)

            # mesa profile (like the flat, but tapers off)
            rho = 1 / (2 * np.pi * bunch_rad**2 * bunch_len) * _jit_flat(z, bunch_len/5.2)

            for k in range(n):
                phi = _jit_linTransform(roots[k], a_phi, b_phi)
                wphi = weights[k]

                # NOTE: optimization- get rid of norm for my own thing
                # may be necessary because numba might throw a fit

                separation = np.array([
                    field_pt[0] - s * np.cos(phi), # x
                    field_pt[1] - s * np.sin(phi), # y
                    field_pt[2] - z])            # z
                integrand = s * rho * separation / np.linalg.norm(separation)**3

                # multiply the fxn by the weights and vol element
                result += ws * wz * wphi * integrand

    # the magic number is 1/8 * 1/ 4 pi epsilon 0
    return 0.125 * COULOMB_K * (b_s - a_s) * (b_z - a_z) * (b_phi - a_phi) * result

@jit(nopython=True, parallel=True)
def _jit_trap_scField(
    field_pt: NDArray[np.float64],
    bunch_rad: float,
    bunch_len: float,
    mesh: NDArray[np.float64],
    rho_line: NDArray[np.float64],
    n: int = 3) -> NDArray[np.float64]:

    # TODO: mayhaps get rid of a_s & a_f for speeeed? is an unnecessary assign
    # but might just get patched away in the JIT optimizations
    s0, s1 = 0, bunch_rad
    z0, z1 = -bunch_len / 2, bunch_len / 2
    phi0, phi1 = 0, 2*np.pi

    ds = (s1 - s0) / n
    dz = (z1 - z0) / n
    dphi = (phi1 - phi0) / n

    result = np.zeros(3, dtype=np.float64)

    for i in prange(n):
        s = s0 + i * ds
        ws = 1 if i == 0 or i == n-1 else 2

        for j in range(n):
            z = z0 + j * dz
            wz = 1 if j == 0 or j == n-1 else 2

            # idz = int((z - mesh[0]) / (mesh[1] - mesh[0]))
            # rho = 0 if s > bunch_rad else rho_line[idz]
            # rho = 1 / (2 * np.pi * bunch_rad**2 * bunch_len)

            # mesa profile (like the flat, but tapers off)
            rho = 1 / (2 * np.pi * bunch_rad**2 * bunch_len) * _jit_flat(z, bunch_len/5.1)

            for k in range(n):
                phi = phi0 + k * dphi
                wphi = 1 if k == 0 or k == n-1 else 2

                # NOTE: optimization- get rid of norm for my own thing
                # may be necessary because numba might throw a fit

                separation = np.array([
                    field_pt[0] - s * np.cos(phi), # x
                    field_pt[1] - s * np.sin(phi), # y
                    field_pt[2] - z])              # z
                integrand = s * rho * separation / np.linalg.norm(separation)**3

                # multiply the fxn by the weights and vol element
                result += ws * wz * wphi * integrand

    # the magic number is 1/8 * 1/ 4 pi epsilon 0
    return 0.125 * COULOMB_K * ds * dz * dphi * result


def call_jit_scField(
    field_pt: NDArray[np.float64],
    bunch_rad: float,
    bunch_len: float,
    rho: multigauss.MultiGaussFit,
    n: int = 3,
    integrator: str = "gaussian") -> NDArray[np.float64]:
    """Computes the Electric Field at carteisan point (field_pt) away from a cylinder of
    radius (bunch_rad) and length (bunch_len). This is for a 1-D bunch along z and
    the optimizations here reflect that. Assumes regular grid spacing. Defaults to
    Gaussian quadrature, but trapezoidal rule is also available."""

    mesh, rho_line = rho.mesh, rho.fitted_line

    if integrator[0] == "g":
        # print("[INFO] Using Gaussian Quadrature.")
        roots, weights = np.polynomial.legendre.leggauss(n)
        return _jit_quad_scField(
            field_pt=field_pt,
            bunch_rad=bunch_rad,
            bunch_len=bunch_len,
            roots=roots,
            weights=weights,
            mesh=mesh,
            rho_line=rho_line,
            n=n)
    elif integrator[0] == "t":
        # print("[INFO] Using Trapezoidal Rule.")
        return _jit_trap_scField(
            field_pt=field_pt,
            bunch_rad=bunch_rad,
            bunch_len=bunch_len,
            mesh=mesh,
            rho_line=rho_line,
            n=n)
    else:
        raise UserWarning("No!")

