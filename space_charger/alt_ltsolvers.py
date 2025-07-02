from numba import jit, prange
import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Constants ---
c = 3E8
EPSILON_0 = 8.85E-12
COULOMB_K = 1 / (4 * np.pi * EPSILON_0)

# --- Utility Functions ---

@jit(nopython=True)
def _jit_linTransform(t: float, a: float, b: float) -> float:
    """Linear transform from [-1,1] to [a,b]"""
    return 0.5 * ((b - a) * t + (a + b))

@jit(nopython=True)
def _jit_heaviside(x1, x2):
    if np.isnan(x1):
        return np.nan
    elif x1 == 0:
        return x2
    elif x1 < 0:
        return 0.0
    else:
        return 1.0

@jit(nopython=True)
def _jit_flat(
    z:float,
    sigma_z:float) -> float:
    Rise = (_jit_heaviside(z+3*sigma_z,0) - _jit_heaviside(2.0*sigma_z+z,1))*np.sin(abs(np.pi/2*(z+3*sigma_z)/(sigma_z)))
    Flat = (_jit_heaviside(z+2.0*sigma_z,1) - _jit_heaviside(z-2.0*sigma_z,1))
    Fall = (_jit_heaviside(z-2.0*sigma_z,1) - _jit_heaviside(z-3*sigma_z,0))*np.sin(abs(np.pi/2*(z-3*sigma_z)/(sigma_z)))

    return (Rise + Flat + Fall)

# --- Gaussian Fit Helpers ---
def _assert_mesh_clean(mesh):
    dx = np.diff(mesh)
    if not (np.allclose(dx, dx[0]) and np.all(dx > 0)):
        raise ValueError("Provided mesh must be uniform and increasing.")

def fit_gaussian_density(
    z_array: NDArray[np.float64],
    nbins: int = 50,
    ngaussians: int = 50,
    width: float = 0.00005,
    mesh: NDArray[np.float64] = None,
    scale: float = 1.0,
    plot: bool = False):
    """
    Fit a sum of Gaussians to a 1D array (e.g. particle z-positions),
    return (xGauss, ampGauss, sigGauss) for use in space charge solver.
    """

    # Histogram particle positions
    histo, bins = np.histogram(z_array, bins=nbins, density=True)

    if mesh is None:
        mesh = np.linspace(bins[0], bins[-1], 100000)
    else:
        _assert_mesh_clean(mesh)

    bin_width = bins[1] - bins[0]
    guesses = np.linspace(bins[1], bins[-2], ngaussians)

    xGauss = np.zeros(ngaussians, dtype=np.float64)
    ampGauss = np.zeros(ngaussians, dtype=np.float64)
    sigGauss = np.full(ngaussians, width, dtype=np.float64)

    normsum = 0.0
    fitted_line = np.zeros_like(mesh)

    for i in range(ngaussians):
        idx = int((guesses[i] - bins[0]) / bin_width)
        idx = np.clip(idx, 0, len(histo) - 1)

        xGauss[i] = (bins[idx] + bins[idx+1]) / 2
        ampGauss[i] = histo[idx]
        fitted_line += ampGauss[i] * np.exp(-(mesh - xGauss[i]) ** 2 / (2 * width**2))
        normsum += ampGauss[i] * (width * np.sqrt(2 * np.pi))

    # Normalize and scale by desired factor (e.g., charge)
    ampGauss = (ampGauss / normsum) * scale
    fitted_line *= (scale / normsum)
    histo *= scale

    if plot:
        plt.figure(figsize=(7, 4))
        plt.stairs(histo, bins, label="Histogram")
        plt.plot(mesh, fitted_line, 'r-', label="Gaussian Fit")
        plt.title("Multi-Gaussian Fit to z-Distribution")
        plt.xlabel("z [m]")
        plt.ylabel("Density")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    return xGauss, ampGauss, sigGauss

@jit(nopython=True, fastmath=True)
def _gauss_sum(z, NGauss, xGauss, ampGauss, sigGauss):
    sum = 0.0
    for i in range(NGauss):
        sum += ampGauss[i] * np.exp(-((z - xGauss[i]) ** 2) / (2 * sigGauss[i] ** 2))
    return sum

@jit(nopython=True, fastmath=True)
def _lmd(z, xGauss, ampGauss, sigGauss):
    # i omit normalizing here because I assume that ampGauss is already normalzed
    NGauss = len(xGauss)
    return _gauss_sum(z, NGauss, xGauss, ampGauss, sigGauss)
# --- Rho Functions ---

@jit(nopython=True)
def _rho_const(z, s, bunch_rad, bunch_len, params):
    return params[0]

@jit(nopython=True)
def _rho_mesa(z, s, bunch_rad, bunch_len, params):
    sigma_z = params[0]
    return 1 / (2 * np.pi * bunch_rad**2 * bunch_len) * _jit_flat(z, sigma_z)

@jit(nopython=True)
def _rho_gauss(z, s, bunch_rad, bunch_len, params):
    if s > bunch_rad:
        return 0.0
    xGauss, ampGauss, sigGauss = params
    return _lmd(z, xGauss, ampGauss, sigGauss)

# --- Unified Integrator Kernel ---

@jit(nopython=True, parallel=True)
def _jit_integrate(
    n: int,
    bunch_rad: float,
    bunch_len: float,
    field_pt: NDArray[np.float64],
    weights: NDArray[np.float64],
    roots: NDArray[np.float64],
    rho_func,
    rho_params: tuple,
    is_trapezoid: bool ) -> NDArray[np.float64]:
    # Integration bounds
    a_s, b_s = 0, bunch_rad
    a_z, b_z = -bunch_len / 2, bunch_len / 2
    a_phi, b_phi = 0, 2 * np.pi

    # Step sizes for trapezoid
    ds = (b_s - a_s) / n
    dz = (b_z - a_z) / n
    dphi = (b_phi - a_phi) / n

    result = np.zeros(3, dtype=np.float64)

    for i in prange(n):
        if is_trapezoid:
            s = a_s + i * ds
            ws = 1 if i == 0 or i == n - 1 else 2
        else:
            s = _jit_linTransform(roots[i], a_s, b_s)
            ws = weights[i]

        for j in range(n):
            if is_trapezoid:
                z = a_z + j * dz
                wz = 1 if j == 0 or j == n - 1 else 2
            else:
                z = _jit_linTransform(roots[j], a_z, b_z)
                wz = weights[j]

            rho = rho_func(z, s, bunch_rad, bunch_len, rho_params)

            for k in range(n):
                if is_trapezoid:
                    phi = a_phi + k * dphi
                    wphi = 1 if k == 0 or k == n - 1 else 2
                else:
                    phi = _jit_linTransform(roots[k], a_phi, b_phi)
                    wphi = weights[k]

                separation = np.array([
                    field_pt[0] - s * np.cos(phi),
                    field_pt[1] - s * np.sin(phi),
                    field_pt[2] - z
                ])

                integrand = s * rho * separation / np.linalg.norm(separation) ** 3
                result += ws * wz * wphi * integrand

    # Scale factor
    if is_trapezoid:
        factor = 0.125 * COULOMB_K * ds * dz * dphi
    else:
        factor = 0.125 * COULOMB_K * (b_s - a_s) * (b_z - a_z) * (b_phi - a_phi)

    return factor * result

# --- Solver ---

def solve_SCFields(
    bunch_rad: float,
    bunch_len: float,
    co_mesh: NDArray[np.float64],
    rho_type: str = "mesa",
    integrator: str = "gaussian",
    n: int = 3,
    gauss_params: dict = None ) -> NDArray[np.float64]:
    """
    Unified solver.
    """
    roots, weights = leggauss(n)

    # Select rho function and parameters
    if rho_type.startswith("c"):
        rho_func = _rho_const
        rho_params = (1 / (2 * np.pi * bunch_rad**2 * bunch_len),)
    elif rho_type.startswith("m"):
        rho_func = _rho_mesa
        rho_params = (bunch_len / 5.2,)
    elif rho_type.startswith("g"):
        if gauss_params is None:
            raise ValueError("gauss_params must be provided for Gaussian rho.")
        rho_func = _rho_gauss
        rho_params = (
            gauss_params["xGauss"],
            gauss_params["ampGauss"],
            gauss_params["sigGauss"]
        )
    else:
        raise ValueError(f"Invalid rho type '{rho_type}'")

    # Choose integration type
    is_trapezoid = integrator.startswith("t")

    x_mesh, y_mesh, z_mesh = co_mesh
    lb_efld_cyl = np.zeros((len(x_mesh), len(y_mesh), len(z_mesh), 3), dtype=np.float64)

    total_iterations = len(z_mesh) * len(y_mesh)

    with tqdm(total=total_iterations, desc="Computing Field") as pbar:
        for i_z, z in enumerate(z_mesh):
            for i_y, y in enumerate(y_mesh):
                for i_x, x in enumerate(x_mesh):
                    lb_efld_cyl[i_x, i_y, i_z] = _jit_integrate(
                        n=n,
                        bunch_rad=bunch_rad,
                        bunch_len=bunch_len,
                        field_pt=np.array([x, y, z]),
                        weights=weights,
                        roots=roots,
                        rho_func=rho_func,
                        rho_params=rho_params,
                        is_trapezoid=is_trapezoid
                    )
                pbar.update(1)

    return lb_efld_cyl