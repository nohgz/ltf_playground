import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from numba import jit

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

    # normalize and scale by any factor
    ampGauss = (ampGauss / normsum) * scale
    fitted_line *= (scale / normsum) * (-1e-8)
    histo *= scale

    if plot:
        # calculate a test value using the things
        
        plt.figure(figsize=(7, 4))
        plt.stairs(histo*(-1e-8), bins, label="Histogram")
        plt.plot(mesh, fitted_line, 'r-', label="Gaussian Fit")


        # val = np.random.default_rng().integers(low=0, high=nbins)
        # thang = _lmd(bins[val], xGauss=xGauss, AmpGauss=ampGauss, SigGauss=sigGauss)
        # plt.plot(bins[val], thang, 'kx')
        
        plt.title("Charge Density [$C$]")
        plt.xlabel("z [$m$]")
        plt.ylabel("Charge Density")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # total_charge = np.trapz(fitted_line, mesh)
    # print(f"Total charge = {total_charge} C, SCALE = {scale}")

    return xGauss, ampGauss, sigGauss

# @jit(nopython=True, fastmath=True)
# def _gauss_sum(z, NGauss, xGauss, ampGauss, sigGauss):
#     sum = 0.0
#     for i in range(NGauss):
#         sum += ampGauss[i] * np.exp(-((z - xGauss[i]) ** 2) / (2 * sigGauss[i] ** 2))
#     return sum

@jit(nopython=True, fastmath=True)
def _gauss_sum(x,NGauss,xGauss,ampGauss,sigGauss):
    NGauss = len(xGauss)
    Sum = 0
    normsum = 0.0
    for i in range(NGauss):
        Sum += ampGauss[i]*np.exp(-(x-xGauss[i])**2/2/sigGauss[i]**2)
        normsum += ampGauss[i]*np.sqrt(2*np.pi*sigGauss[i]**2)
    return Sum/normsum


# @jit(nopython=True, fastmath=True)
# def _lmd(z, xGauss, ampGauss, sigGauss):
#     # i omit normalizing here because I assume that ampGauss is already normalzed
#     NGauss = len(xGauss)
#     return _gauss_sum(z, NGauss, xGauss, ampGauss, sigGauss)

#FIXME: import jit nerd
@jit(nopython=True,fastmath=True)
def _lmd(z,xGauss,AmpGauss,SigGauss,deriv=False):
    NGauss = len(xGauss)
    sigma = SigGauss[0]
    dz = sigma/1000
    fac = 1.0
    return fac*_gauss_sum(z,NGauss,xGauss,AmpGauss,SigGauss)
    
