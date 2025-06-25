import numpy as np
import matplotlib.pyplot as plt

def plotNormalWithLine(s, sig=None, mu=None, title=None, opacity=None):
    _, bins, _ = plt.hist(s, 50, density=True, label = title, alpha = opacity)

    if sig is not None and mu is not None:
        plt.plot(bins, 1/(sig * np.sqrt(2 * np.pi)) *
                    np.exp( - (bins - mu)**2 / (2 * sig**2) ),
                linewidth=2, color='r')

class MultiGaussFit():
    def __init__(
                    self,
                    arr,
                    nbins:int = 50,
                    ngaussians:int = 1,
                    width:float = 10.0,
                    mesh = None
                ):

        # get the histogram and bins arrays from the input array
        histo, bins = np.histogram(arr, bins=nbins, density=True)

        # bins = (bins[1:] + bins[:-1])/2

        # if no mesh specified, then overwrite it with a guess
        if mesh is None:
            self.mesh = np.linspace(bins[0], bins[-1], 100000)
        # if there IS a mesh, then sanitize it
        else:
            print(f"[INFO] GAUSSIAN FIT MESH SPECIFIED. CHECKING...", end="")
            self._mesh_isclean(mesh)
            # if the mesh is clean, then we good
            self.mesh = mesh


        # set up the array that contains where the gaussians will be centered
        eval_pts = np.zeros((ngaussians, 2))

        # first seed where the gaussians should probably be
        guesses = np.linspace(bins[1], bins[-2], ngaussians)

        # create a place to hold that gaussian fit
        fitted_line = 0
        normsum = 0
        for i in range(ngaussians):
            # get the index of the values closest
            lower_index = int((guesses[i] - bins[0])/(bins[1] - bins[0]))

            # then place the gaussian's peak location at
            eval_pts[i] = (
                (bins[lower_index]+bins[lower_index+1])/2,
                histo[lower_index]
                ) # (loc, amplitude)

            # create the gaussians and the normalizing
            fitted_line += eval_pts[i][1] * np.exp(
                    -(self.mesh- eval_pts[i][0])**2/(2 * width**2)
                )
            normsum += eval_pts[i][1] * (width * np.sqrt(2 * np.pi))

        fitted_line /= normsum

        # store the input params for future use
        self.arr = arr
        self.ngaussians = ngaussians
        self.nbins = nbins
        self.width = width

        # store the important calculated things
        self.histo = histo
        self.bins = bins
        self.fitted_line = np.array(fitted_line, dtype=np.float64)

    def _mesh_isclean(self, mesh):
        """Ensures the mesh provided is uniformly spaced and monotonically increasing."""
        #uniform spacing check
        dx = np.diff(mesh)
        if not (np.allclose(dx, dx[0]) and np.all(dx > 0)):
            raise ValueError("Mesh is not clean!!!")
        else:
            print(" (PASS)")

    def __call__(self, val, pltpoint = None):
        # get the closest point on the mesh
        # get the index of the values closest

        # clamp the value to be in the mesh range
        val = np.clip(val, self.mesh[0], self.mesh[-1])

        lower_index = int((val - self.mesh[0])/(self.mesh[1] - self.mesh[0]))

        # if the user wants to plot where they're getting the value
        if pltpoint is not None:
            plt.plot(self.mesh[lower_index], self.fitted_line[lower_index], "kx")

        return self.fitted_line[lower_index]

    def scale_by_factor(self, num):
        """Returns a new MultiGaussFit instance with the same fitted shape
        but scaled by num."""
        # Create a shallow copy
        new = MultiGaussFit.__new__(MultiGaussFit)

        # Copy all fields (except for the fitted line, which we'll scale)
        new.arr = self.arr
        new.nbins = self.nbins
        new.ngaussians = self.ngaussians
        new.width = self.width
        new.mesh = self.mesh
        new.bins = self.bins

        # Scale the PDF by a factor
        new.histo = self.histo * num
        new.fitted_line = self.fitted_line * num

        return new

    def plot_fit(self):
        # plot the where the peaks of the gaussians should be
        plt.stairs(self.histo, self.bins)
        # plt.plot(eval_pts[:,0], eval_pts[:,1], 'kx',  markersize=12)
        plt.plot(self.mesh, self.fitted_line, 'r-')