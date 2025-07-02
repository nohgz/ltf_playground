Main(
    DEBUG = False,
    INTEGRATOR = "Trapezoidal", #Can be "Trapezoidal" or "Gaussian"
    SHOW_GAUSSIAN_FIT = False,
    SAVE_PLOTS = True,
    OUT_PATH = ".",
    SEED_RNG = True
)

Bunch(
    NUM_PARTICLES = 10000,
    SPECIES = "Electron",    # can be "Electron" or "Proton"
    MU_VEL = 2.6E8, #m/s
    SIG_VEL = 5E6,  #m/s
    MU_POS = 0,     #meters
    SIG_POS = 1E-4, #meter
    DISTRIBUTION = "Gaussian", # can be "Uniform", "Mesa", or "Gaussian"
    RADIUS = 1E-4   # meters
)

#notes, 9 mesh points and 64 quad points seem to work decently well (exec time = )
#now 7 mesh points and 48 quad points works better and faster for
# and now 5 and 64 looks to take the cake (exec time = ~11s)
# mu_v = 2.6E8, sig_v = 5E6, rad = 1E-4, integ = trap

#somehow, against all odds, 3 and 16 looks passable :skull:
Mesh(
    MESH_PTS = 5,
    QUAD_PTS = 64
)

GaussFits(
    NUM_BINS = 50,
    NUM_GAUSSIANS = 50,
    WIDTH_GAUSSIANS = 0.00004
)

