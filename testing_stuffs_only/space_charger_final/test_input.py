Main(
    DEBUG = False,
    INTEGRATOR = "trapezoidal",
    SAVE_PLOTS = True,
    OUT_PATH = "."
)

Bunch(
    NUM_PARTICLES = 10000,
    SPECIES = "Electron",    # can be "Electron" or "Proton"
    MU_VEL = 2.6E8, #m/s
    SIG_VEL = 5E6,  #m/s
    MU_POS = 0,     #meters
    SIG_POS = 1E-4, #meter
    DISTRIBUTION = "Uniform", # can be "Uniform", "Mesa", or "Gaussian"
    RADIUS = 1E-4   # meters
)

Mesh(
    MESH_PTS = 7,
    QUAD_PTS = 64
)

GaussFits(
    NUM_BINS = 50,
    NUM_GAUSSIANS = 50,
    WIDTH_GAUSSIANS = 0.00005
)

