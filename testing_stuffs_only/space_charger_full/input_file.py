Main(
    INTEGRATOR = "Trapezoidal", #Can be "Trapezoidal" or "Gaussian"
    SHOW_GAUSSIAN_FIT = True,
    SHOW_MESH = True,
    SAVE_PLOTS = True,
    OUT_PATH = ".",
    SEED_RNG = True
)

Bunch(
    # change the particles to be macroparticles. i.e. define bunch charge and then
    # scale it according to num particles (q = bunchCharge/N)
    NUM_PARTICLES = 5,
    SPECIES = "Electron",    # can be "Electron" or "Proton"
    MU_VEL = 2.6E8, #m/s
    SIG_VEL = 5E6,  #m/s
    DISTRIBUTION = "Gaussian", # can be "Uniform", "Mesa", or "Gaussian"
    RADIUS = 2.5e-6,   # meters
    LENGTH = 1.2e-2,
    CHARGE = -1e-08
)

#notes, 7 mesh points and 64 quad points seem to work decently well for
# mu_v = 2.6E8, sig_v = 5E6, rad = 1E-4, integ = trap
Mesh(
    X_MESH_PTS = 7,
    Y_MESH_PTS = 7,   # set to -1 to set dy = dx
    Z_MESH_PTS = 49,  # set to -1 to set dz = dx
    QUAD_PTS = 48
)

GaussFits(
    NUM_BINS = 50,
    NUM_GAUSSIANS = 50,
    WIDTH_GAUSSIANS = -1 # set to -1 to auto infer based on bunch length (works p well)
)

#notes, 9 mesh points and 64 quad points seem to work decently well (exec time = )
#now 7 mesh points and 48 quad points works better and faster for
# and now 5 and 64 looks to take the cake (exec time = ~11s)
# mu_v = 2.6E8, sig_v = 5E6, rad = 1E-4, integ = trap

#somehow, against all odds, 3 and 16 looks passable :skull:
