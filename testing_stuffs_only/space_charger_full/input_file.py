Main(
    INTEGRATOR = "Trapezoidal", #Can be "Trapezoidal" or "Gaussian"
    SHOW_GAUSSIAN_FIT = False,
    SAVE_PLOTS = True,
    OUT_PATH = ".",
    SEED_RNG = True
)

Bunch(
    # change the particles to be macroparticles. i.e. define bunch charge and then
    # scale it according to num particles (q = bunchCharge/N)
    NUM_PARTICLES = 10000,
    SPECIES = "Electron",    # can be "Electron" or "Proton"
    MU_VEL = 2.6E8, #m/s
    SIG_VEL = 5E6,  #m/s
    MU_POS = 0,     #meters
    SIG_POS = 1E-4, #meter
    DISTRIBUTION = "Gaussian", # can be "Uniform", "Mesa", or "Gaussian"
    RADIUS = 2.5e-6,   # meters
    LENGTH = 5.2e-5,     # set to -1 to auto infer bunch length. recommended
    CHARGE = -1e-08
)

#notes, 5 mesh points and 64 quad points seem to work decently well for
# mu_v = 2.6E8, sig_v = 5E6, rad = 1E-4, integ = trap
Mesh(
    X_MESH_PTS = 5,
    Y_MESH_PTS = 5,   # set to -1 to set dy = dx
    Z_MESH_PTS = 15,  # set to -1 to set dz = dx
    SHOW_MESH = True,
    QUAD_PTS = 64
)

GaussFits(
    NUM_BINS = 50,
    NUM_GAUSSIANS = 50,
    WIDTH_GAUSSIANS = 0.00004
)

#notes, 9 mesh points and 64 quad points seem to work decently well (exec time = )
#now 7 mesh points and 48 quad points works better and faster for
# and now 5 and 64 looks to take the cake (exec time = ~11s)
# mu_v = 2.6E8, sig_v = 5E6, rad = 1E-4, integ = trap

#somehow, against all odds, 3 and 16 looks passable :skull:
