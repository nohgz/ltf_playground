# >>> IMPORT BLOCK >>>
from dataclasses import dataclass
import os
import sys
import tqdm
import numpy as np
from numpy.linalg import norm, matmul
from enum import Enum
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

# custom packages for modularity
import fourvectors as fv
import multigauss as mg
import ltsolvers as ltsolvers
# <<< IMPORT BLOCK <<<

# >>> CONSTANTS >>>
c = 3E8
EPSILON_0 = 8.85E-12
COULOMB_K = 1 / (4 * np.pi * EPSILON_0)
# <<< CONSTANTS <<<

# >>> DATA CLASSES >>>
collected_data = {}

@dataclass
class MainConfig:
    INTEGRATOR: str
    SHOW_GAUSSIAN_FIT: bool
    SAVE_PLOTS: bool
    OUT_PATH: str
    SEED_RNG: bool

@dataclass
class BunchConfig:
    NUM_PARTICLES: int
    SPECIES: str
    MU_VEL: float
    SIG_VEL: float
    MU_POS: float
    SIG_POS: float
    DISTRIBUTION: str
    RADIUS: float
    LENGTH: float

@dataclass
class MeshConfig:
    X_MESH_PTS: int
    Y_MESH_PTS: int
    Z_MESH_PTS: int
    SHOW_MESH: bool
    QUAD_PTS: int

@dataclass
class GaussFitsConfig:
    NUM_BINS: int
    NUM_GAUSSIANS: int
    WIDTH_GAUSSIANS: float
# <<< DATA CLASSES <<<

# >>> PARTICLE CLASSES >>>
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

class BunchParticle():
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
            v0_4v = fv.to_four_velocity(v0_3v)

        ### HANDLE INITIAL POSITIONS

        # if no position specified, assume origin
        if pos0_3v is None or pos0_4v is None:
            pos0_4v = np.array([0, 0, 0, 0])

        # if there is a pos 3v provided, convert it
        if pos0_3v is not None:
            pos0_4v = fv.to_four_position(pos0_3v, 0)

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
            self.pos_4v = matmul(
                    fv.lorentz_matrix_z_4v(other.vel_4v), self.get_separation(other)
                )

    def boost_as_reference(self):
        self.lorentz_boost_to(self)

    def lorentz_boost_from_4v(self, to_vel_4v):
        """Lorentz boost given a four-velocity"""
        lorentz_matrix = fv.lorentz_matrix_z_4v(to_vel_4v)
        self.vel_4v = matmul(lorentz_matrix, self.vel_4v)

    def lorentz_boost_from_3v(self, to_vel_3v):
        """Lorentz boost given a three-velocity"""
        self.lorentz_boost_from_4v(fv.to_four_velocity(to_vel_3v))

    def get_separation(self, other):
        """"get the separation vector between the two positions"""
        return self.pos_4v - other.pos_4v

    ### Getters
    def get_3v(self):
        return fv.to_three_velocity(self.vel_4v)

    def get_3p(self):
        return fv.to_three_position(self.pos_4v)
# <<< PARTICLE CLASS <<<


# >>> FILE PARSING >>>
def get_input_filepath():
    if len(sys.argv) < 2:
        print("Usage: python spacecharger.py <input_file.py>")
        sys.exit(1)

    input_file = sys.argv[1]

    if not os.path.isfile(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    return input_file

def record_block(name):
    def handler(**kwargs):
        collected_data[name] = kwargs
    return handler

# Load the input file into this context
def parse_input_file(filepath):
    global_namespace = {
        'Main': record_block("Main"),
        'Bunch': record_block("Bunch"),
        'Mesh': record_block("Mesh"),
        'GaussFits': record_block("GaussFits")
    }

    with open(filepath, 'r') as f:
        code = f.read()
        exec(code, global_namespace)

    return collected_data

def open_lab_fields(filepath):
    print(f"[LOG] OPENING FIELDS FROM {filepath}...", end="")
    # Load the fields back from the pickle file
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)

    # Retrieve the individual fields
    E = loaded_data['E_lab']
    B = loaded_data['B_lab']

    print("DONE")
    return E, B

#<<< FILE PARSING <<<

# >>> HELPER FUNCTIONS >>>
def closestVal(val, array, dx = None):
    if dx == None:
        dx = array[1] - array[0]

    # clip val to be in the array
    val = np.clip(val, array[0], array[-1])

    return int((val - array[0])/dx)
# <<< HELPER FUNCTIONS <<<

def routine(
    input_file: str = None,
    main_config: dict = None,
    bunch_config: dict = None,
    mesh_config: dict = None,
    gaussfits_config: dict = None ):
    """
    Run the routine either by providing an input file or by passing configs directly.
    You must provide either `input_file` or all 4 configs.
    """
    if input_file is not None:
        # Load from input file
        parse_input_file(input_file)
        main = MainConfig(**collected_data["Main"])
        bunch = BunchConfig(**collected_data["Bunch"])
        mesh = MeshConfig(**collected_data["Mesh"])
        gauss = GaussFitsConfig(**collected_data["GaussFits"])
    else:
        # Validate that all configs are given
        if not all([main_config, bunch_config, mesh_config, gaussfits_config]):
            raise ValueError("If input_file is not provided, you must provide all config dictionaries.")
        main = MainConfig(**main_config)
        bunch = BunchConfig(**bunch_config)
        mesh = MeshConfig(**mesh_config)
        gauss = GaussFitsConfig(**gaussfits_config)

    # Seed RNG if needed
    if main.SEED_RNG:
        np.random.seed(6969)

    ###
    # STEP 1: Initialize the particle array
    ###
    parts = []
    for i in range(bunch.NUM_PARTICLES):
        parts.append(BunchParticle(
            Electron,
            v0_3v=[0, 0, np.clip(np.random.normal(bunch.MU_VEL, bunch.SIG_VEL), 0, c)],
            pos0_3v=[0, 0, np.random.normal(bunch.MU_POS, bunch.SIG_POS)]
        ))

    lab_particle_vel = np.array([p.get_3v() for p in parts])
    lab_particle_pos = np.array([p.get_3p()[0] for p in parts])

    lab_ref_vel = np.mean(lab_particle_vel, axis=0)[0]
    lab_ref_pos = np.mean(lab_particle_pos, axis=0)

    reference = BunchParticle(Reference, v0_3v=lab_ref_vel, pos0_3v=lab_ref_pos)

    ###
    # STEP 2: Lorentz boost
    ###
    for p in parts:
        p.lorentz_boost_to(reference)

    reference.boost_as_reference()

    lb_particle_pos = np.array([p.get_3p()[0] for p in parts])

    ###
    # STEP 3: Gaussian binning
    ###
    xGauss, ampGauss, sigGauss = mg.fit_gaussian_density(
        lb_particle_pos[:, 2],
        nbins=gauss.NUM_BINS,
        ngaussians=gauss.NUM_GAUSSIANS,
        width=gauss.WIDTH_GAUSSIANS,
        plot=main.SHOW_GAUSSIAN_FIT,
        scale=Electron.CHARGE.value
    )

    ###
    # STEP 4: Mesh
    ###
    
    # stupid hacky fix if we need to manually set bunch length
    if bunch.LENGTH == -1:
        bunch.LENGTH = max(lb_particle_pos[:,2]) - min(lb_particle_pos[:,2])

    # the x mesh will always be this
    x_mesh, dx = np.linspace(
        start   = -bunch.RADIUS,
        stop    = bunch.RADIUS,
        num     = mesh.X_MESH_PTS,
        retstep = True
    )

    if mesh.Y_MESH_PTS == -1:
        y_mesh = x_mesh
        dy = dx
    else:
        y_mesh, dy = np.linspace(
                start   = -bunch.RADIUS,
                stop    = bunch.RADIUS,
                num     = mesh.Y_MESH_PTS,
                retstep = True
        )

    if mesh.Z_MESH_PTS == -1: # auto set length
        mesh.Z_MESH_PTS = int(np.round(bunch.LENGTH / dx))  # total points (approx)

        # i need odd numbers of mesh points to center at 0
        if mesh.Z_MESH_PTS % 2 == 0:
            mesh.Z_MESH_PTS += 1

        z_mesh = np.linspace(
            start = - (mesh.Z_MESH_PTS // 2) * dx,
            stop  = + (mesh.Z_MESH_PTS // 2) * dx,
            num   = mesh.Z_MESH_PTS
        )
        dz = dx
    else:
        z_mesh, dz = np.linspace(
                start   = - bunch.LENGTH / 2,
                stop    = + bunch.LENGTH / 2,
                num     = mesh.Z_MESH_PTS,
                retstep = True
        )


    ###
    # STEP 5: Solve fields
    ###
    gauss_params = {
        "xGauss": xGauss,
        "ampGauss": ampGauss,
        "sigGauss": sigGauss
    }

    lb_efld_cyl = ltsolvers.solve_SCFields(
        bunch_rad=bunch.RADIUS,
        bunch_len=bunch.LENGTH,
        co_mesh=(x_mesh, y_mesh, z_mesh),
        rho_type=bunch.DISTRIBUTION.lower(),
        integrator=main.INTEGRATOR.lower(),
        n=mesh.QUAD_PTS,
        gauss_params=gauss_params
    )

    ###
    # STEP 6: Convert cylindrical to Cartesian
    ###
    lb_ER = lb_efld_cyl[..., 0]
    lb_EZ = lb_efld_cyl[..., 2]

    X, Y = np.meshgrid(x_mesh, y_mesh, indexing='ij')
    phi = np.arctan2(Y, X)

    lb_Ex = lb_ER * np.cos(phi)[..., None]
    lb_Ey = lb_ER * np.sin(phi)[..., None]
    lb_Ez = lb_EZ

    lb_efld_cart = np.stack((lb_Ex, lb_Ey, lb_Ez), axis=-1)

    ###
    # STEP 7: Transform to lab frame
    ###
    flat_E = lb_efld_cart.reshape(-1, 3)
    flat_E_boosted = []
    flat_B_boosted = []

    v_norm = norm(lab_ref_vel)

    for E_vec in flat_E:
        E_tr, B_tr = fv.inverseFieldTransform(E_vec, np.zeros(3), v_norm)
        flat_E_boosted.append(E_tr)
        flat_B_boosted.append(B_tr)

    flat_E_boosted = np.array(flat_E_boosted)
    flat_B_boosted = np.array(flat_B_boosted)

    flat_E_boosted[np.isnan(flat_E_boosted)] = 0.0
    flat_B_boosted[np.isnan(flat_B_boosted)] = 0.0

    final_E = np.reshape(flat_E_boosted, lb_efld_cart.shape)
    final_B = np.reshape(flat_B_boosted, lb_efld_cart.shape)

    lab_z_mesh = z_mesh / fv.gamma_3v(lab_ref_vel)

    return final_E, final_B, x_mesh, y_mesh, lab_z_mesh

if __name__ == "__main__":

    """This shows how to use the alternative way of calling spacecharger- good for
    real time stuffs when the bunch is changing and we have to change the mesh up a bit."""
    main_config = dict(
        INTEGRATOR = "Trapezoidal", #Can be "Trapezoidal" or "Gaussian"
        SHOW_GAUSSIAN_FIT = False,
        SAVE_PLOTS = True,
        OUT_PATH = ".",
        SEED_RNG = True
    )

    bunch_config = dict(
        NUM_PARTICLES = 10000,
        SPECIES = "Electron",    # can be "Electron" or "Proton"
        MU_VEL = 2.6E8, #m/s
        SIG_VEL = 5E6,  #m/s
        MU_POS = 0,     #meters
        SIG_POS = 1E-4, #meter
        DISTRIBUTION = "Mesa", # can be "Uniform", "Mesa", or "Gaussian"
        RADIUS = 1E-4,   # meters
        LENGTH = -1     # set to -1 to auto infer bunch length. recommended
    )

    #notes, 5 mesh points and 64 quad points seem to work decently well for
    # mu_v = 2.6E8, sig_v = 5E6, rad = 1E-4, integ = trap
    mesh_config = dict(
        X_MESH_PTS = 5,
        Y_MESH_PTS = 5,   # set to -1 to set dy = dx
        Z_MESH_PTS = 15,  # set to -1 to set dz = dx
        SHOW_MESH = True,
        QUAD_PTS = 64
    )

    gaussfits_config = dict(
        NUM_BINS = 50,
        NUM_GAUSSIANS = 50,
        WIDTH_GAUSSIANS = 0.00004
    )

    E, B, x, y, z = routine(
        main_config=main_config,
        bunch_config=bunch_config,
        mesh_config=mesh_config,
        gaussfits_config=gaussfits_config
    )