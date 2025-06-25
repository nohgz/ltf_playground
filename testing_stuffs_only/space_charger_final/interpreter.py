# >>> IMPORT BLOCK >>>
from dataclasses import dataclass
import os
import sys
import logging
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
    DEBUG: bool
    INTEGRATOR: str
    SAVE_PLOTS: bool
    OUT_PATH: str

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

@dataclass
class MeshConfig:
    MESH_PTS: int
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

def open_fields(filepath):
    print(f"[LOG] OPENING FIELDS FROM {filepath}...", end="")
    # Load the fields back from the pickle file
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)

    # Retrieve the individual fields
    E = loaded_data['E']
    B = loaded_data['B']

    print("DONE")
    return E, B
#<<< FILE PARSING <<<

#>>> INFO PRINTING >>>
def print_logo() -> None:
    if os.get_terminal_size().columns > 65:
        print(" \
 _____                    _____ _                       \n \
|   __|___  __ ___ ___   |     | |_  __ ___ ___ ___ ___ \n \
|__   | . ||. |  _| -_|  |   --|   ||. |  _| . | -_|  _|\n \
|_____|  _|___|___|___|  |_____|_|_|___|_| |_  |___|_|  \n \
      |_|                                  |___|        \n \
--------------------------------------------------------")
    else:
        print(" \
 _____                  \n \
|   __|___  __ ___ ___  \n \
|__   | . ||. |  _| -_| \n \
|_____|  _|___|___|___| \n \
      |_|               \n \
 _____ _                \n \
|     | |_  __ ___ ___ ___ ___ \n \
|   --|   ||. |  _| . | -_|  _|\n \
|_____|_|_|___|_| |_  |___|_|  \n \
                  |___|        \n \
-------------------------------")

def print_inputs(config:dict) -> None:
    # do some nice printing
    for item in config.items():
        name_group = item[0]
        vals_group = item[1]
        print(f"\n>>> {name_group} >>>")

        for val in vals_group:
            print(f"{val}: {vals_group[val]}")
# <<< INFO PRINTING <<<

# >>> HELPER FUNCTIONS >>>
def closestVal(val, array, dx = None):
    if dx == None:
        dx = array[1] - array[0]

    # clip val to be in the array
    val = np.clip(val, array[0], array[-1])

    return int((val - array[0])/dx)
# <<< HELPER FUNCTIONS <<<

if __name__ == '__main__':
    config = parse_input_file('test_input.py')

    # Parse Inputs
    main = MainConfig(**collected_data["Main"])
    bunch = BunchConfig(**collected_data["Bunch"])
    mesh = MeshConfig(**collected_data["Mesh"])
    gauss = GaussFitsConfig(**collected_data["GaussFits"])

    print_logo()
    print_inputs(config)

    # Prepare output directory
    timestamp = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    out_root = os.path.abspath(main.OUT_PATH)
    output_dir = os.path.join(out_root, f"output_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)

    # Setup logging to file + stdout
    log_path = os.path.join(output_dir, 'log.txt')
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    log = logging.info

    ###
    # STEP 1: Initialize the particle array
    ###
    # create an array of NUM_PARTICLES Particle Objects with random velocities
    parts = []

    log("CREATE PARTICLE ARRAY...")
    for i in range(bunch.NUM_PARTICLES):
        # i'm just going to do a whole bunch just going in the z direction
        # and give them a whole lot of positions
        parts.append(BunchParticle(
            Electron,
            # generate a velocity that is clamped
            v0_3v=[0, 0, np.clip(np.random.normal(bunch.MU_VEL, bunch.SIG_VEL),0,c)],
            pos0_3v=[0 ,0, np.random.normal(bunch.MU_POS, bunch.SIG_POS)]
            ))


    # get the velocities and positions of each particle

    log("GET LAB FRAME PARTICLE VELOCITIES & POSITIONS...")
    lab_particle_vel = np.array([particle.get_3v() for particle in parts])
    lab_particle_pos = np.array([particle.get_3p()[0] for particle in parts])

    # get the velocity and position of the reference particle
    lab_ref_vel = np.mean(lab_particle_vel, axis=0)[0]
    lab_ref_pos = np.mean(lab_particle_pos, axis=0)

    # create reference particle
    reference = BunchParticle(Reference, v0_3v = lab_ref_vel, pos0_3v = lab_ref_pos)

    ###
    # STEP 2: Lorentz boost to the reference particle's frame
    ###

    log("LORENTZ BOOST PARTICLE VELOCITIES & POSITIONS...")
    # Perform the lorentz boost for each particle
    for particle in parts:
        particle.lorentz_boost_to(reference)

    reference.boost_as_reference()

    # get the velocities and positions of each particle
    log("GET BOOSTED VELOCITIES & POSITIONS...")
    lb_particle_vel = np.array([particle.get_3v() for particle in parts])
    lb_particle_pos = np.array([particle.get_3p()[0] for particle in parts])

    # get the velocity and position of the reference particle
    lb_ref_vel = np.mean(lb_particle_vel, axis=0)[0]
    lb_ref_pos = np.mean(lb_particle_pos, axis=0)

    ###
    # STEP 3: Bin the distribution of particles in the Lorentz-boosted frame
    ###

    log("GET CHARGE DENSITY...")
    # now, I can get the number density of the particles at a certain position
    lb_mgf = mg.MultiGaussFit(
        lb_particle_pos[:,2],
        nbins=50,
        ngaussians=50,
        width=.00005
        )

    # i'll translate this into a charge density
    charge_density = lb_mgf.scale_by_factor(Electron.CHARGE.value)

    ###
    # STEP 4: Obtain the Longitudinal & Transverse Electric fields
    #
    # Take the bunch of particles to be a cylinder, and slice it up. Each slice
    # will have a charge density determined by the bunched up particles.
    #
    ###

    log("CONSTRUCTING MESHES...")
    bunch_len = charge_density.bins[-1] - charge_density.bins[0]

    tran_mesh, d_tran = np.linspace(
        start   = -bunch.RADIUS,
        stop    = bunch.RADIUS,
        num     = mesh.MESH_PTS,
        retstep = True
    )

    n_z = int(np.round(bunch_len / d_tran))  # total points (approx)
    # i need odd numbers of mesh points to center at 0
    if n_z % 2 == 0:
        n_z += 1

    long_mesh = np.linspace(
        start = - (n_z // 2) * d_tran,
        stop  = + (n_z // 2) * d_tran,
        num   = n_z
    )

    x_mesh, y_mesh = tran_mesh, tran_mesh
    z_mesh = long_mesh

    ### EVALUATE!!! ###
    lb_efld_cyl = np.zeros((len(x_mesh), len(y_mesh), len(z_mesh), 3), dtype=np.float64)

    total_iterations = len(z_mesh) * len(y_mesh)

    with tqdm.tqdm(total=total_iterations, desc="Computing Field") as pbar:
        for i_z, z in enumerate(z_mesh):
            for i_y, y in enumerate(y_mesh):
                for i_x, x in enumerate(x_mesh):
                    lb_efld_cyl[i_x, i_y, i_z] = ltsolvers.call_jit_scField(
                        field_pt=np.array([x, y, z]),
                        bunch_rad=bunch.RADIUS,
                        bunch_len=bunch_len,
                        n=mesh.QUAD_PTS,
                        rho=charge_density,
                        integrator=main.INTEGRATOR
                    )
                pbar.update(1)

    if main.SAVE_PLOTS:
        file_name = "eflds"
        file_path = os.path.join(output_dir, file_name + ".png")
        log(f"SAVING PLOTS TO \"{file_path}\"...")

        idx = closestVal(bunch.RADIUS, x_mesh)
        idy = closestVal(0, y_mesh)
        idz = closestVal(0, z_mesh)

        fig, axs = plt.subplots(1,2, figsize=(14, 5))

        # Radial Efld
        axs[0].grid()
        axs[0].axvline(x=lb_ref_pos[0], label='Bunch Center', color='red')
        axs[0].axvspan(-bunch_len/2, bunch_len/2, alpha=0.2, label='Bunch Length')
        axs[0].plot(z_mesh, lb_efld_cyl[idx, idy, :][:, 0], '.-', label="Radial Electric Field")
        axs[0].set_xlabel("Longitudinal Distance $z$ (m)")
        axs[0].set_ylabel("Electric Field Magnitude (V/m)")
        axs[0].set_title(f"Radial Electric Field ($n$ = {mesh.QUAD_PTS}) (Ref. Frame)")
        axs[0].legend()

        # Longitudinal Efld
        axs[1].grid(True)
        axs[1].axvline(x=lb_ref_pos[0], label='Bunch Center', color='red')
        axs[1].plot(z_mesh, lb_efld_cyl[idx, idy, :][:,2], '.-', label="Longitudinal Electric Field")
        axs[1].axvspan(-bunch_len/2, bunch_len/2, alpha=0.2, label='Bunch Area')
        axs[1].set_xlabel("Longitudinal Distance $z$ (m)")
        axs[1].set_ylabel("Electric Field Magnitude (V/m)")
        axs[1].set_title(f"Longitudinal Electric Field ($n$ = {mesh.QUAD_PTS}) (Ref. Frame)")
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()


    ###
    # Step 6: Convert the fields in the reference particle frame to cartesian,
    #         then convert that into the lab frame using the functions that I defined.
    ###

    log("CONVERTING FIELDS TO CARTESIAN...")

    # Extract cylindrical field components
    lb_ER = lb_efld_cyl[..., 0]
    lb_EZ = lb_efld_cyl[..., 2]

    # Create meshgrids of x and y for phi calculation
    X, Y = np.meshgrid(x_mesh, y_mesh, indexing='ij')
    phi = np.arctan2(Y, X)

    # Compute Ex and Ey using broadcasting
    lb_Ex = lb_ER * np.cos(phi)[..., None]
    lb_Ey = lb_ER * np.sin(phi)[..., None]
    lb_Ez = lb_EZ

    # Stack into final Cartesian E-field
    lb_efld_cart = np.stack((lb_Ex, lb_Ey, lb_Ez), axis=-1)

    log("BOOSTING FIELDS BACK TO LAB FRAME...")

    # Get shape and preallocate final E and B fields
    shape = lb_efld_cart.shape[:3]
    final_E = np.zeros_like(lb_efld_cart)
    final_B = np.zeros_like(lb_efld_cart)

    # Flatten spatial indices for efficiency
    flat_E = lb_efld_cart.reshape(-1, 3)
    flat_E_boosted = []
    flat_B_boosted = []

    v_norm = norm(lab_ref_vel)

    # Apply inverse field transform per point
    for E_vec in flat_E:
        E_tr, B_tr = fv.inverseFieldTransform(E_vec, np.zeros(3), v_norm)
        flat_E_boosted.append(E_tr)
        flat_B_boosted.append(B_tr)

    # Reshape boosted fields back
    final_E = np.reshape(flat_E_boosted, lb_efld_cart.shape)
    final_B = np.reshape(flat_B_boosted, lb_efld_cart.shape)

    # Save to pickle
    fields_filename = f"fields_{timestamp}.pkl"
    fields_path = os.path.join(output_dir, fields_filename)
    log(f"WRITING FIELDS TO \"{fields_path}\"...")

    with open(fields_path, 'wb') as f:
        pickle.dump({'E': final_E, 'B': final_B}, f)

    print("[ DONE ]")