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
    INTEGRATOR: str
    SHOW_GAUSSIAN_FIT: bool
    SHOW_MESH: bool
    SAVE_PLOTS: bool
    OUT_PATH: str
    SEED_RNG: bool

@dataclass
class BunchConfig:
    NUM_PARTICLES: int
    SPECIES: str
    MU_VEL: float
    SIG_VEL: float
    DISTRIBUTION: str
    RADIUS: float
    LENGTH: float
    CHARGE: float

@dataclass
class MeshConfig:
    X_MESH_PTS: int
    Y_MESH_PTS: int
    Z_MESH_PTS: int
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

class Macroparticle(Enum):
    """A mutable particle. Initialized to nothing, as that is up to user choice."""
    NAME   = "MACROPARTICLE"
    MASS   = 0  # kg
    CHARGE = 0  # C

class BunchParticle():
    def __init__(
            self,
            species,
            pos0_3v = None,  # a three-position
            pos0_4v = None,  # a four-position
            v0_3v  = None,   # a three-velocity
            v0_4v  = None,   # a four-velocity
            frame  = "LAB",   # defaults to lab frame
            mass_override = None, # stupid hack for macroparticles
            charge_override = None # stupid hack for macroparticles
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

        # override mass and charge if given
        self.mass = mass_override if mass_override is not None else species.MASS.value
        self.charge = charge_override if charge_override is not None else species.CHARGE.value

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
        main_conf = MainConfig(**collected_data["Main"])
        bunch_conf = BunchConfig(**collected_data["Bunch"])
        mesh_conf = MeshConfig(**collected_data["Mesh"])
        gauss_conf = GaussFitsConfig(**collected_data["GaussFits"])
    else:
        # Validate that all configs are given
        if not all([main_config, bunch_config, mesh_config, gaussfits_config]):
            raise ValueError("If input_file is not provided, you must provide all config dictionaries.")
        main_conf = MainConfig(**main_config)
        bunch_conf = BunchConfig(**bunch_config)
        mesh_conf = MeshConfig(**mesh_config)
        gauss_conf = GaussFitsConfig(**gaussfits_config)

    # Seed RNG if needed
    if main_conf.SEED_RNG:
        np.random.seed(6969)

    print_logo()

    # Prepare output directory
    timestamp = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    out_root = os.path.abspath(main_conf.OUT_PATH)
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

    # seed RNG if needed
    if main_conf.SEED_RNG:
        np.random.seed(6969) # haha funny

    ###
    # STEP 1: Initialize the particle array
    ###
    # create an array of NUM_PARTICLES Particle Objects with random velocities
    parts = []

    # i think this magic number works best for determining particle spread
    # based only on a given bunch length for the lab
    sig_pos = bunch_conf.LENGTH / 14

    if bunch_conf.SPECIES[0].lower == "e":
        log("USING ELECTRONS")
        for i in range(bunch_conf.NUM_PARTICLES):
            parts.append(BunchParticle(
                Electron,
                v0_3v=[0, 0, np.clip(np.random.normal(bunch_conf.MU_VEL, bunch_conf.SIG_VEL), 0, c)],
                pos0_3v=[0, 0, np.random.normal(0, sig_pos)]
            ))
    elif bunch_conf.SPECIES[0].lower == "p":
        log("USING PROTONS")
        for i in range(bunch_conf.NUM_PARTICLES):
            parts.append(BunchParticle(
                Proton,
                v0_3v=[0, 0, np.clip(np.random.normal(bunch_conf.MU_VEL, bunch_conf.SIG_VEL), 0, c)],
                pos0_3v=[0, 0, np.random.normal(0, sig_pos)]
            ))
    else: #use macroparticle
        p_charge = bunch_conf.CHARGE/bunch_conf.NUM_PARTICLES
        p_mass = abs(p_charge/Electron.CHARGE.value)*Electron.MASS.value
        log(f"USING MACROPARTICLES WITH q={p_charge} AND m={p_mass}")
        for i in range(bunch_conf.NUM_PARTICLES):
            parts.append(BunchParticle(
                Macroparticle,
                v0_3v=[
                    0, 0,
                    np.clip(np.random.normal(bunch_conf.MU_VEL, bunch_conf.SIG_VEL), 0, c)
                ],
                pos0_3v=[
                    0, 0,
                    np.random.normal(0, sig_pos)
                ],
                mass_override=p_mass,
                charge_override=p_charge
            ))


    # get the velocities and positions of each particle
    log("GET LAB FRAME PARTICLE VELOCITIES & POSITIONS...")
    lab_particle_vel = np.array([particle.get_3v() for particle in parts])
    lab_particle_pos = np.array([particle.get_3p()[0] for particle in parts])

    # get the velocity and position of the reference particle
    lab_ref_vel = np.mean(lab_particle_vel, axis=0)[0]
    lab_ref_pos = np.mean(lab_particle_pos, axis=0)

    # get the bunch length in the lab frame
    lab_bunch_len = max(lab_particle_pos[:,2]) - min(lab_particle_pos[:,2])

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
    com_particle_vel = np.array([particle.get_3v() for particle in parts])
    com_particle_pos = np.array([particle.get_3p()[0] for particle in parts])

    # get the velocity and position of the reference particle
    com_ref_vel = np.mean(com_particle_vel, axis=0)[0]
    com_ref_pos = np.mean(com_particle_pos, axis=0)

    ###
    # STEP 3: Bin the distribution of particles in the Lorentz-boosted frame
    ###

    log("GET CHARGE DENSITY...")

    # attempt to auto infer gaussian width based on bunch length
    if gauss_conf.WIDTH_GAUSSIANS == -1:
        gauss_conf.WIDTH_GAUSSIANS = 2 * lab_bunch_len / gauss_conf.NUM_BINS

    xGauss, ampGauss, sigGauss = mg.fit_gaussian_density(
        com_particle_pos[:,2],
        nbins=gauss_conf.NUM_BINS,
        ngaussians=gauss_conf.NUM_GAUSSIANS,
        width=gauss_conf.WIDTH_GAUSSIANS,
        plot=main_conf.SHOW_GAUSSIAN_FIT,
        scale=bunch_conf.CHARGE)

    ###
    # STEP 4: Obtain the Longitudinal & Transverse Electric fields
    #
    # Take the bunch of particles to be a cylinder, and slice it up. Each slice
    # will have a charge density determined by the bunched up particles.
    #
    ###

    log("CONSTRUCTING CO-MOVING MESHES...")

    com_bunch_len = max(com_particle_pos[:,2]) - min(com_particle_pos[:,2])

    # the x mesh will always be this
    com_x_mesh, dx = np.linspace(
        start   = -bunch_conf.RADIUS,
        stop    = bunch_conf.RADIUS,
        num     = mesh_conf.X_MESH_PTS,
        retstep = True
    )

    if mesh_conf.Y_MESH_PTS == -1:
        log("AUTO SET Y SPACING")
        com_y_mesh = com_x_mesh
        dy = dx
    else:
        com_y_mesh, dy = np.linspace(
            start   = -bunch_conf.RADIUS,
            stop    = bunch_conf.RADIUS,
            num     = mesh_conf.Y_MESH_PTS,
            retstep = True
        )

    if mesh_conf.Z_MESH_PTS == -1:
        log("AUTO SET Z SPACING")
        mesh_conf.Z_MESH_PTS = int(np.round(com_bunch_len / dx))
        if mesh_conf.Z_MESH_PTS % 2 == 0:
            mesh_conf.Z_MESH_PTS += 1

        com_z_mesh = np.linspace(
            start = -(mesh_conf.Z_MESH_PTS // 2) * dx,
            stop  = +(mesh_conf.Z_MESH_PTS // 2) * dx,
            num   = mesh_conf.Z_MESH_PTS
        )
        dz = dx
    else:
        com_z_mesh, dz = np.linspace(
            start   = -com_bunch_len / 2,
            stop    = +com_bunch_len / 2,
            num     = mesh_conf.Z_MESH_PTS,
            retstep = True
        )

    log(f"MESH SPACING: dx: {dx}, dy: {dy}, dz: {dz}")




    if main_conf.SHOW_MESH:
        # plot the bunches
        import matplotlib.patches as patches

        # --- Compute Grids ---
        X_xy, Y_xy = np.meshgrid(com_x_mesh, com_y_mesh, indexing='ij')
        X_xz, Z_xz = np.meshgrid(com_x_mesh, com_z_mesh, indexing='ij')

        # Circle (bunch cross-section in xy-plane)
        theta = np.linspace(0, 2*np.pi, 200)
        circle_x = bunch_conf.RADIUS * np.cos(theta)
        circle_y = bunch_conf.RADIUS * np.sin(theta)

        # Rectangle (bunch cross-section in xz-plane)
        z_min = -com_bunch_len / 2
        z_max =  com_bunch_len / 2
        x_min = -bunch_conf.RADIUS
        x_max =  bunch_conf.RADIUS

        # --- Plotting ---
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        ### Subplot 1: Transverse (xy) Plane
        axs[0].grid(True)
        axs[0].scatter(X_xy, Y_xy, s=5, label='Mesh Points')
        axs[0].plot(circle_x, circle_y, 'r-', linewidth=2, label='Bunch Radius')
        axs[0].set_xlabel('x (m)')
        axs[0].set_ylabel('y (m)')
        axs[0].set_title('Transverse Mesh Slice (xy-plane at z ~ 0)')
        axs[0].axis('equal')
        # axs[0].legend()

        ### Subplot 2: Longitudinal (xz) Plane
        axs[1].grid(True)
        axs[1].scatter(Z_xz, X_xz, s=5, label='Mesh Points')
        rect = patches.Rectangle(
            (z_min, x_min), com_bunch_len, 2 * bunch_conf.RADIUS,
            linewidth=1, edgecolor='r', facecolor='none', label='Bunch Region'
        )
        axs[1].add_patch(rect)
        axs[1].set_xlabel('z (m)')
        axs[1].set_ylabel('x (m)')
        axs[1].set_title('Longitudinal Mesh Slice (xz-plane at y ~ 0)')
        axs[1].axis('equal')
        axs[1].legend()

        plt.tight_layout()

        if main_conf.SAVE_PLOTS:
            file_name = "com_meshes"
            file_path = os.path.join(output_dir, file_name + ".png")
            log(f"SAVING CO-MOVING MESH DIAGRAMS TO \"{file_path}\"...")

            plt.savefig(file_path)

        plt.show()
        plt.close()


    ### EVALUATE!!! ###
    com_efld_cyl = ltsolvers.solve_SCFields(
        bunch_rad=bunch_conf.RADIUS,
        bunch_len=com_bunch_len,
        co_mesh=(com_x_mesh, com_y_mesh, com_z_mesh),
        rho_type=bunch_conf.DISTRIBUTION.lower(),
        integrator=main_conf.INTEGRATOR.lower(),
        n=mesh_conf.QUAD_PTS,
        gauss_params={
            "xGauss": xGauss,
            "ampGauss": ampGauss,
            "sigGauss": sigGauss
        }
    )

    # save the comoving eflds
    if main_conf.SAVE_PLOTS:
        file_name = "com_eflds"
        file_path = os.path.join(output_dir, file_name + ".png")
        log(f"SAVING CO-MOVING E-FIELD PLOTS TO \"{file_path}\"...")

        idx = closestVal(bunch_conf.RADIUS, com_x_mesh)
        idy = closestVal(0, com_y_mesh)
        idz = closestVal(0, com_z_mesh)

        fig, axs = plt.subplots(1,2, figsize=(14, 5))

        # Radial Efld
        axs[0].grid()
        axs[0].axvline(x=com_ref_pos[0], label='Bunch Center', color='red')
        axs[0].axvspan(-lab_bunch_len/2, lab_bunch_len/2, alpha=0.2, label='Lab Bunch Length')
        axs[0].plot(com_z_mesh, com_efld_cyl[idx, idy, :][:, 0], '.-', label="Radial Electric Field")
        axs[0].set_xlabel("Longitudinal Distance $z$ (m)")
        axs[0].set_ylabel("Electric Field Magnitude (V/m)")
        axs[0].set_title(f"Radial Electric Field ($n$ = {mesh_conf.QUAD_PTS}) (Co. Frame)")
        axs[0].legend()

        # Longitudinal Efld
        axs[1].grid(True)
        axs[1].axvline(x=com_ref_pos[0], label='Bunch Center', color='red')
        axs[1].plot(com_z_mesh, com_efld_cyl[idx, idy, :][:,2], '.-', label="Longitudinal Electric Field")
        axs[1].axvspan(-lab_bunch_len/2, lab_bunch_len/2, alpha=0.2, label='Lab Bunch Length')
        axs[1].set_xlabel("Longitudinal Distance $z$ (m)")
        axs[1].set_ylabel("Electric Field Magnitude (V/m)")
        axs[1].set_title(f"Longitudinal Electric Field ($n$ = {mesh_conf.QUAD_PTS}) (Co. Frame)")
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
    com_ER = com_efld_cyl[..., 0]
    com_EZ = com_efld_cyl[..., 2]

    # Compute Ex and Ey in comoving frame
    X, Y = np.meshgrid(com_x_mesh, com_y_mesh, indexing='ij')
    phi = np.arctan2(Y, X)

    com_Ex = com_ER * np.cos(phi)[..., None]
    com_Ey = com_ER * np.sin(phi)[..., None]
    com_Ez = com_EZ

    com_efld_cart = np.stack((com_Ex, com_Ey, com_Ez), axis=-1)

    log("BOOSTING FIELDS BACK TO LAB FRAME...")

    lab_E = np.zeros_like(com_efld_cart)
    lab_B = np.zeros_like(com_efld_cart)

    flat_E = com_efld_cart.reshape(-1, 3)
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

    lab_E = np.reshape(flat_E_boosted, com_efld_cart.shape)
    lab_B = np.reshape(flat_B_boosted, com_efld_cart.shape)

    # scale the lab frame mesh
    lab_z_mesh = com_z_mesh / fv.gamma_3v(lab_ref_vel)

    # save lab frame eflds
    if main_conf.SAVE_PLOTS:
        file_name = "lab_eflds"
        file_path = os.path.join(output_dir, file_name + ".png")
        log(f"SAVING LAB E-FIELD PLOTS TO \"{file_path}\"...")

        idx = closestVal(bunch_conf.RADIUS, com_x_mesh)
        idy = closestVal(0, com_y_mesh)
        idz = closestVal(0, com_z_mesh)

        fig, axs = plt.subplots(1,2, figsize=(14, 5))

        # Radial Efld
        axs[0].grid()
        axs[0].axvline(x=com_ref_pos[0], label='Bunch Center', color='red')
        axs[0].axvspan(-lab_bunch_len/2, lab_bunch_len/2, alpha=0.2, label='Lab Bunch Length')
        axs[0].plot(lab_z_mesh, lab_E[idx, idy, :][:, 0], '.-', label="Radial Electric Field")
        axs[0].set_xlabel("Longitudinal Distance $z$ (m)")
        axs[0].set_ylabel("Electric Field Magnitude (V/m)")
        axs[0].set_title(f"Radial Electric Field ($n$ = {mesh_conf.QUAD_PTS}) (Lab. Frame)")
        axs[0].legend()

        # Longitudinal Efld
        axs[1].grid(True)
        axs[1].axvline(x=com_ref_pos[0], label='Bunch Center', color='red')
        axs[1].plot(lab_z_mesh, lab_E[idx, idy, :][:,2], '.-', label="Longitudinal Electric Field")
        axs[1].axvspan(-lab_bunch_len/2, lab_bunch_len/2, alpha=0.2, label='Lab Bunch Length')
        axs[1].set_xlabel("Longitudinal Distance $z$ (m)")
        axs[1].set_ylabel("Electric Field Magnitude (V/m)")
        axs[1].set_title(f"Longitudinal Electric Field ($n$ = {mesh_conf.QUAD_PTS}) (Lab. Frame)")
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    # save lab B-fields
    if main_conf.SAVE_PLOTS:
        file_name = "lab_bflds"
        file_path = os.path.join(output_dir, file_name + ".png")
        log(f"SAVING LAB B-FIELD PLOTS TO \"{file_path}\"...")

        idx = closestVal(bunch_conf.RADIUS, com_x_mesh)
        idy = closestVal(0, com_y_mesh)
        idz = closestVal(0, com_z_mesh)

        fig, axs = plt.subplots(1,2, figsize=(14, 5))

        # B_x
        axs[0].grid()
        axs[0].axvline(x=com_ref_pos[0], label='Bunch Center', color='red')
        axs[0].plot(lab_z_mesh, lab_B[idx, idy, :][:, 0], '.-', label="Horizontal Magnetic Field")
        axs[0].axvspan(-lab_bunch_len/2, lab_bunch_len/2, alpha=0.2, label='Bunch Length')
        axs[0].set_xlabel("Longitudinal Distance $z$ (m)")
        axs[0].set_ylabel("Magnetic Field Magnitude (T)")
        axs[0].set_title(f"Horizontal Magnetic Field($n$ = {mesh_conf.QUAD_PTS}) (Lab. Frame)")
        axs[0].legend()

        # B_y
        axs[1].grid(True)
        axs[1].axvline(x=com_ref_pos[0], label='Bunch Center', color='red')
        axs[1].plot(lab_z_mesh, lab_B[idx, idy, :][:,1], '.-', label="Vertical Magnetic Field")
        axs[1].axvspan(-lab_bunch_len/2, lab_bunch_len/2, alpha=0.2, label='Bunch Length')
        axs[1].set_xlabel("Longitudinal Distance $z$ (m)")
        axs[1].set_ylabel("Magnetic Field Magnitude (T)")
        axs[1].set_title(f"Vertical Magnetic Field ($n$ = {mesh_conf.QUAD_PTS}) (Lab. Frame)")
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(file_path)
        plt.close()

    # Save to pickle
    fields_filename = f"fields_{timestamp}.pkl"
    fields_path = os.path.join(output_dir, fields_filename)
    log(f"WRITING FIELDS TO \"{fields_path}\"...")

    with open(fields_path, 'wb') as f:
        pickle.dump({
            "mesh_lab": lab_z_mesh,
            "mesh_co": com_z_mesh,
            "E_lab": lab_E,
            "B_lab": lab_B,
            "E_co": com_efld_cart}, f)

    print("[ DONE ]")

    # log some extra helpful information
    log(f"GAMMA: {fv.gamma_3v(lab_ref_vel)}")
    log(f"LAB BUNCH LEN: {lab_bunch_len}")
    log(f"COMOVING BUNCH LEN: {com_bunch_len}")

if __name__ == "__main__":
    # get system args, if any
    main_config = dict(
        INTEGRATOR = "Trapezoidal", #Can be "Trapezoidal" or "Gaussian"
        SHOW_GAUSSIAN_FIT = True,
        SHOW_MESH = True,
        SAVE_PLOTS = True,
        OUT_PATH = ".",
        SEED_RNG = True
    )

    bunch_config = dict(
        # change the particles to be macroparticles. i.e. define bunch charge and then
        # scale it according to num particles (q = bunchCharge/N)
        NUM_PARTICLES = 5,
        SPECIES = "Electron",    # can be "Electron" or "Proton"
        MU_VEL = 2.6E8, #m/s
        SIG_VEL = 5E6,  #m/s
        DISTRIBUTION = "Gaussian", # can be "Uniform", "Mesa", or "Gaussian"
        RADIUS = 0.02097928077291609, # meters
        LENGTH = 0.003631584759359322,
        CHARGE = -1e-08
    )


    #notes, 7 mesh points and 64 quad points seem to work decently well for
    # mu_v = 2.6E8, sig_v = 5E6, rad = 1E-4, integ = trap
    mesh_config = dict(
        X_MESH_PTS = 5,
        Y_MESH_PTS = 5,   # set to -1 to set dy = dx
        Z_MESH_PTS = 49,  # set to -1 to set dz = dx
        QUAD_PTS = 48
    )

    gaussfits_config = dict(
        NUM_BINS = 50,
        NUM_GAUSSIANS = 50,
        WIDTH_GAUSSIANS = -1 # set to -1 to auto infer based on bunch length (works p well)
    )

    routine(
        main_config=main_config,
        bunch_config=bunch_config,
        mesh_config=mesh_config,
        gaussfits_config=gaussfits_config
    )

    # alternatively, we can pass in an input file like so
    # routine(sys.argv[1])
