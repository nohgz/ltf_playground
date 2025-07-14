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

# >>> HELPER FUNCTIONS >>>
def closestVal(val, array, dx = None):
    if dx == None:
        dx = array[1] - array[0]

    # clip val to be in the array
    val = np.clip(val, array[0], array[-1])

    return int((val - array[0])/dx)
# <<< HELPER FUNCTIONS <<<

def routine(
    main_config: dict = None,
    bunch_config: dict = None,
    mesh_config: dict = None,
    gaussfits_config: dict = None ):
    """
    Run the routine by passing configs directly.
    """

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

    print(f"NPARTS {bunch_conf.NUM_PARTICLES}")

    ###
    # STEP 1: Initialize the particle array
    ###
    # create an array of NUM_PARTICLES Particle Objects with random velocities
    parts = []

    # i think this magic number works best for determining particle spread
    # based only on a given bunch length for the lab
    sig_pos = bunch_conf.LENGTH / 3

    if bunch_conf.SPECIES[0].lower == "e":
        for i in range(bunch_conf.NUM_PARTICLES):
            parts.append(BunchParticle(
                Electron,
                v0_3v=[0, 0, np.clip(np.random.normal(bunch_conf.MU_VEL, bunch_conf.SIG_VEL), 0, c)],
                pos0_3v=[0, 0, np.random.normal(0, sig_pos)]
            ))
    elif bunch_conf.SPECIES[0].lower == "p":
        for i in range(bunch_conf.NUM_PARTICLES):
            parts.append(BunchParticle(
                Proton,
                v0_3v=[0, 0, np.clip(np.random.normal(bunch_conf.MU_VEL, bunch_conf.SIG_VEL), 0, c)],
                pos0_3v=[0, 0, np.random.normal(0, sig_pos)]
            ))
    else: #use macroparticle
        p_charge = bunch_conf.CHARGE/bunch_conf.NUM_PARTICLES
        p_mass = abs(p_charge/Electron.CHARGE.value)*Electron.MASS.value
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
    lab_particle_vel = np.array([particle.get_3v() for particle in parts])
    lab_particle_pos = np.array([particle.get_3p()[0] for particle in parts])

    # get the velocity and position of the reference particle
    # lab_ref_vel = np.mean(lab_particle_vel, axis=0)[0]
    lab_ref_vel = parts[0].vel_4v[-1]
    print(lab_ref_vel)

    lab_ref_pos = np.mean(lab_particle_pos, axis=0)

    # get the bunch length in the lab frame
    lab_bunch_len = max(lab_particle_pos[:,2]) - min(lab_particle_pos[:,2])

    # create reference particle
    reference = BunchParticle(Reference, v0_3v = lab_ref_vel, pos0_3v = lab_ref_pos)

    ###
    # STEP 2: Lorentz boost to the reference particle's frame
    ###

    # Perform the lorentz boost for each particle
    for particle in parts:
        particle.lorentz_boost_to(reference)

    reference.boost_as_reference()

    # get the velocities and positions of each particle
    com_particle_vel = np.array([particle.get_3v() for particle in parts])
    com_particle_pos = np.array([particle.get_3p()[0] for particle in parts])

    # get the velocity and position of the reference particle
    com_ref_vel = np.mean(com_particle_vel, axis=0)[0]
    com_ref_pos = np.mean(com_particle_pos, axis=0)

    ###
    # STEP 3: Bin the distribution of particles in the Lorentz-boosted frame
    ###

    # attempt to auto infer gaussian width based on bunch length
    if gauss_conf.WIDTH_GAUSSIANS == -1:
        gauss_conf.WIDTH_GAUSSIANS = 2 * lab_bunch_len / gauss_conf.NUM_BINS


    print(f"FIT PARAMS \n BINS: {gauss_conf.NUM_BINS} \n NGAUS: {gauss_conf.NUM_GAUSSIANS} \n WIDTH: {gauss_conf.WIDTH_GAUSSIANS} \n SCALE: {bunch_conf.CHARGE}")

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

    com_bunch_len = max(com_particle_pos[:,2]) - min(com_particle_pos[:,2])

    # the x mesh will always be this
    com_x_mesh, dx = np.linspace(
        start   = -bunch_conf.RADIUS,
        stop    = bunch_conf.RADIUS,
        num     = mesh_conf.X_MESH_PTS,
        retstep = True
    )

    if mesh_conf.Y_MESH_PTS == -1:
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


    ###
    # Step 6: Convert the fields in the reference particle frame to cartesian,
    #         then convert that into the lab frame using the functions that I defined.
    ###

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


    print(f"lab_zmeshran -> {lab_z_mesh[0]} TO {lab_z_mesh[-1]}")
    print(f"com_blen -> {com_bunch_len}")

    return lab_E, lab_B, com_x_mesh, com_y_mesh, lab_z_mesh