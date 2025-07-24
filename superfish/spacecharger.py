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
from scipy.interpolate import RegularGridInterpolator

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
    SHOW_GAUSSIAN_FIT: bool
    SHOW_MESH: bool
    OUT_PATH: str
    SEED_RNG: bool

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
    parts,
    main_config: dict = None,
    mesh_config: dict = None,
    gaussfits_config: dict = None ):
    """
    Run the routine by passing configs directly.
    """

    # Validate that all configs are given
    if not all([main_config, mesh_config, gaussfits_config]):
        raise ValueError("If input_file is not provided, you must provide all config dictionaries.")

    main_conf = MainConfig(**main_config)
    mesh_conf = MeshConfig(**mesh_config)
    gauss_conf = GaussFitsConfig(**gaussfits_config)

    # Seed RNG if needed
    if main_conf.SEED_RNG:
        np.random.seed(6969)

    ###
    # STEP 1: Initialize the particle array
    ###

    alt_parts = []

    # this is so stupid i should really just extend the class or something
    for i in range(len(parts)):
        alt_parts.append(BunchParticle(
            Macroparticle,
            v0_3v=parts[i].vel,
            pos0_3v=parts[i].pos,
            mass_override=parts[i].m,
            charge_override=parts[i].q
        ))

    # get the velocities and positions of each particle
    lab_particle_vel = np.array([particle.get_3v() for particle in alt_parts])
    lab_particle_pos = np.array([particle.get_3p()[0] for particle in alt_parts])

    # get the velocity and position of the reference particle
    lab_ref_vel = fv.to_three_velocity(alt_parts[0].vel_4v)
    lab_ref_pos = lab_particle_pos[0]

    if main_conf.DEBUG:
        print(f"sample 3v: {alt_parts[2].get_3v()}")
        print(f"lab_ref_vel: {lab_ref_vel}, lab_ref_pos {lab_ref_pos}")
        print(f"γ (gamma): {fv.gamma_3v(lab_ref_vel)}")

        # try to get the total charge
        tot_charge = 0.0
        for particle in alt_parts:
            tot_charge += particle.charge

        print(f"total particle charge: {tot_charge}")

    # get the bunch length in the lab frame
    lab_bunch_len = max(lab_particle_pos[:,2]) - min(lab_particle_pos[:,2])
    # print(f"labbunchlencalc --> {max(lab_particle_pos[:,2])} - {min(lab_particle_pos[:,2])} = {lab_bunch_len}")

    # create reference particle
    reference = BunchParticle(Reference, v0_3v = lab_ref_vel, pos0_3v = lab_ref_pos)

    ###
    # STEP 2: Lorentz boost to the reference particle's frame
    ###

    # Perform the lorentz boost for each particle
    for particle in alt_parts:
        particle.lorentz_boost_to(reference)

    reference.boost_as_reference()

    # get the velocities and positions of each particle
    com_particle_vel = np.array([particle.get_3v() for particle in alt_parts])
    com_particle_pos = np.array([particle.get_3p()[0] for particle in alt_parts])

    # get the bunch length
    com_bunch_len = max(com_particle_pos[:,2]) - min(com_particle_pos[:,2])
    com_radii = np.sqrt(com_particle_pos[:,0]**2 + com_particle_pos[:,1]**2)

    # RMS radius
    bunch_rad_rms = np.sqrt(np.mean(com_radii**2))

    # Max radius
    bunch_rad_max = np.max(com_radii)

    #FIXME: TEMPORARY TESTING
    bunch_rad_rms = bunch_rad_max

    if main_conf.DEBUG:
        print(f"! Reference Particle Position: {reference.get_3p()}")
        print(f"Bunch RMS radius: {bunch_rad_rms}")
        print(f"Bunch Max radius: {bunch_rad_max}")

    ###
    # STEP 3: Bin the distribution of particles in the Lorentz-boosted frame
    ###

    # attempt to auto infer gaussian width based on bunch length
    if gauss_conf.WIDTH_GAUSSIANS == -1:
        gauss_conf.WIDTH_GAUSSIANS = com_bunch_len / gauss_conf.NUM_BINS

    if main_conf.DEBUG:
        print(f"FIT PARAMS \n BINS: {gauss_conf.NUM_BINS} \n NGAUS: {gauss_conf.NUM_GAUSSIANS}\
          \n WIDTH: {gauss_conf.WIDTH_GAUSSIANS} \n SCALE: {len(parts) * parts[0].q}")

    xGauss, ampGauss, sigGauss = mg.fit_gaussian_density(
        z_array=com_particle_pos[:,2],
        nbins=gauss_conf.NUM_BINS,
        ngaussians=gauss_conf.NUM_GAUSSIANS,
        width=gauss_conf.WIDTH_GAUSSIANS,
        plot=main_conf.SHOW_GAUSSIAN_FIT,
        scale=1) # charge of beam

    ###
    # STEP 4: Obtain the Longitudinal & Transverse Electric fields
    #
    # Take the bunch of particles to be a cylinder, and slice it up. Each slice
    # will have a charge density determined by the bunched up particles.
    #
    ###

    com_x_mesh, dx = np.linspace(
        start   = -bunch_rad_rms,
        stop    = bunch_rad_rms,
        num     = mesh_conf.X_MESH_PTS,
        retstep = True
    )

    com_y_mesh, dy = np.linspace(
        start   = -bunch_rad_rms,
        stop    = bunch_rad_rms,
        num     = mesh_conf.Y_MESH_PTS,
        retstep = True
    )

    com_z_mesh, dz = np.linspace(
        start   = min(com_particle_pos[:,2]),
        stop    = max(com_particle_pos[:,2]),
        num     = mesh_conf.Z_MESH_PTS,
        retstep = True
    )

    ### EVALUATE!!! ###
    com_efld_cyl = ltsolvers.solve_SCFields(
        bunch_rad=bunch_rad_rms,
        bunch_len=com_bunch_len,
        co_mesh=(com_x_mesh, com_y_mesh, com_z_mesh),
        rho_type="gaussian",
        integrator=main_conf.INTEGRATOR.lower(),
        n=mesh_conf.QUAD_PTS,
        gauss_params={
            "xGauss": xGauss,
            "ampGauss": ampGauss,
            "sigGauss": sigGauss
        }
    )

    # EXTRA PLOTTING
    if main_conf.DEBUG:
        idx = closestVal(bunch_rad_rms, com_x_mesh)
        idy = closestVal(0, com_y_mesh)
        idz = closestVal(0, com_z_mesh)

        fig, axs = plt.subplots(1,2, figsize=(14, 5))

        # Radial Efld
        axs[0].grid()
        axs[0].axvspan(-lab_bunch_len/2, lab_bunch_len/2, alpha=0.2, label='Lab Bunch Length')
        axs[0].plot(com_z_mesh, com_efld_cyl[idx, idy, :][:, 0], '.-', label="Radial Electric Field")
        axs[0].set_xlabel("Longitudinal Distance $z$ (m)")
        axs[0].set_ylabel("Electric Field Magnitude (V/m)")
        axs[0].set_title(f"Radial Electric Field ($n$ = {mesh_conf.QUAD_PTS}) (Co. Frame)")
        axs[0].legend()

        # Longitudinal Efld
        axs[1].grid(True)
        axs[1].plot(com_z_mesh, com_efld_cyl[idx, idy, :][:,2], '.-', label="Longitudinal Electric Field")
        axs[1].axvspan(-lab_bunch_len/2, lab_bunch_len/2, alpha=0.2, label='Lab Bunch Length')
        axs[1].set_xlabel("Longitudinal Distance $z$ (m)")
        axs[1].set_ylabel("Electric Field Magnitude (V/m)")
        axs[1].set_title(f"Longitudinal Electric Field ($n$ = {mesh_conf.QUAD_PTS}) (Co. Frame)")
        axs[1].legend()

        plt.tight_layout()
        plt.show()

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

    # Build interpolators
    Ex_interp = RegularGridInterpolator(
        (com_x_mesh, com_y_mesh, com_z_mesh),
        com_efld_cart[...,0]
    )

    Ey_interp = RegularGridInterpolator(
        (com_x_mesh, com_y_mesh, com_z_mesh),
        com_efld_cart[...,1]
    )

    Ez_interp = RegularGridInterpolator(
        (com_x_mesh, com_y_mesh, com_z_mesh),
        com_efld_cart[...,2]
    )

    # Evaluate E-field in comoving frame at particle positions
    E_com_at_particles = np.stack([
        Ex_interp(com_particle_pos),
        Ey_interp(com_particle_pos),
        Ez_interp(com_particle_pos)
    ], axis=1)

    # Lorentz transform each field vector to lab frame
    lab_E_at_particles = []
    lab_B_at_particles = []

    # print("parts[0].q", alt_parts[0].charge, "  len(alt)", len(alt_parts))


    for E_com in E_com_at_particles:
        E_lab, B_lab = fv.inverseFieldTransform(E_com, np.zeros(3), lab_ref_vel)
        lab_E_at_particles.append(E_lab)
        lab_B_at_particles.append(B_lab)

    lab_E_at_particles = np.array(lab_E_at_particles)
    lab_B_at_particles = np.array(lab_B_at_particles)

    if main_conf.DEBUG:
        print("[ CYCLE DONE ] \n \n")

    return lab_E_at_particles, lab_B_at_particles