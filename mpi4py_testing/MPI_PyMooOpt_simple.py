from numba import njit, jit, prange, set_num_threads, get_num_threads
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
import os
from mpi4py import MPI
import numba_mpi as nbMPI
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.isres import ISRES
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize
from multiprocess import Pool
import math
import numpy
import pickle
import sys
numpy.math = math


eps0 = 8.854e-12
Keps0 = 1/(4*np.pi*eps0)
c = 3e8
beam_charge = -1e-9
energy = 100.0 #MeV (mod this), keep at 40 MeV now
R = 2.19       # rad of curv of bend
k = 1/R
phi = np.linspace(0.01/180*np.pi,25/180*np.pi,40,endpoint=False)
shldmin = 1e-2
shldmax = 0.1
shldgap = 1e-2
Nimages = 0
NGauss = 32 # num of gaussians to discretize profile (mod this)
Nwake = 101 # num of pts in the comoving mesh
sigma_z = 300e-6 #2e-3/6 #300e-6
m = 9.1e-31
c = 3e8
Np = 1e3
Nbins = 50
sigmax = sigma_z # the sigma of each gaussian (mod this)
sigma = sigma_z

STOP_CODE = -999.14

# *** MPI STUFF ***
mpi_rank = nbMPI.rank()
mpi_size = nbMPI.size()

# This stuff is used in find_wake_def_shape_local
workloads = [Nwake // mpi_size for i in range(mpi_size)]


for i in range(Nwake % mpi_size):
    workloads[i] += 1

if mpi_rank == 0:
    print(mpi_size, workloads)

starts = [0]
for w in workloads[:-1]:
    starts.append(starts[-1]+w)
ends = [s+w for s,w in zip(starts, workloads)]

mpi_wake_start = starts[mpi_rank]
mpi_wake_end = ends[mpi_rank]

print(f"RANK: {mpi_rank} WAKE PART: {mpi_wake_start}-{mpi_wake_end}")
# *** END MPI STUFF ***


# also doesn't seem to be used
def flat(z,sigma_z):
    norm = 2*(2*sigma_z/np.pi) + 4*sigma_z
    Rise = (np.heaviside(z+3*sigma_z,0) - np.heaviside(2.0*sigma_z+z,1))*np.sin(abs(np.pi/2*(z+3*sigma_z)/(sigma_z)))
    Flat = (np.heaviside(z+2.0*sigma_z,1) - np.heaviside(z-2.0*sigma_z,1))
    Fall = (np.heaviside(z-2.0*sigma_z,1) - np.heaviside(z-3*sigma_z,0))*np.sin(abs(np.pi/2*(z-3*sigma_z)/(sigma_z)))

    return (Rise + Flat + Fall)


# doesn't seem to be used
# @njit(cache=True)
# def gauss_un(z,sigma_z):
#     return np.exp(-(z)**2/(2*sigma_z**2))


@njit(cache=True)
def gauss_sum(x,NGauss,xGauss,AmpGauss,SigGauss):
    #NGauss = len(xGauss)
    sum = np.zeros(np.size(x)) #.reshape([np.size(x),])
    normsum = 0.0
    for i in range(NGauss):
        sum += AmpGauss[i]*np.exp(-(x-xGauss[i])**2/2/SigGauss[i]**2)
        normsum += AmpGauss[i]*np.sqrt(2*np.pi*SigGauss[i]**2)
    return sum/normsum

@njit(cache=True)
def lmd(z,params,xGauss,AmpGauss,SigGauss,deriv=False):
    NGauss = len(xGauss)
    sigma = SigGauss[0]
    dz = sigma/1000
    Np = params[1]
    charge = params[2]
    #deriv = params[3]
    fac = charge #/(sigma * np.sqrt(2 * np.pi))
    #fac = charge/sigma/np.sqrt(2*np.pi)
    if deriv==True:
        #return fac*(flat(z+dz,sigma) - flat(z-dz,sigma))/(2*dz)
        return fac*(gauss_sum(z+dz,NGauss,xGauss,AmpGauss,SigGauss) - gauss_sum(z-dz,NGauss,xGauss,AmpGauss,SigGauss))/(2*dz)
        #return fac/(2*dz)*(np.exp(-(z+dz)**2/(2*sigma**2)) - np.exp(-(z-dz)**2/(2*sigma**2)))
    #return fac*np.exp(-z**2/(2*sigma**2))
    return fac*gauss_sum(z,NGauss,xGauss,AmpGauss,SigGauss) #*flat(z,sigma)

@njit(cache=True)
def find_wake_mayes_images(s,R,phi,energy,N,H,xGauss,AmpGauss,SigGauss):
    sL = R*phi**3/24.0

    gmma = energy/0.511
    beta = np.sqrt(1-1/gmma**2)

    alph = np.linspace(k*s+phi,k*s,1001)
    dalph = alph[1]-alph[0]
    sum = 0
    for n in range(1,N+1):
        sgn = (-1)**n
        ralph = np.sqrt(2-2*np.cos(alph)+(n*k*H)**2)
        #print(ralph)
        salph = 1/k*(k*s - alph + beta*ralph)

        Integ = (beta**2*np.cos(alph)-1)/ralph*lmd(salph,[sigma,Np,1.0],xGauss,AmpGauss,SigGauss,deriv=True)
        IntContrib = np.sum((Integ[1:]+Integ[:-1])/2)*dalph
        BndContrib = -k*(lmd(salph[-1],[sigma,Np,1.0],xGauss,AmpGauss,SigGauss)[0]/ralph[-1] \
                         - lmd(salph[0],[sigma,Np,1.0],xGauss,AmpGauss,SigGauss)[0]/ralph[0])
        sum += 2*sgn*(IntContrib + BndContrib)

    return Keps0*sum

@njit(fastmath=True, cache=True)
def find_wake_mayes_ss(s,R,phi,energy,xGauss,AmpGauss,SigGauss):

    sL = R*phi**3/24.0
    k = 1/R

    gmma = energy/0.511
    beta = np.sqrt(1-1/gmma**2)
    lb = 10*sigma_z

    alph = np.linspace(k*s+phi,k*s,51)

    dalph = alph[1]-alph[0]
    ralph = np.sqrt(2-2*np.cos(alph))

    ii = np.where(np.abs(alph) < 1e-3)
    jj = np.where(np.abs(alph) >= 1e-3)


    salph = 1/k*(k*s - alph + beta*ralph)

    Integrand = np.zeros(len(alph))

    Integrand[jj] = (beta**2*np.cos(alph[jj])-1)/ralph[jj] - alph[jj]/np.abs(alph[jj])/gmma**2*(1-beta*np.sin(alph[jj])/ralph[jj])/(alph[jj]-beta*ralph[jj]) #- (beta**2-1)/alph
    Integrand[ii] = (1/(2*gmma**2)+alph[ii]**2/8)*(8*gmma**2*alph[ii]+gmma**4*alph[ii]**2)/(6+2*gmma**2*alph[ii]**2 + gmma**4*alph[ii]**4/4)

    Integ = (Integrand*lmd(salph,[sigma,Np,1.0],xGauss,AmpGauss,SigGauss,deriv=True))
    IntContrib = np.sum(Integ)*dalph

    if sL < 1e-12:
        sL = k*lb

    sum = IntContrib + (lmd(s-sL,[sigma,Np,1.0],xGauss,AmpGauss,SigGauss,False)[0]\
                        -lmd(s-4*sL,[sigma,Np,1.0],xGauss,AmpGauss,SigGauss,False)[0])/sL**(1/3)

    return Keps0*sum

@njit(fastmath=True)
def find_wake_mayes(s,R,phi,energy,N,H,xGauss,AmpGauss,SigGauss):
    # Main CSR Wake
    eCSR = find_wake_mayes_ss(s,R,phi,energy,xGauss,AmpGauss,SigGauss)
    eShld = find_wake_mayes_images(s,R,phi,energy,N,H,xGauss,AmpGauss,SigGauss)

    return eCSR+eShld

@njit
def find_wake_def_shape_local(gap, z, Nimages, R, phi, energy, xGauss, AmpGauss, SigGauss):
    Wake_local = np.zeros(mpi_wake_end - mpi_wake_start)
    for local_i, global_i in enumerate(range(mpi_wake_start, mpi_wake_end)):
        Wake_local[local_i] = find_wake_mayes(
            z[global_i], R, phi, energy, Nimages, gap,
            xGauss, AmpGauss, SigGauss
        )
    return Wake_local

@njit()
def evolve_distribution(
    gap, z, beam_charge, Nwake, Nimages, R, phi, energy,
    xGauss, AmpGauss, SigGauss):
    assert mpi_rank == 0, "only rank 0 allowed!"


    Ns = len(phi)
    # DO NOT PARALLELIZE PHI EVER BRO IT WORKS IN 1-D BUT MIGHT NOT WORK IN 3-D
    ds = R * (phi[1] - phi[0])
    energies = np.zeros(Nwake - 1)

    for i in range(1, Ns):
        send_buf = np.empty(1 + NGauss, dtype=np.float64)

        # pack it into a 1-D buffer
        send_buf[0] = phi[i-1]
        send_buf[1:1+NGauss] = AmpGauss #TODO: Perhaps one broadcast of AmpGauss? can be done to save some chatter-time

        # Broadcast phi + profile to all ranks
        nbMPI.bcast(send_buf, root=0)

        # Rank 0 computes its own chunk
        Wake_local = find_wake_def_shape_local(
            gap=gap,
            z=z,
            Nimages=Nimages,
            R=R,
            phi=phi[i-1],
            energy=energy,
            xGauss=xGauss,
            AmpGauss=AmpGauss,
            SigGauss=SigGauss
        )

        print("LWL:", len(Wake_local))
        # Gather wake chunks from all ranks
        # size = total number of ranks
        recv_buf = np.zeros((mpi_size, len(Wake_local)), dtype=np.float64)
        nbMPI.gather(Wake_local, recv_buf, count=len(Wake_local), root=0)

        # if mpi_rank == 0:
        #     print(recv_buf)

        wake = np.zeros(Nwake)
        idx = 0

        for chunk in recv_buf:
            print(chunk)
            wake[idx:idx+len(chunk)] = chunk
            idx += len(chunk)

        energies += 0.5 * (wake[1:] + wake[:-1]) * ds * beam_charge / 1e6

    return energies

@njit
def worker_loop():
    if mpi_rank == 0:
        print("Rank 0 must not be a worker! D:<")

    no_buf = np.empty(1, dtype=np.float64) # acts like None
    recv_buf = np.empty(1+2*NGauss, dtype=np.float64)

    while True:
        nbMPI.bcast(recv_buf, root=0)

        if recv_buf[0] == STOP_CODE:
            break

        phi_val = recv_buf[0]
        AmpGauss = recv_buf[1:1+NGauss].copy()

        Wake_local = find_wake_def_shape_local(
            gap=shldgap,
            z=g_z,
            Nimages=Nimages,
            R=R,
            phi=phi_val,
            energy=energy,
            xGauss=g_xGauss,
            AmpGauss=AmpGauss,
            SigGauss=g_sigGauss
        )

        print("NEH",Wake_local.shape)
        nbMPI.gather(Wake_local, no_buf, count=len(Wake_local), root=0)


class CSROptProblem(ElementwiseProblem):

    def __init__(self,BeamCharge,sigma,energy,R,Phi,ShldMin,ShldMax,NGauss,AmpLim,SigLim,Nwake,**kwargs):
        super().__init__(n_var=NGauss,
                         n_obj=1,
                         n_ieq_constr=0,
                         xl=np.array([AmpLim[0] for i in range(NGauss)]),
                         #,linac_phases[0]-PhaseDiff,linac_phases[1]-PhaseDiff,linac_phases[2]-PhaseDiff,linac_phases[3]-PhaseDiff,linac_phases[4]-PhaseDiff,linac_phases[5]-PhaseDiff]),
                         xu=np.array([AmpLim[1] for i in range(NGauss)]),**kwargs)
                        #,linac_phases[0]+PhaseDiff,linac_phases[1]+PhaseDiff,linac_phases[2]+PhaseDiff,linac_phases[3]+PhaseDiff,linac_phases[4]+PhaseDiff,linac_phases[5]+PhaseDiff]))

        self.BeamCharge = BeamCharge
        self.R = R
        self.Phi = Phi
        self.ShldMin = ShldMin
        self.ShldMax = ShldMax
        self.NGauss = NGauss
        self.sigma = sigma
        self.xGauss = np.linspace(-4*sigma,4*sigma,self.NGauss,endpoint=True)
        self.Nwake = Nwake
        self.energy = energy
        self.SigLim = SigLim
        self.SigGauss = self.SigLim * np.ones(self.NGauss)

    def _evaluate(self, x, out, *args, **kwargs):
        if mpi_rank == 0:
            self.AmpGauss = np.array(x[:self.NGauss])
            z = np.linspace(-5*self.sigma, 5*self.sigma, self.Nwake)

            Lmd = lmd(
                z,
                [self.sigma, 1.0, 1.0],
                self.xGauss,
                self.AmpGauss,
                self.SigGauss,
                deriv=False
            )

            energies = evolve_distribution(
                gap=1e-2,
                z=z,
                beam_charge=self.BeamCharge,
                Nwake=self.Nwake,
                Nimages=0,
                R=self.R,
                phi=self.Phi,
                energy=self.energy,
                xGauss=self.xGauss,
                AmpGauss=self.AmpGauss,
                SigGauss=self.SigGauss
            )

            mean = np.sum(energies * 0.5*(Lmd[1:] + Lmd[:-1]) * (z[1] - z[0]))
            out["F"] = np.sqrt(np.sum((energies - mean)**2 * 0.5*(Lmd[1:] + Lmd[:-1]) * (z[1] - z[0])))

problem = CSROptProblem(beam_charge,sigma_z,energy,R,phi,shldmin,shldmax,NGauss,[1e-3,1.0],sigmax,Nwake)

# define some globals for the MPI stuff
g_xGauss = problem.xGauss
g_sigGauss = problem.SigGauss
g_z = np.linspace(-5 * problem.sigma, 5 * problem.sigma, problem.Nwake)

#algorithm = NSGA2( # params of genetic algo
#    pop_size=100,
#    n_offsprings=50,
#    sampling=FloatRandomSampling(),
#    crossover=SBX(prob=0.92, eta=25),
#    mutation=PM(eta=25),
#    eliminate_duplicates=True
#)

#CMAES(x0=np.array([np.random.uniform(low = 1e-2, high = 1.0, size=NGauss)]\
#                      +[np.random.uniform(low = sigmax/4, high = sigmax, size=NGauss)]))
#

algorithm = ISRES(n_offsprings=50, rule=1.0 / 7.0, gamma=0.85, alpha=0.2)
termination = get_termination("n_gen", 8) # could relax to 100

#    -- old stuff --
#    Ncores = 128
#    tot_threads = get_num_threads()
#    set_num_threads(32)
#    n_threads = 32
#    pool = Pool(n_threads)
#    runner = StarmapParallelization(pool.starmap)

if __name__ == "__main__":
    if mpi_rank == 0:

        starttime = nbMPI.wtime()
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=1,
            save_history=False,
            verbose=True
        )
        # After optimizer is done, send STOP
        nbMPI.bcast(np.array([STOP_CODE], dtype=np.float64), root=0)
        endtime = nbMPI.wtime()

        print(f"TIME: {endtime-starttime}")

        with open("opt_out.pkl", "wb") as f:
            pickle.dump(res, f)

    else:
        # Other ranks do their own thing (which is wait for instruction from 0)
        worker_loop()
        pass