import numba
from numba import njit, prange
from mpi4py import MPI
import numpy as np

@njit(numba.float32(numba.float32))
def _jit_fxn(x):
    return np.sinh(x)

@njit(numba.float32(numba.float32, numba.float32, numba.uint16))
def _jit_integrate(x_lo, x_hi, n):
    """Trapezoidal rule integrator"""
    h = (x_hi - x_lo) / (n - 1)

    sum = (_jit_fxn(x_lo) + _jit_fxn(x_hi)) / 2
    for i in prange(1, n - 1):
        sum += _jit_fxn(x_lo + i * h)

    return h * sum

def main():
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    # interval of integration
    INTERVAL = np.array([0, 5])
    TRAP_PTS = 128

    interval_length = INTERVAL[1] - INTERVAL[0]
    workload = interval_length / world_size

    my_start = INTERVAL[0] + my_rank * workload
    my_end = my_start + workload
    
    # load balancing
    base = TRAP_PTS // world_size
    if my_rank < TRAP_PTS % world_size:
        my_pts = base + 1
    else:
        my_pts = base


    print(f"RANK: {my_rank}, start: {my_start}, end: {my_end}, NUM POINTS: {my_pts}")
    
    # perform each nodes work
    start_time = MPI.Wtime()
    local_sum = _jit_integrate(my_start, my_end, int(128/world_size))
    end_time = MPI.Wtime()

    if my_rank == 0:
        print(f"sum time: {end_time-start_time}")

    sendbuf = np.array([local_sum], dtype='float32')
    recvbuf = np.zeros(1, dtype='float32')
    world_comm.Reduce(sendbuf, recvbuf, op=MPI.SUM, root=0)

    if my_rank == 0:
        print("Sum:", recvbuf[0])

if __name__ == "__main__":
    main()
