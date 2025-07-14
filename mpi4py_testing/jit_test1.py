import numba
from numba import njit, prange
from mpi4py import MPI
import numpy as np

# TODO: Add a function that we can integrate over then MPI that

# added eager compilation
@njit(numba.int32(numba.int32, numba.int32), parallel=True)
def _jit_add(lo, hi):
    sum = 0
    for i in prange(lo, hi):
        sum += i
    return sum

def main():
    """
    what the problem should be

    sum = 0
    for i in range(N):
        sum += i
    """
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    N = 10000
    print(f"world {world_size} rank {my_rank}")

    # determine workload for each rank
    workloads = [N // world_size for i in range(world_size)]
    for i in range(N % world_size):
        workloads[i] += 1
    my_start = 0
    for i in range( my_rank ):
        my_start += workloads[my_rank]
    my_end = my_start + workloads[my_rank]

    # perform each nodes work
    start_time = MPI.Wtime()
    local_sum = _jit_add(my_start, my_end)
    end_time = MPI.Wtime()

    if my_rank == 0:
        print(f"sum time: {end_time-start_time}")

    sendbuf = np.array([local_sum], dtype='i')
    recvbuf = np.zeros(1, dtype='i')

    world_comm.Reduce(sendbuf, recvbuf, op=MPI.SUM, root=0)

    if my_rank == 0:
        print("Sum:", recvbuf[0])

if __name__ == "__main__":
    main()
