import numpy as np
from numba_mpi import recv, send, allgather
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def distributed_vector_square(vector_size):
    """
    Splits a vector across MPI ranks, squares each element, and gathers the results.
    Only rank 0 returns the full result; other ranks return None.
    """
    # Only rank 0 initializes the full data
    if rank == 0:
        full_vector = np.random.rand(vector_size).astype(np.float64)
        # Split the vector into chunks
        chunks = np.array_split(full_vector, size)
    else:
        chunks = None

    # Scatter the chunks
    local_chunk = comm.scatter(chunks, root=0)

    # Perform computation: square the local chunk
    local_result = local_chunk ** 2

    # Gather all the local results back to rank 0
    gathered = comm.gather(local_result, root=0)

    if rank == 0:
        # Concatenate the results in order
        final_result = np.concatenate(gathered)
        return final_result
    else:
        return None

if __name__ == "__main__":
    result = distributed_vector_square(16)
    if rank == 0:
        print("Squared vector:", result)
