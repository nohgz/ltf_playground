from mpi4py import MPI

def main():
    world_comm = MPI.COMM_WORLD
    world_size = world_comm.Get_size()
    my_rank = world_comm.Get_rank()

    print(f"World Size: {world_size} \t Rank : {my_rank}")

if __name__ == "__main__":
    main()
    
