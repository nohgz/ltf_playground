import timeit, mpi4py, numba, numpy as np, numba_mpi


N_TIMES = 10000
RTOL = 1e-3

@numba.jit
def get_pi_part(n_intervals=1000000, rank=0, size=1):
    h = 1 / n_intervals
    partial_sum = 0.0
    for i in range(rank + 1, n_intervals, size):
        x = h * (i - 0.5)
        partial_sum += 4 / (1 + x**2)
    return h * partial_sum

@numba.jit
def pi_numba_mpi(n_intervals):
    pi = np.array([0.])
    part = np.empty_like(pi)
    for _ in range(N_TIMES):
        part[0] = get_pi_part(n_intervals, numba_mpi.rank(), numba_mpi.size())
        numba_mpi.allreduce(part, pi, numba_mpi.Operator.SUM)
        assert abs(pi[0] - np.pi) / np.pi < RTOL

def pi_mpi4py(n_intervals):
    pi = np.array([0.])
    part = np.empty_like(pi)
    for _ in range(N_TIMES):
        part[0] = get_pi_part(n_intervals, mpi4py.MPI.COMM_WORLD.rank, mpi4py.MPI.COMM_WORLD.size)
        mpi4py.MPI.COMM_WORLD.Allreduce(part, (pi, mpi4py.MPI.DOUBLE), op=mpi4py.MPI.SUM)
        assert abs(pi[0] - np.pi) / np.pi < RTOL

plot_x = [x for x in range(1, 11)]
plot_y = {'numba_mpi': [], 'mpi4py': []}
for x in plot_x:
    for impl in plot_y:
        plot_y[impl].append(min(timeit.repeat(
            f"pi_{impl}(n_intervals={N_TIMES // x})",
            globals=locals(),
            number=1,
            repeat=10
        )))

if numba_mpi.rank() == 0:
    from matplotlib import pyplot
    pyplot.figure(figsize=(8.3, 3.5), tight_layout=True)
    pyplot.plot(plot_x, np.array(plot_y['mpi4py'])/np.array(plot_y['numba_mpi']), marker='o')
    pyplot.xlabel('number of MPI calls per interval')
    pyplot.ylabel('mpi4py/numba-mpi wall-time ratio')
    pyplot.title(f'mpiexec -np {numba_mpi.size()}')
    pyplot.grid()
    pyplot.savefig('readme_plot.svg')