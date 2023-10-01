from time import sleep
from mpi4py.futures import MPIPoolExecutor
from mpi4py.MPI import Get_processor_name, COMM_WORLD


def func(n):
    comm = COMM_WORLD
    rank = comm.Get_rank()
    processor = Get_processor_name()
    print(f'rank={rank} processor={processor} n={n} sleep(10)')
    sleep(10)

def main():
    comm = COMM_WORLD
    rank = comm.Get_rank()
    processor = Get_processor_name()
    size = comm.Get_size()
    print(f'rank={rank} processor={processor} size={size}')

    executor = MPIPoolExecutor()
    for i in range(11, 15):
        executor.submit(func, i)
    print('before sleep(20)')
    sleep(20)
    print('end main()')


def worker():
    MPIPoolExecutor()


if __name__ == "__main__":
    main()


if __name__ == "__worker__":
    worker()
