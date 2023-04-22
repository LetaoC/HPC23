#include <stdio.h>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    if (argc < 2)
    {
        printf("Usage: mpirun ./pingpong <loop-time:N>");
        abort();
    }

    int Nrepeat = atoi(argv[1]);

    int Nsize = 500000;
    int *array = (int *)malloc(Nsize * sizeof(int));
    for (int i = 0; i < Nsize; i++)
        array[i] = 10;
    MPI_Barrier(MPI_COMM_WORLD);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &namelen);
    printf("Process %d on %s out of %d\n", rank, processor_name, size);
    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();

    for (int repeat = 0; repeat < Nrepeat; repeat++)
    {
        MPI_Status status;
        if (rank == 0)
        {
            MPI_Send(&array, 1, MPI_INT, 1, repeat, MPI_COMM_WORLD);
            MPI_Recv(&array, 1, MPI_INT, size - 1, repeat, MPI_COMM_WORLD, &status);
            //printf("Iteration %d: sum = %d\n", repeat, array[0]);
        }
        else
        {

            MPI_Recv(&array, 1, MPI_INT, rank - 1, repeat, MPI_COMM_WORLD, &status);
            //array[0] += rank;
            MPI_Send(&array, 1, MPI_INT, (rank + 1) % size, repeat, MPI_COMM_WORLD);
        }
    }

    if (rank == 0)
    {
        tt = MPI_Wtime() - tt;
        printf("Bandwidth: %e GB/s\n", (Nsize * Nrepeat) / tt / 1e9);
        printf("Total time: %f s\n", tt);
    }
    MPI_Finalize();
    return 0;
}