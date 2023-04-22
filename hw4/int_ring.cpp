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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &namelen);
    printf("Process %d on %s out of %d\n", rank, processor_name, size);
    MPI_Barrier(MPI_COMM_WORLD);

    double tt = MPI_Wtime();

    int sum = 0;
    for (int repeat = 0; repeat < Nrepeat; repeat++)
    {
        MPI_Status status;
        if (rank == 0)
        {
            MPI_Send(&sum, 1, MPI_INT, 1, repeat, MPI_COMM_WORLD);
            MPI_Recv(&sum, 1, MPI_INT, size - 1, repeat, MPI_COMM_WORLD, &status);
            //printf("Iteration %d: sum = %d\n", repeat, sum);
        }
        else
        {

            MPI_Recv(&sum, 1, MPI_INT, rank - 1, repeat, MPI_COMM_WORLD, &status);
            sum += rank;
            MPI_Send(&sum, 1, MPI_INT, (rank + 1) % size, repeat, MPI_COMM_WORLD);
        }
    }

    if (rank == 0)
    {
        tt = MPI_Wtime() - tt;
        printf("latency: %e ms\n", tt / Nrepeat * 1000);
        printf("Total time: %f s\n", tt);
    }
    MPI_Finalize();
    return 0;
}
