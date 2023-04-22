#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <cstdlib>
#include <math.h>
#include <mpi.h>

void scan_seq(int* prefix_sum, const int* A, int n){
    if (n == 0) return;
    prefix_sum[0] = A[0];
    for (int i = 1; i < n; i++){
        prefix_sum[i] = prefix_sum[i-1] + A[i];
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(processor_name, &namelen);
    printf("Process %d on %s out of %d\n", rank, processor_name, size);
    double tt = MPI_Wtime();
    int Nsize = 100000;
    int *data = NULL, *local_data = NULL, *local_scan = NULL, *offsets = NULL;

    if (rank == 0)
    {
        // Fill data on rank 0
        data = (int *)malloc(Nsize * sizeof(int));
        for (int i = 0; i < Nsize; i++)
        {
            data[i] = rand() %10000;
            //printf("ind %d is %d\n", i, data[i]);
        }
    }
    //int *prefix_sum = (int *)malloc(Nsize * sizeof(int));
    //scan_seq(prefix_sum, data, Nsize);
    //for (int i = 0; i < Nsize; i++) printf("scan result %d is %d\n",i, prefix_sum[i]);

    local_data = (int *)malloc(Nsize / size * sizeof(int));
    MPI_Scatter(data, Nsize / size, MPI_INT, local_data, Nsize / size, MPI_INT, 0, comm);

    local_scan = (int *)malloc(Nsize / size * sizeof(int));
    local_scan[0] = local_data[0];
    for (int i = 1; i < Nsize / size; i++)
    {
        local_scan[i] = local_scan[i - 1] + local_data[i];
    }

    offsets = (int *)malloc(size * sizeof(int));
    MPI_Allgather(&local_scan[Nsize / size - 1], 1, MPI_INT, offsets, 1, MPI_INT, comm);

    int pre_sum = 0;
    for (int i = 0; i < rank; i++)
    {
        pre_sum += offsets[i];
    }
    for (int i = 0; i < Nsize / size; i++)
    {
        local_scan[i] += pre_sum;
    }

    MPI_Gather(local_scan, Nsize / size, MPI_INT, data, Nsize / size, MPI_INT, 0, MPI_COMM_WORLD);
    

    if (rank == 0)
    {
        int err= 0;
        //for (int i = 0; i < Nsize; i++) err = std::max(err, std::abs(data[i] - prefix_sum[i]));
        //for (int i = 0; i < Nsize; i++) printf("mpi scan %d is %d\n", i, data[i]);
        //printf("error = %d\n", err);
        tt = MPI_Wtime() - tt;
        printf("Total time = %d\n",tt);
    }
    MPI_Finalize();
    return 0;
}