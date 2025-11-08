#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);           // Initialize MPI

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  // Total processes

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);  // Process rank

    printf("Hello from rank %d out of %d processes\n", world_rank, world_size);

    MPI_Finalize();                   // Finalize MPI
    return 0;
}
