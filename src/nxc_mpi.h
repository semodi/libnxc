#ifndef MPI_H
#define MPI_H
#ifdef MPI
int MPI_COMM_WORLD;
int MPI_SIZE;
int MPI_RANK;
int MPI_SUM;
int MPI_MAX;
int MPI_DOUBLE;
int MPI_INTEGER;
int mpierror = 0;
extern "C" {
void mpi_comm_rank_(int *,int *,int *);
void mpi_allreduce_(void *, void *, int *, int *,int *, int *, int *);
void mpi_comm_size_(int *, int *, int*);
void init_nxc_mpi_(int * mpi_comm_world, int * mpi_sum, int * mpi_max, int * mpi_double, int * mpi_int){
    MPI_COMM_WORLD = *mpi_comm_world;
    mpi_comm_rank_(&MPI_COMM_WORLD, &MPI_RANK, &mpierror);
    mpi_comm_size_(&MPI_COMM_WORLD, &MPI_SIZE, &mpierror);
    MPI_SUM = *mpi_sum;
    MPI_MAX = *mpi_max;
    MPI_DOUBLE = *mpi_double;
    MPI_INTEGER = *mpi_int;
    // std::cout << "Initializing MPI in C for rank " << MPI_RANK << std::endl;
  }
}
#endif
#endif
