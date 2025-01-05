
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

#define N 2000 // Matrix dimensions (NxN)

// Initialize matrices A, B, and C (only on rank 0)
void init(double*& A, double*& B, double*& C) {
    A = new double[N * N];
    B = new double[N * N];
    C = new double[N * N];

    for (unsigned i = 0; i < N * N; i++) {
        A[i] = double(i) / N;
        B[i] = double(i) / N;
        C[i] = 0.0;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *A = nullptr, *B = nullptr, *C = nullptr;
    double *local_A = nullptr, *local_C = nullptr;

    int rows_per_process = N / size;
    int leftover_rows = N % size;

    if (rank == 0) {
        // Initialize matrices A, B, and C
        init(A, B, C);
    }

    // Allocate memory for local matrices
    int local_rows = rows_per_process + (rank < leftover_rows ? 1 : 0);
    local_A = new double[local_rows * N];
    local_C = new double[local_rows * N]();

    // Scatter rows of A among processes
    int* sendcounts = nullptr;
    int* displs = nullptr;

    if (rank == 0) {
        sendcounts = new int[size];
        displs = new int[size];

        int offset = 0;
        for (int i = 0; i < size; i++) {
            int rows = rows_per_process + (i < leftover_rows ? 1 : 0);
            sendcounts[i] = rows * N;
            displs[i] = offset;
            offset += rows * N;
        }
    }

    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, local_A, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Broadcast matrix B to all processes
    if (rank == 0) {
        MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    } else {
        B = new double[N * N];
        MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Perform local computation of C = A * B
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                local_C[i * N + j] += local_A[i * N + k] * B[k * N + j];
            }
        }
    }

    // Gather results back to rank 0
    MPI_Gatherv(local_C, local_rows * N, MPI_DOUBLE, C, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Cleanup
    delete[] local_A;
    delete[] local_C;
    if (rank == 0) {
        delete[] sendcounts;
        delete[] displs;
        delete[] A;
        delete[] B;
        delete[] C;
    } else {
        delete[] B;
    }

    MPI_Finalize();
    return 0;
}
