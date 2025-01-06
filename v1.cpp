#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

// pass -DN <val> to define N without editing this file
#ifndef N
#define N 2000 // Matrix dimension
#endif

void init(double*& A, double*& B, double*& C) {
    A = new double[N * N];
    B = new double[N * N];
    C = new double[N * N];

    for (unsigned i = 0; i < N * N; i++) {
        A[i] = double(i) / N;
        B[i] = double(i) / N;
    }
}

int main() {
    MPI_Init(nullptr, nullptr);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int rows_per_proc = N / size;
    int remainder = N % size; 
    int rows = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;
    // int offset = rank * rows_per_proc + std::min(rank, remainder);

    double* A_local = new double[rows * N];
    double* B = new double[N * N];
    double* C_local = new double[rows * N];
    
    double *A = nullptr, *C = nullptr;
    if (rank == 0) {
        init(A, B, C);
    }

    double start = MPI_Wtime();

    // each cmputer has to get matrx B
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int* sendcounts = new int[size];
    int* displs = new int[size];
    
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (i < remainder ? rows_per_proc + 1 : rows_per_proc) * N;
        displs[i] = (i * rows_per_proc + std::min(i, remainder)) * N;
    }

    // We distribute each row of A to a different computer
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, A_local, rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // THis is the actual mult. O(N^3) 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < N; j++) {
            C_local[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C_local[i * N + j] += A_local[i * N + k] * B[k * N + j];
            }
        }
    }

    // send the results to the master proc.
    MPI_Gatherv(C_local, rows * N, MPI_DOUBLE,
                C, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "=====================================================\n";
        std::cout << "Distributed Matrix Multiplication\n";
        std::cout << "Matrix Size N: " << N << "\n";
        std::cout << "Number of Processes: " << size << "\n";
        std::cout << "Execution time: " << end - start << "\n";

        // Basic sum check
        double sum = 0;
        for (unsigned i = 0; i < N * N; ++i) {
            sum += C[i];
        }
        std::cout << "Sum: " << sum << std::endl;
        std::cout << "=====================================================\n";

        delete[] A;
        delete[] C;
    }

    // Cleanup
    delete[] A_local;
    delete[] B;
    delete[] C_local;
    delete[] sendcounts;
    delete[] displs;

    MPI_Finalize();
    return 0;
}
