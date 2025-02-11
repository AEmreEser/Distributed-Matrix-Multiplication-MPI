#include <iostream>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include <vector>
#include <algorithm>

#ifndef N
#define N 2000  
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 24  
#endif

/** 
    Ahmet Emre Eser - 2025 
*/

using namespace std;

void init(double*& A, double*& B, double*& C) {
    
    A = (double*)aligned_alloc(64, N * N * sizeof(double));
    B = (double*)aligned_alloc(64, N * N * sizeof(double));
    C = (double*)aligned_alloc(64, N * N * sizeof(double));

    for (unsigned i = 0; i < N * N; i++) {
        A[i] = double(i) / N; // double(std::rand() % 1000) / 10;
        B[i] = double(i) / N; // double(std::rand() % 1000) / 10;
    }
}


void multiply_blocked(const double* A_local, const double* B, double* C_local, int rows) {
    vector<double> B_block(BLOCK_SIZE * BLOCK_SIZE);
    
    fill_n(C_local, rows * N, 0.0);
    
    for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            for (int i = 0; i < rows; i++) {
                for (int j = jj; j < min(jj + BLOCK_SIZE, N); j++) {
                    double sum = C_local[i * N + j];
                    #pragma unroll(3) // old value: 8
                    for (int k = kk; k < min(kk + BLOCK_SIZE, N); k++) {
                        sum += A_local[i * N + k] * B[k * N + j];
                    }
                    C_local[i * N + j] = sum;
                }
            }
        }
    }
}

int main() {
    MPI_Init(nullptr, nullptr);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    MPI_Datatype column_type;
    MPI_Type_vector(N, 1, N, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    int rows_per_proc = N / size;
    int remainder = N % size;
    int rows = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;
    // int offset = rank * rows_per_proc + min(rank, remainder);

    double* A_local = (double*)aligned_alloc(64, rows * N * sizeof(double));
    double* B = (double*)aligned_alloc(64, N * N * sizeof(double));
    double* C_local = (double*)aligned_alloc(64, rows * N * sizeof(double));
    
    double *A = nullptr, *C = nullptr;
    if (rank == 0) {
        init(A, B, C);
    }

    double start = MPI_Wtime();
    
    MPI_Request bcast_request;
    MPI_Ibcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD, &bcast_request);
    
    vector<int> sendcounts(size);
    vector<int> displs(size);
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (i < remainder ? rows_per_proc + 1 : rows_per_proc) * N;
        displs[i] = (i * rows_per_proc + min(i, remainder)) * N;
    }
    
    MPI_Scatterv(A, sendcounts.data(), displs.data(), MPI_DOUBLE, A_local, rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Wait(&bcast_request, MPI_STATUS_IGNORE);

    multiply_blocked(A_local, B, C_local, rows);

    MPI_Request gather_request;
    MPI_Igatherv(C_local, rows * N, MPI_DOUBLE, C, sendcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD, &gather_request);

    MPI_Wait(&gather_request, MPI_STATUS_IGNORE);

    double end = MPI_Wtime();

    if (rank == 0) {
        cout << "=====================================================\n";
        cout << "Distributed Matrix Multiplication\n";
        cout << "Matrix Size N: " << N << "\n";
        cout << "Number of Processes: " << size << "\n";
        cout << "Block Size: " << BLOCK_SIZE << "\n";
        cout << "Execution time: " << end - start << "\n";

        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (unsigned i = 0; i < N * N; ++i) {
            sum += C[i];
        }
        cout << "Sum: " << sum << endl;
        cout << "=====================================================\n";

        free(A);
        free(C);
    }

    free(A_local);
    free(B);
    free(C_local);
    MPI_Type_free(&column_type);

    MPI_Finalize();
    return 0;
}
