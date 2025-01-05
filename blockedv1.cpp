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
#define BLOCK_SIZE 32  
#endif

void init(double*& A, double*& B, double*& C) {
    
    A = (double*)aligned_alloc(64, N * N * sizeof(double));
    B = (double*)aligned_alloc(64, N * N * sizeof(double));
    C = (double*)aligned_alloc(64, N * N * sizeof(double));

    
    for (unsigned j = 0; j < N; j++) {
        for (unsigned i = 0; i < N; i++) {
            B[j * N + i] = double(i * N + j) / N;
        }
    }
    
    
    for (unsigned i = 0; i < N * N; i++) {
        A[i] = double(i) / N;
    }
}


void multiply_blocked(const double* A_local, const double* B, double* C_local, 
                     int my_rows) {
    std::vector<double> B_block(BLOCK_SIZE * BLOCK_SIZE);
    
    
    std::fill_n(C_local, my_rows * N, 0.0);
    
    
    for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
        for (int kk = 0; kk < N; kk += BLOCK_SIZE) {
            
            for (int i = 0; i < my_rows; i++) {
                for (int j = jj; j < std::min(jj + BLOCK_SIZE, N); j++) {
                    double sum = C_local[i * N + j];
                    #pragma unroll(8)  
                    for (int k = kk; k < std::min(kk + BLOCK_SIZE, N); k++) {
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
    int my_rows = (rank < remainder) ? rows_per_proc + 1 : rows_per_proc;
    int my_offset = rank * rows_per_proc + std::min(rank, remainder);

    
    double* A_local = (double*)aligned_alloc(64, my_rows * N * sizeof(double));
    double* B = (double*)aligned_alloc(64, N * N * sizeof(double));
    double* C_local = (double*)aligned_alloc(64, my_rows * N * sizeof(double));
    
    double *A = nullptr, *C = nullptr;
    if (rank == 0) {
        init(A, B, C);
    }

    double start = MPI_Wtime();

    
    MPI_Request bcast_request;
    MPI_Ibcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD, &bcast_request);

    
    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; i++) {
        sendcounts[i] = (i < remainder ? rows_per_proc + 1 : rows_per_proc) * N;
        displs[i] = (i * rows_per_proc + std::min(i, remainder)) * N;
    }

    
    MPI_Scatterv(A, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 A_local, my_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    
    MPI_Wait(&bcast_request, MPI_STATUS_IGNORE);

    
    multiply_blocked(A_local, B, C_local, my_rows);

    
    MPI_Request gather_request;
    MPI_Igatherv(C_local, my_rows * N, MPI_DOUBLE,
                 C, sendcounts.data(), displs.data(), MPI_DOUBLE,
                 0, MPI_COMM_WORLD, &gather_request);

    
    MPI_Wait(&gather_request, MPI_STATUS_IGNORE);

    double end = MPI_Wtime();

    if (rank == 0) {
        std::cout << "=====================================================\n";
        std::cout << "Optimized Distributed Matrix Multiplication\n";
        std::cout << "Matrix Size N: " << N << "\n";
        std::cout << "Number of Processes: " << size << "\n";
        std::cout << "Block Size: " << BLOCK_SIZE << "\n";
        std::cout << "Execution time: " << end - start << "\n";

        
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum)
        for (unsigned i = 0; i < N * N; ++i) {
            sum += C[i];
        }
        std::cout << "Sum: " << sum << std::endl;
        std::cout << "=====================================================\n";

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
