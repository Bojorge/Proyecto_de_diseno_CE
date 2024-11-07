#include "matrix_mult_BLAS.h"
#include "xf_blas/gemm.hpp"
#include "xf_blas/handle.hpp" 

void matrix_mult_BLAS(RawDataT *A, RawDataT *B, RawDataT *C, int M, int N, int P) {
    #pragma HLS INTERFACE m_axi offset=slave port=A bundle=gmem0
    #pragma HLS INTERFACE m_axi offset=slave port=B bundle=gmem1
    #pragma HLS INTERFACE m_axi offset=slave port=C bundle=gmem2
    #pragma HLS INTERFACE s_axilite register port=M
    #pragma HLS INTERFACE s_axilite register port=N
    #pragma HLS INTERFACE s_axilite register port=P
    #pragma HLS INTERFACE s_axilite register port=return

    // Initialize BLAS handle
    xf::blas::Handle<float> blas_handle;  // Adjust to appropriate type

    // Define matrices as xf_blas matrices
    xf::blas::Matrix<float> matA(M, N, A);  // Adjust to specific types as needed
    xf::blas::Matrix<float> matB(N, P, B);
    xf::blas::Matrix<float> matC(M, P, C);

    // Perform matrix multiplication with BLAS
    xf::blas::gemm(blas_handle, matA, matB, matC);
}
