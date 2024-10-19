#include "matrix_mult.h"

// Función del acelerador de multiplicación de matrices
// A, B y C son punteros a matrices en formato FP16 (half)
// M es el número de filas de la matriz A
// N es el número de columnas de la matriz A y filas de la matriz B
// P es el número de columnas de la matriz B
void matrix_mult_accel(const half *A, const half *B, half *C, int M, int N, int P) {
    for (int i = 0; i < M; i++) {
        #pragma HLS PIPELINE II=1
        
        for (int j = 0; j < P; j++) {
            half sum = 0;
            
            for (int k = 0; k < N; k++) {
                #pragma HLS UNROLL factor=4
                sum += A[i * N + k] * B[k * P + j];
            } 
            C[i * P + j] = sum;
        }
    }
}

/*
static void matrix_mult_accel(const DataT *A, const DataT *B, DataT *C, int M, int N, int P) {
    DataT temp_A[256];
    DataT temp_B[256];

    DataT result = 0;

    for (int i = 0; i < M; i++) {
        #pragma HLS PIPELINE
        for (int j = 0; j < P; j++) {
            result = 0;
            for (int k = 0; k < N; k++) {
                #pragma HLS unroll
                temp_A[k] = A[i * N + k];
                temp_B[k] = B[k * P + j];

                result += temp_A[k] * temp_B[k];
            }
            C[i * P + j] = result; 
        }
    }
}
*/

// Función principal de multiplicación de matrices
void matrix_mult(RawDataT *A, RawDataT *B, RawDataT *C, uint16_t M, uint16_t N, uint16_t P) {
    #pragma HLS INTERFACE m_axi offset=slave port=A bundle=gmem0
    #pragma HLS INTERFACE m_axi offset=slave port=B bundle=gmem1
    #pragma HLS INTERFACE m_axi offset=slave port=C bundle=gmem2
    #pragma HLS INTERFACE s_axilite register port=M
    #pragma HLS INTERFACE s_axilite register port=N
    #pragma HLS INTERFACE s_axilite register port=P
    #pragma HLS INTERFACE s_axilite register port=return

    matrix_mult_accel(A, B, C, M, N, P);
}

