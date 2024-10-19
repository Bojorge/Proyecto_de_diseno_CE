



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

