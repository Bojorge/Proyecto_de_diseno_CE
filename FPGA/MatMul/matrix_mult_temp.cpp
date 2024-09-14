#include "matrix_mult.h"

// Función acelerada para la multiplicación de matrices
static void matrix_mult_accel(RawDataT *a, RawDataT *b, RawDataT *c, uint16_t a_rows, uint16_t b_cols, uint16_t c_cols) {
    uint16_t b_cols_shift = b_cols >> kShiftData;
    uint16_t c_cols_shift = c_cols >> kShiftData;

    matrix_mult_samples:
    for (uint16_t ay = 0; ay < a_rows; ++ay) {

        matrix_mult_layers:
        RawDataT valpacket = 0;
        for (uint16_t cx = 0; cx < c_cols; ++cx) {
            #pragma HLS pipeline
            DataT val = 0.f;

            matrix_mult_perceptron:
            for (uint16_t bx = 0; bx < b_cols_shift; ++bx) {
                RawDataT a_raw = a[ay * b_cols_shift + bx];
                RawDataT b_raw = b[cx * b_cols_shift + bx];
                for (uint16_t p = 0; p < kPackets; ++p) {
                    #pragma HLS unroll
                    uint16_t poff_low = p * kDataWidth;
                    uint16_t poff_high = poff_low + kDataWidth - 1;
                    
                    DataT a, b;

                    a.V = a_raw(poff_high, poff_low);
                    b.V = b_raw(poff_high, poff_low);

                    val += a * b;
                }
            }

            // Obtener los índices
            uint16_t cx_mod = cx & (kPackets - 1);
            uint16_t cx_div = cx >> kShiftData;
            uint16_t val_mod = (cx + 1) & (kPackets - 1);

            // Escribir según sea necesario
            uint16_t poff_low = cx_mod * kDataWidth;
            uint16_t poff_high = poff_low + kDataWidth - 1;

            valpacket(poff_high, poff_low) = val.V;

            // Transmitir si se ha completado
            if (val_mod == 0) {
                c[cx_div + ay * c_cols_shift] = valpacket;
                valpacket = 0;
            }
        }
    }
}

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
