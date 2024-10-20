#ifndef MATRIX_MULT_H
#define MATRIX_MULT_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <stdint.h>
#include <hls_stream.h>
using DataT = ap_fixed<16, 12>;


extern "C" {
    void matrix_mult(const DataT *A, const DataT *B, DataT *C, int M, int N, int P);
}

#endif // MATRIX_MULT_H


