#ifndef MATRIX_MULT_H
#define MATRIX_MULT_H

#include <hls_half.h>

#ifdef __cplusplus
extern "C" {
#endif

void matrix_mult(const half *A, const half *B, half *C, int M, int N, int P);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_MULT_H
