#ifndef MATRIX_MULT_H
#define MATRIX_MULT_H

#include <hls_half.h>
#include <stdint.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>

static constexpr uint16_t  kBusWidth = 64;
static constexpr uint16_t  kDataWidth = 16;
static constexpr uint16_t  kPackets = kBusWidth / kDataWidth;
static constexpr uint16_t  kShiftData = 2;

using RawDataT = ap_uint<kBusWidth>;
using DataT = half;

extern "C" {
void matrix_mult(RawDataT *a, RawDataT *b, RawDataT *c, uint16_t  a_rows, uint16_t  b_cols, uint16_t  c_cols);
}

#endif // MATRIX_MULT_H