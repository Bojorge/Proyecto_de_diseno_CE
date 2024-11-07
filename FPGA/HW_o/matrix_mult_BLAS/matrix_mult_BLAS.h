#ifndef MATRIX_MULT_H
#define MATRIX_MULT_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <stdint.h>
#include <hls_stream.h>

static constexpr int kBusWidth = 64;
static constexpr int kDataWidth = 16;
static constexpr int kDataInt = 6;
static constexpr int kPackets = kBusWidth / kDataWidth;
static constexpr int kShiftData = 2; 

using RawDataT = ap_uint<kBusWidth>;
using StreamT = hls::stream<RawDataT>;
using DataT = ap_fixed<kDataWidth, kDataInt>;

extern "C" {
    void matrix_mult_BLAS(const RawDataT *A, const RawDataT *B, RawDataT *C, int M, int N, int P);
}

#endif // MATRIX_MULT_H


