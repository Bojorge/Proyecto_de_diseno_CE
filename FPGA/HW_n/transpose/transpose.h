#ifndef __TRANSPOSE_H__
#define __TRANSPOSE_H__

#include <stdint.h>
#include <ap_int.h>
#include <hls_stream.h>

static constexpr int kBusWidth = 64;
static constexpr int kDataWidth = 16;
static constexpr int kPackets = kBusWidth / kDataWidth;

using RawDataT = ap_uint<kBusWidth>;
using StreamT = hls::stream<RawDataT>;

extern "C" {
void transpose(RawDataT *input, RawDataT *output, int rows, int cols);
}

#endif // __TRANSPOSE_H__
