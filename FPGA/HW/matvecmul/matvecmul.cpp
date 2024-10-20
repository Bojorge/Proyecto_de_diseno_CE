/*
 * Copyright 2022-2024
 * Author: Luis G. Leon-Vega <luis.leon@ieee.org>
 */

#include "matvecmul.h"

#include "hls_math.h"

static constexpr int kPacketsPerReplica = kPackets / kReplicas;
static constexpr int kARowsPacketised = kARows / kPackets;
static constexpr int kARowsReplicated = kARows / kReplicas;
static constexpr int kBColsPacketised = kBCols / kPackets;
static constexpr int kASize = kARows * kBCols * sizeof(DataT) * kPackets;
static constexpr int kBSize = kBCols * sizeof(DataT) * kPackets;
static constexpr int kCSize = kARows * sizeof(DataT) * kPackets;

template <int NT, int N>
struct AddPairs {
  static DataT Execute(DataT vec[NT], int idx = 0) {
#pragma HLS inline off
    DataT lhs = AddPairs<NT, N / 2>::Execute(vec, idx);
    DataT rhs = AddPairs<NT, N / 2>::Execute(vec, N / 2 + idx);
    return lhs + rhs;
  }
};

template <int NT>
struct AddPairs<NT, 1> {
  static DataT Execute(DataT vec[NT], int idx) {
#pragma HLS inline off
    return vec[idx];
  }
};

static void matvecmul_gemm_stream(StreamT &a, StreamT &b, StreamSingleT &c,
                                  const int b_cols) {
#pragma HLS INLINE off
#pragma HLS PIPELINE

  DataT res[kPackets];
#pragma HLS ARRAY_PARTITION dim = 0 type = complete variable = res

  AccT tres{0};
  for (int p = 0; p < kPackets; ++p) {
#pragma HLS unroll
    res[p] = DataT{0};
  }

gemv_reduce:
  for (int b_col = 0; b_col < b_cols; b_col += kPackets) {  // k
#pragma HLS LOOP_TRIPCOUNT min = kBColsPacketised max = kBColsPacketised avg = \
    kBColsPacketised
#pragma HLS PIPELINE
    RawDataT a_packet = a.read();
    RawDataT b_packet = b.read();

  gemv_reduce_packet:
    for (int p = 0; p < kPackets; ++p) {
#pragma HLS LOOP_TRIPCOUNT min = kPackets max = kPackets avg = kPackets
#pragma HLS UNROLL
      const int low = p * kDataWidth;
      const int high = low + kDataWidth - 1;
      AccT a_val, b_val;
      GET_RAW(a_val) = a_packet(high, low);
      GET_RAW(b_val) = b_packet(high, low);
      res[p] += GET_NUMBER(a_val) * GET_NUMBER(b_val);
    }
  }

  GET_NUMBER(tres) = AddPairs<kPackets, kPackets>::Execute(res);
  c.write(GET_RAW(tres));
}

static void matvecmul_gemm(StreamT &a, StreamT &b,
                           StreamSingleT &c, const int a_rows,
                           const int b_cols) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable = a type = complete dim = 0
#pragma HLS ARRAY_PARTITION variable = b type = complete dim = 0
#pragma HLS ARRAY_PARTITION variable = c type = complete dim = 0

  // TODO: the rows of C are grouped given that stream C is parallel
  // The number of iterations is proportional to the parallelism
gemv_c_rows:
  for (int c_row = 0; c_row < a_rows; c_row += kReplicas) {  // m
#pragma HLS LOOP_TRIPCOUNT min = kARowsReplicated max = kARowsReplicated avg = \
    kARowsReplicated
#pragma HLS PIPELINE
    // TODO: the streams take place here
  gemv_c_streams:
    for (int s = 0; s < kReplicas; ++s) {
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT min = kReplicas max = kReplicas avg = kReplicas
      matvecmul_gemm_stream(a, b, c, b_cols);
    }
  }
}

static void matvecmul_to_stream_a(RawDataT *a, StreamT &sa,
                                  const int rows, const int cols) {
#pragma HLS INLINE off
  const int tcols = cols / kPackets;

  // Repeated matrix transmission: The rows are interleaved in streams (or
  // replicas)
a_rows:
  for (int row = 0; row < rows; row += kReplicas) {
#pragma HLS LOOP_TRIPCOUNT min = kARowsReplicated max = kARowsReplicated avg = \
    kARowsReplicated
    // Transmit columns
  a_cols:
    for (int col = 0; col < tcols; ++col) {
#pragma HLS LOOP_TRIPCOUNT min = kBColsPacketised max = kBColsPacketised avg = \
    kBColsPacketised
#pragma HLS LOOP_FLATTEN
#pragma HLS PIPELINE
      // Interleaved row access -> probably not the clevest but solvable
    a_streams:
      for (int s = 0; s < kReplicas; ++s) {
#pragma HLS LOOP_TRIPCOUNT min = kReplicas max = kReplicas avg = kReplicas
#pragma HLS UNROLL
        const int row_shift = (row + s) * tcols;
        const int cols_shift = col;
        const int shift = cols_shift + row_shift;
        RawDataT packet = a[shift];
        sa.write(packet);
      }
    }
  }
}

static void matvecmul_to_stream_b(RawDataT *a, StreamT &sa,
                                  const int cols, const int rep_rows) {
#pragma HLS INLINE off
  const int tcols = cols / kPackets;

  // Repeated row transmission: This is determined by the number of rows of C.
  // We need to transmit the column multiple times per stream. Even more when
  // We have fewer streams
b_mat_reps:
  for (int rep_row = 0; rep_row < rep_rows; rep_row += kReplicas) {
#pragma HLS LOOP_TRIPCOUNT min = kARowsReplicated max = kARowsReplicated avg = \
    kARowsReplicated
#pragma HLS PIPELINE
    // Transmit columns: Transposed columns are in packets in this case!
  b_mat_cols:
    for (int col = 0; col < tcols; ++col) {
#pragma HLS LOOP_TRIPCOUNT min = kBColsPacketised max = kBColsPacketised avg = \
    kBColsPacketised
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE
      const int row_shift = 0;
      const int cols_shift = col;
      const int shift = cols_shift + row_shift;
      RawDataT packet = a[shift];
    b_mat_streams:
      for (int s = 0; s < kReplicas; ++s) {
#pragma HLS UNROLL
#pragma HLS LOOP_TRIPCOUNT min = kReplicas max = kReplicas avg = kReplicas
        sa.write(packet);
      }
    }
  }
}

static void matvecmul_from_stream(RawDataT *a, StreamSingleT &sa,
                                  const int rows) {
#pragma HLS INLINE off
  const int trows = rows / kPackets;
c_rows:
  for (int i = 0; i < trows; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = kARowsPacketised max = kARowsPacketised avg = \
    kARowsPacketised
    RawDataT packet;
  c_packets:
    for (int p = 0; p < kPackets; p += kReplicas) {
#pragma HLS LOOP_TRIPCOUNT min = kPacketsPerReplica max = \
    kPacketsPerReplica avg = kPacketsPerReplica
#pragma HLS PIPELINE
    c_streams:
      // Compute the offset
      // The data is assumed to come interleaved
      const int olow = p * kDataWidth;
      for (int s = 0; s < kReplicas; ++s) {
#pragma HLS LOOP_TRIPCOUNT min = kReplicas max = kReplicas avg = kReplicas
#pragma HLS UNROLL
        const int low = olow + s * kDataWidth;
        const int high = low + kDataWidth - 1;
        packet(high, low) = sa.read();
      }
      a[i] = packet;
    }
  }
}

extern "C" {

/**
 * matrix: (rows, cols)
 * a: input (samples, inputs)
 * b: weights (outputs, inputs) assumed transposed [1, inputs]
 * c: output (samples, outputs). Assumed [samples, 1]
 */
void matvecmul(RawDataT *a, RawDataT *b, RawDataT *c, int a_rows, int b_cols,
               int c_cols) {
#pragma HLS INTERFACE m_axi offset = slave port = a depth = kASize bundle = \
    gmem0
#pragma HLS INTERFACE m_axi offset = slave port = b depth = kBSize bundle = \
    gmem1
#pragma HLS INTERFACE m_axi offset = slave port = c depth = kCSize bundle = \
    gmem2
#pragma HLS INTERFACE s_axilite register port = a_rows
#pragma HLS INTERFACE s_axilite register port = b_cols
#pragma HLS INTERFACE s_axilite register port = c_cols
#pragma HLS INTERFACE s_axilite register port = return

  // TODO: Make this dynamic through the directive file. Here, we assume two
  // rows at a time
  // TODO: A stream is in charge of a row, whereas B stream is redundant
  static StreamT stream_a;
#pragma HLS ARRAY_PARTITION dim = 0 type = complete variable = stream_a
#pragma HLS STREAM variable=stream_a depth=16 type=fifo

  static StreamT stream_b;
#pragma HLS ARRAY_PARTITION dim = 0 type = complete variable = stream_b
#pragma HLS STREAM variable=stream_b depth=16 type=fifo

  // TODO: Make this dynamic through the directive file. Here we assume FLOAT32
  static StreamSingleT stream_out;
#pragma HLS ARRAY_PARTITION dim = 0 type = complete variable = stream_out
#pragma HLS STREAM variable=stream_out depth=16 type=fifo

#pragma HLS dataflow
  matvecmul_to_stream_a(a, stream_a, a_rows, b_cols);
  matvecmul_to_stream_b(b, stream_b, b_cols, a_rows);

  matvecmul_gemm(stream_a, stream_b, stream_out, a_rows, b_cols);

  matvecmul_from_stream(c, stream_out, a_rows);
}
}
