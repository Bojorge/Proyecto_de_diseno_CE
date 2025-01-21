#include "transpose.h"

#ifdef USE_AXI_STREAM
static void transpose_accel(StreamT &input_stream, StreamT &output_stream, int rows, int cols) {
  RawDataT buffer[1024]; // Buffer temporal para almacenar datos
  
  // Cargar datos en el buffer
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < (cols >> 2); ++c) {
#pragma HLS pipeline
      buffer[c + r * (cols >> 2)] = input_stream.read();
    }
  }
  
  // Realizar la transposición escribiendo los datos en el stream de salida
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < (rows >> 2); ++r) {
#pragma HLS pipeline
      output_stream.write(buffer[r + c * (rows >> 2)]);
    }
  }
}
#else
static void transpose_accel(RawDataT *input, RawDataT *output, int rows, int cols) {
  RawDataT buffer[1024]; // Buffer temporal para almacenar datos
  
  // Cargar datos en el buffer
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < (cols >> 2); ++c) {
#pragma HLS pipeline
      buffer[c + r * (cols >> 2)] = input[r * (cols >> 2) + c];
    }
  }
  
  // Realizar la transposición escribiendo los datos en la matriz de salida
  for (int c = 0; c < cols; ++c) {
    for (int r = 0; r < (rows >> 2); ++r) {
#pragma HLS pipeline
      output[c * (rows >> 2) + r] = buffer[r + c * (rows >> 2)];
    }
  }
}
#endif

extern "C" {

/**
 * Transpose function for a 2D matrix represented in 1D format
 * input: pointer to the input matrix
 * output: pointer to the output matrix
 * rows: number of rows in the input matrix
 * cols: number of columns in the input matrix
 */
void transpose(RawDataT *input, RawDataT *output, int rows, int cols) {
#pragma HLS INTERFACE m_axi offset=slave port=input bundle=gmem0
#pragma HLS INTERFACE m_axi offset=slave port=output bundle=gmem1
#pragma HLS INTERFACE s_axilite register port=rows
#pragma HLS INTERFACE s_axilite register port=cols
#pragma HLS INTERFACE s_axilite register port=return

#ifdef USE_AXI_STREAM
  static StreamT input_stream, output_stream;
#pragma HLS stream variable=input_stream depth=128
#pragma HLS stream variable=output_stream depth=128
#pragma HLS dataflow
  
  // Cargar y transponer usando flujos
  load_data(input, input_stream, rows, cols);
  transpose_accel(input_stream, output_stream, rows, cols);
  store_data(output, output_stream, rows, cols);
#else
  transpose_accel(input, output, rows, cols);
#endif
}

}
