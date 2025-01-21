#ifndef DIV_H
#define DIV_H

#include <hls_stream.h>
#include "ap_int.h"
#include "ap_fixed.h"
#include <stdint.h>

// Definiciones necesarias
using RawDataT = ap_uint<512>;  // Ancho de datos
using DataT = ap_fixed<16, 6>;
constexpr int kPackets = 16;    // Número de elementos en un paquete
constexpr int kDataWidth = 16;  // Ancho de cada dato en bits

// Función principal para la división
extern "C" void div(RawDataT *in1, RawDataT *in2, RawDataT *out, uint64_t size);

#endif // DIV_H
