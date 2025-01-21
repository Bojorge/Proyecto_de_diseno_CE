#include "div.h"

// Función para cargar datos de entrada en el flujo 'inStream'
static void load_input(RawDataT *in, hls::stream<RawDataT> &inStream, uint64_t size) {
  // Calcula la cantidad de paquetes de datos
  const uint64_t size_raw = size / kPackets;
  // Itera sobre cada paquete y lo envía al flujo
  for (uint64_t i = 0; i < size_raw; ++i) {
#pragma HLS PIPELINE  // Aplica la directiva HLS PIPELINE para mejorar el rendimiento
    inStream << in[i]; // Encola cada dato en el flujo 'inStream'
  }
}

// Función para realizar la división de los datos de entrada
static void compute_div(hls::stream<RawDataT> &in1_stream, hls::stream<RawDataT> &in2_stream, hls::stream<RawDataT> &out_stream, uint64_t size) {
  // Itera en bloques de tamaño 'kPackets'
  for (uint64_t i = 0; i < size; i += kPackets) {
#pragma HLS PIPELINE  // Optimiza el rendimiento mediante pipelining
    RawDataT raw_in1 = in1_stream.read(); // Lee un paquete de datos del flujo de entrada 1
    RawDataT raw_in2 = in2_stream.read(); // Lee un paquete de datos del flujo de entrada 2
    RawDataT raw_out = 0; // Inicializa el paquete de salida

    // Itera a través de cada sub-dato en el paquete
    for (int p = 0; p < kPackets; ++p) {
#pragma HLS UNROLL  // Desenrolla el bucle para mejorar el rendimiento
      int poff_low = p * kDataWidth;       // Posición inicial del sub-dato en el paquete
      int poff_high = poff_low + kDataWidth - 1; // Posición final del sub-dato

      // Extrae cada sub-dato en bits del paquete de entrada
      ap_uint<kDataWidth> in1_bits = raw_in1(poff_high, poff_low);
      ap_uint<kDataWidth> in2_bits = raw_in2(poff_high, poff_low);

      // Convierte el valor de in1_bits e in2_bits a tipo DataT para la operación de división
      DataT in1 = static_cast<DataT>(in1_bits.to_int());
      DataT in2 = static_cast<DataT>(in2_bits.to_int());

      // Realiza la división y maneja el caso en que el divisor sea 0
      DataT out = (in2 != 0) ? static_cast<DataT>(in1 / in2) : static_cast<DataT>(0);

      // Convierte el resultado de la división a bits y lo almacena en el paquete de salida
      ap_uint<kDataWidth> out_bits = *reinterpret_cast<ap_uint<kDataWidth>*>(&out);
      raw_out(poff_high, poff_low) = out_bits;
    }
    out_stream << raw_out; // Envía el paquete procesado al flujo de salida
  }
}

// Función para almacenar el resultado final desde el flujo de salida en el arreglo 'out'
static void store_result(RawDataT *out, hls::stream<RawDataT> &out_stream, uint64_t size) {
  // Calcula la cantidad de paquetes de datos
  const uint64_t size_raw = size / kPackets;
  // Lee cada dato del flujo de salida y lo almacena en el arreglo
  for (uint64_t i = 0; i < size_raw; ++i) {
#pragma HLS PIPELINE  // Aplica la directiva PIPELINE para optimización
    out[i] = out_stream.read();
  }
}

extern "C" {

// Función principal del kernel que realiza la operación de división
void div(RawDataT *in1, RawDataT *in2, RawDataT *out, uint64_t size) {
#pragma HLS INTERFACE m_axi port=in1 bundle=gmem0 // Interfaz para la entrada 'in1' usando AXI
#pragma HLS INTERFACE m_axi port=in2 bundle=gmem1 // Interfaz para la entrada 'in2' usando AXI
#pragma HLS INTERFACE m_axi port=out bundle=gmem2 // Interfaz para la salida 'out' usando AXI
#pragma HLS INTERFACE s_axilite register port=size // Interfaz AXI-Lite para el parámetro 'size'
#pragma HLS INTERFACE s_axilite register port=return // Interfaz AXI-Lite para retornar

  // Declaración de flujos de datos para las entradas y la salida
  hls::stream<RawDataT> in1_stream("in1_stream");
  hls::stream<RawDataT> in2_stream("in2_stream");
  hls::stream<RawDataT> out_stream("out_stream");
#pragma HLS stream variable=in1_stream depth=32 // Define la profundidad del flujo
#pragma HLS stream variable=in2_stream depth=32
#pragma HLS stream variable=out_stream depth=32

#pragma HLS dataflow // Activa el paralelismo en el diseño de HLS
  load_input(in1, in1_stream, size);        // Carga los datos de 'in1' en el flujo 'in1_stream'
  load_input(in2, in2_stream, size);        // Carga los datos de 'in2' en el flujo 'in2_stream'
  compute_div(in1_stream, in2_stream, out_stream, size); // Realiza la operación de división
  store_result(out, out_stream, size);      // Almacena el resultado en el arreglo 'out'
}

}
