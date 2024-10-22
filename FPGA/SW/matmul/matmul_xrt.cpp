#include <xrt/xrt.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <iostream>
#include <vector>
#include <cstdlib> 

constexpr int kBusWidth = 64;
constexpr int kDataWidth = 16;
constexpr int kPackets = kBusWidth / kDataWidth;

using RawDataT = uint64_t; // ap_uint<kBusWidth> en HLS, equivalente a uint64_t en C++
using DataT = int16_t;

int main(int argc, char** argv) {
    int M = 2;  // Número de filas de la matriz A y C
    int N = 3;  // Número de columnas de la matriz A y filas de la matriz B
    int P = 2;  // Número de columnas de la matriz B y C

    // Tamaños de las matrices (ajustados para el ancho del bus de datos)
    int size_A = M * (N / kPackets) * sizeof(RawDataT);
    int size_B = N * (P / kPackets) * sizeof(RawDataT);
    int size_C = M * (P / kPackets) * sizeof(RawDataT);

    // Reservar memoria alineada para las matrices A, B y C
    RawDataT* A = static_cast<RawDataT*>(std::aligned_alloc(64, size_A));
    RawDataT* B = static_cast<RawDataT*>(std::aligned_alloc(64, size_B));
    RawDataT* C = static_cast<RawDataT*>(std::aligned_alloc(64, size_C));

    // Verificar si la memoria fue asignada correctamente
    if (!A || !B || !C) {
        std::cerr << "Error: no se pudo asignar memoria alineada." << std::endl;
        return EXIT_FAILURE;
    }

    // Inicializar matrices A y B con datos de prueba
    A[0] = 0x0002000100030002; // Datos de la matriz A en formato empaquetado (2x3)
    A[1] = 0x0006000500040003;
    
    B[0] = 0x00080007000A0009; // Datos de la matriz B en formato empaquetado (3x2)
    B[1] = 0x000C000B000E000D;

    // Inicializar el dispositivo XRT
    xrt::device device(0);
    xrt::uuid xclbin_uuid = device.load_xclbin("matmul.xclbin");
    xrt::kernel kernel(device, xclbin_uuid, "matmul");

    // Crear buffers en la FPGA para A, B y C
    xrt::bo buffer_a(device, A, size_A, kernel.group_id(0));
    xrt::bo buffer_b(device, B, size_B, kernel.group_id(1));
    xrt::bo buffer_c(device, C, size_C, kernel.group_id(2));

    // Sincronizar los buffers de entrada (A y B) desde el host al dispositivo
    buffer_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Ejecutar el kernel con los parámetros de tamaño
    auto run = kernel(buffer_a, buffer_b, buffer_c, M, N, P);
    run.wait(); // Esperar a que termine la ejecución del kernel

    // Sincronizar el buffer de salida (C) desde el dispositivo al host
    buffer_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Mostrar los resultados (matriz C)
    std::cout << "Matriz C (resultado en punto fijo):" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P / kPackets; ++j) {
            std::cout << std::hex << C[i * (P / kPackets) + j] << " "; // Mostrar los datos en formato hexadecimal
        }
        std::cout << std::endl;
    }

    // Liberar la memoria alineada
    std::free(A);
    std::free(B);
    std::free(C);

    return 0;
}
