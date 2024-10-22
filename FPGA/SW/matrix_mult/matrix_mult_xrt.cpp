#include <xrt/xrt.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <vector>
#include <iostream>
#include <cstdlib> // Para std::aligned_alloc y std::free

#define FIXED_SCALE (1 << 4)
using DataT = int16_t;

// Función para convertir de float a punto fijo (16 bits)
int16_t floatToFixed(float f) {
    return static_cast<int16_t>(f * FIXED_SCALE);
}

// Función para convertir de punto fijo a float
float fixedToFloat(int16_t fixedValue) {
    return static_cast<float>(fixedValue) / FIXED_SCALE;
}

int main(int argc, char** argv) {
    int M = 2; // Número de filas de la matriz A y C
    int N = 3; // Número de columnas de la matriz A y filas de la matriz B
    int P = 2; // Número de columnas de la matriz B y C

    // Alinear buffers para A, B y C
    size_t size_A = M * N * sizeof(DataT);
    size_t size_B = N * P * sizeof(DataT);
    size_t size_C = M * P * sizeof(DataT);

    // Crear memoria alineada
    DataT* A = static_cast<DataT*>(std::aligned_alloc(64, size_A));
    DataT* B = static_cast<DataT*>(std::aligned_alloc(64, size_B));
    DataT* C = static_cast<DataT*>(std::aligned_alloc(64, size_C));

    // Verificar si la memoria fue asignada correctamente
    if (!A || !B || !C) {
        std::cerr << "Error: no se pudo asignar memoria alineada." << std::endl;
        return EXIT_FAILURE;
    }

    // Inicializar matrices de entrada
    A[0] = floatToFixed(1.0f); A[1] = floatToFixed(2.0f); A[2] = floatToFixed(3.0f);
    A[3] = floatToFixed(4.0f); A[4] = floatToFixed(5.0f); A[5] = floatToFixed(6.0f); // A en punto fijo (2x3)

    B[0] = floatToFixed(7.0f); B[1] = floatToFixed(8.0f);
    B[2] = floatToFixed(9.0f); B[3] = floatToFixed(10.0f);
    B[4] = floatToFixed(11.0f); B[5] = floatToFixed(12.0f); // B en punto fijo (3x2)

    // Inicializar el dispositivo XRT
    xrt::device device(0);
    xrt::uuid xclbin_uuid = device.load_xclbin("matrix_mult.xclbin");
    xrt::kernel kernel(device, xclbin_uuid, "matrix_mult");

    // Crear buffers para A, B y C
    xrt::bo buffer_a(device, A, size_A, kernel.group_id(0));
    xrt::bo buffer_b(device, B, size_B, kernel.group_id(1));
    xrt::bo buffer_c(device, C, size_C, kernel.group_id(2));

    // Sincronizar los buffers de entrada (A y B) desde el host al dispositivo
    buffer_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Ejecutar el kernel
    auto run = kernel(buffer_a, buffer_b, buffer_c, M, N, P);
    run.wait(); // Esperar a que termine la ejecución del kernel

    // Sincronizar el buffer de salida desde el dispositivo al host
    buffer_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Mostrar los resultados (matriz C)
    std::cout << "Matriz C (resultado en punto fijo):" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            std::cout << fixedToFloat(C[i * P + j]) << " "; // Convertir a float para mostrar el resultado
        }
        std::cout << std::endl;
    }

    // Liberar la memoria alineada
    std::free(A);
    std::free(B);
    std::free(C);

    return 0;
}
