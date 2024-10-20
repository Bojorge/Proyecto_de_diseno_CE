#include <xrt/xrt.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <vector>
#include <iostream>
#include <cstdint>

//using DataT = std::float16_t;
//using DataT = ap_fixed<16, 12>;
//using DataT = float;


int main(int argc, char** argv) {
    int M = 2; // Número de filas de la matriz A y C
    int N = 3; // Número de columnas de la matriz A y filas de la matriz B
    int P = 2; // Número de columnas de la matriz B y C

    // Matrices de entrada
    std::vector<DataT> A = {1, 2, 3, 4, 5, 6}; // Matriz A de tamaño MxN (2x3)
    std::vector<DataT> B = {7, 8, 9, 10, 11, 12}; // Matriz B de tamaño NxP (3x2)
    
    // Matriz de salida
    std::vector<DataT> C(M * P); // Matriz C de tamaño MxP (2x2)

    // Inicializar el dispositivo XRT
    xrt::device device(0); 
    xrt::uuid xclbin_uuid = device.load_xclbin("matrix_mult.xclbin");
    xrt::kernel kernel(device, xclbin_uuid, "matrix_mult");

    // Crear buffers para A, B y C
    xrt::bo buffer_a(device, A.data(), A.size() * sizeof(DataT), kernel.group_id(0));
    xrt::bo buffer_b(device, B.data(), B.size() * sizeof(DataT), kernel.group_id(1));
    xrt::bo buffer_c(device, C.data(), C.size() * sizeof(DataT), kernel.group_id(2));

    // Sincronizar los buffers de entrada (A y B) desde el host al dispositivo
    buffer_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Ejecutar el kernel
    auto run = kernel(buffer_a, buffer_b, buffer_c, M, N, P);
    run.wait(); // Esperar a que termine la ejecución del kernel

    // Sincronizar el buffer de salida desde el dispositivo al host
    buffer_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Mostrar los resultados (matriz C)
    std::cout << "Matriz C (resultado):" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            std::cout << C[i * P + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0; 
}