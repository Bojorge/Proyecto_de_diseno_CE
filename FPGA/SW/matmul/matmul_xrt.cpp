#include <xrt/xrt.h>
#include <vector>
#include <iostream>

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

    // Matrices de entrada
    std::vector<DataT> A = {floatToFixed(1.0f), floatToFixed(2.0f), floatToFixed(3.0f),
                            floatToFixed(4.0f), floatToFixed(5.0f), floatToFixed(6.0f)}; // A en punto fijo (2x3)

    std::vector<DataT> B = {floatToFixed(7.0f), floatToFixed(8.0f), floatToFixed(9.0f),
                            floatToFixed(10.0f), floatToFixed(11.0f), floatToFixed(12.0f)}; // B en punto fijo (3x2)

    
    // Matriz de salida
    std::vector<DataT> C(M * P); // Matriz C de tamaño MxP (2x2)

    // Inicializar el dispositivo XRT
    xrt::device device(0); 
    xrt::uuid xclbin_uuid = device.load_xclbin("matmul.xclbin");
    xrt::kernel kernel(device, xclbin_uuid, "matmul");

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
    std::cout << "Matriz C (resultado en punto fijo):" << std::endl;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            std::cout << fixedToFloat(C[i * P + j]) << " "; // Convertir a float para mostrar el resultado
        }
        std::cout << std::endl;
    }

    return 0; 
}
