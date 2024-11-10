#include <iostream>
#include <vector>
#include <cmath>
#include <xrt/xrt.h>
#include <xrt/xrt/xrt_kernel.h>
#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <cstdlib>

#define FIXED_SCALE (1 << 4) // Escala para el punto fijo
using DataT = int16_t;

// Función para convertir de float a punto fijo (16 bits)
int16_t floatToFixed(float f) {
    return static_cast<int16_t>(f * FIXED_SCALE);
}

// Función para convertir de punto fijo a float
float fixedToFloat(int16_t fixedValue) {
    return static_cast<float>(fixedValue) / FIXED_SCALE;
}

void printMatrix(const std::vector<DataT>& matrix, int rows, int cols, const std::string& name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << fixedToFloat(matrix[i * cols + j]) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    const char* xclbinFilename = "matrix_mult.xclbin";

    // Abrir el dispositivo y cargar el archivo .xclbin
    xrt::device device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbinFilename);

    // Crear el kernel a partir de la definición en .xclbin
    xrt::kernel kernel = xrt::kernel(device, uuid, "matrix_mult");

    // Inicializar las matrices de entrada y salida
    int M = 5, N = 5, P = 5;
    std::vector<DataT> A(M * N, floatToFixed(1.0f)); // Rellena A con valor fijo
    std::vector<DataT> B(N * P, floatToFixed(1.0f)); // Rellena B con valor fijo
    std::vector<DataT> C(M * P, floatToFixed(0.0f)); // Inicializa C en cero
    std::vector<DataT> C_expected(M * P, floatToFixed(0.0f)); // Resultado esperado en software

    // Imprimir matrices de entrada
    printMatrix(A, M, N, "A");
    printMatrix(B, N, P, "B");

    // Calcular el resultado esperado en software
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += fixedToFloat(A[i * N + k]) * fixedToFloat(B[k * P + j]);
            }
            C_expected[i * P + j] = floatToFixed(sum);
        }
    }

    // Crear buffers en la memoria global para las matrices
    xrt::bo buffer_A = xrt::bo(device, A.size() * sizeof(DataT), kernel.group_id(0));
    xrt::bo buffer_B = xrt::bo(device, B.size() * sizeof(DataT), kernel.group_id(1));
    xrt::bo buffer_C = xrt::bo(device, C.size() * sizeof(DataT), kernel.group_id(2));

    // Copiar datos de las matrices A y B a la memoria del dispositivo
    buffer_A.write(A.data());
    buffer_B.write(B.data());
    buffer_A.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    buffer_B.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Ejecutar el kernel
    auto run = kernel(buffer_A, buffer_B, buffer_C, M, N, P);
    run.wait();

    // Leer el resultado de la memoria del dispositivo a C
    buffer_C.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    buffer_C.read(C.data());
    for (auto& elem : C) {
        elem /= FIXED_SCALE;
    }

    // Mostrar los resultados obtenidos y compararlos con el esperado
    std::cout << "Matrix C (Expected):\n";
    printMatrix(C_expected, M, P, "C_expected");

    std::cout << "Matrix C (Hardware Result):\n";
    printMatrix(C, M, P, "C (Hardware)");

    // Verificación de resultados
    bool passed = true;
    for (int i = 0; i < M * P; ++i) {
        float expected = fixedToFloat(C_expected[i]);
        float obtained = fixedToFloat(C[i]);
        if (std::abs(expected - obtained) > 0.1f) {
            std::cout << "Mismatch at index " << i << ": Expected " << expected << ", Obtained " << obtained << std::endl;
            passed = false;
        }
    }

    if (passed) {
        std::cout << "TEST PASSED\n";
    } else {
        std::cout << "TEST FAILED\n";
    }

    return 0;
}
