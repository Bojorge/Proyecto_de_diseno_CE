#include <iostream>
#include <vector>
#include <xrt/xrt/xrt_kernel.h>
#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>
#include <ap_fixed.h>

using DataT = ap_fixed<16, 12>;

#define DATA_SIZE 256 // Tamaño de los datos

int main(int argc, char** argv) {
    const char* xclbinFilename = "matrix_mult.xclbin"; 

    // Abrir el dispositivo y cargar el archivo .xclbin
    xrt::device device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbinFilename);

    // Crear el kernel a partir de la definición en .xclbin
    xrt::kernel kernel = xrt::kernel(device, uuid, "matrix_mult");

    // Inicializar las matrices de entrada y salida
    int M = 16, N = 16, P = 16; // Ejemplo de tamaño de matrices
    std::vector<DataT> A(M * N, DataT(1.0)); // Rellena A con valores 1.0
    std::vector<DataT> B(N * P, DataT(1.0)); // Rellena B con valores 1.0
    std::vector<DataT> C(M * P, DataT(0.0)); // Inicializa C en cero

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

    std::cout << "Resultado de C[0]: " << C[0] << std::endl;
    std::cout << "Resultado de C[1]: " << C[1] << std::endl;
    std::cout << "Resultado de C[M*P-1]: " << C[M*P-1] << std::endl;

    return 0;
}
