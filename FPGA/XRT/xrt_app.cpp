#include <xrt/xrt.h> // Incluye la biblioteca XRT para gestionar dispositivos y buffers en FPGA.
#include <xrt/experimental/xrt_kernel.h> // Incluye las funcionalidades experimentales de XRT para manejar kernels.
#include <vector> // Incluye la biblioteca estándar de C++ para utilizar el contenedor vector.

int main(int argc, char** argv) {
    // Definición de las dimensiones de las matrices
    int M = 2; // Número de filas de la matriz A y C
    int N = 3; // Número de columnas de la matriz A y filas de la matriz B
    int P = 2; // Número de columnas de la matriz B y C

    // Inicialización de las matrices A y B con valores específicos
    std::vector<half> A = {1, 2, 3, 4, 5, 6}; // Matriz A de tamaño MxN (2x3)
    std::vector<half> B = {7, 8, 9, 10, 11, 12}; // Matriz B de tamaño NxP (3x2)
    
    // Vector para almacenar el resultado de la multiplicación de matrices
    std::vector<half> C(M * P); // Matriz C de tamaño MxP (2x2)

    // Cargar el bitstream en la FPGA
    // 'device(0)' crea un objeto dispositivo, seleccionando el dispositivo FPGA con ID 0.
    xrt::device device(0);
    
    // Carga el archivo binario del kernel (xclbin) en la FPGA y devuelve un UUID.
    xrt::uuid xclbin_uuid = device.load_xclbin("krnl_mult.xclbin");

    // Crear el kernel
    // Se crea un objeto kernel asociándolo al dispositivo y al bitstream cargado.
    // El nombre "matrix_mult" debe coincidir con el nombre del kernel en el archivo xclbin.
    xrt::kernel kernel = xrt::kernel(device, xclbin_uuid, "matrix_mult");

    // Crear buffers de entrada y salida
    // Se crean objetos buffer (bo) para almacenar las matrices A, B, y C en la memoria del dispositivo.
    // Los buffers se asignan a los grupos de argumentos correspondientes del kernel.

    // Buffer para la matriz A
    xrt::bo buffer_a = xrt::bo(device, A.data(), A.size() * sizeof(half), kernel.group_id(0));

    // Buffer para la matriz B
    xrt::bo buffer_b = xrt::bo(device, B.data(), B.size() * sizeof(half), kernel.group_id(1));

    // Buffer para la matriz C (resultado)
    xrt::bo buffer_c = xrt::bo(device, C.data(), C.size() * sizeof(half), kernel.group_id(2));

    // Ejecutar el kernel
    // Llama al kernel con los buffers y las dimensiones M, N, P como argumentos.
    auto run = kernel(buffer_a, buffer_b, buffer_c, M, N, P);
    
    // Esperar a que la ejecución del kernel finalice
    run.wait();

    // Leer los resultados desde el dispositivo
    // Sincroniza el buffer de salida C con la memoria del host para que se pueda leer el resultado.
    buffer_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Mostrar los resultados
    // Itera sobre la matriz resultante C y la imprime en formato de matriz.
    for (int i = 0; i < M; ++i) { // Itera sobre las filas de la matriz C
        for (int j = 0; j < P; ++j) { // Itera sobre las columnas de la matriz C
            std::cout << C[i * P + j] << " "; // Imprime el valor de cada elemento
        }
        std::cout << std::endl; // Salto de línea al final de cada fila
    }

    return 0; // Indica que el programa finalizó correctamente
}
