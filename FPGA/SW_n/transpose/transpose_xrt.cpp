#include <iostream>
#include <vector>
#include <xrt/xrt.h>
#include <xrt/xrt/xrt_kernel.h>
#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>

constexpr int kBusWidth = 64;
constexpr int kDataWidth = 16;
constexpr int kPackets = kBusWidth / kDataWidth;

using RawDataT = uint64_t; // Para mapear ap_uint<kBusWidth> en el host

// Función para imprimir matrices
void print_matrix(const std::vector<RawDataT> &matrix, int rows, int cols, const std::string &name) {
    std::cout << "Matrix " << name << ":\n";
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::cout << matrix[r * cols + c] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Función para realizar la transposición en el host (para validar)
void transpose_cpu(const std::vector<RawDataT> &input, std::vector<RawDataT> &output, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            output[c * rows + r] = input[r * cols + c];
        }
    }
}

int main(int argc, char **argv) {
    try {
        // Nombre del archivo xclbin
        const char *xclbinFilename = "transpose.xclbin";

        // Dimensiones de la matriz
        const int rows = 4;
        const int cols = 4;
        const size_t matrix_size = rows * cols;
        const size_t data_size = matrix_size * sizeof(RawDataT);

        // Inicializar dispositivo y cargar xclbin
        xrt::device device = xrt::device(0);
        auto uuid = device.load_xclbin(xclbinFilename);

        // Crear el kernel
        xrt::kernel kernel = xrt::kernel(device, uuid, "transpose");

        // Crear buffers de entrada y salida
        xrt::bo input_bo = xrt::bo(device, data_size, kernel.group_id(0));
        xrt::bo output_bo = xrt::bo(device, data_size, kernel.group_id(1));

        // Crear y llenar datos de entrada
        std::vector<RawDataT> input(matrix_size);
        std::vector<RawDataT> output(matrix_size, 0);
        std::vector<RawDataT> expected_output(matrix_size, 0);

        for (size_t i = 0; i < matrix_size; ++i) {
            input[i] = static_cast<RawDataT>(i + 1); // Llenar con valores secuenciales
        }

        // Cargar datos de entrada en el buffer
        input_bo.write(input.data());
        input_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Ejecutar kernel
        auto run = kernel(input_bo, output_bo, rows, cols);
        run.wait();

        // Leer los datos de salida
        output_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        output_bo.read(output.data());

        // Validar resultados en el host
        transpose_cpu(input, expected_output, rows, cols);

        // Imprimir matrices
        print_matrix(input, rows, cols, "Input");
        print_matrix(output, cols, rows, "Output (Hardware Result)");
        print_matrix(expected_output, cols, rows, "Expected Output");

        // Comparar salida de hardware con resultado esperado
        bool passed = true;
        for (size_t i = 0; i < matrix_size; ++i) {
            if (output[i] != expected_output[i]) {
                std::cerr << "Mismatch at index " << i << ": Expected " << expected_output[i]
                          << ", Obtained " << output[i] << "\n";
                passed = false;
            }
        }

        if (passed) {
            std::cout << "TEST PASSED\n";
        } else {
            std::cout << "TEST FAILED\n";
        }

    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
