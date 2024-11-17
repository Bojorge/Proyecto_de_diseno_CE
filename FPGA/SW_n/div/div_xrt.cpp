#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <xrt/xrt.h>
#include <xrt/xrt/xrt_kernel.h>
#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>

#define FIXED_SCALE (1 << 10) // Escala para el punto fijo
using DataT = int16_t;

// Función para convertir de float a punto fijo (16 bits)
int16_t floatToFixed(float f) {
    return static_cast<int16_t>(f * FIXED_SCALE);
}

// Función para convertir de punto fijo a float
float fixedToFloat(int16_t fixedValue) {
    return static_cast<float>(fixedValue) / FIXED_SCALE;
}

void print_data(const std::vector<DataT>& in1, int rows, int cols, const std::string& name) {
    std::cout << "Matrix " << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << fixedToFloat(in1[i * cols + j])*FIXED_SCALE << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    try {
        const char* xclbinFilename = "div.xclbin";

        // Dimensiones de las matrices
        const int rows = 5;
        const int cols = 5;
        const int size = rows * cols;
        const size_t data_size = size * sizeof(uint16_t);

        // Inicializa dispositivo y kernel
        xrt::device device = xrt::device(0);
        auto uuid = device.load_xclbin(xclbinFilename);
        
        // Crear el kernel a partir de la definición en .xclbin
        xrt::kernel kernel = xrt::kernel(device, uuid, "div");


        // Buffers para datos de entrada y salida
        xrt::bo in1_bo = xrt::bo(device, data_size, kernel.group_id(0));
        xrt::bo in2_bo = xrt::bo(device, data_size, kernel.group_id(1));
        xrt::bo out_bo = xrt::bo(device, data_size, kernel.group_id(2));

        // Llenar datos de prueba
        std::vector<DataT> in1(size, 20); // Entrada 1
        std::vector<DataT> in2(size, 5); // Entrada 2
        std::vector<DataT> out(size, 0); // Inicializar salida

        in1_bo.write(in1.data());
        in2_bo.write(in2.data());

        // Sincroniza buffers de entrada al dispositivo
        in1_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        in2_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Ejecutar kernel
        auto run = kernel(in1_bo, in2_bo, out_bo, size);
        run.wait();

        // Sincroniza el buffer de salida de vuelta al host
        out_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        out_bo.read(out.data());
        
        for (auto& elem : out) {
            elem /= (FIXED_SCALE);
        }
        

        // Imprimir matrices
        print_data(in1, rows, cols, "Input 1");
        print_data(in2, rows, cols, "Input 2");
        print_data(out, rows, cols, "Output (Hardware Result)");
        

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
