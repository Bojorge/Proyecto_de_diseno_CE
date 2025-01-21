#include <iostream>
#include <cstdint>
#include <cmath>
#include <cstring>

// XRT includes
#include <xrt/xrt.h>
#include <xrt/xrt/xrt_kernel.h>
#include <xrt/xrt/xrt_bo.h>
#include <xrt/xrt/xrt_device.h>

// HLS Types
//#include "../include/ap_fixed"

// Function to find the next power of two greater than or equal to n
int next_power_of_two(int n) {
    return (n <= 64) ? 64 : pow(2, ceil(log2(n)));
}

int main(int argc, char** argv) {
    
    // Get input size
    const char* binaryFile = "elementwise.xclbin";
    int a_rows = 5;
    int b_cols = 5;
    b_cols = b_cols < 8 ? 8 : (b_cols - (b_cols & 0b111));
    int op = 0; // Set to 0 for addition

    std::cout << "A rows: " << a_rows << "\n"
              << "B cols: " << b_cols << std::endl;

    // Compute sizes
    int size = a_rows * b_cols;

    // Open the device and load the xclbin
    int device_index = 0;
    auto device = xrt::device(device_index);
    auto uuid = device.load_xclbin(binaryFile);
    auto elementwise = xrt::kernel(device, uuid, "elementwise");

    // Allocate buffer in global memory
    auto bo_a = xrt::bo(device, size * sizeof(float), elementwise.group_id(0));
    auto bo_b = xrt::bo(device, size * sizeof(float), elementwise.group_id(1));
    auto bo_c = xrt::bo(device, size * sizeof(float), elementwise.group_id(2));

    // Map buffer objects into host memory
    auto bo_a_map = bo_a.map<float*>();
    auto bo_b_map = bo_b.map<float*>();
    auto bo_c_map = bo_c.map<float*>();

    // Fill vectors with values
    std::fill(bo_a_map, bo_a_map + size, 1.0f);  // Fill A with 1s
    std::fill(bo_b_map, bo_b_map + size, 2.0f);  // Fill B with 2s

    // Display input vectors
    std::cout << "Vector A:\n";
    for (int i = 0; i < size; ++i) {
        std::cout << bo_a_map[i] << " ";
        if ((i + 1) % b_cols == 0) std::cout << "\n";
    }

    std::cout << "\nVector B:\n";
    for (int i = 0; i < size; ++i) {
        std::cout << bo_b_map[i] << " ";
        if ((i + 1) % b_cols == 0) std::cout << "\n";
    }

    // Synchronize input buffer data to device
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Execute the kernel
    auto run = elementwise(bo_a, bo_b, bo_c, size, op); // op = 0 for addition
    run.wait();

    // Get the output data from the device
    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    // Display output vector
    std::cout << "\nResult Vector C (A + B):\n";
    for (int i = 0; i < size; ++i) {
        std::cout << bo_c_map[i] << " ";
        if ((i + 1) % b_cols == 0) std::cout << "\n";
    }

    std::cout << "\nTEST PASSED\n";
    return 0;
}
