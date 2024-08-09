#include <iostream>
#include <string>
#include "posix_sockets.h"

int main() {
    const int num_iterations = 1000;
    std::string message;

    // Variables para almacenar los máximos valores
    long maxRAMUsage = 0;
    double maxUserCPU = 0.0;
    double maxSystemCPU = 0.0;

    for (int i = 0; i < num_iterations; ++i) {
        message = "mensaje #" + std::to_string(i);
        std::cout << message << std::endl;
        send_message("127.0.0.1", "12345", message.c_str());

        long ramUsage = getRAMUsage();
        double userCPU, systemCPU;
        getCPUUsage(userCPU, systemCPU);

        // Actualizar los máximos valores
        if (ramUsage > maxRAMUsage) maxRAMUsage = ramUsage;
        if (userCPU > maxUserCPU) maxUserCPU = userCPU;
        if (systemCPU > maxSystemCPU) maxSystemCPU = systemCPU;
    }

    std::cout << "-------------------------------" << std::endl;
    std::cout << "RAM: " << maxRAMUsage << " KB" << std::endl;
    std::cout << "CPU usuario: " << maxUserCPU << " s" << std::endl;
    std::cout << "CPU sistema: " << maxSystemCPU << " s" << std::endl;

    return 0;
}
