#include <Poco/SharedMemory.h>
#include <Poco/Exception.h>
#include <Poco/File.h>
#include <Poco/Semaphore.h>
#include <iostream>
#include <cstring>
#include <string>
#include <chrono>
#include <thread>
#include <sys/resource.h>

const std::string SHARED_MEMORY_NAME = "MySharedMemory";
const std::size_t SHARED_MEMORY_SIZE = 65536; 
const std::size_t BLOCK_SIZE = 1024;



long getRAMUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // Devuelve el uso máximo de RAM en kilobytes
}

void getCPUUsage(double &userCPU, double &systemCPU) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    userCPU = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;  // tiempo de CPU en modo usuario en segundos
    systemCPU = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6; // tiempo de CPU en modo sistema en segundos
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    // Variables para almacenar los máximos valores
    long maxRAMUsage = 0;
    double maxUserCPU = 0.0;
    double maxSystemCPU = 0.0;

    try {
        Poco::SharedMemory sharedMemory(SHARED_MEMORY_NAME, SHARED_MEMORY_SIZE, Poco::SharedMemory::AM_WRITE, nullptr, true);

        for (int i = 0; i < 100; ++i) {
            std::string sharedData = "MESSAGE #" + std::to_string(i);
            std::cout << "WRITING <- " << sharedData << std::endl;

            // Limpiar el bloque de memoria compartida y copiar los datos
            std::memset(sharedMemory.begin(), '\0', BLOCK_SIZE);
            std::memcpy(sharedMemory.begin(), sharedData.data(), sharedData.size());

            // Medir y actualizar los máximos valores
            long ramUsage = getRAMUsage();
            double userCPU, systemCPU;
            getCPUUsage(userCPU, systemCPU);

            if (ramUsage > maxRAMUsage) maxRAMUsage = ramUsage;
            if (userCPU > maxUserCPU) maxUserCPU = userCPU;
            if (systemCPU > maxSystemCPU) maxSystemCPU = systemCPU;

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

    } catch (Poco::Exception& ex) {
        std::cerr << "Poco exception: " << ex.displayText() << std::endl;
        return -1;
    }
    // Registrar el tiempo de fin
    auto end = std::chrono::high_resolution_clock::now();
    // Calcular la duración
    std::chrono::duration<double> duration = end - start;
    std::cout << "El programa tardó " << duration.count() << " segundos en ejecutarse." << std::endl;

    std::cout << "-------------------------------" << std::endl;
    std::cout << "RAM: " << maxRAMUsage << " KB" << std::endl;
    std::cout << "CPU usuario: " << maxUserCPU << " s" << std::endl;
    std::cout << "CPU sistema: " << maxSystemCPU << " s" << std::endl;
    
    return 0;
}