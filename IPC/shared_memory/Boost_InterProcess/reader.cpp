#include <boost/interprocess/managed_shared_memory.hpp>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <stdio.h>
#include <chrono>
#include <thread>
#include <sys/resource.h>

using namespace boost::interprocess;

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

int main (int argc, char *argv[])
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto start = std::chrono::high_resolution_clock::now();
    
    char buffer[1024];
    
    memset(buffer, '\0', sizeof(buffer));
    
    // Variables para almacenar los máximos valores
    long maxRAMUsage = 0;
    double maxUserCPU = 0.0;
    double maxSystemCPU = 0.0;

    managed_shared_memory segment (open_only, "MySharedMemory");
    managed_shared_memory::handle_t handle = 240; //hardcodeado, esto se debe pasar como parametro

    //Get buffer local address from handle
    void *msg = segment.get_address_from_handle(handle);

    for(int i=0;i<100;i++){
        std::cout << "READING -> ";
        std::cout << (char*)msg << std::endl;

        // Medir y actualizar los máximos valores
        long ramUsage = getRAMUsage();
        double userCPU, systemCPU;
        getCPUUsage(userCPU, systemCPU);

        if (ramUsage > maxRAMUsage) maxRAMUsage = ramUsage;
        if (userCPU > maxUserCPU) maxUserCPU = userCPU;
        if (systemCPU > maxSystemCPU) maxSystemCPU = systemCPU;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
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