#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>
#include <sys/resource.h>

const char* SHARED_MEMORY_NAME = "/MySharedMemory";
const size_t SHARED_MEMORY_SIZE = 65536;

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
    
    // Crear y abrir el segmento de memoria compartida
    int fd = shm_open(SHARED_MEMORY_NAME, O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        perror("shm_open");
        return -1;
    }

    // Establecer el tamaño del segmento de memoria compartida
    if (ftruncate(fd, SHARED_MEMORY_SIZE) == -1) {
        perror("ftruncate");
        close(fd);
        return -1;
    }

    // Mapear el segmento de memoria compartida en el espacio de direcciones del proceso
    void* ptr = mmap(NULL, SHARED_MEMORY_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return -1;
    }

    // Variables para almacenar los máximos valores
    long maxRAMUsage = 0;
    double maxUserCPU = 0.0;
    double maxSystemCPU = 0.0;

    for (int i = 0; i < 100; ++i) {
        std::string sharedData = "MESSAGE #" + std::to_string(i);
        std::cout << "WRITING <- " << sharedData << std::endl;
        
        memset(ptr, '\0', SHARED_MEMORY_SIZE); // Limpia la memoria compartida
        memcpy(ptr, sharedData.c_str(), sharedData.size()); // Copia los datos a la memoria compartida

        // Medir y actualizar los máximos valores
        long ramUsage = getRAMUsage();
        double userCPU, systemCPU;
        getCPUUsage(userCPU, systemCPU);

        if (ramUsage > maxRAMUsage) maxRAMUsage = ramUsage;
        if (userCPU > maxUserCPU) maxUserCPU = userCPU;
        if (systemCPU > maxSystemCPU) maxSystemCPU = systemCPU;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Desvincular y cerrar
    if (munmap(ptr, SHARED_MEMORY_SIZE) == -1) {
        perror("munmap");
    }
    close(fd);
    
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