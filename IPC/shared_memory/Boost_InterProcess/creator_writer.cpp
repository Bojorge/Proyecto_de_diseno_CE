#include <boost/interprocess/managed_shared_memory.hpp>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <chrono>
#include <thread> 
#include <sys/resource.h>

#define SharedMemory "MySharedMemory"

using namespace boost::interprocess;

long getRAMUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // Devuelve el uso máximo de RAM en kilobytes
}

void getCPUUsage(double &userCPU, double &systemCPU) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    // Parte entera del tiempo en segundos  +  parte fraccionaria del tiempo en microsegundos, convertida a segundos
    userCPU = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;  // tiempo de CPU en modo usuario en segundos
    systemCPU = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6; // tiempo de CPU en modo sistema en segundos
}

int main (int argc, char *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // Buffer para almacenar datos a compartir
    char buffer[1024];
    
    // Inicializa los buffers con ceros (caracteres nulos)
    memset(buffer, '\0', sizeof(buffer));
    
    // Variables para almacenar los máximos valores
    long maxRAMUsage = 0;
    double maxUserCPU = 0.0;
    double maxSystemCPU = 0.0;

    // Estructura para eliminar la memoria compartida cuando el programa termina
    struct shm_remove
    {
        shm_remove() { shared_memory_object::remove(SharedMemory); } // Elimina la memoria compartida al inicio
        ~shm_remove() { shared_memory_object::remove(SharedMemory); } // Elimina la memoria compartida al finalizar
    } remover; // Instancia de la estructura para que funcione automáticamente

    // Crea un segmento de memoria compartida llamado "MySharedMemory" con un tamaño de 64 KB
    managed_shared_memory segment(create_only, SharedMemory, 65536);

    // Asigna 1 KB de la memoria compartida y guarda la dirección en shptr
    void * shptr = segment.allocate(1024);

    /*
        IMPORTANTE
     * Obtiene un identificador de la dirección base que puede usarse para 
     * identificar cualquier byte del segmento de memoria compartida, 
     * incluso si se mapea en direcciones base diferentes en otro proceso.
     */
    //managed_shared_memory::handle_t handle = segment.get_handle_from_address(shptr);
    //std::cout << "Shared Memory handle : " << handle << std::endl;

    // Bucle infinito para interactuar con el usuario hasta que se seleccione "exit"
    for (int i=0;i<100;i++) {        
        memset(buffer, '\0', sizeof(buffer)); // Reinicia el buffer de escritura
        std::string sharedData = "MESSAGE #" + std::to_string(i);
        std::cout << "WRITING <- " << sharedData << std::endl;
        strncpy(buffer, sharedData.c_str(), sizeof(buffer) - 1);
        memset(shptr, '\0', 1024); // Limpia el bloque de memoria compartida
        memcpy((char*)shptr, buffer, strlen(buffer)); // Copia los datos a la memoria compartida

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