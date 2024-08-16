#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <thread>

const char* SHARED_MEMORY_NAME = "/MySharedMemory";
const size_t SHARED_MEMORY_SIZE = 65536;

int main() {
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

    // Escribir en la memoria compartida en un bucle
    for (int i = 0; i < 10; ++i) {
        std::string sharedData = "ABCDEFGH " + std::to_string(i);
        std::cout << "WRITING <- " << sharedData << std::endl;
        
        memset(ptr, '\0', SHARED_MEMORY_SIZE); // Limpia la memoria compartida
        memcpy(ptr, sharedData.c_str(), sharedData.size()); // Copia los datos a la memoria compartida

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Desvincular y cerrar
    if (munmap(ptr, SHARED_MEMORY_SIZE) == -1) {
        perror("munmap");
    }
    close(fd);
    
    return 0;
}
