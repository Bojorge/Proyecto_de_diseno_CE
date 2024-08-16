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
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Abrir el segmento de memoria compartida
    int fd = shm_open(SHARED_MEMORY_NAME, O_RDONLY, 0666);
    if (fd == -1) {
        perror("shm_open");
        return -1;
    }

    // Mapear el segmento de memoria compartida en el espacio de direcciones del proceso
    void* ptr = mmap(NULL, SHARED_MEMORY_SIZE, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return -1;
    }
    
    for (int i = 0; i < 10; ++i) {
        std::cout << "READING -> ";
        std::cout << static_cast<char*>(ptr) << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Desvincular y cerrar
    if (munmap(ptr, SHARED_MEMORY_SIZE) == -1) {
        perror("munmap");
    }
    close(fd);

    // Eliminar el segmento de memoria compartida (opcional, según tus necesidades)
    shm_unlink(SHARED_MEMORY_NAME);
    
    return 0;
}
