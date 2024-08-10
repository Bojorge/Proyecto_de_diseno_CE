#include <iostream>
#include <cstdlib>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include "shared_memory.hpp"

namespace bip = boost::interprocess;

bool destroy_memory_block(const char *name) {
    try {
        bip::shared_memory_object::remove(name);
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error al eliminar el bloque de memoria compartida: " << e.what() << std::endl;
        return false;
    }
}

bool destroy_semaphore(const char *name) {
    try {
        bip::named_semaphore::remove(name);
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error al eliminar el semáforo: " << e.what() << std::endl;
        return false;
    }
}

int main(int argc, char *argv[]) 
{
    if (argc != 1) {
        std::cerr << "Uso: " << argv[0] << " (no args)" << std::endl;
        return EXIT_FAILURE;
    }

    // Eliminar bloques de memoria compartida
    bool struct_destroyed = destroy_memory_block(STRUCT_LOCATION);
    bool buffer_destroyed = destroy_memory_block(BUFFER_LOCATION);

    // Eliminar semáforos
    bool sem_read_destroyed = destroy_semaphore(SEM_READ_PROCESS_FNAME);
    bool sem_write_destroyed = destroy_semaphore(SEM_WRITE_PROCESS_FNAME);

    if (struct_destroyed) {
        std::cout << "Bloque de memoria compartida destruido: " << STRUCT_LOCATION << std::endl;
    } else {
        std::cerr << "No se pudo destruir el bloque de memoria compartida: " << STRUCT_LOCATION << std::endl;
    }

    if (buffer_destroyed) {
        std::cout << "Bloque de memoria compartida destruido: " << BUFFER_LOCATION << std::endl;
    } else {
        std::cerr << "No se pudo destruir el bloque de memoria compartida: " << BUFFER_LOCATION << std::endl;
    }

    if (sem_read_destroyed) {
        std::cout << "Semáforo destruido: " << SEM_READ_PROCESS_FNAME << std::endl;
    } else {
        std::cerr << "No se pudo destruir el semáforo: " << SEM_READ_PROCESS_FNAME << std::endl;
    }

    if (sem_write_destroyed) {
        std::cout << "Semáforo destruido: " << SEM_WRITE_PROCESS_FNAME << std::endl;
    } else {
        std::cerr << "No se pudo destruir el semáforo: " << SEM_WRITE_PROCESS_FNAME << std::endl;
    }

    return EXIT_SUCCESS;
}
