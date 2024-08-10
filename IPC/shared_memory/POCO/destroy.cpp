#include <iostream>
#include <cstdlib>
#include <Poco/Exception.h>
#include <Poco/NamedSemaphore.h>
#include <Poco/SharedMemory.h>
#include "shared_memory.hpp"

#define STRUCT_FILENAME "struct_location"
#define BUFFER_FILENAME "buffer_location"

bool destroy_memory_block(const std::string &location) {
    try {
        // Intentar eliminar la memoria compartida
        Poco::SharedMemory::remove(location);
        return true;
    } catch (const Poco::Exception &e) {
        std::cerr << "Error destroying memory block: " << e.displayText() << std::endl;
        return false;
    }
}

bool destroy_semaphore(const std::string &name) {
    try {
        Poco::NamedSemaphore::remove(name);
        return true;
    } catch (const Poco::Exception &e) {
        std::cerr << "Error destroying semaphore: " << e.displayText() << std::endl;
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
    bool struct_destroyed = destroy_memory_block(STRUCT_FILENAME);
    bool buffer_destroyed = destroy_memory_block(BUFFER_FILENAME);

    // Eliminar semáforos
    bool sem_read_destroyed = destroy_semaphore(SEM_READ_PROCESS_FNAME);
    bool sem_write_destroyed = destroy_semaphore(SEM_WRITE_PROCESS_FNAME);

    if (struct_destroyed) {
        std::cout << "Destroyed block: " << STRUCT_FILENAME << std::endl;
    } else {
        std::cerr << "Could not destroy block: " << STRUCT_FILENAME << std::endl;
    }

    if (buffer_destroyed) {
        std::cout << "Destroyed block: " << BUFFER_FILENAME << std::endl;
    } else {
        std::cerr << "Could not destroy block: " << BUFFER_FILENAME << std::endl;
    }

    if (sem_read_destroyed) {
        std::cout << "Destroyed semaphore: " << SEM_READ_PROCESS_FNAME << std::endl;
    } else {
        std::cerr << "Could not destroy semaphore: " << SEM_READ_PROCESS_FNAME << std::endl;
    }

    if (sem_write_destroyed) {
        std::cout << "Destroyed semaphore: " << SEM_WRITE_PROCESS_FNAME << std::endl;
    } else {
        std::cerr << "Could not destroy semaphore: " << SEM_WRITE_PROCESS_FNAME << std::endl;
    }

    return EXIT_SUCCESS;
}
