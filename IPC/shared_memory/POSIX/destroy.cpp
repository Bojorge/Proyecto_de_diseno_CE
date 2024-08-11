#include <iostream>
#include <cstdlib>
#include <cstring>
#include "shared_memory.h"

#define STRUCT_FILENAME "creator.cpp"
#define BUFFER_FILENAME "destroy.cpp"


int main(int argc, char *argv[]) 
{
    if (argc != 1) {
        std::cerr << "Uso: " << argv[0] << " (no args)" << std::endl;
        return EXIT_FAILURE;
    }

    // Eliminar bloques de memoria compartida
    bool struct_destroyed = destroy_memory_block(STRUCT_FILENAME);
    bool buffer_destroyed = destroy_memory_block(BUFFER_FILENAME);

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

    return EXIT_SUCCESS;
}
