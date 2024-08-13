#include "shared_memory.hpp"

int main() {
    // Tamaño del buffer para 100 Sentences
    std::size_t sizeBuffer = 100 * sizeof(Sentence); 

    // Eliminar el bloque de memoria compartida si existe
    if (!destroy_mem_block(BUFFER_LOCATION)) {
        std::cerr << "Error al destruir el bloque de memoria compartida." << std::endl;
        return EXIT_FAILURE;
    }

    // Inicializar el bloque de memoria compartida
    init_mem_block(BUFFER_LOCATION, sizeBuffer);

    // Crear semáforos para controlar el acceso concurrente
    if (!create_semaphore(SEM_READ_PROCESS_NAME, 1)) {
        std::cerr << "Error al crear el semáforo de lectura del proceso." << std::endl;
        return EXIT_FAILURE;
    }
    
    if (!create_semaphore(SEM_WRITE_PROCESS_NAME, 1)) {
        std::cerr << "Error al crear el semáforo de escritura del proceso." << std::endl;
        return EXIT_FAILURE;
    }

    if (!create_semaphore(SEM_READ_VARIABLE_NAME, 1)) {
        std::cerr << "Error al crear el semáforo de lectura de variables." << std::endl;
        return EXIT_FAILURE;
    }
    
    if (!create_semaphore(SEM_WRITE_VARIABLE_NAME, 1)) {
        std::cerr << "Error al crear el semáforo de escritura de variables." << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Memoria compartida y semáforos inicializados correctamente." << std::endl;

    return EXIT_SUCCESS;
}
