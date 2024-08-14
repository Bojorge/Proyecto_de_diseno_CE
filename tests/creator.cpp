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

    check_shared_memory_size(BUFFER_LOCATION);

    Sentence *buffer = attach_buffer(BUFFER_LOCATION);

    // Verificar la alineación
    std::uintptr_t bufferAddress = reinterpret_cast<std::uintptr_t>(buffer);
    std::size_t alignment = alignof(Sentence);
    if (bufferAddress % alignment == 0) {
        std::cout << "La memoria está alineada correctamente. Alignment = " << alignment << std::endl;
    }

    std::cout << "El buffer esta en la dirección: " << buffer << std::endl;
    std::cout << "&buffer : " << &buffer << std::endl;

    buffer->character = 'A';
    std::cout << "character: " << buffer->character << std::endl;

    return EXIT_SUCCESS;
}



