#include <boost/interprocess/managed_shared_memory.hpp>
#include <iostream>
#include <cstdlib> // Para std::system
#include <sstream>
#include <chrono> // Para std::chrono::seconds
#include <thread> // Para std::this_thread::sleep_for

#define SharedMemory "MySharedMemory"

using namespace boost::interprocess;

int main (int argc, char *argv[])
{
    // Buffer para almacenar datos a compartir y opciones del menú
    char buffer[1024];
    
    // Inicializa los buffers con ceros (caracteres nulos)
    memset(buffer, '\0', sizeof(buffer));
    
    // Estructura para eliminar la memoria compartida cuando el programa termina
    struct shm_remove
    {
        shm_remove() { shared_memory_object::remove(SharedMemory); } // Elimina la memoria compartida al inicio
        ~shm_remove() { shared_memory_object::remove(SharedMemory); } // Elimina la memoria compartida al finalizar
    } remover; // Instancia de la estructura para que funcione automáticamente

    // Crea un segmento de memoria compartida llamado "MySharedMemory" con un tamaño de 64 KB
    managed_shared_memory segment(create_only, SharedMemory, 65536);

    // Obtiene la cantidad de memoria libre antes de la asignación
    managed_shared_memory::size_type free_memory_before = segment.get_free_memory();

    // Asigna 1 KB de la memoria compartida y guarda la dirección en shptr
    void * shptr = segment.allocate(1024);

    // Obtiene la cantidad de memoria libre después de la asignación
    managed_shared_memory::size_type free_memory_after = segment.get_free_memory();

    // Muestra la cantidad de memoria libre antes y después de la asignación
    std::cout << "Free Memory Before  : " << free_memory_before << std::endl;
    std::cout << "Free Memory After   : " << free_memory_after << std::endl;
    std::cout << "Free Memory Diff.   : " << free_memory_before - free_memory_after << std::endl;

    /*
     * Obtiene un identificador de la dirección base que puede usarse para 
     * identificar cualquier byte del segmento de memoria compartida, 
     * incluso si se mapea en direcciones base diferentes en otro proceso.
     */
    managed_shared_memory::handle_t handle = segment.get_handle_from_address(shptr);
    std::cout << "Shared Memory handle : " << handle << std::endl;

    // Bucle infinito para interactuar con el usuario hasta que se seleccione "exit"
    for (int i=0;i<10;i++) {        
        memset(buffer, '\0', sizeof(buffer)); // Reinicia el buffer de escritura
        std::string sharedData = "ABCDEFGH " + std::to_string(i);
        std::cout << "WRITING <- " << sharedData << std::endl;
        strncpy(buffer, sharedData.c_str(), sizeof(buffer) - 1);
        memset(shptr, '\0', 1024); // Limpia el bloque de memoria compartida
        memcpy((char*)shptr, buffer, strlen(buffer)); // Copia los datos a la memoria compartida

        // Esperar 1 segundo antes de intentar nuevamente
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return 0;
}