#include <boost/interprocess/managed_shared_memory.hpp>
#include <iostream>
#include <cstdlib> // Para std::system
#include <sstream>

#define SharedMemory "MySharedMemory"

using namespace boost::interprocess;

int main (int argc, char *argv[])
{
    // Buffer para almacenar datos a compartir y opciones del menú
    char buffer[1024];
    char opt[10];

    // Inicializa los buffers con ceros (caracteres nulos)
    memset(buffer, '\0', sizeof(buffer));
    memset(opt, '\0', sizeof(opt));

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
    for (;;) {
        std::cout << "Select Option (write/read/exit) : ";
        memset(opt, '\0', sizeof(opt)); // Reinicia el buffer de opciones
        std::cin.getline(opt, sizeof(opt)); // Lee la opción del usuario
        
        if ( !strcmp(opt, "write") ){
            // Si se selecciona "write", escribe datos en la memoria compartida
            std::cout << "-> Shared Memory : ";
            memset(buffer, '\0', sizeof(buffer)); // Reinicia el buffer de escritura
            std::cin.getline(buffer, sizeof(buffer)); // Lee los datos del usuario
            memset(shptr, '\d', 1024); // Limpia el bloque de memoria compartida
            memcpy((char*)shptr, buffer, strlen(buffer)); // Copia los datos a la memoria compartida
        }
        else if ( !strcmp(opt, "read") ){
            // Si se selecciona "read", lee datos de la memoria compartida
            std::cout << "<- Share Memory : ";
            std::cout << (char*)shptr << std::endl; // Muestra los datos almacenados en la memoria compartida
        }
        else {
            // Si se selecciona cualquier otra cosa (incluyendo "exit"), termina el programa
            break;
        }
    }
    return 0;
}
