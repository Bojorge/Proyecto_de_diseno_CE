#include "shared_memory.hpp"

namespace bip = boost::interprocess;

// Inicializa el bloque de memoria compartida
void init_mem_block(const char *shm_name, std::size_t sizeBuffer) {
    try {
        // Crear el objeto de memoria compartida
        bip::shared_memory_object shm(bip::create_only, shm_name, bip::read_write);

        // Configurar el tamaño del segmento de memoria compartida
        shm.truncate(sizeBuffer);

        // Mapear el segmento de memoria compartida en el espacio de direcciones del proceso
        bip::mapped_region region(shm, bip::read_write);

        // Acceder al buffer en la memoria compartida
        void* addr = region.get_address();
        std::memset(addr, 0, sizeBuffer); // Inicializar el buffer a cero

        std::cout << "Bloque de memoria compartida inicializado correctamente." << std::endl;
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Error al inicializar el bloque de memoria compartida: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Adjunta al buffer en memoria compartida
Sentence* attach_buffer(const char *shm_name) {
    try {
        bip::shared_memory_object shm(bip::open_only, shm_name, bip::read_write);
        bip::mapped_region region(shm, bip::read_write);
        void* addr = region.get_address();

        return static_cast<Sentence*>(addr);
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Error al adjuntar el buffer compartido: " << e.what() << std::endl;
        return nullptr;
    }
}

// Desvincula el bloque de memoria compartida (solo para `managed_shared_memory`)
bool detach_mem_block(const char *location) {
    // Para `shared_memory_object`, no hay una función directa para "desvincular".
    // Solo se debe destruir el segmento de memoria compartida en el sistema de archivos.
    return true;
}

// Destruye un bloque de memoria compartida
bool destroy_mem_block(const char *shm_name) {
    try {
        bip::shared_memory_object::remove(shm_name);
        return true;
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Error al intentar destruir el bloque de memoria compartida: " << e.what() << std::endl;
        return false;
    }
}

// Crea un semáforo
bool create_semaphore(const char *name, unsigned int initial_count) {
    try {
        bip::named_semaphore::remove(name); // Elimina el semáforo si ya existe
        bip::named_semaphore sem(bip::create_only, name, initial_count);
        return true;
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Error al intentar crear el semáforo: " << e.what() << std::endl;
        return false;
    }
}

// Destruye un semáforo
bool destroy_semaphore(const char *name) {
    try {
        bip::named_semaphore::remove(name);
        return true;
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Error al intentar destruir el semáforo: " << e.what() << std::endl;
        return false;
    }
}

// Obtiene un semáforo
bip::named_semaphore* get_semaphore(const char *name) {
    try {
        bip::named_semaphore *sem = new bip::named_semaphore(bip::open_only, name);
        return sem;
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Error al obtener el semáforo: " << e.what() << std::endl;
        return nullptr;
    }
}

void check_shared_memory_size(const char* location) {
    try {
        // Abrir el objeto de memoria compartida
        boost::interprocess::shared_memory_object shm_obj(boost::interprocess::open_only, location, boost::interprocess::read_only);
        
        // Obtener el tamaño del bloque de memoria compartida
        boost::interprocess::offset_t size;
        shm_obj.get_size(size);
        
        std::cout << "El tamaño del bloque de memoria compartida en " << location << " es de: " << size << " bytes." << std::endl;
    } catch (const boost::interprocess::interprocess_exception& e) {
        std::cerr << "Error al intentar obtener el tamaño de la memoria compartida: " << e.what() << std::endl;
    }
}