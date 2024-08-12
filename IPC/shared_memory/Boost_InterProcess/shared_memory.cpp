#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>
#include "shared_memory.hpp"

namespace bip = boost::interprocess;

// Obtiene el uso de RAM en KB
long getRAMUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

// Obtiene el uso de CPU en segundos
void getCPUUsage(double &userCPU, double &systemCPU) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    userCPU = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;
    systemCPU = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
}

SharedData* attach_struct(const char *struct_location) {
    try {
        bip::managed_shared_memory segment(bip::open_only, struct_location);
        SharedData* sharedData = segment.find<SharedData>("SharedData").first;

        if (sharedData == nullptr) {
            std::cerr << "Error: la estructura compartida no se encontró." << std::endl;
            return nullptr;
        }

        return sharedData;
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Error al adjuntar la estructura compartida: " << e.what() << std::endl;
        return nullptr;
    }
}

Sentence* attach_buffer(const char *buffer_location) {
    try {
        bip::managed_shared_memory segment(bip::open_only, buffer_location);
        Sentence* buffer = segment.find<Sentence>("SentenceBuffer").first;

        if (buffer == nullptr) {
            std::cerr << "Error: el buffer compartido no se encontró." << std::endl;
            return nullptr;
        }

        return buffer;
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Error al adjuntar el buffer compartido: " << e.what() << std::endl;
        return nullptr;
    }
}

bool detach_struct(const char *struct_location) {
    try {
        bip::managed_shared_memory segment(bip::open_only, struct_location);
        segment.destroy<SharedData>("SharedData");
        return true;
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Error al intentar desvincular la estructura: " << e.what() << std::endl;
        return false;
    }
}

bool detach_buffer(const char *buffer_location) {
    try {
        bip::managed_shared_memory segment(bip::open_only, buffer_location);
        segment.destroy<Sentence>("SentenceBuffer");
        return true;
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Error al intentar desvincular el buffer: " << e.what() << std::endl;
        return false;
    }
}

bool destroy_memory_block(const char *location) {
    try {
        bip::shared_memory_object::remove(location);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error al intentar eliminar el bloque de memoria: " << e.what() << std::endl;
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

void init_mem_block(const char *struct_location, const char *buffer_location, std::size_t sizeStruct, std::size_t sizeBuffer) {
    try {
        // Eliminar bloques de memoria compartida existentes
        destroy_memory_block(struct_location);
        destroy_memory_block(buffer_location);

        // Crear y configurar el objeto de memoria compartida para la estructura
        boost::interprocess::shared_memory_object shm_struct(boost::interprocess::create_only, struct_location, boost::interprocess::read_write);
        shm_struct.truncate(sizeStruct);
        boost::interprocess::mapped_region region_struct(shm_struct, boost::interprocess::read_write);
        
        // Inicializar la estructura en la memoria compartida
        SharedData* sharedData = new (region_struct.get_address()) SharedData();

        // Crear y configurar el objeto de memoria compartida para el buffer
        boost::interprocess::shared_memory_object shm_buffer(boost::interprocess::create_only, buffer_location, boost::interprocess::read_write);
        shm_buffer.truncate(sizeBuffer);
        boost::interprocess::mapped_region region_buffer(shm_buffer, boost::interprocess::read_write);
        
        // Inicializar el buffer en la memoria compartida
        std::size_t numSentences = sizeBuffer / sizeof(Sentence);
        Sentence* sentenceBuffer = new (region_buffer.get_address()) Sentence[numSentences];

        std::cout << "Bloques de memoria compartida inicializados correctamente." << std::endl;
    } catch (const boost::interprocess::interprocess_exception& e) {
        std::cerr << "Error al inicializar los bloques de memoria compartida: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}