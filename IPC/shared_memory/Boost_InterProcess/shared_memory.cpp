#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <boost/interprocess/containers/vector.hpp>
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
        // Abre la memoria compartida existente
        bip::managed_shared_memory segment(bip::open_only, struct_location);

        // Encuentra el objeto SharedData
        SharedData* sharedData = segment.find<SharedData>("SharedData").first;

        if (sharedData == nullptr) {
            std::cerr << "Error al adjuntar la estructura compartida." << std::endl;
            return nullptr;
        }

        return sharedData;
    } catch (const std::exception& e) {
        std::cerr << "Error al adjuntar la estructura compartida: " << e.what() << std::endl;
        return nullptr;
    }
}

Sentence* attach_buffer(const char *buffer_location) {
    try {
        // Abre la memoria compartida existente
        bip::managed_shared_memory segment(bip::open_only, buffer_location);

        // Encuentra el buffer de Sentences
        Sentence* buffer = segment.find<Sentence>("SentenceBuffer").first;

        if (buffer == nullptr) {
            std::cerr << "Error al adjuntar el buffer compartido." << std::endl;
            return nullptr;
        }

        return buffer;
    } catch (const std::exception& e) {
        std::cerr << "Error al adjuntar el buffer compartido: " << e.what() << std::endl;
        return nullptr;
    }
}

void init_mem_block(const char *struct_location, const char *buffer_location, int sizeStruct, int sizeBuffer) {
    try {
        // Crear la memoria compartida para la estructura
        bip::managed_shared_memory struct_segment(bip::create_only, struct_location, sizeStruct);

        // Crear la memoria compartida para el buffer
        bip::managed_shared_memory buffer_segment(bip::create_only, buffer_location, sizeBuffer);

        // Inicializar la estructura en la memoria compartida
        struct_segment.construct<SharedData>("SharedData")();

        // Inicializar el buffer en la memoria compartida
        buffer_segment.construct<Sentence>("SentenceBuffer")[sizeBuffer / sizeof(Sentence)]();

        std::cout << "Bloques de memoria compartida inicializados correctamente." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error al inicializar los bloques de memoria compartida: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

bool detach_struct(const char *struct_location) {
    try {
        bip::managed_shared_memory segment(bip::open_only, struct_location);
        segment.destroy<SharedData>("SharedData");
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error al intentar desvincular la estructura: " << e.what() << std::endl;
        return false;
    }
}

bool detach_buffer(const char *buffer_location) {
    try {
        bip::managed_shared_memory segment(bip::open_only, buffer_location);
        segment.destroy<Sentence>("SentenceBuffer");
        return true;
    } catch (const std::exception& e) {
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
