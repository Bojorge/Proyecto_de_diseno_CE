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

// Inicializa el bloque de memoria compartida
void init_mem_block(const char *struct_location, const char *buffer_location, int sizeStruct, int sizeBuffer) {
    using namespace boost::interprocess;
    
    // Crear segmento de memoria compartida para SharedData
    managed_shared_memory shm(create_only, struct_location, sizeStruct);
    shm.construct<SharedData>("SharedData")();
    
    // Crear segmento de memoria compartida para Sentence
    managed_shared_memory shmBuffer(create_only, buffer_location, sizeBuffer);
    shmBuffer.construct<Sentence>("SentenceArray")[sizeBuffer / sizeof(Sentence)]; // Ajuste del tamaño del buffer
}

// Adjunta la estructura compartida
SharedData *attach_struct(const char *struct_location) {
    using namespace boost::interprocess;
    
    managed_shared_memory shm(open_only, struct_location);
    SharedData *sharedData = shm.find<SharedData>("SharedData").first;
    return sharedData;
}

// Adjunta el buffer
Sentence *attach_buffer(const char *buffer_location, int sizeBuffer) {
    using namespace boost::interprocess;
    
    managed_shared_memory shmBuffer(open_only, buffer_location);
    Sentence *buffer = shmBuffer.find<Sentence>("SentenceArray").first;
    return buffer;
}

// Desconecta la estructura compartida
bool detach_struct(SharedData *sharedStruct) {
    // La memoria compartida es administrada por Boost.Interprocess y no requiere desconexión explícita.
    return true;
}

// Desconecta el buffer
bool detach_buffer(Sentence *buffer) {
    // La memoria compartida es administrada por Boost.Interprocess y no requiere desconexión explícita.
    return true;
}

// Destruye el bloque de memoria compartida
bool destroy_memory_block(const char *filename) {
    using namespace boost::interprocess;
    
    try {
        shared_memory_object::remove(filename);
        return true;
    } catch (...) {
        return false;
    }
}
