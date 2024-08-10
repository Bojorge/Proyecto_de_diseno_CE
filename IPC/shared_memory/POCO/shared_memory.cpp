#include "shared_memory.hpp"
#include <Poco/Exception.h>
#include <iostream>
#include <sys/resource.h>

long getRAMUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

void getCPUUsage(double &userCPU, double &systemCPU) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    userCPU = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;
    systemCPU = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
}

void init_mem_block(const std::string &struct_location, const std::string &buffer_location, int sizeStruct, int sizeBuffer) {
    try {
        // Crear la memoria compartida para estructura y buffer
        Poco::SharedMemory sharedMemStruct(struct_location, sizeStruct);
        Poco::SharedMemory sharedMemBuffer(buffer_location, sizeBuffer);

        // Crear la región mapeada
        Poco::SharedMemory::MappedRegion regionStruct(sharedMemStruct);
        Poco::SharedMemory::MappedRegion regionBuffer(sharedMemBuffer);

        // Inicializar la estructura compartida
        SharedData *sharedData = static_cast<SharedData*>(regionStruct.begin());
        sharedData->bufferSize = sizeBuffer / sizeof(Sentence);
        sharedData->writeIndex = 0;
        sharedData->readIndex = 0;
        sharedData->readingFileIndex = 0;
        sharedData->clientBlocked = 0;
        sharedData->recBlocked = 0;
        sharedData->charsTransferred = 0;
        sharedData->charsRemaining = 0;
        sharedData->memUsed = sizeStruct + sizeBuffer;
        sharedData->clientUserTime = 0;
        sharedData->clientKernelTime = 0;
        sharedData->recUserTime = 0;
        sharedData->recKernelTime = 0;
        sharedData->writingFinished = false;
        sharedData->readingFinished = false;
        sharedData->statsInited = false;

        // Crear semáforos
        // Estos nombres y tamaños deben ser adaptados según los requerimientos de la aplicación
        Semaphore sem_read(1);  // Inicializa con 1 para control de acceso
        Semaphore sem_write(0); // Inicializa con 0, espera a ser liberado

        // Crear semáforos para variables (si es necesario)
        for (int i = 0; i < sharedData->bufferSize; ++i) {
            std::string sem_read_name = std::string(SEM_READ_VARIABLE_FNAME) + std::to_string(i);
            std::string sem_write_name = std::string(SEM_WRITE_VARIABLE_FNAME) + std::to_string(i);
            Semaphore sem_temp_read(0); // Inicializa según sea necesario
            Semaphore sem_temp_write(1); // Inicializa según sea necesario
        }

    } catch (const Poco::Exception &e) {
        std::cerr << "Error initializing memory block: " << e.displayText() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

SharedData *attach_struct(const std::string &struct_location) {
    try {
        Poco::SharedMemory sharedMemStruct(struct_location, sizeof(SharedData));
        Poco::SharedMemory::MappedRegion region(sharedMemStruct);
        return static_cast<SharedData*>(region.begin());
    } catch (const Poco::Exception &e) {
        std::cerr << "Error attaching to shared struct: " << e.displayText() << std::endl;
        return nullptr;
    }
}

Sentence *attach_buffer(const std::string &buffer_location) {
    try {
        Poco::SharedMemory sharedMemBuffer(buffer_location, sizeof(Sentence) * 100);
        Poco::SharedMemory::MappedRegion region(sharedMemBuffer);
        return static_cast<Sentence*>(region.begin());
    } catch (const Poco::Exception &e) {
        std::cerr << "Error attaching to shared buffer: " << e.displayText() << std::endl;
        return nullptr;
    }
}

bool detach_struct(SharedData *sharedStruct) {
    // No es necesario en POCO, ya que la memoria se administra automáticamente
    return true;
}

bool detach_buffer(Sentence *buffer) {
    // No es necesario en POCO, ya que la memoria se administra automáticamente
    return true;
}

bool destroy_memory_block(const std::string &location) {
    try {
        Poco::SharedMemory::remove(location);
        return true;
    } catch (const Poco::Exception &e) {
        std::cerr << "Error destroying memory block: " << e.displayText() << std::endl;
        return false;
    }
}
