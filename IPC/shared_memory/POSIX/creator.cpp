#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <semaphore.h>
#include <fcntl.h>
#include "shared_memory.h"

void init_empty_struct(SharedData *sharedData, int numChars) {
    sharedData->bufferSize = numChars;
    sharedData->writeIndex = 0;
    sharedData->readIndex = 0;
    sharedData->readingFileIndex = 0;
    sharedData->clientBlocked = 0;
    sharedData->recBlocked = 0;
    sharedData->charsTransferred = 0;
    sharedData->charsRemaining = 0;
    sharedData->clientUserTime = 0;
    sharedData->clientKernelTime = 0;
    sharedData->recUserTime = 0;
    sharedData->recKernelTime = 0;
    sharedData->memUsed = sizeof(SharedData) + (sizeof(Sentence) * numChars);
    sharedData->writingFinished = false;
    sharedData->readingFinished = false;
    sharedData->statsInited = false;
}

void printResourceUsage() {
    long ramUsage = getRAMUsage();
    double userCPU, systemCPU;
    getCPUUsage(userCPU, systemCPU);

    std::cout << "Uso de RAM: " << ramUsage << " KB" << std::endl;
    std::cout << "Uso de CPU - Modo Usuario: " << userCPU << " s" << std::endl;
    std::cout << "Uso de CPU - Modo Sistema: " << systemCPU << " s" << std::endl;
}

int main(int argc, char *argv[]) 
{
    // Eliminar bloques de memoria compartida y semáforos existentes
    destroy_memory_block(STRUCT_LOCATION);
    destroy_memory_block(BUFFER_LOCATION);

    int numChars = 10;

    using namespace boost::interprocess;

    // Crear y abrir semáforos
    named_semaphore sem_read(create_only, SEM_READ_PROCESS_FNAME, 1);
    named_semaphore sem_write(create_only, SEM_WRITE_PROCESS_FNAME, 0);

    for (int i = 0; i < numChars; i++) {
        std::string sem_write_name = std::string(SEM_WRITE_VARIABLE_FNAME) + std::to_string(i);
        std::string sem_read_name = std::string(SEM_READ_VARIABLE_FNAME) + std::to_string(i);

        named_semaphore sem_temp_write(create_only, sem_write_name.c_str(), 1);
        named_semaphore sem_temp_read(create_only, sem_read_name.c_str(), 0);
    }

    // Inicializar bloques de memoria compartida
    init_mem_block(STRUCT_LOCATION, BUFFER_LOCATION, sizeof(SharedData), numChars * sizeof(Sentence));

    // Adjuntar a los bloques de memoria compartida
    SharedData *sharedStruct = attach_struct(STRUCT_LOCATION);
    if (sharedStruct == nullptr) {
        std::cerr << "Error al adjuntar al bloque de memoria compartida." << std::endl;
        exit(EXIT_FAILURE);
    }

    Sentence *buffer = attach_buffer(BUFFER_LOCATION);
    if (buffer == nullptr) {
        std::cerr << "Error al adjuntar al bloque de memoria compartida." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Inicializar la estructura compartida
    init_empty_struct(sharedStruct, numChars);

    // Visualizar el bloque de memoria
    while (true) {
        sem_read.wait();  // Esperar en el semáforo de lectura

        std::cout << "\033[0;0H\033[2J"; // Mover el cursor a la posición (0, 0) y borrar la pantalla
        std::cout.flush();
        for (int i = 0; i < numChars; i++) {
            std::cout << "buffer[" << i << "] = \"" << buffer[i].character << "\" | time: " << buffer[i].time << std::endl;
        }

        std::cout << "-------------------------------------------------" << std::endl;
        std::cout.flush();

        sem_write.post(); // Liberar el semáforo de escritura

        printResourceUsage();
    }

    // Destruir el segmento de memoria compartida y semáforos
    destroy_memory_block(STRUCT_LOCATION);
    destroy_memory_block(BUFFER_LOCATION);

    return 0;
}
