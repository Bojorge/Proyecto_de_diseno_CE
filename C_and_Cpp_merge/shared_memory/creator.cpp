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

int main(int argc, char *argv[]) 
{
    destroy_memory_block(STRUCT_LOCATION);
    destroy_memory_block(BUFFER_LOCATION);
    
    int numChars = 10;
    // std::cout << "Ingrese la cantidad de caracteres a compartir: ";
    // std::cin >> numChars;

    // Set the semaphores
    sem_unlink(SEM_READ_PROCESS_FNAME);
    sem_unlink(SEM_WRITE_PROCESS_FNAME);

    sem_t *sem_read = sem_open(SEM_READ_PROCESS_FNAME, O_CREAT, 0644, 1);
    if (sem_read == SEM_FAILED) {
        perror("sem_open/read");
        exit(EXIT_FAILURE);
    }
    sem_t *sem_write = sem_open(SEM_WRITE_PROCESS_FNAME, O_CREAT, 0644, 0);
    if (sem_write == SEM_FAILED) {
        perror("sem_open/write");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numChars; i++) {
        // Make every semaphore name for each buffer space
        std::string sem_write_name = std::string(SEM_WRITE_VARIABLE_FNAME) + std::to_string(i);
        std::string sem_read_name = std::string(SEM_READ_VARIABLE_FNAME) + std::to_string(i);

        // Unlink them to prevent
        sem_unlink(sem_write_name.c_str());
        sem_unlink(sem_read_name.c_str());

        // Init each semaphore
        sem_t *sem_temp_write = sem_open(sem_write_name.c_str(), O_CREAT, 0644, 1);
        if (sem_temp_write == SEM_FAILED) {
            perror("sem_open/variables");
            exit(EXIT_FAILURE);
        }

        sem_t *sem_temp_read = sem_open(sem_read_name.c_str(), O_CREAT, 0644, 0);
        if (sem_temp_read == SEM_FAILED) {
            perror("sem_open/variables");
            exit(EXIT_FAILURE);
        }
    }

    // Initialize shared mem blocks
    init_mem_block(STRUCT_LOCATION, BUFFER_LOCATION, sizeof(SharedData), numChars * sizeof(Sentence));

    // Attach to struct shared mem block
    SharedData *sharedStruct = attach_struct(STRUCT_LOCATION, sizeof(SharedData));
    if (sharedStruct == nullptr) {
        std::cerr << "Error al adjuntar al bloque de memoria compartida." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Attach to buffer mem block
    Sentence *buffer = attach_buffer(BUFFER_LOCATION, numChars * sizeof(Sentence));
    if (buffer == nullptr) {
        std::cerr << "Error al adjuntar al bloque de memoria compartida." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Initialize empty struct for the shared mem block
    init_empty_struct(sharedStruct, numChars);

    // Start visualization of mem block
    while (true) {
        sem_wait(sem_read);
        std::cout << "\033[0;0H\033[2J"; // Mover el cursor a la posición (0, 0) y borrar la pantalla
        std::cout.flush();
        for (int i = 0; i < numChars; i++) {
            std::cout << "buffer[" << i << "] = \"" << buffer[i].character << "\" | time: " << buffer[i].time << std::endl;
        }

        std::cout << "-------------------------------------------------" << std::endl;
        std::cout.flush();

        sem_post(sem_write);
    }

    detach_struct(sharedStruct);
    detach_buffer(buffer);

    // Destroy the shared mem block and semaphores
    sem_close(sem_read);
    sem_close(sem_write);

    destroy_memory_block(STRUCT_LOCATION);
    destroy_memory_block(BUFFER_LOCATION);

    return 0;
}
