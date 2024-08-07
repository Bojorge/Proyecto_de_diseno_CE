#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <semaphore.h>
#include <string>
#include "shared_memory.h"

void read_memory(sem_t *sem_read, sem_t *sem_write, SharedData *sharedData, Sentence *buffer, int interval) {
    while (!sharedData->writingFinished) {
        sem_wait(sem_write);

        // Obtener el semáforo para el espacio de lectura
        std::string sem_read_name = std::string(SEM_READ_VARIABLE_FNAME) + std::to_string(sharedData->readIndex);
        sem_t *sem_var_read = sem_open(sem_read_name.c_str(), 0);
        if (sem_var_read == SEM_FAILED) {
            perror("sem_open/variables");
            exit(EXIT_FAILURE);
        }

        // Obtener el semáforo para el espacio de escritura
        std::string sem_write_name = std::string(SEM_WRITE_VARIABLE_FNAME) + std::to_string(sharedData->readIndex);
        sem_t *sem_var_write = sem_open(sem_write_name.c_str(), 0);
        if (sem_var_write == SEM_FAILED) {
            perror("sem_open/variables");
            exit(EXIT_FAILURE);
        }

        // Leer después de comprobar si el semáforo está abierto
        sem_wait(sem_var_read);

        // Imprimir en la consola el índice del buffer, el carácter y la hora recuperada
        int index = sharedData->readIndex;
        std::cout << "Leyendo del buffer:\nbuffer[" << index << "] = \"" << buffer[index].character << "\" | tiempo: " << buffer[index].time << std::endl;

        // Borrar el carácter leído del buffer
        buffer[index].character = '\0';
        strcpy(buffer[index].time, "");

        // Actualizar las variables compartidas
        sharedData->charsTransferred++;
        sharedData->readIndex = (sharedData->readIndex + 1) % sharedData->bufferSize;

        // Publicar para que se pueda escribir de nuevo en el espacio
        sem_post(sem_var_write);
        sem_post(sem_read);

        // Esperar el intervalo especificado antes de la próxima lectura
        sleep(interval);
    }
}

int main() {
    // Open semaphores that were already created
    sem_t *sem_read = sem_open(SEM_READ_PROCESS_FNAME, 0);
    if (sem_read == SEM_FAILED) {
        perror("sem_open/creator");
        exit(EXIT_FAILURE);
    }
    sem_t *sem_write = sem_open(SEM_WRITE_PROCESS_FNAME, 0);
    if (sem_write == SEM_FAILED) {
        perror("sem_open/client");
        exit(EXIT_FAILURE);
    }

    // Connect to shared mem block
    SharedData *sharedData = attach_struct(STRUCT_LOCATION, sizeof(SharedData));
    if (sharedData == nullptr) {
        std::cerr << "ERROR: no se pudo acceder al bloque" << std::endl;
        return -1;
    }

    Sentence *buffer = attach_buffer(BUFFER_LOCATION, (sharedData->bufferSize * sizeof(Sentence)));
    if (buffer == nullptr) {
        std::cerr << "ERROR: no se pudo acceder al bloque" << std::endl;
        return -1;
    }

    int interval = 2;

    read_memory(sem_read, sem_write, sharedData, buffer, interval);

    // Detach from memory after finishing
    detach_struct(sharedData);
    detach_buffer(buffer);

    return 0;
}
