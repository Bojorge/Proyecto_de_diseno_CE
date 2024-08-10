#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include <semaphore.h>
#include <fcntl.h>
#include "shared_memory.h"

void printResourceUsage() {
    long ramUsage = getRAMUsage();
    double userCPU, systemCPU;
    getCPUUsage(userCPU, systemCPU);

    std::cout << "Uso de RAM: " << ramUsage << " KB" << std::endl;
    std::cout << "Uso de CPU - Modo Usuario: " << userCPU << " s" << std::endl;
    std::cout << "Uso de CPU - Modo Sistema: " << systemCPU << " s" << std::endl;
}

void insert_char(sem_t *sem_read, sem_t *sem_write, SharedData *sharedData, Sentence *buffer) {
    char character;

    // Variables para manejar la entrada dinámica
    std::string dynamic_string;

    while (true) {
        // Esperar hasta que el semáforo de escritura esté disponible
        sem_wait(sem_write);

        // Reiniciar el string dinámico para la próxima iteración
        dynamic_string.clear();

        // Leer un carácter de la entrada estándar
        std::cout << "\n \n * Ingrese un carácter (Ctrl+D para terminar): ";
        if ((character = std::cin.get()) == EOF) {
            std::cout << "Fin de entrada." << std::endl;
            break;
        }
        std::cin.get(); // Consumir el '\n' después del carácter ingresado

        // Concatenar el carácter al string dinámico
        dynamic_string += character;

        // Obtener el semáforo para el espacio de escritura
        std::string sem_write_name = std::string(SEM_WRITE_VARIABLE_FNAME) + std::to_string(sharedData->writeIndex);
        sem_t *sem_var_write = sem_open(sem_write_name.c_str(), 0);
        if (sem_var_write == SEM_FAILED) {
            perror("sem_open/variables");
            exit(EXIT_FAILURE);
        }

        // Obtener el semáforo para el espacio de lectura
        std::string sem_read_name = std::string(SEM_READ_VARIABLE_FNAME) + std::to_string(sharedData->writeIndex);
        sem_t *sem_var_read = sem_open(sem_read_name.c_str(), 0);
        if (sem_var_read == SEM_FAILED) {
            perror("sem_open/variables");
            exit(EXIT_FAILURE);
        }

        // Esperar el semáforo para el espacio de escritura
        sem_wait(sem_var_write);

        // Obtener la marca de tiempo actual en el formato deseado
        time_t current_time;
        struct tm *timeinfo;
        time(&current_time);
        timeinfo = localtime(&current_time);
        strftime(buffer[sharedData->writeIndex].time, MAX_TIME_LENGTH, "%b %d %Y %H:%M:%S", timeinfo);

        // Asignar el carácter al buffer
        int index = sharedData->writeIndex;
        //std::cout << "Caracter ingresado: " << character << std::endl;
        buffer[sharedData->writeIndex].character = character;
        //std::cout << "Agregando a buffer:\nbuffer[" << index << "] = \"" << buffer[index].character << "\" | tiempo: " << buffer[index].time << "\n------------------------\n";

        // Actualizar los índices compartidos
        sharedData->writeIndex = (sharedData->writeIndex + 1) % sharedData->bufferSize;

        // Post para que la variable pueda ser leída
        sem_post(sem_var_read);  
        sem_post(sem_read);

        printResourceUsage();
    }

    sharedData->writingFinished = true;
}

int main() 
{   
    // Open semaphores that were already created
    sem_t *sem_read = sem_open(SEM_READ_PROCESS_FNAME, 0);
    if (sem_read == SEM_FAILED) {
        perror("sem_open/read");
        exit(EXIT_FAILURE);
    }
    sem_t *sem_write = sem_open(SEM_WRITE_PROCESS_FNAME, 0);
    if (sem_write == SEM_FAILED) {
        perror("sem_open/write");
        exit(EXIT_FAILURE);
    }

    // Connect to shared mem struct
    SharedData *sharedData = attach_struct(STRUCT_LOCATION, sizeof(SharedData));
    if (sharedData == nullptr) {
        std::cerr << "ERROR: no se pudo acceder al bloque" << std::endl;
        return -1;
    }

    // Connect to shared mem buffer
    Sentence *buffer = attach_buffer(BUFFER_LOCATION, (sharedData->bufferSize * sizeof(Sentence)));
    if (buffer == nullptr) {
        std::cerr << "ERROR: no se pudo acceder al bloque" << std::endl;
        return -1;
    }

    insert_char(sem_read, sem_write, sharedData, buffer);

    // Detach from memory after finishing
    detach_struct(sharedData);
    detach_buffer(buffer);
    return 0;
}
