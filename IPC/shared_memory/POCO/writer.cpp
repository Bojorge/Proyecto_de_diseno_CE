#include <iostream>
#include <cstdlib>
#include <string>
#include <ctime>
#include <Poco/Exception.h>
#include <Poco/NamedSemaphore.h>
#include <Poco/SharedMemory.h>
#include <Poco/ScopedLock.h>
#include "shared_memory.hpp"

void printResourceUsage() {
    long ramUsage = getRAMUsage();
    double userCPU, systemCPU;
    getCPUUsage(userCPU, systemCPU);

    std::cout << "Uso de RAM: " << ramUsage << " KB" << std::endl;
    std::cout << "Uso de CPU - Modo Usuario: " << userCPU << " s" << std::endl;
    std::cout << "Uso de CPU - Modo Sistema: " << systemCPU << " s" << std::endl;
}

void insert_char(Poco::NamedSemaphore &sem_read, Poco::NamedSemaphore &sem_write, SharedData *sharedData, Sentence *buffer) {
    char character;
    std::string dynamic_string;

    while (true) {
        sem_write.wait();  // Esperar hasta que el semáforo de escritura esté disponible

        dynamic_string.clear();  // Reiniciar el string dinámico para la próxima iteración

        std::cout << "\n \n * Ingrese un carácter (Ctrl+D para terminar): ";
        if ((character = std::cin.get()) == EOF) {
            std::cout << "Fin de entrada." << std::endl;
            break;
        }
        std::cin.get();  // Consumir el '\n' después del carácter ingresado

        dynamic_string += character;

        // Obtener el semáforo para el espacio de escritura
        std::string sem_write_name = std::string(SEM_WRITE_VARIABLE_FNAME) + std::to_string(sharedData->writeIndex);
        Poco::NamedSemaphore sem_var_write(Poco::NamedSemaphore::OPEN_ONLY, sem_write_name);

        // Obtener el semáforo para el espacio de lectura
        std::string sem_read_name = std::string(SEM_READ_VARIABLE_FNAME) + std::to_string(sharedData->writeIndex);
        Poco::NamedSemaphore sem_var_read(Poco::NamedSemaphore::OPEN_ONLY, sem_read_name);

        sem_var_write.wait();  // Esperar el semáforo para el espacio de escritura

        // Obtener la marca de tiempo actual
        time_t current_time;
        struct tm *timeinfo;
        time(&current_time);
        timeinfo = localtime(&current_time);
        strftime(buffer[sharedData->writeIndex].time, MAX_TIME_LENGTH, "%b %d %Y %H:%M:%S", timeinfo);

        // Asignar el carácter al buffer
        int index = sharedData->writeIndex;
        buffer[sharedData->writeIndex].character = character;

        // Actualizar los índices compartidos
        sharedData->writeIndex = (sharedData->writeIndex + 1) % sharedData->bufferSize;

        sem_var_read.post();  // Liberar el semáforo de lectura
        sem_read.post();      // Liberar el semáforo de lectura general

        printResourceUsage();
    }

    sharedData->writingFinished = true;
}

int main() 
{
    try {
        // Abrir semáforos que ya fueron creados
        Poco::NamedSemaphore sem_read(Poco::NamedSemaphore::OPEN_ONLY, SEM_READ_PROCESS_FNAME);
        Poco::NamedSemaphore sem_write(Poco::NamedSemaphore::OPEN_ONLY, SEM_WRITE_PROCESS_FNAME);

        // Conectar con el bloque de memoria compartida
        SharedData *sharedData = attach_struct(STRUCT_LOCATION);
        if (sharedData == nullptr) {
            std::cerr << "ERROR: no se pudo acceder al bloque" << std::endl;
            return -1;
        }

        // Conectar con el buffer de memoria compartida
        Sentence *buffer = attach_buffer(BUFFER_LOCATION);
        if (buffer == nullptr) {
            std::cerr << "ERROR: no se pudo acceder al bloque" << std::endl;
            return -1;
        }

        insert_char(sem_read, sem_write, sharedData, buffer);

        // Desconectar de la memoria después de finalizar
        detach_struct(sharedData);
        detach_buffer(buffer);

    } catch (const Poco::Exception &e) {
        std::cerr << "Error: " << e.displayText() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
