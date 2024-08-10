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

void read_memory(Poco::NamedSemaphore &sem_read, Poco::NamedSemaphore &sem_write, SharedData *sharedData, Sentence *buffer, int interval) {
    while (!sharedData->writingFinished) {
        sem_write.wait();  // Esperar hasta que el semáforo de escritura esté disponible

        // Obtener el semáforo para el espacio de lectura
        std::string sem_read_name = std::string(SEM_READ_VARIABLE_FNAME) + std::to_string(sharedData->readIndex);
        Poco::NamedSemaphore sem_var_read(Poco::NamedSemaphore::OPEN_ONLY, sem_read_name);

        // Obtener el semáforo para el espacio de escritura
        std::string sem_write_name = std::string(SEM_WRITE_VARIABLE_FNAME) + std::to_string(sharedData->readIndex);
        Poco::NamedSemaphore sem_var_write(Poco::NamedSemaphore::OPEN_ONLY, sem_write_name);

        sem_var_read.wait();  // Esperar el semáforo para el espacio de lectura

        // Imprimir en la consola el índice del buffer, el carácter y la hora recuperada
        int index = sharedData->readIndex;
        std::cout << "\n \n *** Leyendo:\nbuffer[" << index << "] = \"" << buffer[index].character << "\" | tiempo: " << buffer[index].time << std::endl;

        printResourceUsage();

        // Borrar el carácter leído del buffer
        buffer[index].character = '\0';
        strcpy(buffer[index].time, "");

        // Actualizar las variables compartidas
        sharedData->charsTransferred++;
        sharedData->readIndex = (sharedData->readIndex + 1) % sharedData->bufferSize;

        sem_var_write.post();  // Liberar el semáforo de escritura
        sem_read.post();      // Liberar el semáforo de lectura general

        // Esperar el intervalo especificado antes de la próxima lectura
        Poco::Thread::sleep(interval * 1000); // `sleep` en segundos se convierte a milisegundos
    }
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

        int interval = 2;

        read_memory(sem_read, sem_write, sharedData, buffer, interval);

        // Desconectar de la memoria después de finalizar
        detach_struct(sharedData);
        detach_buffer(buffer);

    } catch (const Poco::Exception &e) {
        std::cerr << "Error: " << e.displayText() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
