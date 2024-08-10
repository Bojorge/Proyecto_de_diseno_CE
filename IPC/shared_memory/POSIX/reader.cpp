#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include "shared_memory.hpp"

namespace bip = boost::interprocess;

void printResourceUsage() {
    long ramUsage = getRAMUsage();
    double userCPU, systemCPU;
    getCPUUsage(userCPU, systemCPU);

    std::cout << "Uso de RAM: " << ramUsage << " KB" << std::endl;
    std::cout << "Uso de CPU - Modo Usuario: " << userCPU << " s" << std::endl;
    std::cout << "Uso de CPU - Modo Sistema: " << systemCPU << " s" << std::endl;
}

void read_memory(bip::named_semaphore &sem_read, bip::named_semaphore &sem_write, SharedData *sharedData, Sentence *buffer, int interval) {
    while (!sharedData->writingFinished) {
        sem_write.wait();

        // Obtener el semáforo para el espacio de lectura
        std::string sem_read_name = std::string(SEM_READ_VARIABLE_FNAME) + std::to_string(sharedData->readIndex);
        bip::named_semaphore sem_var_read(bip::open_only, sem_read_name.c_str());

        // Obtener el semáforo para el espacio de escritura
        std::string sem_write_name = std::string(SEM_WRITE_VARIABLE_FNAME) + std::to_string(sharedData->readIndex);
        bip::named_semaphore sem_var_write(bip::open_only, sem_write_name.c_str());

        // Esperar el semáforo para el espacio de lectura
        sem_var_read.wait();

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

        // Publicar para que se pueda escribir de nuevo en el espacio
        sem_var_write.post();
        sem_read.post();

        // Esperar el intervalo especificado antes de la próxima lectura
        sleep(interval);
    }
}

int main() {
    using namespace boost::interprocess;

    // Abrir semáforos ya creados
    bip::named_semaphore sem_read(bip::open_only, SEM_READ_PROCESS_FNAME);
    bip::named_semaphore sem_write(bip::open_only, SEM_WRITE_PROCESS_FNAME);

    // Conectar a la estructura de memoria compartida
    SharedData *sharedData = attach_struct(STRUCT_LOCATION);
    if (sharedData == nullptr) {
        std::cerr << "ERROR: no se pudo acceder al bloque" << std::endl;
        return -1;
    }

    // Conectar al buffer de memoria compartida
    Sentence *buffer = attach_buffer(BUFFER_LOCATION, sharedData->bufferSize * sizeof(Sentence));
    if (buffer == nullptr) {
        std::cerr << "ERROR: no se pudo acceder al bloque" << std::endl;
        return -1;
    }

    int interval = 2;

    read_memory(sem_read, sem_write, sharedData, buffer, interval);

    // Desconectar de la memoria después de terminar
    detach_struct(sharedData);
    detach_buffer(buffer);

    return 0;
}
