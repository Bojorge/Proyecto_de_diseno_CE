#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include "shared_memory.hpp"

void printResourceUsage() {
    long ramUsage = getRAMUsage();
    double userCPU, systemCPU;
    getCPUUsage(userCPU, systemCPU);

    std::cout << "Uso de RAM: " << ramUsage << " KB" << std::endl;
    std::cout << "Uso de CPU - Modo Usuario: " << userCPU << " s" << std::endl;
    std::cout << "Uso de CPU - Modo Sistema: " << systemCPU << " s" << std::endl;
}

void read_memory(boost::interprocess::named_semaphore &sem_read, 
                 boost::interprocess::named_semaphore &sem_write, 
                 SharedData *sharedData, 
                 Sentence *buffer, 
                 int interval) 
{
    while (!sharedData->writingFinished) {
        // Esperar hasta que el semáforo de escritura esté disponible
        sem_write.wait(); 

        // Obtener el semáforo para el espacio de lectura
        std::string sem_read_name = std::string(SEM_READ_VARIABLE_FNAME) + std::to_string(sharedData->readIndex);
        boost::interprocess::named_semaphore sem_var_read(boost::interprocess::open_only, sem_read_name.c_str());

        // Obtener el semáforo para el espacio de escritura
        std::string sem_write_name = std::string(SEM_WRITE_VARIABLE_FNAME) + std::to_string(sharedData->readIndex);
        boost::interprocess::named_semaphore sem_var_write(boost::interprocess::open_only, sem_write_name.c_str());

        // Esperar el semáforo para el espacio de lectura
        sem_var_read.wait();

        // Imprimir en la consola el índice del buffer, el carácter y la hora recuperada
        int index = sharedData->readIndex;
        std::cout << "\n \n *** Leyendo:\nbuffer[" << index << "] = \"" << buffer[index].character << "\" | tiempo: " << buffer[index].time << std::endl;

        printResourceUsage();

        // Borrar el carácter leído del buffer
        buffer[index].character = '\0';
        std::strcpy(buffer[index].time, "");

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

    // Conectar a los semáforos que ya fueron creados
    named_semaphore sem_read(open_only, SEM_READ_PROCESS_FNAME);
    named_semaphore sem_write(open_only, SEM_WRITE_PROCESS_FNAME);

    // Conectar al bloque de memoria compartida para SharedData
    managed_shared_memory shm(open_only, STRUCT_LOCATION);
    SharedData *sharedData = shm.find<SharedData>("SharedData").first;
    if (sharedData == nullptr) {
        std::cerr << "ERROR: no se pudo acceder al bloque" << std::endl;
        return -1;
    }

    // Conectar al bloque de memoria compartida para Sentence
    managed_shared_memory shmBuffer(open_only, BUFFER_LOCATION);
    Sentence *buffer = shmBuffer.find<Sentence>("SentenceArray").first;
    if (buffer == nullptr) {
        std::cerr << "ERROR: no se pudo acceder al bloque" << std::endl;
        return -1;
    }

    int interval = 2;

    read_memory(sem_read, sem_write, sharedData, buffer, interval);

    return 0;
}
