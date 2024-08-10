#include <iostream>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include "shared_memory.hpp"

namespace bip = boost::interprocess;

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
    try {
        int numChars = 10; // Tamaño de buffer

        // Crear memoria compartida
        bip::managed_shared_memory segment(bip::create_only, STRUCT_LOCATION, sizeof(SharedData) + numChars * sizeof(Sentence));

        // Inicializar semáforos
        bip::named_semaphore sem_read(bip::create_only, SEM_READ_PROCESS_FNAME, 1);
        bip::named_semaphore sem_write(bip::create_only, SEM_WRITE_PROCESS_FNAME, 0);

        // Inicializar semáforos para cada buffer
        for (int i = 0; i < numChars; i++) {
            std::string sem_write_name = std::string(SEM_WRITE_VARIABLE_FNAME) + std::to_string(i);
            std::string sem_read_name = std::string(SEM_READ_VARIABLE_FNAME) + std::to_string(i);

            bip::named_semaphore sem_temp_write(bip::create_only, sem_write_name.c_str(), 1);
            bip::named_semaphore sem_temp_read(bip::create_only, sem_read_name.c_str(), 0);
        }

        // Obtener punteros a la estructura y al buffer
        SharedData *sharedStruct = segment.find_or_construct<SharedData>("SharedData")();
        
        // Crear buffer en la memoria compartida
        segment.construct<Sentence>("Buffer")[numChars];

        // Obtener el puntero al buffer
        Sentence *buffer = segment.find<Sentence>("Buffer").first;

        // Inicializar estructura compartida
        init_empty_struct(sharedStruct, numChars);

        // Visualizar bloque de memoria
        while (true) {
            sem_read.wait();
            std::cout << "\033[0;0H\033[2J"; // Mover el cursor a la posición (0, 0) y borrar la pantalla
            std::cout.flush();
            for (int i = 0; i < numChars; i++) {
                std::cout << "buffer[" << i << "] = \"" << buffer[i].character << "\" | time: " << buffer[i].time << std::endl;
            }

            std::cout << "-------------------------------------------------" << std::endl;
            std::cout.flush();

            sem_write.post();

            printResourceUsage();
        }
    } catch (const bip::interprocess_exception &e) {
        std::cerr << "Boost.Interprocess exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
