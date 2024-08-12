#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
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

void create_and_init_semaphores(const char *segment_name, int numChars) {
    try {
        // Crear e inicializar semáforos globales
        bip::shared_memory_object shm_sems(bip::create_only, segment_name, bip::read_write);
        shm_sems.truncate(65536); // Ajustar tamaño según sea necesario
        bip::mapped_region region_sems(shm_sems, bip::read_write);
        
        bip::interprocess_semaphore* sem_read = new (region_sems.get_address()) bip::interprocess_semaphore(1);
        bip::interprocess_semaphore* sem_write = new (region_sems.get_address() + sizeof(bip::interprocess_semaphore)) bip::interprocess_semaphore(0);

        // Crear semáforos para cada espacio en el buffer
        for (int i = 0; i < numChars; i++) {
            std::string sem_write_name = std::string("SemWriteVar") + std::to_string(i);
            std::string sem_read_name = std::string("SemReadVar") + std::to_string(i);

            bip::interprocess_semaphore* sem_write_var = new (region_sems.get_address() + sizeof(bip::interprocess_semaphore) * 2 + i * sizeof(bip::interprocess_semaphore)) bip::interprocess_semaphore(1);
            bip::interprocess_semaphore* sem_read_var = new (region_sems.get_address() + sizeof(bip::interprocess_semaphore) * 2 + (numChars + i) * sizeof(bip::interprocess_semaphore)) bip::interprocess_semaphore(0);
        }
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Error al crear e inicializar semáforos: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

void destroy_semaphores(const char *segment_name, int numChars) {
    try {
        bip::shared_memory_object shm_sems(bip::open_only, segment_name, bip::read_write);
        bip::mapped_region region_sems(shm_sems, bip::read_write);
        
        // Destruir semáforos globales
        bip::interprocess_semaphore* sem_read = static_cast<bip::interprocess_semaphore*>(region_sems.get_address());
        bip::interprocess_semaphore* sem_write = static_cast<bip::interprocess_semaphore*>(region_sems.get_address() + sizeof(bip::interprocess_semaphore));

        sem_read->~interprocess_semaphore();
        sem_write->~interprocess_semaphore();

        // Destruir semáforos para cada espacio en el buffer
        for (int i = 0; i < numChars; i++) {
            bip::interprocess_semaphore* sem_write_var = static_cast<bip::interprocess_semaphore*>(region_sems.get_address() + sizeof(bip::interprocess_semaphore) * 2 + i * sizeof(bip::interprocess_semaphore));
            bip::interprocess_semaphore* sem_read_var = static_cast<bip::interprocess_semaphore*>(region_sems.get_address() + sizeof(bip::interprocess_semaphore) * 2 + (numChars + i) * sizeof(bip::interprocess_semaphore));

            sem_write_var->~interprocess_semaphore();
            sem_read_var->~interprocess_semaphore();
        }
        
        // Eliminar el segmento de memoria compartida
        bip::shared_memory_object::remove(segment_name);
    } catch (const bip::interprocess_exception& e) {
        std::cerr << "Error al destruir semáforos: " << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    int numChars = 10;

    // Nombre del segmento de semáforos
    const char *sems_segment_name = "SemaphoresSegment";

    // Crear e inicializar semáforos
    create_and_init_semaphores(sems_segment_name, numChars);

    // Inicializar bloques de memoria compartida
    init_mem_block(STRUCT_LOCATION, BUFFER_LOCATION, sizeof(SharedData), numChars * sizeof(Sentence));

    std::cout << " ---  AQUI  ---" << std::endl;
    
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

    // Obtener semáforos desde el segmento de memoria compartida
    bip::shared_memory_object shm_sems(bip::open_only, sems_segment_name, bip::read_write);
    bip::mapped_region region_sems(shm_sems, bip::read_write);

    bip::interprocess_semaphore* sem_read = static_cast<bip::interprocess_semaphore*>(region_sems.get_address());
    bip::interprocess_semaphore* sem_write = static_cast<bip::interprocess_semaphore*>(region_sems.get_address() + sizeof(bip::interprocess_semaphore));

    // Visualizar el bloque de memoria
    while (true) {
        sem_read->wait();  // Esperar en el semáforo de lectura

        std::cout << "\033[0;0H\033[2J"; // Mover el cursor a la posición (0, 0) y borrar la pantalla
        std::cout.flush();
        for (int i = 0; i < numChars; i++) {
            std::cout << "buffer[" << i << "] = \"" << buffer[i].character << "\" | time: " << buffer[i].time << std::endl;
        }

        std::cout << "-------------------------------------------------" << std::endl;
        std::cout.flush();

        sem_write->post(); // Liberar el semáforo de escritura

        printResourceUsage();
    }

    // Destruir semáforos y liberar recursos
    destroy_semaphores(sems_segment_name, numChars);
    destroy_memory_block(STRUCT_LOCATION);
    destroy_memory_block(BUFFER_LOCATION);

    return 0;
}
