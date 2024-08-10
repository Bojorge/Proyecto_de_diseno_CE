#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
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

void insert_char(bip::named_semaphore &sem_read, bip::named_semaphore &sem_write, SharedData *sharedData, Sentence *buffer) {
    char character;

    // Variables para manejar la entrada dinámica
    std::string dynamic_string;

    while (true) {
        // Esperar hasta que el semáforo de escritura esté disponible
        sem_write.wait();

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
        bip::named_semaphore sem_var_write(bip::open_only, sem_write_name.c_str());

        // Obtener el semáforo para el espacio de lectura
        std::string sem_read_name = std::string(SEM_READ_VARIABLE_FNAME) + std::to_string(sharedData->writeIndex);
        bip::named_semaphore sem_var_read(bip::open_only, sem_read_name.c_str());

        // Esperar el semáforo para el espacio de escritura
        sem_var_write.wait();

        // Obtener la marca de tiempo actual en el formato deseado
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

        // Post para que la variable pueda ser leída
        sem_var_read.post();
        sem_read.post();

        printResourceUsage();
    }

    sharedData->writingFinished = true;
}

int main() 
{
    using namespace boost::interprocess;

    // Crear y abrir semáforos
    named_semaphore sem_read(open_only, SEM_READ_PROCESS_FNAME);
    named_semaphore sem_write(open_only, SEM_WRITE_PROCESS_FNAME);

    int numChars = 10;

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

    // Escribir datos en el buffer
    for (int i = 0; i < numChars; i++) {
        sem_write.wait(); // Esperar en el semáforo de escritura

        // Escribir datos en el buffer (ejemplo simple)
        buffer[i].character = 'A' + (i % 26);
        std::sprintf(buffer[i].time, "%d", i);

        sem_read.post(); // Liberar el semáforo de lectura
    }

    // Liberar recursos
    sem_read.post(); // Asegurarse de que el semáforo de lectura se libera al final

    // No es necesario desconectar explícitamente en Boost.Interprocess

    // Eliminar el segmento de memoria compartida y semáforos
    destroy_memory_block(STRUCT_LOCATION);
    destroy_memory_block(BUFFER_LOCATION);

    // Cerrar los semáforos
    sem_unlink(SEM_READ_PROCESS_FNAME);
    sem_unlink(SEM_WRITE_PROCESS_FNAME);
    for (int i = 0; i < numChars; i++) {
        std::string sem_write_name = std::string(SEM_WRITE_VARIABLE_FNAME) + std::to_string(i);
        std::string sem_read_name = std::string(SEM_READ_VARIABLE_FNAME) + std::to_string(i);
        sem_unlink(sem_write_name.c_str());
        sem_unlink(sem_read_name.c_str());
    }

    return 0;
}
