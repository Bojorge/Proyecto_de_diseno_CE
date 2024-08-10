#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
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

void insert_char(boost::interprocess::named_semaphore &sem_read, 
                 boost::interprocess::named_semaphore &sem_write, 
                 SharedData *sharedData, 
                 Sentence *buffer) 
{
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
        boost::interprocess::named_semaphore sem_var_write(boost::interprocess::open_only, sem_write_name.c_str());

        // Obtener el semáforo para el espacio de lectura
        std::string sem_read_name = std::string(SEM_READ_VARIABLE_FNAME) + std::to_string(sharedData->writeIndex);
        boost::interprocess::named_semaphore sem_var_read(boost::interprocess::open_only, sem_read_name.c_str());

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

    // Conectar a los semáforos que ya fueron creados
    named_semaphore sem_read(open_only, SEM_READ_PROCESS_FNAME);
    named_semaphore sem_write(open_only, SEM_WRITE_PROCESS_FNAME);

    // Conectar al bloque de memoria compartida para SharedData
    managed_shared_memory shm(create_only, STRUCT_LOCATION, 65536); // Ajusta el tamaño si es necesario
    SharedData *sharedData = shm.find<SharedData>("SharedData").first;
    if (sharedData == nullptr) {
        std::cerr << "ERROR: no se pudo acceder al bloque" << std::endl;
        return -1;
    }

    // Conectar al bloque de memoria compartida para Sentence
    managed_shared_memory shmBuffer(create_only, BUFFER_LOCATION, 65536); // Ajusta el tamaño si es necesario
    Sentence *buffer = shmBuffer.find<Sentence>("SentenceArray").first;
    if (buffer == nullptr) {
        std::cerr << "ERROR: no se pudo acceder al bloque" << std::endl;
        return -1;
    }

    insert_char(sem_read, sem_write, sharedData, buffer);

    // Desconectar de la memoria compartida después de finalizar
    // Destruir bloques de memoria si es necesario aquí (descomentar si se desea destruir automáticamente)
    // shared_memory_object::remove(STRUCT_LOCATION);
    // shared_memory_object::remove(BUFFER_LOCATION);

    return 0;
}
