#include <Poco/SharedMemory.h>
#include <Poco/Exception.h>
#include <Poco/File.h>
#include <Poco/Semaphore.h>
#include <iostream>
#include <cstring>
#include <string>
#include <chrono> // Para std::chrono::seconds
#include <thread> // Para std::this_thread::sleep_for

const std::string FILE_NAME = "shared_memory.dat";
const std::size_t SHARED_MEMORY_SIZE = 65536; // Tamaño de la memoria compartida
const std::size_t BLOCK_SIZE = 1024; // Tamaño del bloque de datos
Poco::Semaphore writeSemaphore(1, 1); // Semáforo de escritura (1 valor inicial)

int main() {
    try {
        Poco::File file(FILE_NAME);
        if (!file.exists()) {
            file.createFile(); // Crear el archivo si no existe
        }

        // Establecer el tamaño del archivo
        file.setSize(SHARED_MEMORY_SIZE);

        Poco::SharedMemory sharedMemory(file, Poco::SharedMemory::AM_WRITE);

        for (int i = 0; i < 10; ++i) {
            // Construir el mensaje para escribir en la memoria compartida
            std::string sharedData = "ABCDEFGH " + std::to_string(i);
            std::cout << "WRITING <- " << sharedData << std::endl;

            // Esperar semáforo para acceder a la memoria
            writeSemaphore.wait();

            // Limpiar el bloque de memoria compartida y copiar los datos
            std::memset(sharedMemory.begin(), '\0', BLOCK_SIZE);
            std::memcpy(sharedMemory.begin(), sharedData.data(), sharedData.size());

            // Liberar semáforo
            writeSemaphore.set();

            // Esperar 1 segundo antes de intentar nuevamente
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

    } catch (Poco::Exception& ex) {
        std::cerr << "Poco exception: " << ex.displayText() << std::endl;
        return -1;
    }

    return 0;
}
