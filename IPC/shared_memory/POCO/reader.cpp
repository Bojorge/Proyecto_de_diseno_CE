#include <Poco/SharedMemory.h>
#include <Poco/File.h>
#include <Poco/Exception.h>
#include <Poco/Semaphore.h>
#include <iostream>
#include <cstring>
#include <string>
#include <chrono>
#include <thread>

const std::string FILE_NAME = "shared_memory.dat";
const std::size_t SHARED_MEMORY_SIZE = 65536; // Tamaño de la memoria compartida

int main() {
    try {
        // Esperar 1 segundo antes de iniciar
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // Crear el archivo si no existe
        Poco::File file(FILE_NAME);
        if (!file.exists()) {
            file.createFile();
        }

        // Establecer el tamaño del archivo
        file.setSize(SHARED_MEMORY_SIZE);

        // Abrir la memoria compartida en modo de solo lectura
        Poco::SharedMemory sharedMemory(file, Poco::SharedMemory::AM_READ);

        // Definir la dirección del bloque en memoria compartida
        // Nota: En POCO no se usa un "handle" como en Boost. Usamos una dirección fija.
        void *msg = sharedMemory.begin(); 

        for (int i = 0; i < 10; i++) {
            std::cout << "READING -> ";
            std::cout << static_cast<char*>(msg) << std::endl;

            // Esperar 1 segundo antes de intentar nuevamente
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    } catch (Poco::Exception& ex) {
        std::cerr << "Poco exception: " << ex.displayText() << std::endl;
        return -1;
    }

    return 0;
}
