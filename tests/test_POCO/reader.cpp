#include <Poco/SharedMemory.h>
#include <Poco/Exception.h>
#include <Poco/File.h>
#include <Poco/Semaphore.h>
#include <iostream>
#include <string>

const std::string FILE_NAME = "shared_memory.dat";
const std::size_t SHARED_MEMORY_SIZE = 65536; // Tamaño de la memoria compartida
Poco::Semaphore writeSemaphore(1, 1); // Semáforo de escritura

int main() {
    try {
        Poco::File file(FILE_NAME);
        if (!file.exists()) {
            std::cerr << "Shared memory file does not exist. Please run the writer first." << std::endl;
            return -1;
        }

        Poco::SharedMemory sharedMemory(file, Poco::SharedMemory::AM_READ);

        for (int i = 0; i < 10; ++i) {
            writeSemaphore.wait(); // Esperar a que el escritor libere el semáforo

            std::string content(static_cast<char*>(sharedMemory.begin()), SHARED_MEMORY_SIZE);
            std::cout << "Read " << (i + 1) << ": Shared Memory Content: " << content.c_str() << std::endl;

            writeSemaphore.set(); // Liberar el semáforo para que el escritor pueda escribir nuevamente

            // Esperar un corto tiempo para evitar lecturas demasiado rápidas consecutivas
            //std::cout << "Press enter to read the next or wait automatically: ";
            //std::cin.ignore();
        }

    } catch (Poco::Exception& ex) {
        std::cerr << "Poco exception: " << ex.displayText() << std::endl;
        return -1;
    }

    return 0;
}
