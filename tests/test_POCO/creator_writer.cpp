#include <Poco/SharedMemory.h>
#include <Poco/Exception.h>
#include <Poco/File.h>
#include <Poco/Semaphore.h>
#include <iostream>
#include <cstring>
#include <string>

const std::string FILE_NAME = "shared_memory.dat";
const std::size_t SHARED_MEMORY_SIZE = 65536; // Tamaño de la memoria compartida
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

        std::string inputData = "ABCDEFGHIJ"; // Cadena de 10 caracteres
        if (inputData.size() > 10) {
            std::cerr << "Data size exceeds 10 characters." << std::endl;
            return -1;
        }

        for (int i = 0; i < 10; ++i) {
            writeSemaphore.wait(); // Esperar semáforo para acceder a la memoria

            std::memcpy(sharedMemory.begin(), inputData.data(), inputData.size());
            std::memset(static_cast<char*>(sharedMemory.begin()) + inputData.size(), 0, SHARED_MEMORY_SIZE - inputData.size());

            std::cout << "Iteration " << i + 1 << ": Written '" << inputData << "' to shared memory." << std::endl;

            writeSemaphore.set(); // Liberar semáforo
        }

    } catch (Poco::Exception& ex) {
        std::cerr << "Poco exception: " << ex.displayText() << std::endl;
        return -1;
    }

    return 0;
}
