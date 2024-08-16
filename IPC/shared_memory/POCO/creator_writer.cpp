#include <Poco/SharedMemory.h>
#include <Poco/Exception.h>
#include <Poco/File.h>
#include <Poco/Semaphore.h>
#include <iostream>
#include <cstring>
#include <string>
#include <chrono>
#include <thread>

const std::string FILE_NAME = "shared_memory.dat";
const std::size_t SHARED_MEMORY_SIZE = 65536; 
const std::size_t BLOCK_SIZE = 1024;

int main() {
    try {
        Poco::File file(FILE_NAME);
        if (!file.exists()) {
            file.createFile(); // Crear el archivo si no existe
        }

        file.setSize(SHARED_MEMORY_SIZE);

        Poco::SharedMemory sharedMemory(file, Poco::SharedMemory::AM_WRITE);

        for (int i = 0; i < 10; ++i) {
            std::string sharedData = "MESSAGE #" + std::to_string(i);
            std::cout << "WRITING <- " << sharedData << std::endl;

            // Limpiar el bloque de memoria compartida y copiar los datos
            std::memset(sharedMemory.begin(), '\0', BLOCK_SIZE);
            std::memcpy(sharedMemory.begin(), sharedData.data(), sharedData.size());

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

    } catch (Poco::Exception& ex) {
        std::cerr << "Poco exception: " << ex.displayText() << std::endl;
        return -1;
    }

    return 0;
}
