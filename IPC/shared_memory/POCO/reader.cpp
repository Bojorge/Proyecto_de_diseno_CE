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
const std::size_t SHARED_MEMORY_SIZE = 65536;

int main() {
    try {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Crear el archivo si no existe
        Poco::File file(FILE_NAME);
        if (!file.exists()) {
            file.createFile();
        }

        file.setSize(SHARED_MEMORY_SIZE);

        // Abrir la memoria compartida en modo de solo lectura
        Poco::SharedMemory sharedMemory(file, Poco::SharedMemory::AM_READ);

        void *msg = sharedMemory.begin(); 

        for (int i = 0; i < 10; i++) {
            std::cout << "READING -> ";
            std::cout << static_cast<char*>(msg) << std::endl;

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } catch (Poco::Exception& ex) {
        std::cerr << "Poco exception: " << ex.displayText() << std::endl;
        return -1;
    }

    return 0;
}
