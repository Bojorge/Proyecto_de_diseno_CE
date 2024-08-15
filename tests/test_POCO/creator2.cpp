#include "Poco/SharedMemory.h"
#include <Poco/Exception.h>
#include <iostream>
#include <cstring>

const std::string MEMORY_NAME = "shared_memory";
const std::size_t MEMORY_SIZE = 65536;

int main() {
    try {
        // Crear o conectar a un segmento de memoria compartida por nombre
        Poco::SharedMemory sharedMemory(MEMORY_NAME, MEMORY_SIZE, Poco::SharedMemory::AM_READ | Poco::SharedMemory::AM_WRITE, 0, true);

        // Inicializar la memoria compartida
        std::memset(sharedMemory.begin(), 0, MEMORY_SIZE);

        // Interacción del usuario
        char opt[10];
        char buffer[1024];

        while (true) {
            std::cout << "Select Option (write/read/exit): ";
            std::cin.getline(opt, sizeof(opt));

            if (std::strcmp(opt, "write") == 0) {
                std::cout << "Enter text to write: ";
                std::cin.getline(buffer, sizeof(buffer));
                std::memcpy(sharedMemory.begin(), buffer, std::strlen(buffer));
            } else if (std::strcmp(opt, "read") == 0) {
                std::cout << "Shared Memory Content: ";
                std::cout << static_cast<char*>(sharedMemory.begin()) << std::endl;
            } else if (std::strcmp(opt, "exit") == 0) {
                break;
            } else {
                std::cout << "Invalid option." << std::endl;
            }
        }
    } catch (Poco::Exception& ex) {
        std::cerr << "Poco exception: " << ex.displayText() << std::endl;
        return -1;
    }

    return 0;
}
