#include "Poco/SharedMemory.h"
#include <Poco/Exception.h>
#include "Poco/File.h"
#include <iostream>
#include <cstring> // Para memset, memcpy
#include <cstdlib> // Para exit

const std::string FILE_NAME = "shared_memory.dat";
const std::size_t MEMORY_SIZE = 65536;

int main() {
    try {
        // Abre el archivo existente para la memoria compartida
        Poco::File file(FILE_NAME);

        if (!file.exists()) {
            std::cerr << "Shared memory file does not exist. Please run the creator first." << std::endl;
            return -1;
        }

        // Se conecta a un segmento de memoria compartida basado en el archivo
        Poco::SharedMemory sharedMemory(file, Poco::SharedMemory::AM_READ | Poco::SharedMemory::AM_WRITE);

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
