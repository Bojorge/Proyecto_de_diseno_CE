#include "Poco/SharedMemory.h"
#include "Poco/File.h"
#include <iostream>
#include <cstring>

const std::string FILE_NAME = "shared_memory.dat";
const std::size_t MEMORY_SIZE = 65536;

int main() {
    try {
        // Crear un archivo y establecer su tamaño
        Poco::File file(FILE_NAME);
        if (!file.exists()) {
            file.createFile();
        }
        file.setSize(MEMORY_SIZE); // Establece el tamaño del archivo

        // Mapear el archivo a memoria compartida
        Poco::SharedMemory sharedMemory(file, Poco::SharedMemory::AM_READ | Poco::SharedMemory::AM_WRITE);

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
