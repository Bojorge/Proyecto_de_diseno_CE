#include <Poco/SharedMemory.h>
#include <Poco/Exception.h>
#include <Poco/File.h>
#include <iostream>
#include <cstring>  // For std::memcpy
#include <string>

const std::string FILE_NAME = "shared_memory.dat";
const std::size_t SHARED_MEMORY_SIZE = 65536; // Tamaño conocido de la memoria compartida

int main() {
    try {
        Poco::File file(FILE_NAME);
        if (!file.exists()) {
            file.createFile(); // Crear el archivo si no existe
        }

        // Establecer el tamaño del archivo
        file.setSize(SHARED_MEMORY_SIZE);

        Poco::SharedMemory sharedMemory(file, Poco::SharedMemory::AM_WRITE);

        std::string option;
        std::string inputData;

        while (true) {
            std::cout << "Select Option (write/read/exit): ";
            std::cin >> option;
            std::cin.ignore(); // Limpiar el buffer de entrada

            if (option == "write") {
                std::cout << "Enter text to write: ";
                std::getline(std::cin, inputData);
                if (inputData.size() > SHARED_MEMORY_SIZE) {
                    std::cerr << "Data size exceeds shared memory size." << std::endl;
                    continue;
                }
                std::memcpy(sharedMemory.begin(), inputData.data(), inputData.size());
                std::memset(static_cast<char*>(sharedMemory.begin()) + inputData.size(), 0, SHARED_MEMORY_SIZE - inputData.size()); // Rellenar el resto
            } else if (option == "read") {
                std::string content(static_cast<char*>(sharedMemory.begin()), SHARED_MEMORY_SIZE);
                std::cout << "Shared Memory Content: " << content.c_str() << std::endl;
            } else if (option == "exit") {
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
