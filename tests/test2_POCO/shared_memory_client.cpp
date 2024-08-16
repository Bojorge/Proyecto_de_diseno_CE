#include <Poco/File.h>
#include "Poco/Poco.h"
#include <Poco/Exception.h>
#include <iostream>
#include <string>
#include "MySharedMemoryHandler.h"

const std::string FILE_NAME = "shared_memory.dat";
const std::string SHARED_MEMORY_NAME = "shared_memory";
const std::size_t MEMORY_SIZE = 65536;

int main() {
    try {
        // Conectar al archivo para la memoria compartida
        Poco::File file(FILE_NAME);
        if (!file.exists()) {
            std::cerr << "Shared memory file does not exist. Please run the manager first." << std::endl;
            return -1;
        }

        // Crear un manejador de memoria compartida usando el constructor con archivo
        MySharedMemoryHandler handlerFile(file, Poco::SharedMemory::AM_READ | Poco::SharedMemory::AM_WRITE);

        // Crear un manejador de memoria compartida usando el constructor con nombre y tamaño
        MySharedMemoryHandler handlerName(SHARED_MEMORY_NAME, MEMORY_SIZE, Poco::SharedMemory::AM_READ | Poco::SharedMemory::AM_WRITE);

        // Crear un manejador de memoria compartida usando el constructor por defecto
        MySharedMemoryHandler handlerDefault;

        // Crear un manejador de memoria compartida usando el constructor de copia
        MySharedMemoryHandler handlerCopy(handlerName);

        // Mostrar la información de cada manejador
        handlerFile.displayInfo();
        handlerName.displayInfo();
        handlerDefault.displayInfo();
        handlerCopy.displayInfo();

        std::string option;
        std::string inputData;

        while (true) {
            std::cout << "Select Option (write/read/exit): ";
            std::cin >> option;
            std::cin.ignore(); // Limpiar el buffer de entrada

            if (option == "write") {
                std::cout << "Enter text to write: ";
                std::getline(std::cin, inputData);
                handlerFile.writeToMemory(inputData); // Usando handlerFile para escribir
            } else if (option == "read") {
                std::cout << "Shared Memory Content: " << handlerFile.readFromMemory() << std::endl; // Usando handlerFile para leer
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
