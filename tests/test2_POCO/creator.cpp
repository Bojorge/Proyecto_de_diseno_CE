#include <Poco/SharedMemory.h>
#include <Poco/File.h>
#include <Poco/Exception.h>
#include <iostream>
#include <string>
#include "MySharedMemoryHandler.h"

const std::string FILE_NAME = "shared_memory.dat";

int main() {
    try {
        // Crear las instancias necesarias
        const std::string memoryName = "shared_memory";
        Poco::File file(FILE_NAME);

        // Instanciar el objeto MySharedMemoryHandler
        MySharedMemoryHandler handler(memoryName, file);

        // Usar el objeto
        handler.displayInfo();

    } catch (Poco::Exception& ex) {
        std::cerr << "Poco exception: " << ex.displayText() << std::endl;
        return -1;
    }


    return 0;
}
