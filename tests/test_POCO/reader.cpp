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
            std::cerr << "Shared memory file does not exist. Please run the creator first." << std::endl;
            return -1;
        }

        Poco::SharedMemory sharedMemory(file, Poco::SharedMemory::AM_READ);

        std::string content(static_cast<char*>(sharedMemory.begin()), SHARED_MEMORY_SIZE);
        std::cout << "Shared Memory Content: " << content.c_str() << std::endl;

    } catch (Poco::Exception& ex) {
        std::cerr << "Poco exception: " << ex.displayText() << std::endl;
        return -1;
    }

    return 0;
}
