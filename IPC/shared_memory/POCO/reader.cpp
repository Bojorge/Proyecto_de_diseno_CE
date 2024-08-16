#include <Poco/SharedMemory.h>
#include <Poco/File.h>
#include <Poco/Exception.h>
#include <Poco/Semaphore.h>
#include <iostream>
#include <cstring>
#include <string>
#include <chrono>
#include <thread>

const std::string SHARED_MEMORY_NAME = "MySharedMemory";
const std::size_t SHARED_MEMORY_SIZE = 65536;

int main() {
    try {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        Poco::SharedMemory sharedMemory(SHARED_MEMORY_NAME, SHARED_MEMORY_SIZE, Poco::SharedMemory::AM_READ, nullptr, false);

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
