#include "shared_memory.hpp"
#include <thread>
#include <chrono>
#include <iostream>

int main() {
    try {
        create_shared_memory();
        std::cout << "Shared memory created successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to create shared memory: " << e.what() << std::endl;
        return 1;
    }

    int count = 0;
    while (true) {
        try {
            std::string message = "Message " + std::to_string(count++);
            write_to_shared_memory(message);
            std::cout << "Produced: " << message << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Exception while writing to shared memory: " << e.what() << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1)); // Simula el tiempo de producción
    }

    return 0;
}
