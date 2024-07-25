#include "shared_memory.hpp"
#include <thread>
#include <chrono>
#include <iostream>

int main() {
    while (true) {
        try {
            std::string message = read_from_shared_memory();
            std::cout << "Consumed: " << message << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Exception while reading from shared memory: " << e.what() << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::seconds(2)); // Simula el tiempo de consumo
    }

    return 0;
}
