#include <iostream>
#include <thread>
#include <boost/asio.hpp>
#include "sockets.hpp"

const char* PORT = "12345";

void server_thread() {
    start_server(PORT);
}

int main() {
    try {
        auto start = std::chrono::high_resolution_clock::now();
    
        std::thread server(server_thread);
        server.join();

        // Registrar el tiempo de fin
        auto end = std::chrono::high_resolution_clock::now();
        // Calcular la duración
        std::chrono::duration<double> duration = end - start;
        std::cout << "El programa tardó " << duration.count() << " segundos en ejecutarse." << std::endl;

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
