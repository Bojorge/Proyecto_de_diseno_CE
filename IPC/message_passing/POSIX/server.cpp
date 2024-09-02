#include <iostream>
#include <thread>
#include "sockets.hpp"

void server_thread() {
    start_server("12345");
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::thread server(server_thread);
    server.join();

    // Registrar el tiempo de fin
    auto end = std::chrono::high_resolution_clock::now();
    // Calcular la duración
    std::chrono::duration<double> duration = end - start;
    std::cout << "El programa tardó " << duration.count() << " segundos en ejecutarse." << std::endl;

    return 0;
}
