#include <Poco/Net/ServerSocket.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Net/SocketStream.h>
#include <Poco/Exception.h>
#include <iostream>
#include <thread>
#include "sockets.hpp"

using Poco::Net::ServerSocket;
using Poco::Net::StreamSocket;
using Poco::Net::SocketStream;
using Poco::Exception;

const char* PORT = "12345";

void server_thread() {
    start_server(PORT);
}

int main() {
    try {
        auto start = std::chrono::high_resolution_clock::now();

        // Crear y ejecutar el hilo del servidor
        std::thread server(server_thread);
        server.join(); 

        // Registrar el tiempo de fin
        auto end = std::chrono::high_resolution_clock::now();
        // Calcular la duración
        std::chrono::duration<double> duration = end - start;
        std::cout << "El programa tardó " << duration.count() << " segundos en ejecutarse." << std::endl;
    } catch (const Poco::Exception& e) {
        std::cerr << "Poco Exception: " << e.displayText() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception: " << e.what() << std::endl;
    }

    return 0;
}
