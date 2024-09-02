#include <Poco/Net/StreamSocket.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/Net/SocketStream.h>
#include <Poco/Exception.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <thread>
#include "sockets.hpp"  // Asegúrate de que este archivo incluya las declaraciones necesarias

using Poco::Net::StreamSocket;
using Poco::Net::SocketAddress;
using Poco::Net::SocketStream;
using Poco::Exception;

int main() {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto start = std::chrono::high_resolution_clock::now();

    const int num_iterations = 1000;
    std::string message;

    // Variables para almacenar los máximos valores
    long maxRAMUsage = 0;
    double maxUserCPU = 0.0;
    double maxSystemCPU = 0.0;

    for (int i = 0; i < num_iterations; ++i) {
        message = "mensaje #" + std::to_string(i);
        std::cout << "Enviando: " << message << std::endl;

        try {
            // Llama a la función send_message desde sockets.hpp
            send_message("127.0.0.1", "12345", message.c_str());
        } catch (const Exception& e) {
            std::cerr << "Error al enviar mensaje: " << e.displayText() << std::endl;
        }

        // Medir y actualizar los máximos valores
        long ramUsage = getRAMUsage();
        double userCPU, systemCPU;
        getCPUUsage(userCPU, systemCPU);

        if (ramUsage > maxRAMUsage) maxRAMUsage = ramUsage;
        if (userCPU > maxUserCPU) maxUserCPU = userCPU;
        if (systemCPU > maxSystemCPU) maxSystemCPU = systemCPU;
    }
    // Registrar el tiempo de fin
    auto end = std::chrono::high_resolution_clock::now();
    // Calcular la duración
    std::chrono::duration<double> duration = end - start;
    std::cout << "El programa tardó " << duration.count() << " segundos en ejecutarse." << std::endl;

    std::cout << "-------------------------------" << std::endl;
    std::cout << "RAM: " << maxRAMUsage << " KB" << std::endl;
    std::cout << "CPU usuario: " << maxUserCPU << " s" << std::endl;
    std::cout << "CPU sistema: " << maxSystemCPU << " s" << std::endl;

    return 0;
}
