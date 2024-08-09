#include <Poco/Net/ServerSocket.h>
#include <Poco/Net/StreamSocket.h>
#include <Poco/Net/SocketStream.h>
#include <Poco/Exception.h>
#include <iostream>
#include <thread>
#include <vector>
#include <sys/resource.h>
#include "sockets.hpp"

using Poco::Net::ServerSocket;
using Poco::Net::StreamSocket;
using Poco::Net::SocketStream;

// Funciones para obtener el uso de RAM y CPU
long getRAMUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss; // Devuelve el uso máximo de RAM en kilobytes
}

void getCPUUsage(double &userCPU, double &systemCPU) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    userCPU = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;  // tiempo de CPU en modo usuario en segundos
    systemCPU = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6; // tiempo de CPU en modo sistema en segundos
}

void start_server(const char* port) {
    const int num_iterations = 1000;
    int message_count = 0;

    // Convertir el puerto de const char* a int
    int portNumber = std::stoi(port);

    // Crear el socket del servidor
    ServerSocket serverSocket(portNumber);

    // Variables para almacenar los máximos valores
    long maxRAMUsage = 0;
    double maxUserCPU = 0.0;
    double maxSystemCPU = 0.0;

    // Función para manejar cada cliente
    auto handle_client = [&maxRAMUsage, &maxUserCPU, &maxSystemCPU](StreamSocket socket) {
        try {
            char data[1024];
            int bytesReceived = socket.receiveBytes(data, sizeof(data));

            while (bytesReceived > 0) {
                std::cout << "\n * Mensaje recibido >>> " << std::string(data, bytesReceived) << std::endl;
                bytesReceived = socket.receiveBytes(data, sizeof(data));
            }
        } catch (Poco::Exception& e) {
            std::cerr << "Exception in session: " << e.displayText() << std::endl;
        }
    };

    // Hilo para medir el uso máximo de recursos
    std::thread monitor_thread([&maxRAMUsage, &maxUserCPU, &maxSystemCPU]() {
        while (true) {
            long ramUsage = getRAMUsage();
            double userCPU, systemCPU;
            getCPUUsage(userCPU, systemCPU);

            if (ramUsage > maxRAMUsage) maxRAMUsage = ramUsage;
            if (userCPU > maxUserCPU) maxUserCPU = userCPU;
            if (systemCPU > maxSystemCPU) maxSystemCPU = systemCPU;

            std::this_thread::sleep_for(std::chrono::seconds(1)); // Intervalo de muestreo
        }
    });

    std::cout << "Server listening on port " << port << std::endl;

    while (message_count < num_iterations) {
        try {
            // Aceptar conexión
            StreamSocket clientSocket = serverSocket.acceptConnection();
            
            // Llamar a la función handle_client en un nuevo hilo
            std::thread([handle_client = std::move(handle_client), socket = std::move(clientSocket)]() mutable {
                handle_client(std::move(socket));
            }).detach();

            message_count++;
        } catch (Poco::Exception& e) {
            std::cerr << "Server exception: " << e.displayText() << std::endl;
        }
    }

    // Esperar a que el hilo de monitoreo termine (esto podría mejorarse si es necesario)
    monitor_thread.detach();

    std::cout << "\n *** Se han recibido " << num_iterations << " mensajes. Cerrando servidor..." << std::endl;
    std::cout << "RAM: " << maxRAMUsage << " KB" << std::endl;
    std::cout << "CPU usuario: " << maxUserCPU << " s" << std::endl;
    std::cout << "CPU sistema: " << maxSystemCPU << " s" << std::endl;
}

void send_message(const char* server_ip, const char* port, const char* message) {
    try {
        Poco::Net::SocketAddress address(server_ip, std::stoi(port));
        Poco::Net::StreamSocket socket(address);
        Poco::Net::SocketStream stream(socket);

        stream << message << std::flush;
    } catch (Poco::Exception& e) {
        std::cerr << "Exception: " << e.displayText() << std::endl;
    }
}
