#include "sockets.hpp"
#include <iostream>
#include <boost/asio.hpp>
#include <thread>
#include <vector>
#include <sys/resource.h>

using boost::asio::ip::tcp;

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

    boost::asio::io_context io_context;
    tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), std::stoi(port)));

    // Variables para almacenar los máximos valores
    long maxRAMUsage = 0;
    double maxUserCPU = 0.0;
    double maxSystemCPU = 0.0;

    // Función para manejar cada cliente
    auto handle_client = [&maxRAMUsage, &maxUserCPU, &maxSystemCPU](tcp::socket socket) {
        try {
            char data[1024];
            boost::system::error_code error;
            
            while (true) {
                size_t length = socket.read_some(boost::asio::buffer(data), error);
                if (error == boost::asio::error::eof) {
                    break; // Connection closed cleanly by peer.
                } else if (error) {
                    throw boost::system::system_error(error); // Some other error.
                }
                std::cout << "\n * Mensaje recibido >>> " << std::string(data, length) << std::endl;
            }
        } catch (std::exception& e) {
            std::cerr << "Exception in session: " << e.what() << std::endl;
        }
    };

    std::cout << "Server listening on port " << port << std::endl;

    while (message_count < num_iterations) {
        tcp::socket socket(io_context);
        acceptor.accept(socket);

        // Llama a la función handle_client en un nuevo hilo
        std::thread([handle_client = std::move(handle_client), socket = std::move(socket), &maxRAMUsage, &maxUserCPU, &maxSystemCPU]() mutable {
            handle_client(std::move(socket));

            // Medir y actualizar los máximos valores
            long ramUsage = getRAMUsage();
            double userCPU, systemCPU;
            getCPUUsage(userCPU, systemCPU);

            if (ramUsage > maxRAMUsage) maxRAMUsage = ramUsage;
            if (userCPU > maxUserCPU) maxUserCPU = userCPU;
            if (systemCPU > maxSystemCPU) maxSystemCPU = systemCPU;
        }).detach();

        message_count++;
    }
    std::cout << "\n *** Se han recibido " << num_iterations << " mensajes. Cerrando servidor..." << std::endl;

    std::cout << "RAM: " << maxRAMUsage << " KB" << std::endl;
    std::cout << "CPU usuario: " << maxUserCPU << " s" << std::endl;
    std::cout << "CPU sistema: " << maxSystemCPU << " s" << std::endl;
}


void send_message(const char* server_ip, const char* port, const char* message) {
    boost::asio::io_context io_context;
    tcp::resolver resolver(io_context);
    tcp::resolver::results_type endpoints = resolver.resolve(server_ip, port);

    tcp::socket socket(io_context);
    boost::asio::connect(socket, endpoints);

    boost::asio::write(socket, boost::asio::buffer(message, std::strlen(message)));
}
