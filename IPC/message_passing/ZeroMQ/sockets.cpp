#include "sockets.hpp"
#include <iostream>
#include <zmq.hpp>
#include <thread>
#include <vector>
#include <sys/resource.h>

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

    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    std::string endpoint = "tcp://*:" + std::string(port);
    socket.bind(endpoint.c_str());

    // Variables para almacenar los máximos valores
    long maxRAMUsage = 0;
    double maxUserCPU = 0.0;
    double maxSystemCPU = 0.0;

    std::cout << "Server listening on port " << port << std::endl;

    while (message_count < num_iterations) {
        zmq::message_t request;

        // Esperar a recibir el mensaje
        zmq::recv_result_t result = socket.recv(request, zmq::recv_flags::none);
        if (!result.has_value()) {
            std::cerr << "Error receiving message." << std::endl;
            continue;
        }

        std::string recv_msg(static_cast<char*>(request.data()), request.size());
        std::cout << "\n * Mensaje recibido >>> " << recv_msg << std::endl;

        // Responder al cliente
        std::string reply_msg = "Message received";
        zmq::message_t reply(reply_msg.size());
        memcpy(reply.data(), reply_msg.data(), reply_msg.size());
        socket.send(reply, zmq::send_flags::none);

        // Medir y actualizar los máximos valores
        long ramUsage = getRAMUsage();
        double userCPU, systemCPU;
        getCPUUsage(userCPU, systemCPU);

        if (ramUsage > maxRAMUsage) maxRAMUsage = ramUsage;
        if (userCPU > maxUserCPU) maxUserCPU = userCPU;
        if (systemCPU > maxSystemCPU) maxSystemCPU = systemCPU;

        message_count++;
    }

    std::cout << "\n *** Se han recibido " << num_iterations << " mensajes. Cerrando servidor..." << std::endl;
    std::cout << "RAM: " << maxRAMUsage << " KB" << std::endl;
    std::cout << "CPU usuario: " << maxUserCPU << " s" << std::endl;
    std::cout << "CPU sistema: " << maxSystemCPU << " s" << std::endl;
}

void send_message(const char* server_ip, const char* port, const char* message) {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REQ);
    std::string endpoint = "tcp://" + std::string(server_ip) + ":" + std::string(port);
    socket.connect(endpoint.c_str());

    zmq::message_t request(strlen(message));
    memcpy(request.data(), message, strlen(message));
    socket.send(request, zmq::send_flags::none);

    zmq::message_t reply;
    zmq::recv_result_t result = socket.recv(reply, zmq::recv_flags::none);
    if (!result.has_value()) {
        std::cerr << "Error receiving reply." << std::endl;
        return;
    }

    //std::string reply_msg(static_cast<char*>(reply.data()), reply.size());
    //std::cout << "Received reply: " << reply_msg << std::endl;
}
