#ifndef SOCKETS_HPP
#define SOCKETS_HPP

#include <boost/asio.hpp>

// Función para iniciar el servidor usando Boost.Asio
void start_server(const char* port);

// Función para enviar un mensaje usando Boost.Asio
void send_message(const char* server_ip, const char* port, const char* message);

// Funciones para obtener el uso de RAM y CPU (implementación permanece igual)
long getRAMUsage();
void getCPUUsage(double &userCPU, double &systemCPU);

#endif // SOCKETS_HPP
