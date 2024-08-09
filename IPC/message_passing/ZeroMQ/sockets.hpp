#ifndef SOCKETS_HPP
#define SOCKETS_HPP

#include <string>

// Función para iniciar el servidor usando ZeroMQ
void start_server(const char* port);

// Función para enviar un mensaje usando ZeroMQ
void send_message(const char* server_ip, const char* port, const char* message);

// Funciones para obtener el uso de RAM y CPU
long getRAMUsage();
void getCPUUsage(double &userCPU, double &systemCPU);

#endif // SOCKETS_HPP
