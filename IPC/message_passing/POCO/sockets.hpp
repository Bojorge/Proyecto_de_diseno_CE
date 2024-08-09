#ifndef SOCKETS_HPP
#define SOCKETS_HPP

#include <Poco/Net/StreamSocket.h>
#include <Poco/Net/ServerSocket.h>
#include <Poco/Net/SocketStream.h>

// Función para iniciar el servidor usando POCO
void start_server(const char* port);

// Función para enviar un mensaje usando POCO
void send_message(const char* server_ip, const char* port, const char* message);

// Funciones para obtener el uso de RAM y CPU (implementación permanece igual)
long getRAMUsage();
void getCPUUsage(double &userCPU, double &systemCPU);

#endif // SOCKETS_HPP
