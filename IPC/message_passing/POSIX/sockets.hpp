#ifndef SOCKETS_HPP
#define SOCKETS_HPP

void start_server(const char* port);
void send_message(const char* server_ip, const char* port, const char* message);
long getRAMUsage();
void getCPUUsage(double &userCPU, double &systemCPU);

#endif // SOCKETS_HPP
