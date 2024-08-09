#ifndef POSIX_SOCKETS_H
#define POSIX_SOCKETS_H

#ifdef __cplusplus
extern "C" {
#endif

void start_server(const char* port);
void send_message(const char* server_ip, const char* port, const char* message);
long getRAMUsage();
void getCPUUsage(double &userCPU, double &systemCPU);

#ifdef __cplusplus
}
#endif

#endif // POSIX_SOCKETS_H
