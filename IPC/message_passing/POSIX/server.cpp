#include <iostream>
#include <thread>
#include "posix_sockets.h"

void server_thread() {
    start_server("12345");
}

int main() {
    std::thread server(server_thread);
    server.join();
    return 0;
}
