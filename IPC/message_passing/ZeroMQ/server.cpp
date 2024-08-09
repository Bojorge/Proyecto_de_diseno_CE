#include <iostream>
#include <thread>
#include <boost/asio.hpp>
#include "sockets.hpp"

const char* PORT = "12345";

void server_thread() {
    start_server(PORT);
}

int main() {
    try {
        std::thread server(server_thread);
        server.join();
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
