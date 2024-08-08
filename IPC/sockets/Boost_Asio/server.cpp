#include <iostream>
#include <thread>
#include <boost/asio.hpp>
#include "sockets.hpp"

void server_thread() {
    start_server("12345");
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
