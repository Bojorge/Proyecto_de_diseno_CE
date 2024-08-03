#include <iostream>
#include <string>
#include "posix_sockets.h"

int main() {
    const int num_iterations = 5;
    std::string message;

    for (int i = 0; i < num_iterations; ++i) {
        std::cout << "Escribir el mensaje (" << (i + 1) << "/" << num_iterations << ") >>> ";
        std::getline(std::cin, message);

        send_message("127.0.0.1", "12345", message.c_str());
    }

    return 0;
}
