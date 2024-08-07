#include <iostream>
#include <string>
#include "posix_sockets.h"

int main() {
    const int num_iterations = 1000;
    std::string message;

    for (int i = 0; i < num_iterations; ++i) {
        std::cout << "Iteración " << i << " --------------------------------------" << std::endl;
        message = "mensaje #" + std::to_string(i);
        send_message("127.0.0.1", "12345", message.c_str());
    }

    return 0;
}
