#include <iostream>
#include <string>
#include "posix_sockets.h"

int main() {
    std::string message;
    std::cout << "Enter message: ";
    std::getline(std::cin, message);

    send_message("127.0.0.1", "12345", message.c_str());

    return 0;
}
