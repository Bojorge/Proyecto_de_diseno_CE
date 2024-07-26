#include <Poco/Net/StreamSocket.h>
#include <Poco/Net/SocketAddress.h>
#include <Poco/Net/SocketStream.h>
#include <iostream>

using Poco::Net::StreamSocket;
using Poco::Net::SocketAddress;
using Poco::Net::SocketStream;

const int PORT = 12345;
const std::string SERVER_IP = "127.0.0.1";

void send_message(const std::string& message) {
    try {
        SocketAddress serverAddr(SERVER_IP, PORT);
        StreamSocket socket(serverAddr);
        SocketStream str(socket);
        str << message << std::endl;
        str.flush();
    } catch (Poco::Exception& e) {
        std::cerr << "Client exception: " << e.displayText() << std::endl;
    }
}

int main() {
    std::string message;

    while (true) {
        std::cout << "Enter message: ";
        std::getline(std::cin, message);

        if (message == "exit") {
            break;
        }

        send_message(message);
    }

    return 0;
}
