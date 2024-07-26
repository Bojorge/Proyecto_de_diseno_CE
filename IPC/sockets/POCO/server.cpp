#include <Poco/Net/ServerSocket.h>
#include <Poco/Net/SocketStream.h>
#include <Poco/Net/StreamSocket.h>
#include <iostream>
#include <thread>

using Poco::Net::ServerSocket;
using Poco::Net::StreamSocket;
using Poco::Net::SocketStream;

const int PORT = 12345;

void session(StreamSocket socket) {
    try {
        char data[1024];
        int bytesReceived = socket.receiveBytes(data, sizeof(data));
        while (bytesReceived > 0) {
            std::cout << "Received: " << std::string(data, bytesReceived) << std::endl;
            bytesReceived = socket.receiveBytes(data, sizeof(data));
        }
    } catch (Poco::Exception& e) {
        std::cerr << "Exception in thread: " << e.displayText() << std::endl;
    }
}

void start_server() {
    try {
        ServerSocket serverSocket(PORT);
        for (;;) {
            StreamSocket socket = serverSocket.acceptConnection();
            std::thread(session, std::move(socket)).detach();
        }
    } catch (Poco::Exception& e) {
        std::cerr << "Server exception: " << e.displayText() << std::endl;
    }
}

int main() {
    start_server();
    return 0;
}
