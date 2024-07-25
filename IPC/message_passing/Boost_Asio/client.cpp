#include <boost/asio.hpp>
#include <iostream>

using boost::asio::ip::tcp;

const int PORT = 12345;
const std::string SERVER_IP = "127.0.0.1";

void send_message(boost::asio::io_context& io_context, const std::string& message) {
    tcp::resolver resolver(io_context);
    tcp::resolver::results_type endpoints = resolver.resolve(SERVER_IP, std::to_string(PORT));

    tcp::socket socket(io_context);
    boost::asio::connect(socket, endpoints);

    boost::asio::write(socket, boost::asio::buffer(message));
}

int main() {
    try {
        boost::asio::io_context io_context;
        std::string message;

        while (true) {
            std::cout << "Enter message: ";
            std::getline(std::cin, message);

            if (message == "exit") {
                break;
            }

            send_message(io_context, message);
        }
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
