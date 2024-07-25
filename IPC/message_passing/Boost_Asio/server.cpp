#include <boost/asio.hpp>
#include <iostream>

using boost::asio::ip::tcp;

const int PORT = 12345;

void session(tcp::socket sock) {
    try {
        for (;;) {
            char data[1024];
            boost::system::error_code error;
            size_t length = sock.read_some(boost::asio::buffer(data), error);
            if (error == boost::asio::error::eof) {
                break; // Connection closed cleanly by peer.
            } else if (error) {
                throw boost::system::system_error(error); // Some other error.
            }
            std::cout << "Received: " << std::string(data, length) << std::endl;
        }
    } catch (std::exception& e) {
        std::cerr << "Exception in thread: " << e.what() << std::endl;
    }
}

void start_server(boost::asio::io_context& io_context) {
    tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), PORT));
    for (;;) {
        tcp::socket socket(io_context);
        acceptor.accept(socket);
        std::thread(session, std::move(socket)).detach();
    }
}

int main() {
    try {
        boost::asio::io_context io_context;
        start_server(io_context);
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    return 0;
}
