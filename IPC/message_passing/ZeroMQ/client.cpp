#include <zmq.hpp>
#include <iostream>

void send_message(const std::string& message) {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REQ);
    socket.connect("tcp://localhost:5555");

    zmq::message_t request(message.size());
    memcpy(request.data(), message.data(), message.size());

    // Enviar el mensaje
    socket.send(request, zmq::send_flags::none);

    // Esperar la respuesta
    zmq::message_t reply;
    socket.recv(reply, zmq::recv_flags::none);
    std::string reply_msg(static_cast<char*>(reply.data()), reply.size());
    std::cout << "Received: " << reply_msg << std::endl;
}

int main() {
    try {
        std::string message;

        while (true) {
            std::cout << "Enter message: ";
            std::getline(std::cin, message);

            if (message == "exit") {
                break;
            }

            send_message(message);
        }
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
