#include <zmq.hpp>
#include <iostream>
#include <thread>

void start_server() {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    socket.bind("tcp://*:5555");

    while (true) {
        zmq::message_t request;

        // Esperar a recibir el mensaje
        socket.recv(request, zmq::recv_flags::none);
        std::string recv_msg(static_cast<char*>(request.data()), request.size());
        std::cout << "Received: " << recv_msg << std::endl;

        // Responder al cliente
        std::string reply_msg = "Message received";
        zmq::message_t reply(reply_msg.size());
        memcpy(reply.data(), reply_msg.data(), reply_msg.size());
        socket.send(reply, zmq::send_flags::none);
    }
}

int main() {
    try {
        std::thread server_thread(start_server);
        server_thread.join();
    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    return 0;
}
