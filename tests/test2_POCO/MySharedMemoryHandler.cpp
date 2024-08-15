#include "MySharedMemoryHandler.h"

MySharedMemoryHandler::MySharedMemoryHandler(const std::string &name, const Poco::File &file)
    : _name(name), _file(file) {
    // Aquí puedes inicializar otros miembros o realizar operaciones
}

void MySharedMemoryHandler::displayInfo() const {
    std::cout << "Shared Memory Name: " << _name << std::endl;
    std::cout << "File Path: " << _file.path() << std::endl;
}
