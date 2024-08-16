#include "MySharedMemoryHandler.h"
#include <Poco/Exception.h>
#include <cstring> // Para std::memcpy

// Constructor por defecto
MySharedMemoryHandler::MySharedMemoryHandler() : _sharedMemory() {}

// Constructor con archivo
MySharedMemoryHandler::MySharedMemoryHandler(const Poco::File &file, Poco::SharedMemory::AccessMode mode, const void *addrHint)
    : _file(file), _sharedMemory(file, mode, addrHint) {
    if (!_file.exists()) {
        _file.createFile(); // Crear el archivo si no existe
    }
}

// Constructor con nombre y tamaño
MySharedMemoryHandler::MySharedMemoryHandler(const std::string &name, std::size_t size, Poco::SharedMemory::AccessMode mode, const void *addrHint, bool server)
    : _name(name), _file(name), _sharedMemory(name, size, mode, addrHint, server) {
    if (!_file.exists()) {
        _file.createFile(); // Crear el archivo si no existe
    }
}

// Constructor de copia
MySharedMemoryHandler::MySharedMemoryHandler(const MySharedMemoryHandler &other)
    : _name(other._name), _file(other._file), _sharedMemory(other._sharedMemory) {
    // No se necesita inicializar nada adicional ya que _sharedMemory es una copia de la original
}

MySharedMemoryHandler& MySharedMemoryHandler::operator=(const MySharedMemoryHandler &other) {
    if (this != &other) {
        _name = other._name;
        _file = other._file;
        _sharedMemory = other._sharedMemory;
    }
    return *this;
}

void MySharedMemoryHandler::displayInfo() const {
    std::cout << "Shared Memory Name: " << _name << std::endl;
    std::cout << "File Path: " << _file.path() << std::endl;
}

void MySharedMemoryHandler::writeToMemory(const std::string &data) {
    std::memcpy(_sharedMemory.begin(), data.c_str(), data.size() + 1); // +1 para el carácter nulo
}

std::string MySharedMemoryHandler::readFromMemory() const {
    return std::string(static_cast<char*>(_sharedMemory.begin()));
}
