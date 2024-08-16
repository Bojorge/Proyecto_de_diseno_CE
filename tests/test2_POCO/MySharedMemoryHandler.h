#ifndef MYSHAREDMEMORYHANDLER_H
#define MYSHAREDMEMORYHANDLER_H

#include <Poco/SharedMemory.h>
#include "Poco/Poco.h"
#include <Poco/File.h>
#include <iostream>
#include <string>

class MySharedMemoryHandler {
public:
    // Constructor por defecto
    MySharedMemoryHandler();

    // Constructor con archivo
    MySharedMemoryHandler(const Poco::File &file, Poco::SharedMemory::AccessMode mode, const void *addrHint = nullptr);

    // Constructor con nombre y tamaño
    MySharedMemoryHandler(const std::string &name, std::size_t size, Poco::SharedMemory::AccessMode mode, const void *addrHint = nullptr, bool server = true);

    // Constructor de copia
    MySharedMemoryHandler(const MySharedMemoryHandler &other);

    // Operador de asignación
    MySharedMemoryHandler& operator=(const MySharedMemoryHandler &other);

    void displayInfo() const;

    // Métodos para leer y escribir en la memoria compartida
    void writeToMemory(const std::string &data);
    std::string readFromMemory() const;

private:
    std::string _name;
    Poco::File _file;
    Poco::SharedMemory _sharedMemory;
};

#endif // MYSHAREDMEMORYHANDLER_H
