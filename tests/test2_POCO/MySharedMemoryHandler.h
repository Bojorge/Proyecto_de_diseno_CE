#ifndef MYSHAREDMEMORYHANDLER_H
#define MYSHAREDMEMORYHANDLER_H

#include <Poco/SharedMemory.h>
#include <Poco/File.h>
#include <iostream>
#include <string>

class MySharedMemoryHandler {
public:
    // Constructor que toma una referencia constante a std::string y Poco::File
    MySharedMemoryHandler(const std::string &name, const Poco::File &file);

    void displayInfo() const;

private:
    std::string _name;
    Poco::File _file;
};

#endif // MYSHAREDMEMORYHANDLER_H
