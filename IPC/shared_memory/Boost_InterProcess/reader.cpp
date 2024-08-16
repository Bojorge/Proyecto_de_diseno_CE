#include <boost/interprocess/managed_shared_memory.hpp>
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <stdio.h>
#include <chrono>
#include <thread>

using namespace boost::interprocess;

int main (int argc, char *argv[])
{
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    char buffer[1024];
    
    memset(buffer, '\0', sizeof(buffer));
    
    managed_shared_memory segment (open_only, "MySharedMemory");
    managed_shared_memory::handle_t handle = 240; //hardcodeado, esto se debe pasar como parametro

    //Get buffer local address from handle
    void *msg = segment.get_address_from_handle(handle);

    for(int i=0;i<10;i++){
        std::cout << "READING -> ";
        std::cout << (char*)msg << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
    
}