#include <boost/interprocess/managed_shared_memory.hpp>
#include <iostream>
#include <cstdlib> //std::system
#include <sstream>
#include <stdio.h>
#include <chrono> // Para std::chrono::seconds
#include <thread> // Para std::this_thread::sleep_for

using namespace boost::interprocess;

int main (int argc, char *argv[])
{
    // Esperar 1 segundo antes de iniciar
        std::this_thread::sleep_for(std::chrono::seconds(1));

    char buffer[1024];
    
    memset(buffer, '\0', sizeof(buffer));
    

    managed_shared_memory segment (open_only, "MySharedMemory");
    managed_shared_memory::handle_t handle = 240; //hardcodeado, esto se debe pasar como parametro


    //Get buffer local address from handle
    void *msg = segment.get_address_from_handle(handle);

    for(int i=0;i<10;i++){
        std::cout << "READING -> ";
        std::cout << (char*)msg << std::endl;

        // Esperar 1 segundo antes de intentar nuevamente
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return 0;
    
}