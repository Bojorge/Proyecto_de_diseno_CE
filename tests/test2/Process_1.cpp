#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <iostream>

using boost::interprocess::read_write;
using boost::interprocess::create_only;

int main() {
    std::cout << "P1 started" << std::endl;

    boost::interprocess::shared_memory_object shm(create_only, "SharedMem", read_write);
    shm.truncate(1000);

    boost::interprocess::mapped_region region(shm, read_write);

    // Write 1 in all the memory
    std::memset(region.get_address(), 1, region.get_size());

    std::cout << "P1 done" << std::endl;

    return 0;
}
