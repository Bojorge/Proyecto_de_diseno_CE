#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <iostream>

using boost::interprocess::read_only;
using boost::interprocess::open_only;

int main() {
    std::cout << "P2 started" << std::endl;

    boost::interprocess::mapped_region region;
    while (region.get_size() == 0u) {
        try {
            boost::interprocess::shared_memory_object shm(open_only, "SharedMem", read_only);
            region = boost::interprocess::mapped_region(shm, read_only);
        } catch(...) {
            std::cout << "Failed to access shared memory" << std::endl;
        }
    }

    const std::byte* mem = static_cast<const std::byte*>(region.get_address());

    // Check the memory
    bool isError = false;
    for(std::size_t i = 0; i < region.get_size(); ++i) {
        const std::byte data = *(mem + i);
        if(data != std::byte{1}) {
            isError = true;
            break;
        }
    }

    boost::interprocess::shared_memory_object::remove("SharedMem");
    std::cout << "P2 done, error = " << isError << std::endl;

    return 0;
}
