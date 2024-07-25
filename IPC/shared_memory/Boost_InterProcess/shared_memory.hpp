#ifndef SHARED_MEMORY_HPP
#define SHARED_MEMORY_HPP

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <iostream>
#include <cstring>  // Para strncpy

namespace bip = boost::interprocess;

typedef bip::allocator<char, bip::managed_shared_memory::segment_manager> CharAllocator;
typedef bip::basic_string<char, std::char_traits<char>, CharAllocator> SharedString;

struct SharedMemoryBuffer {
    enum { BufferSize = 1024 };
    typedef bip::interprocess_mutex Mutex;
    typedef bip::interprocess_condition Condition;

    Mutex mutex;
    Condition cond_empty;
    Condition cond_full;
    bool full;
    char buffer[BufferSize];

    SharedMemoryBuffer() : full(false) {
        std::memset(buffer, 0, BufferSize);
    }
};

void create_shared_memory() {
    bip::shared_memory_object::remove("MySharedMemory");
    try {
        bip::managed_shared_memory segment(bip::create_only, "MySharedMemory", 65536);
        segment.construct<SharedMemoryBuffer>("SharedMemoryBuffer")();
        std::cout << "Shared memory and buffer created successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Exception while creating shared memory: " << e.what() << std::endl;
        throw;
    }
}

SharedMemoryBuffer* get_shared_memory() {
    try {
        bip::managed_shared_memory segment(bip::open_only, "MySharedMemory");
        SharedMemoryBuffer* buffer = segment.find<SharedMemoryBuffer>("SharedMemoryBuffer").first;
        if (!buffer) {
            throw std::runtime_error("SharedMemoryBuffer not found.");
        }
        return buffer;
    } catch (const std::exception& e) {
        std::cerr << "Exception while accessing shared memory: " << e.what() << std::endl;
        throw;
    }
}

void write_to_shared_memory(const std::string& message) {
    SharedMemoryBuffer* shm_buffer = get_shared_memory();
    if (shm_buffer == nullptr) {
        std::cerr << "Shared memory buffer not found" << std::endl;
        return;
    }

    bip::scoped_lock<SharedMemoryBuffer::Mutex> lock(shm_buffer->mutex);
    while (shm_buffer->full) {
        shm_buffer->cond_full.wait(lock);
    }

    std::strncpy(shm_buffer->buffer, message.c_str(), SharedMemoryBuffer::BufferSize);
    shm_buffer->full = true;
    shm_buffer->cond_empty.notify_one();
    std::cout << "Message written to shared memory: " << message << std::endl;
}

std::string read_from_shared_memory() {
    SharedMemoryBuffer* shm_buffer = get_shared_memory();
    if (shm_buffer == nullptr) {
        std::cerr << "Shared memory buffer not found" << std::endl;
        return "";
    }

    bip::scoped_lock<SharedMemoryBuffer::Mutex> lock(shm_buffer->mutex);
    while (!shm_buffer->full) {
        shm_buffer->cond_empty.wait(lock);
    }

    std::string message(shm_buffer->buffer);
    shm_buffer->full = false;
    shm_buffer->cond_full.notify_one();
    std::cout << "Message read from shared memory: " << message << std::endl;
    return message;
}

#endif // SHA
