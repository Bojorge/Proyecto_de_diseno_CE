#include <iostream>
#include <thread>
#include <chrono>
#include "posix_shared_memory.h"

const char* SHARED_MEMORY_NAME = "/my_shared_memory";
const char* SEM_EMPTY = "/sem_empty";
const char* SEM_FULL = "/sem_full";
const size_t SHARED_MEMORY_SIZE = 1024;

void consumer() {
    char buffer[SHARED_MEMORY_SIZE];

    for (int i = 0; i < 3; ++i) {
        wait_semaphore(SEM_FULL);
        read_from_shared_memory(SHARED_MEMORY_NAME, buffer, SHARED_MEMORY_SIZE);
        std::cout << "Message from shared memory: " << buffer << std::endl;
        post_semaphore(SEM_EMPTY);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    cleanup_shared_memory(SHARED_MEMORY_NAME);
    unlink_semaphore(SEM_EMPTY);
    unlink_semaphore(SEM_FULL);
}

int main() {
    consumer();
    return 0;
}
