#include <iostream>
#include <thread>
#include <chrono>
#include "posix_shared_memory.h"

const char* SHARED_MEMORY_NAME = "/my_shared_memory";
const char* SEM_EMPTY = "/sem_empty";
const char* SEM_FULL = "/sem_full";
const size_t SHARED_MEMORY_SIZE = 1024;

void producer() {
    create_shared_memory(SHARED_MEMORY_NAME, SHARED_MEMORY_SIZE);
    create_semaphore(SEM_EMPTY);
    create_semaphore(SEM_FULL);

    const char* messages[] = { "Message 1", "Message 2", "Message 3" };
    int message_count = 3;

    for (int i = 0; i < message_count; ++i) {
        wait_semaphore(SEM_EMPTY);
        write_to_shared_memory(SHARED_MEMORY_NAME, messages[i], SHARED_MEMORY_SIZE);
        post_semaphore(SEM_FULL);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    cleanup_shared_memory(SHARED_MEMORY_NAME);
    unlink_semaphore(SEM_EMPTY);
    unlink_semaphore(SEM_FULL);
}

int main() {
    producer();
    return 0;
}
