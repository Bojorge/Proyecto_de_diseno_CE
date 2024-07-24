#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include "posix.h"

void write_to_shared_memory() {
    const char* message = "- Este mensaje fue escrito en memoria compartida -";
    int fd = shm_open(SHARED_MEMORY_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(fd, SHARED_MEMORY_SIZE);
    void* ptr = mmap(0, SHARED_MEMORY_SIZE, PROT_WRITE, MAP_SHARED, fd, 0);
    sprintf(ptr, "%s", message);
    munmap(ptr, SHARED_MEMORY_SIZE);
    close(fd);
}

void read_from_shared_memory(char* buffer, size_t buffer_size) {
    int fd = shm_open(SHARED_MEMORY_NAME, O_RDONLY, 0666);
    void* ptr = mmap(0, SHARED_MEMORY_SIZE, PROT_READ, MAP_SHARED, fd, 0);
    strncpy(buffer, (char*)ptr, buffer_size);
    munmap(ptr, SHARED_MEMORY_SIZE);
    close(fd);
}

void cleanup_shared_memory() {
    shm_unlink(SHARED_MEMORY_NAME);
}
