#include "posix_shared_memory.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <semaphore.h>
#include <stdio.h>
#include <string.h>

void create_shared_memory(const char* name, size_t size) {
    int fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        perror("shm_open");
        return;
    }
    if (ftruncate(fd, size) == -1) {
        perror("ftruncate");
        close(fd);
        return;
    }
    close(fd);
}

void write_to_shared_memory(const char* name, const char* message, size_t size) {
    int fd = shm_open(name, O_RDWR, 0666);
    if (fd == -1) {
        perror("shm_open");
        return;
    }
    void* ptr = mmap(0, size, PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }
    memcpy(ptr, message, size);
    munmap(ptr, size);
    close(fd);
}

void read_from_shared_memory(const char* name, char* buffer, size_t size) {
    int fd = shm_open(name, O_RDONLY, 0666);
    if (fd == -1) {
        perror("shm_open");
        return;
    }
    void* ptr = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return;
    }
    memcpy(buffer, ptr, size);
    munmap(ptr, size);
    close(fd);
}

void cleanup_shared_memory(const char* name) {
    shm_unlink(name);
}

void create_semaphore(const char* name) {
    sem_t* sem = sem_open(name, O_CREAT, 0666, 0);
    if (sem == SEM_FAILED) {
        perror("sem_open");
        return;
    }
    sem_close(sem);
}

void wait_semaphore(const char* name) {
    sem_t* sem = sem_open(name, 0);
    if (sem == SEM_FAILED) {
        perror("sem_open");
        return;
    }
    sem_wait(sem);
    sem_close(sem);
}

void post_semaphore(const char* name) {
    sem_t* sem = sem_open(name, 0);
    if (sem == SEM_FAILED) {
        perror("sem_open");
        return;
    }
    sem_post(sem);
    sem_close(sem);
}

void close_semaphore(const char* name) {
    sem_t* sem = sem_open(name, 0);
    if (sem == SEM_FAILED) {
        perror("sem_open");
        return;
    }
    sem_close(sem);
}

void unlink_semaphore(const char* name) {
    sem_unlink(name);
}
