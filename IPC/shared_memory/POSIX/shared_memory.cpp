#include <iostream> 
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/resource.h>
#include "shared_memory.h"

#define IPC_RESULT_ERROR (-1)

long getRAMUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;
}

void getCPUUsage(double &userCPU, double &systemCPU) {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    userCPU = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;
    systemCPU = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
}

static int get_shared_block(const char *location, int size) {
    key_t key;
    key = ftok(location, 0);

    if (key == IPC_RESULT_ERROR) {
        return IPC_RESULT_ERROR;
    }

    return shmget(key, size, 0644 | IPC_CREAT);
}

SharedData * attach_struct(const char *struct_location, int size) {
    int shared_block_id = get_shared_block(struct_location, size);

    if (shared_block_id == IPC_RESULT_ERROR) {
        return NULL;
    }

    SharedData *sharedData = (SharedData *)shmat(shared_block_id, NULL, 0);

    return sharedData;
}

Sentence * attach_buffer(const char *buffer_location, int size) {
    int shared_block_id = get_shared_block(buffer_location, size);

    if (shared_block_id == IPC_RESULT_ERROR) {
        return NULL;
    }

    Sentence *sharedData = (Sentence *)shmat(shared_block_id, NULL, 0);

    return sharedData;
}

void init_mem_block(const char *struct_location, const char *buffer_location, int sizeStruct, int sizeBuffer) {
    int struct_block_id = get_shared_block(struct_location, sizeStruct);
    int buffer_block_id = get_shared_block(buffer_location, sizeBuffer);

    if (struct_block_id == IPC_RESULT_ERROR) {
        std::cerr << "Error al obtener identificador del bloque compartido struct." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (buffer_block_id == IPC_RESULT_ERROR) {
        std::cerr << "Error al obtener identificador del bloque compartido buffer." << std::endl;
        exit(EXIT_FAILURE);
    }
}

bool detach_struct(SharedData *block) {
    return (shmdt(block) != -1);
}

bool detach_buffer(Sentence *buffer) {
    return (shmdt(buffer) != -1);
}

bool destroy_memory_block(const char *location) {
    int shared_block_id = get_shared_block(location, 0);

    if (shared_block_id == IPC_RESULT_ERROR) {
        return false;
    }

    return (shmctl(shared_block_id, IPC_RMID, NULL) != IPC_RESULT_ERROR);
}
