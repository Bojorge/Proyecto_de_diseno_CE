#include <iostream> 
#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#include "shared_memory.h"

#define IPC_RESULT_ERROR (-1)

static int get_shared_block(char *location, int size) {
    key_t key;
    key = ftok(location, 0);

    if (key == IPC_RESULT_ERROR) {
        return IPC_RESULT_ERROR;
    }

    return shmget(key, size, 0644 | IPC_CREAT);
}

SharedData * attach_struct(char *struct_location, int size) {
    int shared_block_id = get_shared_block(struct_location, size);

    if (shared_block_id == IPC_RESULT_ERROR) {
        return NULL;
    }

    SharedData *sharedData = (SharedData *)shmat(shared_block_id, NULL, 0);

    return sharedData;
}

Sentence * attach_buffer(char *buffer_location, int size) {
    int shared_block_id = get_shared_block(buffer_location, size);

    if (shared_block_id == IPC_RESULT_ERROR) {
        return NULL;
    }

    Sentence *sharedData = (Sentence *)shmat(shared_block_id, NULL, 0);

    return sharedData;
}

// c | time: Apr 19 2024 19:00:00
void init_mem_block(char *struct_location, char *buffer_location, int sizeStruct, int sizeBuffer) {
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

bool destroy_memory_block(char *location) {
    int shared_block_id = get_shared_block(location, 0);

    if (shared_block_id == IPC_RESULT_ERROR) {
        return false;
    }

    return (shmctl(shared_block_id, IPC_RMID, NULL) != IPC_RESULT_ERROR);
}
