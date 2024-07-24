#include <stdio.h>
#include "posix.h"

int main() {
    printf("Write to shared memory \n");
    write_to_shared_memory();

    printf("---------------------------------------- \n");
    char buffer[SHARED_MEMORY_SIZE];
    read_from_shared_memory(buffer, sizeof(buffer));

    printf("Read from shared memory: %s\n", buffer);

    cleanup_shared_memory();
    return 0;
}
