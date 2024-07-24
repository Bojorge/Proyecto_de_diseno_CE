#ifndef SHARED_MEMORY_H
#define SHARED_MEMORY_H

#define SHARED_MEMORY_NAME "/my_shared_memory"
#define SHARED_MEMORY_SIZE 1024

void write_to_shared_memory();
void read_from_shared_memory(char* buffer, size_t buffer_size);
void cleanup_shared_memory();

#endif // SHARED_MEMORY_H
