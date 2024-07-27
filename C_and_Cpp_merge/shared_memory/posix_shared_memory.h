#ifndef POSIX_SHARED_MEMORY_H
#define POSIX_SHARED_MEMORY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void create_shared_memory(const char* name, size_t size);
void write_to_shared_memory(const char* name, const char* message, size_t size);
void read_from_shared_memory(const char* name, char* buffer, size_t size);
void cleanup_shared_memory(const char* name);

void create_semaphore(const char* name);
void wait_semaphore(const char* name);
void post_semaphore(const char* name);
void close_semaphore(const char* name);
void unlink_semaphore(const char* name);

#ifdef __cplusplus
}
#endif

#endif // POSIX_SHARED_MEMORY_H
