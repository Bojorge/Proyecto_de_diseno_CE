#ifndef SHARED_MEMORY_HPP
#define SHARED_MEMORY_HPP

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>

namespace bip = boost::interprocess;

struct SharedData {
    int bufferSize;
    int writeIndex, readIndex;
    int readingFileIndex;
    int clientBlocked, recBlocked;
    int charsTransferred, charsRemaining;
    int memUsed;
    int clientUserTime, clientKernelTime;
    int recUserTime, recKernelTime;
    bool writingFinished, readingFinished, statsInited;
};

#define MAX_TIME_LENGTH 21

struct Sentence {
    char character;
    char time[MAX_TIME_LENGTH];
};

long getRAMUsage();
void getCPUUsage(double &userCPU, double &systemCPU);
void init_mem_block(const char *struct_location, const char *buffer_location, std::size_t sizeStruct, std::size_t sizeBuffer);
SharedData *attach_struct(const char *struct_location);
Sentence *attach_buffer(const char *buffer_location);
bool detach_struct(SharedData *sharedStruct);
bool detach_buffer(Sentence *buffer);
bool destroy_memory_block(const char *location);
bool destroy_semaphore(const char *name);

// Variables
#define STRUCT_LOCATION "shared_data_segment"
#define BUFFER_LOCATION "shared_buffer_segment"

#define SEM_READ_PROCESS_FNAME "/myprocessread"
#define SEM_WRITE_PROCESS_FNAME "/myprocesswrite"
#define SEM_READ_VARIABLE_FNAME "/mybufferreadvariable"
#define SEM_WRITE_VARIABLE_FNAME "/mybufferwritevariable"

#define MAX_LENGTH 100

#endif
