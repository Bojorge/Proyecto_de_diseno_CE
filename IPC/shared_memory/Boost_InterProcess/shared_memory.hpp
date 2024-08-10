#ifndef SHARED_MEMORY_HPP
#define SHARED_MEMORY_HPP

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>

namespace bip = boost::interprocess;

// Structs
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

// Funciones
long getRAMUsage();
void getCPUUsage(double &userCPU, double &systemCPU);

// Funciones Boost.Interprocess
void init_mem_block(const char *struct_location, const char *buffer_location, int sizeStruct, int sizeBuffer);
SharedData *attach_struct(const char *struct_location);
Sentence *attach_buffer(const char *buffer_location, int sizeBuffer);
bool detach_struct(SharedData *sharedStruct);
bool detach_buffer(Sentence *buffer);
bool destroy_memory_block(const char *filename);

// Variables
#define STRUCT_LOCATION "shared_memory_struct"
#define BUFFER_LOCATION "shared_memory_buffer"

#define SEM_READ_PROCESS_FNAME "myprocessread"
#define SEM_WRITE_PROCESS_FNAME "myprocesswrite"
#define SEM_READ_VARIABLE_FNAME "mybufferreadvariable"
#define SEM_WRITE_VARIABLE_FNAME "mybufferwritevariable"

#define MAX_LENGTH 100

#endif
