#ifndef SHARED_MEMORY_HPP
#define SHARED_MEMORY_HPP

#include <Poco/SharedMemory.h>
#include <Poco/Semaphore.h>
#include <Poco/Exception.h>
#include <iostream>
#include <cstdlib>
#include <string>

using Poco::SharedMemory;
using Poco::Semaphore;

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
void init_mem_block(const std::string &struct_location, const std::string &buffer_location, int sizeStruct, int sizeBuffer);
SharedData *attach_struct(const std::string &struct_location);
Sentence *attach_buffer(const std::string &buffer_location);
bool detach_struct(SharedData *sharedStruct);
bool detach_buffer(Sentence *buffer);
bool destroy_memory_block(const std::string &location);

// Variables
#define STRUCT_LOCATION "shared_struct"
#define BUFFER_LOCATION "shared_buffer"

#define SEM_READ_PROCESS_FNAME "sem_read_process"
#define SEM_WRITE_PROCESS_FNAME "sem_write_process"
#define SEM_READ_VARIABLE_FNAME "sem_read_variable_"
#define SEM_WRITE_VARIABLE_FNAME "sem_write_variable_"

#define MAX_LENGTH 100

#endif // SHARED_MEMORY_HPP
