#ifndef SHARED_MEMORY_HPP
#define SHARED_MEMORY_HPP

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/sync/named_semaphore.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>

namespace bip = boost::interprocess;


// Tamaño máximo para la cadena de tiempo
#define MAX_TIME_LENGTH 21

// Estructura para almacenar datos en el buffer
struct Sentence {
    char character;
    char time[MAX_TIME_LENGTH];
};

// Funciones para manejar la memoria compartida
void init_mem_block(const char *buffer_location, std::size_t sizeBuffer);
Sentence* attach_buffer(const char *buffer_location);
bool detach_mem_block(const char *location);
bool destroy_mem_block(const char *location);

// Funciones para manejar semáforos
bool create_semaphore(const char *name, unsigned int initial_count);
bool destroy_semaphore(const char *name);
bip::named_semaphore* get_semaphore(const char *name);

// Variables de configuración
#define BUFFER_LOCATION "shared_buffer_segment"

// Nombres para los semáforos
#define SEM_READ_PROCESS_NAME "/myprocessread"
#define SEM_WRITE_PROCESS_NAME "/myprocesswrite"
#define SEM_READ_VARIABLE_NAME "/mybufferreadvariable"
#define SEM_WRITE_VARIABLE_NAME "/mybufferwritevariable"

// Longitud máxima para datos generales
#define MAX_LENGTH 100

#endif
