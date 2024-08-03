#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "shared_memory.h"

void insert_char(sem_t *sem_read, sem_t *sem_write, SharedData *sharedData, Sentence *buffer) {
    char character;

    // Variables para manejar la entrada dinámica
    char *dynamic_string = NULL;
    size_t length = 0;

    while (true) {
        // Esperar hasta que el semáforo de escritura esté disponible
        sem_wait(sem_write);

        // Reiniciar el string dinámico para la próxima iteración
        free(dynamic_string);
        dynamic_string = NULL;
        length = 0;

        // Leer un carácter de la entrada estándar
        printf("Ingrese un carácter (Ctrl+D para terminar): ");
        if ((character = getchar()) == EOF) {
            printf("Fin de entrada.\n");
            break;
        }
        getchar(); // Consumir el '\n' después del carácter ingresado

        // Concatenar el carácter al string dinámico
        char *temp = realloc(dynamic_string, (length + 1) * sizeof(char));
        if (temp == NULL) {
            fprintf(stderr, "Error: No se pudo realocar la memoria\n");
            free(dynamic_string);
            exit(EXIT_FAILURE);
        }
        dynamic_string = temp;
        dynamic_string[length++] = character;

        // Obtener el semáforo para el espacio de escritura
        char sem_write_name[MAX_LENGTH];
        sprintf(sem_write_name, "%s%d", SEM_WRITE_VARIABLE_FNAME, sharedData->writeIndex);

        sem_t *sem_var_write = sem_open(sem_write_name, 0);
        if (sem_var_write == SEM_FAILED) {
            perror("sem_open/variables");
            exit(EXIT_FAILURE);
        }

        // Obtener el semáforo para el espacio de lectura
        char sem_read_name[MAX_LENGTH];
        sprintf(sem_read_name, "%s%d", SEM_READ_VARIABLE_FNAME, sharedData->writeIndex);

        sem_t *sem_var_read = sem_open(sem_read_name, 0);
        if (sem_var_read == SEM_FAILED) {
            perror("sem_open/variables");
            exit(EXIT_FAILURE);
        }

        // Esperar el semáforo para el espacio de escritura
        sem_wait(sem_var_write);

        // Obtener la marca de tiempo actual en el formato deseado
        time_t current_time;
        struct tm *timeinfo;
        time(&current_time);
        timeinfo = localtime(&current_time);
        strftime(buffer[sharedData->writeIndex].time, MAX_TIME_LENGTH, "%b %d %Y %H:%M:%S", timeinfo);

        // Asignar el carácter al buffer
        int index = sharedData->writeIndex;
        printf("Caracter ingresado: %c\n", character);
        buffer[sharedData->writeIndex].character = character;
        printf("Agregando a buffer:\nbuffer[%d] = \"%c\" | tiempo: %s\n------------------------\n", index, buffer[index].character, buffer[index].time);

        // Actualizar los índices compartidos
        sharedData->writeIndex = (sharedData->writeIndex + 1) % sharedData->bufferSize;

        // Post para que la variable pueda ser leída
        sem_post(sem_var_read);  
        sem_post(sem_read);
    }

    sharedData->writingFinished = true;
}



int main() 
{   
    // Open semaphores that were already created
    sem_t *sem_read = sem_open(SEM_READ_PROCESS_FNAME, 0);
    if (sem_read == SEM_FAILED) {
        perror("sem_open/read");
        exit(EXIT_FAILURE);
    }
    sem_t *sem_write = sem_open(SEM_WRITE_PROCESS_FNAME, 0);
    if (sem_write == SEM_FAILED) {
        perror("sem_open/write");
        exit(EXIT_FAILURE);
    }

    // Connect to shared mem struct
    SharedData *sharedData = attach_struct(STRUCT_LOCATION, sizeof(SharedData));
    if (sharedData == NULL) {
        printf("ERROR: no se pudo acceder al bloque\n");
        return -1;
    }

    // Connect to shared mem buffer
    Sentence *buffer = attach_buffer(BUFFER_LOCATION, (sharedData->bufferSize * sizeof(Sentence)));
    if (buffer == NULL) {
        printf("ERROR: no se pudo acceder al bloque\n");
        return -1;
    }


    insert_char(sem_read, sem_write, sharedData, buffer);

    // Detach from memory after finishing
    detach_struct(sharedData);
    detach_buffer(buffer);
    return 0;
}
