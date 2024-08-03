#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "shared_memory.h"

void insert_manually(FILE *file, sem_t *sem_read, sem_t *sem_write, SharedData *sharedData, Sentence *buffer) {
    char character;
    
    // Obtener la longitud del archivo
    fseek(file, 0, SEEK_END);
    long file_length = ftell(file);
    rewind(file);

    // Variables para leer restante de archivo
    char *dynamic_string = NULL;
    size_t buffer_size = 0;
    size_t length = 0;

    while (ftell(file) < file_length) {
        // Esperar hasta que se presione Enter y se tenga el semaforo libre para escribir en ese espacio
        if (getchar() == '\n') {
            sem_wait(sem_write);

            // Mover el puntero del archivo al índice de lectura
            fseek(file, sharedData->readingFileIndex, SEEK_SET);

            // Reiniciar el string dinámico para la próxima iteración
            free(dynamic_string);
            dynamic_string = NULL;
            buffer_size = 0;
            length = 0;

            // Leer el contenido completo del archivo y concatenarlo al string dinámico
            char character2;
            while ((character2 = fgetc(file)) != EOF) {
                // Concatenar el caracter al final del string dinámico
                buffer_size++;
                char *temp = realloc(dynamic_string, buffer_size * sizeof(char));
                if (temp == NULL) {
                    fprintf(stderr, "Error: No se pudo realocar la memoria\n");
                    free(dynamic_string);
                    exit(EXIT_FAILURE);
                }
                dynamic_string = temp;
                dynamic_string[length++] = character2;
            }

            // Get semaphore for said writing space, to check if writing is available
            char sem_write_name[MAX_LENGTH];
            sprintf(sem_write_name, "%s%d", SEM_WRITE_VARIABLE_FNAME, sharedData->writeIndex);

            sem_t *sem_var_write = sem_open(sem_write_name, 0);
            if (sem_var_write == SEM_FAILED) {
                perror("sem_open/variables");
                exit(EXIT_FAILURE);
            }

            // Get semaphore for said reading space, to post after writing
            char sem_read_name[MAX_LENGTH];
            sprintf(sem_read_name, "%s%d", SEM_READ_VARIABLE_FNAME, sharedData->writeIndex);

            sem_t *sem_var_read = sem_open(sem_read_name, 0);
            if (sem_var_read == SEM_FAILED) {
                perror("sem_open/variables");
                exit(EXIT_FAILURE);
            }

            // Write after checking if semaphore is open
            sem_wait(sem_var_write);

            // Obtener la marca de tiempo actual en el formato deseado
            time_t current_time;
            struct tm *timeinfo;
            time(&current_time);
            timeinfo = localtime(&current_time);
            strftime(buffer[sharedData->writeIndex].time, MAX_TIME_LENGTH, "%b %d %Y %H:%M:%S", timeinfo);

            // Mover el puntero del archivo al índice de lectura
            fseek(file, sharedData->readingFileIndex, SEEK_SET);

            // Conseguir caracter del archivo
            character = fgetc(file);

            // Asignar el carácter al buffer
            int index = sharedData->writeIndex;
            printf("Restante por transferir: %s\n", dynamic_string);
            buffer[sharedData->writeIndex].character = character;
            printf("Agregando a buffer:\nbuffer[%d] = \"%c\" | tiempo: %s\n------------------------\n", index, buffer[index].character, buffer[index].time);

            // Actualizar los índices compartidos
            sharedData->writeIndex = (sharedData->writeIndex + 1) % sharedData->bufferSize;
            sharedData->readingFileIndex++;

            // Post so that variable could be read
            sem_post(sem_var_read);  
            sem_post(sem_read);
        }
    }
    sharedData->writingFinished = true;
}

void insert_automatically(FILE *file, int interval, sem_t *sem_read, sem_t *sem_write, SharedData *sharedData, Sentence *buffer) {
    char character;
    
    // Obtener la longitud del archivo
    fseek(file, 0, SEEK_END);
    long file_length = ftell(file);
    rewind(file);

    // Variables para leer restante de archivo
    char *dynamic_string = NULL;
    size_t buffer_size = 0;
    size_t length = 0;

    while (ftell(file) < file_length) { // Mientras no se llegue al final del archivo
        
        sem_wait(sem_write);

        // Mover el puntero del archivo al índice de lectura
        fseek(file, sharedData->readingFileIndex, SEEK_SET);

        // Reiniciar el string dinámico para la próxima iteración
        free(dynamic_string);
        dynamic_string = NULL;
        buffer_size = 0;
        length = 0;

        // Leer el contenido completo del archivo y concatenarlo al string dinámico
        char character2;
        while ((character2 = fgetc(file)) != EOF) {
            // Concatenar el caracter al final del string dinámico
            buffer_size++;
            char *temp = realloc(dynamic_string, buffer_size * sizeof(char));
            if (temp == NULL) {
                fprintf(stderr, "Error: No se pudo realocar la memoria\n");
                free(dynamic_string);
                exit(EXIT_FAILURE);
            }
            dynamic_string = temp;
            dynamic_string[length++] = character2;
        }

        // Get semaphore for said writing space, to check if writing is available
        char sem_write_name[MAX_LENGTH];
        sprintf(sem_write_name, "%s%d", SEM_WRITE_VARIABLE_FNAME, sharedData->writeIndex);

        sem_t *sem_var_write = sem_open(sem_write_name, 0);
        if (sem_var_write == SEM_FAILED) {
            perror("sem_open/variables");
            exit(EXIT_FAILURE);
        }

        // Get semaphore for said reading space, to post after writing
        char sem_read_name[MAX_LENGTH];
        sprintf(sem_read_name, "%s%d", SEM_READ_VARIABLE_FNAME, sharedData->writeIndex);

        sem_t *sem_var_read = sem_open(sem_read_name, 0);
        if (sem_var_read == SEM_FAILED) {
            perror("sem_open/variables");
            exit(EXIT_FAILURE);
        }

        // Write after checking if semaphore is open
        sem_wait(sem_var_write);

        // Obtener la marca de tiempo actual en el formato deseado
        time_t current_time;
        struct tm *timeinfo;
        time(&current_time);
        timeinfo = localtime(&current_time);
        strftime(buffer[sharedData->writeIndex].time, MAX_TIME_LENGTH, "%b %d %Y %H:%M:%S", timeinfo);
        
        // Mover el puntero del archivo al índice de lectura
        fseek(file, sharedData->readingFileIndex, SEEK_SET);

        // Conseguir caracter del archivo
        character = fgetc(file);

        // Asignar el carácter al buffer
        int index = sharedData->writeIndex;
        printf("Restante por transferir: %s\n", dynamic_string);
        buffer[sharedData->writeIndex].character = character;
        printf("Agregando a buffer:\nbuffer[%d] = \"%c\" | tiempo: %s\n------------------------\n", index, buffer[index].character, buffer[index].time);

        // Actualizar los índices compartidos
        sharedData->writeIndex = (sharedData->writeIndex + 1) % sharedData->bufferSize;
        sharedData->readingFileIndex++;

        // Post so that variable could be read
        sem_post(sem_var_read);
        sem_post(sem_read);

        sleep(interval); // Esperar el intervalo especificado
    }
    sharedData->writingFinished = true;
}

int main(int argc, char *argv[]) 
{   
    // Check for specified file and mode same as interval
    if (argc < 3 || argc > 4) {
        printf("Uso: %s <archivo> <modo> [intervalo]s\n", argv[0]);
        printf("Modo: 0 = Manual, 1 = Automático\n");
        return -1;
    }

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

    // Read the file
    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        printf("Error opening file %s\n",argv[1]);
        return -1;
    }
    
    int mode = atoi(argv[2]);
    if (mode == 0) { // Modo manual
        insert_manually(file, sem_read, sem_write, sharedData, buffer);
    } else if (mode == 1) { // Modo automático
        if (argc == 4) { // Se requiere el intervalo
            int interval = atoi(argv[3]);
            if (interval <= 0) {
                printf("Intervalo no válido\n");
                return -1;
            }
            insert_automatically(file, interval, sem_read, sem_write, sharedData, buffer);
        } else {
            printf("Intervalo no especificado\n");
            return -1;
        }
    } else {
        printf("Modo no válido\n");
        return -1;
    }

    // Detach from memory after finishing
    detach_struct(sharedData);
    detach_buffer(buffer);
    return 0;
}
