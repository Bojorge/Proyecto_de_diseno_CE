#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <semaphore.h>

#include "shared_memory.h"

void read_manually(sem_t *sem_read, sem_t *sem_write, SharedData *sharedData, Sentence *buffer) {
    char *dynamic_string = NULL;
    size_t buffer_size = 0;
    size_t length = 0;

    while (!sharedData->writingFinished) {
        // Esperar hasta que se presione enter
        if (getchar() == '\n') {
            // Una vez presionado enter esperamos a ver si se puede escribir
            sem_wait(sem_write);

            // Read the file
            FILE *file = fopen("memory_read.txt", "r+");
            if (file == NULL) {
                printf("Error opening file %s\n", "memory_read.txt");
            }

            // Reiniciar el string dinámico para la próxima iteración
            free(dynamic_string);
            dynamic_string = NULL;
            buffer_size = 0;
            length = 0;

            // Leer el contenido completo del archivo y concatenarlo al string dinámico
            char character;
            while ((character = fgetc(file)) != EOF) {
                // Concatenar el caracter al final del string dinámico
                buffer_size++;
                char *temp = realloc(dynamic_string, buffer_size * sizeof(char));
                if (temp == NULL) {
                    fprintf(stderr, "Error: No se pudo realocar la memoria\n");
                    free(dynamic_string);
                    exit(EXIT_FAILURE);
                }
                dynamic_string = temp;
                dynamic_string[length++] = character;
            }
            buffer_size ++;
            char *temp = realloc(dynamic_string, buffer_size * sizeof(char));
            dynamic_string = temp;

            // Get semaphore for said reading space, to check if reading is available
            char sem_read_name[MAX_LENGTH];
            sprintf(sem_read_name, "%s%d", SEM_READ_VARIABLE_FNAME, sharedData->writeIndex);

            sem_t *sem_var_read = sem_open(sem_read_name, 0);
            if (sem_var_read == SEM_FAILED) {
                perror("sem_open/variables");
                exit(EXIT_FAILURE);
            }

            // Get semaphore for said writing space, to post after reading
            char sem_write_name[MAX_LENGTH];
            sprintf(sem_write_name, "%s%d", SEM_WRITE_VARIABLE_FNAME, sharedData->writeIndex);

            sem_t *sem_var_write = sem_open(sem_write_name, 0);
            if (sem_var_write == SEM_FAILED) {
                perror("sem_open/variables");
                exit(EXIT_FAILURE);
            }

            // Read after checking if semaphore is open
            sem_wait(sem_var_read);

            // Print into console buffer index, character and time retreived
            int index = sharedData->readIndex;
            printf("Leyendo del buffer: \nbuffer[%d] = \"%c\" | tiempo: %s\n", index, buffer[index].character, buffer[index].time);

            dynamic_string[length] = buffer[index].character;
            printf("Archivo reconstruido: %s\n--------------------------\n", dynamic_string);

            // Add character to end of file
            fseek(file, 0, SEEK_END);
            fputc(buffer[index].character, file);

            // Borramos el caracter leido del buffer
            buffer[index].character = '\0';
            strcpy(buffer[index].time, "");
            
            // Actualizamos variables
            sharedData->charsTransferred++;
            sharedData->readIndex = (sharedData->readIndex + 1) % sharedData->bufferSize;

            // Cerramos archivo de reconstruccion
            fclose(file);

            // Posteamos los semaforos
            sem_post(sem_var_write);
            sem_post(sem_read);
        }
        
    }
}

void read_automatically(sem_t *sem_read, sem_t *sem_write, SharedData *sharedData, Sentence *buffer, int interval) {
    char *dynamic_string = NULL;
    size_t buffer_size = 0;
    size_t length = 0;

    while (!sharedData->writingFinished) {
        sem_wait(sem_write);

        // Read the file
        FILE *file = fopen("memory_read.txt", "r+");
        if (file == NULL) {
            printf("Error opening file %s\n", "memory_read.txt");
        }

        // Reiniciar el string dinámico para la próxima iteración
        free(dynamic_string);
        dynamic_string = NULL;
        buffer_size = 0;
        length = 0;

        // Leer el contenido completo del archivo y concatenarlo al string dinámico
        char character;
        while ((character = fgetc(file)) != EOF) {
            // Concatenar el caracter al final del string dinámico
            buffer_size++;
            char *temp = realloc(dynamic_string, buffer_size * sizeof(char));
            if (temp == NULL) {
                fprintf(stderr, "Error: No se pudo realocar la memoria\n");
                free(dynamic_string);
                exit(EXIT_FAILURE);
            }
            dynamic_string = temp;
            dynamic_string[length++] = character;
        }
        buffer_size ++;
        char *temp = realloc(dynamic_string, buffer_size * sizeof(char));
        dynamic_string = temp;

        // Get semaphore for said reading space, to check if reading is available
        char sem_read_name[MAX_LENGTH];
        sprintf(sem_read_name, "%s%d", SEM_READ_VARIABLE_FNAME, sharedData->writeIndex);

        sem_t *sem_var_read = sem_open(sem_read_name, 0);
        if (sem_var_read == SEM_FAILED) {
            perror("sem_open/variables");
            exit(EXIT_FAILURE);
        }

        // Get semaphore for said writing space, to post after reading
        char sem_write_name[MAX_LENGTH];
        sprintf(sem_write_name, "%s%d", SEM_WRITE_VARIABLE_FNAME, sharedData->writeIndex);

        sem_t *sem_var_write = sem_open(sem_write_name, 0);
        if (sem_var_write == SEM_FAILED) {
            perror("sem_open/variables");
            exit(EXIT_FAILURE);
        }

        // Read after checking if semaphore is open
        sem_wait(sem_var_read);

        // Print into console buffer index, character and time retreived
        int index = sharedData->readIndex;
        printf("Leyendo del buffer: \nbuffer[%d] = \"%c\" | tiempo: %s\n", index, buffer[index].character, buffer[index].time);

        dynamic_string[length] = buffer[index].character;
        printf("Archivo reconstruido: %s\n--------------------------\n", dynamic_string);

        // Add character to end of file
        fseek(file, 0, SEEK_END);
        fputc(buffer[index].character, file);

        // Borramos el caracter leido del buffer
        buffer[index].character = '\0';
        strcpy(buffer[index].time, "");
        
        // Actualizamos variables
        sharedData->charsTransferred++;
        sharedData->readIndex = (sharedData->readIndex + 1) % sharedData->bufferSize;

        // Cerramos archivo de reconstruccion
        fclose(file);

        // Posteamos los semaforos
        sem_post(sem_var_write);
        sem_post(sem_read);

        // Esperar el intervalo especificado antes de la próxima lectura
        sleep(interval);
    }
}



int main(int argc, char *argv[]) {
    // Verificar número de argumentos
    if (argc < 2 || argc > 3) {
        printf("Uso: %s <modo> [intervalo]s\n", argv[0]);
        printf("Modo: 0 = Manual, 1 = Automático\n");
        return -1;
    }

    int mode = atoi(argv[1]);
    int interval = 0;
    if (mode == 1 && argc == 3) {
        interval = atoi(argv[2]);
        if (interval <= 0) {
            printf("Intervalo no válido\n");
            return -1;
        }
    }

    // Open semaphores that were already created
    sem_t *sem_read = sem_open(SEM_READ_PROCESS_FNAME, 0);
    if (sem_read == SEM_FAILED) {
        perror("sem_open/creator");
        exit(EXIT_FAILURE);
    }
    sem_t *sem_write = sem_open(SEM_WRITE_PROCESS_FNAME, 0);
    if (sem_write == SEM_FAILED) {
        perror("sem_open/client");
        exit(EXIT_FAILURE);
    }

    // Connect to shared mem block
    SharedData *sharedData = attach_struct(STRUCT_LOCATION, sizeof(SharedData));
    if (sharedData == NULL) {
        printf("ERROR: no se pudo acceder al bloque\n");
        return -1;
    }

    Sentence *buffer = attach_buffer(BUFFER_LOCATION, (sharedData->bufferSize * sizeof(Sentence)));
    if (buffer == NULL) {
        printf("ERROR: no se pudo acceder al bloque\n");
        return -1;
    }

    // Realizar llamadas a funciones según el modo especificado
    if (mode == 0) {
        read_manually(sem_read, sem_write, sharedData, buffer);
    } else if (mode == 1) {
        read_automatically(sem_read, sem_write, sharedData, buffer, interval);
    } else {
        printf("Modo no válido\n");
        return -1;
    }

    // Detach from memory after finishing
    detach_struct(sharedData);
    detach_buffer(buffer);

    return 0;
}

