# Compilador
CC = gcc

# Flags del compilador
CFLAGS = -Wall -Wextra -std=c11 -O2 -g

# Archivo de salida
TARGET = my_code

# Archivos fuente
SRCS = my_code.c

# Regla de construcción
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)

# Limpiar archivos compilados
clean:
	rm -f $(TARGET)
