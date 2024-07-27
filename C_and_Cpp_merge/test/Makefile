# Nombre del ejecutable
TARGET = programa

# Compiladores
CC = gcc
CXX = g++

# Archivos fuente
C_SOURCES = funciones_c.c
CPP_SOURCES = main.cpp

# Archivos objeto
C_OBJECTS = $(C_SOURCES:.c=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)

# Reglas de construcción
all: $(TARGET)

$(TARGET): $(C_OBJECTS) $(CPP_OBJECTS)
	$(CXX) -o $@ $^

%.o: %.c
	$(CC) -c $< -o $@

%.o: %.cpp
	$(CXX) -c $< -o $@

clean:
	rm -f $(C_OBJECTS) $(CPP_OBJECTS) $(TARGET)

# Reglas para evitar que se consideren los archivos de cabecera
.PHONY: all clean
