# Nombre del ejecutable
TARGET_SERVER = server
TARGET_CLIENT = client

# Compiladores
CC = gcc
CXX = g++

# Archivos fuente
C_SOURCES = posix_sockets.c
CPP_SOURCES_SERVER = server.cpp
CPP_SOURCES_CLIENT = client.cpp

# Archivos objeto
C_OBJECTS = $(C_SOURCES:.c=.o)
CPP_OBJECTS_SERVER = $(CPP_SOURCES_SERVER:.cpp=.o)
CPP_OBJECTS_CLIENT = $(CPP_SOURCES_CLIENT:.cpp=.o)

# Reglas de construcción
all: $(TARGET_SERVER) $(TARGET_CLIENT)

$(TARGET_SERVER): $(C_OBJECTS) $(CPP_OBJECTS_SERVER)
	$(CXX) -o $@ $^

$(TARGET_CLIENT): $(C_OBJECTS) $(CPP_OBJECTS_CLIENT)
	$(CXX) -o $@ $^

%.o: %.c
	$(CC) -c $< -o $@

%.o: %.cpp
	$(CXX) -c $< -o $@

clean:
	rm -f $(C_OBJECTS) $(CPP_OBJECTS_SERVER) $(CPP_OBJECTS_CLIENT) $(TARGET_SERVER) $(TARGET_CLIENT)

# Reglas para evitar que se consideren los archivos de cabecera
.PHONY: all clean
