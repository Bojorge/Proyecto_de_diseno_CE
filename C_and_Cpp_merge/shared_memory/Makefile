# Nombre del ejecutable
TARGET_PRODUCER = producer
TARGET_CONSUMER = consumer

# Compiladores
CC = gcc
CXX = g++

# Archivos fuente
C_SOURCES = posix_shared_memory.c
CPP_SOURCES_PRODUCER = producer.cpp
CPP_SOURCES_CONSUMER = consumer.cpp

# Archivos objeto
C_OBJECTS = $(C_SOURCES:.c=.o)
CPP_OBJECTS_PRODUCER = $(CPP_SOURCES_PRODUCER:.cpp=.o)
CPP_OBJECTS_CONSUMER = $(CPP_SOURCES_CONSUMER:.cpp=.o)

# Reglas de construcción
all: $(TARGET_PRODUCER) $(TARGET_CONSUMER)

$(TARGET_PRODUCER): $(C_OBJECTS) $(CPP_OBJECTS_PRODUCER)
	$(CXX) -o $@ $^

$(TARGET_CONSUMER): $(C_OBJECTS) $(CPP_OBJECTS_CONSUMER)
	$(CXX) -o $@ $^

%.o: %.c
	$(CC) -c $< -o $@

%.o: %.cpp
	$(CXX) -c $< -o $@

clean:
	rm -f $(C_OBJECTS) $(CPP_OBJECTS_PRODUCER) $(CPP_OBJECTS_CONSUMER) $(TARGET_PRODUCER) $(TARGET_CONSUMER)

# Reglas para evitar que se consideren los archivos de cabecera
.PHONY: all clean
