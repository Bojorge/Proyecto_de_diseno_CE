# Nombre del compilador
CXX = g++

# Flags de compilación
CXXFLAGS = -std=c++17 -Wall -I/usr/include

# Librerías necesarias
LDFLAGS = -lrt -lpthread -lboost_system -lboost_filesystem

# Ejecutables
EXECUTABLES = P1 P2

# Archivos fuente
SOURCES_P1 = Process_1.cpp
SOURCES_P2 = Process_2.cpp

# Objetivos
all: $(EXECUTABLES)

P1: $(SOURCES_P1)
	$(CXX) $(CXXFLAGS) $(SOURCES_P1) -o P1 $(LDFLAGS)

P2: $(SOURCES_P2)
	$(CXX) $(CXXFLAGS) $(SOURCES_P2) -o P2 $(LDFLAGS)

# Limpiar compilaciones previas
clean:
	rm -f $(EXECUTABLES) *.o

# Eliminar el objeto de memoria compartida
remove_shared_memory:
	rm -f /dev/shm/SharedMem

.PHONY: all clean remove_shared_memory
