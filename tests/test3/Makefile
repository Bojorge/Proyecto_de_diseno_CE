# Nombre de los ejecutables
TARGETS = shared_memory user_memory

# Compilador y flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -I/usr/include
LDFLAGS = -lrt -lpthread -lboost_system -lboost_filesystem

# Objetivos
all: $(TARGETS)

# Reglas para compilar cada programa
shared_memory: shared_memory.o
	$(CXX) -o $@ $^ $(LDFLAGS)

user_memory: user_memory.o
	$(CXX) -o $@ $^ $(LDFLAGS)

# Reglas para compilar los archivos fuente
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regla para limpiar los archivos generados
clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
