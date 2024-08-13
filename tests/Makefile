# Definir el compilador y las banderas
CXX = g++
CXXFLAGS = -g -std=c++17 -I/usr/include/boost
LDFLAGS = -lboost_system -lrt
OBJS = shared_memory.o creator.o
EXE = creator

# Regla por defecto
all: $(EXE)

# Regla para compilar el archivo objeto shared_memory.o
shared_memory.o: shared_memory.cpp shared_memory.hpp
	$(CXX) $(CXXFLAGS) -c shared_memory.cpp -o shared_memory.o

# Regla para compilar el archivo objeto creator.o
creator.o: creator.cpp shared_memory.hpp
	$(CXX) $(CXXFLAGS) -c creator.cpp -o creator.o

# Regla para construir el ejecutable creator
creator: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o creator $(LDFLAGS)

# Regla para limpiar los archivos generados
clean:
	rm -f *.o $(EXE)
