CXX     = g++
CXXFLAGS = -g -std=c++17 -I/usr/include/boost
LDFLAGS  = -lboost_system -lrt
OBJS    = shared_memory.o
EXE     = creator writer reader

# Regla por defecto
all: $(EXE)

# Regla para compilar los archivos objeto
shared_memory.o: shared_memory.cpp
	$(CXX) $(CXXFLAGS) -c shared_memory.cpp -o shared_memory.o

# Reglas para compilar los ejecutables
creator: creator.cpp shared_memory.o
	$(CXX) $(CXXFLAGS) creator.cpp shared_memory.o -o creator $(LDFLAGS)

writer: writer.cpp shared_memory.o
	$(CXX) $(CXXFLAGS) writer.cpp shared_memory.o -o writer $(LDFLAGS)

reader: reader.cpp shared_memory.o
	$(CXX) $(CXXFLAGS) reader.cpp shared_memory.o -o reader $(LDFLAGS)

# Regla para limpiar los archivos generados
clean:
	rm -f *.o $(EXE)
