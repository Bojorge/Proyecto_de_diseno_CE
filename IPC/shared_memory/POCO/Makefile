CXX     = g++
CXXFLAGS = -g -I/path/to/poco/include -L/path/to/poco/lib -lPocoFoundation -lPocoUtil -lPocoNamedSem -lPocoSharedMemory
OBJS    = shared_memory.o creator.o writer.o reader.o
EXE     = creator writer reader

# Regla por defecto
all: $(EXE)

# Regla para compilar los archivos objeto
shared_memory.o: shared_memory.cpp
	$(CXX) $(CXXFLAGS) -c shared_memory.cpp -o shared_memory.o

creator.o: creator.cpp
	$(CXX) $(CXXFLAGS) -c creator.cpp -o creator.o

writer.o: writer.cpp
	$(CXX) $(CXXFLAGS) -c writer.cpp -o writer.o

reader.o: reader.cpp
	$(CXX) $(CXXFLAGS) -c reader.cpp -o reader.o

# Reglas para compilar los ejecutables
creator: creator.o shared_memory.o
	$(CXX) $(CXXFLAGS) creator.o shared_memory.o -o creator

writer: writer.o shared_memory.o
	$(CXX) $(CXXFLAGS) writer.o shared_memory.o -o writer

reader: reader.o shared_memory.o
	$(CXX) $(CXXFLAGS) reader.o shared_memory.o -o reader

# Regla para limpiar los archivos generados
clean:
	rm -f *.o $(EXE)
