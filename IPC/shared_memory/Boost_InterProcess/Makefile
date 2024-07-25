# Nombre de los ejecutables
PRODUCER = producer
CONSUMER = consumer

# Compiladores
CXX = g++

# Opciones de compilación
CXXFLAGS = -I/usr/include -lrt -lboost_system -lboost_thread -lpthread

# Archivos fuente
CPP_SOURCES = producer.cpp consumer.cpp shared_memory.hpp

# Reglas de construcción
all: $(PRODUCER) $(CONSUMER)

$(PRODUCER): producer.cpp shared_memory.hpp
	$(CXX) -o $(PRODUCER) producer.cpp $(CXXFLAGS)

$(CONSUMER): consumer.cpp shared_memory.hpp
	$(CXX) -o $(CONSUMER) consumer.cpp $(CXXFLAGS)

clean:
	rm -f $(PRODUCER) $(CONSUMER)

# Reglas para evitar que se consideren los archivos de cabecera
.PHONY: all clean
