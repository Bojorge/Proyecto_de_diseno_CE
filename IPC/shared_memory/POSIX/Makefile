# Definición de variables
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Ofast
LDFLAGS =

# Nombres de los archivos fuente y ejecutables
SRCS = creator_writer.cpp reader.cpp
OBJS = $(SRCS:.cpp=.o)
EXES = creator_writer reader

# Regla por defecto
all: $(EXES)

# Regla para compilar el ejecutable creator_writer
creator_writer: creator_writer.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Regla para compilar el ejecutable reader
reader: reader.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Regla para compilar archivos .cpp en archivos .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Regla para limpiar archivos generados
clean:
	rm -f $(OBJS) $(EXES)

run: $(TARGETS)
	gnome-terminal -- bash -c "./creator_writer; exec bash" & \
	gnome-terminal -- bash -c "./reader; exec bash" &

.PHONY: all clean run
