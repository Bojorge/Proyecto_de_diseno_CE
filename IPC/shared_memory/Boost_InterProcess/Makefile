# Compilador
CXX = g++

# Opciones de compilación
CXXFLAGS = -std=c++17 -Wall -I/usr/include/boost -Ofast

# Opciones de enlace
LIBS = -L/usr/lib/x86_64-linux-gnu -lboost_system -lboost_filesystem

# Archivos fuente
SRCS = creator_writer.cpp reader.cpp

# Archivos objeto
OBJS = $(SRCS:.cpp=.o)

# Nombre de los ejecutables
TARGETS = creator_writer reader

# Regla por defecto
all: $(TARGETS)

# Regla para crear los ejecutables
$(TARGETS): %: %.o
	$(CXX) $(CXXFLAGS) $< -o $@ $(LIBS)

# Regla para compilar archivos fuente a archivos objeto
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regla para limpiar los archivos generados
clean:
	rm -f $(TARGETS) $(OBJS)

run: $(TARGETS)
	gnome-terminal -- bash -c "./creator_writer; exec bash" & \
	gnome-terminal -- bash -c "./reader; exec bash" &

.PHONY: all clean run
