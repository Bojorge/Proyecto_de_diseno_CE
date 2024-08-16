# Compilador
CXX = g++

# Opciones de compilación
CXXFLAGS = -std=c++17 -Wall -I/usr/include/poco

# Opciones de enlace
LIBS = -L/usr/lib/x86_64-linux-gnu -lPocoFoundation -lPocoUtil
# Archivos fuente
SRCS = creator.cpp reader.cpp

# Archivos objeto
OBJS = $(SRCS:.cpp=.o)

# Nombre de los ejecutables
TARGETS = creator reader

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
