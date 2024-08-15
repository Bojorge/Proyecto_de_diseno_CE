# Compilador
CXX = g++

# Opciones de compilación
CXXFLAGS = -std=c++17 -Wall -I/usr/include/poco

# Opciones de enlace
LIBS = -L/usr/lib/x86_64-linux-gnu -lPocoFoundation -lPocoUtil 

# Archivos fuente
SRCS = creator.cpp MySharedMemoryHandler.cpp

# Archivos objeto
OBJS = $(SRCS:.cpp=.o)

# Nombre del ejecutable
TARGET = program

# Regla por defecto
all: $(TARGET)

# Regla para crear el ejecutable
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LIBS)

# Regla para compilar archivos fuente a archivos objeto
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regla para limpiar los archivos generados
clean:
	rm -f $(TARGET) $(OBJS)
