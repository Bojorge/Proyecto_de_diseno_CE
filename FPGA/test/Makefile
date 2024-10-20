# Nombre del compilador
CXX = g++

# Flags del compilador
CXXFLAGS = -std=c++11 -Wall

# Archivos fuente
SRC = main.cpp fixed_point.cpp

# Archivo ejecutable
TARGET = fixed_point_test

# Regla por defecto
all: $(TARGET)

# Regla para compilar el programa
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Regla para limpiar los archivos generados
clean:
	rm -f $(TARGET)
