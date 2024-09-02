# Nombre del ejecutable
TARGET = matrix_mult_stdlib

# Archivos fuente
SRC = matrix_mult_stdlib.cpp

# Flags del compilador
CXXFLAGS = -Wall

# Regla principal para compilar el ejecutable
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Regla para limpiar archivos generados
clean:
	rm -f $(TARGET) *.o
