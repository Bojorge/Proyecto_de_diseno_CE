# Archivos fuente y ejecutables
SRC1 = matrix_mult_stdlib.cpp
SRC2 = sys_resource_test.cpp
TARGET1 = matrix_mult_stdlib
TARGET2 = sys_resource_test

# Flags del compilador
CXXFLAGS = -Wall

# Regla principal para compilar ambos ejecutables
all: $(TARGET1) $(TARGET2)

# Regla para compilar matrix_mult_stdlib
$(TARGET1): $(SRC1)
	$(CXX) $(CXXFLAGS) -o $(TARGET1) $(SRC1)

# Regla para compilar sys_resource_test
$(TARGET2): $(SRC2)
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(SRC2)

# Regla para limpiar archivos generados
clean:
	rm -f $(TARGET1) $(TARGET2) *.o
