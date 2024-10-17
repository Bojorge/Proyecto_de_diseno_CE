# Nombre del ejecutable
TARGET = graph_extractor

# Archivos fuente
SRCS = main.cpp graph.cpp extract_context.cpp graph_builder.cpp mock_context.cpp mock_graph.cpp graph_to_file.cpp

# Compilador
CXX = g++

# Opciones de compilación
CXXFLAGS = -std=c++11 -Ofast -Wall

# Regla principal para compilar el ejecutable
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Regla para limpiar los archivos generados
clean:
	rm -f $(TARGET)

# Regla para ejecutar el programa
run: $(TARGET)
	./$(TARGET) $(ARG)

# Regla para ejecutar el programa con argumento
run_with_arg: $(TARGET)
	@read -p "Ingrese el nombre del archivo GGUF: " ARG; ./$(TARGET) $$ARG

.PHONY: clean run run_with_arg
