# g++ -o matrix_mult_app matrix_mult_app.cpp -lxrt_core -lxrt_coreutil -lhls_half -std=c++17


# Nombre del ejecutable que se generará
TARGET = matrix_mult_app

# Archivos fuente
SRC = main.cpp

# Flags del compilador
# -Wall: Habilita todos los warnings del compilador para ayudar en la depuración.
CXXFLAGS = -Wall

# Librerías necesarias
# -lxrt_core: Enlaza la aplicación con la biblioteca XRT core, que es necesaria para interactuar con la FPGA.
LDLIBS = -lxrt_core

# Regla principal del Makefile que especifica cómo compilar el ejecutable.
# Este comando indica que el archivo binario se compila a partir de los archivos fuente
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDLIBS)

# Regla para limpiar el directorio de los archivos generados (.o y el ejecutable).
# Se usa típicamente antes de recompilar todo el proyecto desde cero.
clean:
	rm -f $(TARGET) *.o
