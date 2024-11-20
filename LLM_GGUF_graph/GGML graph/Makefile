# Compilador
CC = gcc

# Flags del compilador
CFLAGS = -Wall -Wextra -std=c11 -O2 -g -pg  # Añadir -pg para habilitar gprof
LDFLAGS = -lm -pg  # Añadir -pg también aquí

# Archivo de salida
TARGET = gguf_context

# Archivos fuente
SRCS = gguf_context.c

# Archivo de entrada (puedes sobrescribirlo desde la línea de comandos)
FILE = ../law-llm.Q2_K.gguf

# Regla de construcción
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS) $(LDFLAGS)

# Regla para generar el árbol de llamadas completo
call_tree: $(TARGET)
	@echo "Ejecutando el programa para generar gmon.out..."
	./$(TARGET) $(FILE)  # Ejecuta el programa con el archivo de entrada
	@echo "Generando archivo de texto con gprof..."
	gprof -b -p $(TARGET) > call_tree.txt  # Genera el informe detallado de tiempos (flat profile)
	gprof -b -q $(TARGET) >> call_tree.txt  # Añade el árbol de llamadas completo
	@echo "Generando archivo .dot con gprof2dot..."
	gprof $(TARGET) gmon.out | gprof2dot -o call_tree.dot  # Genera el archivo .dot
	@echo "Generando gráfico visual del árbol de llamadas..."
	dot -Tpng call_tree.dot -o call_tree.png  # Convierte el archivo .dot en un archivo .png
	@echo "Árbol de llamadas generado como call_tree.png y call_tree.dot"


# Limpiar archivos compilados y de perfilado
clean:
	rm -f $(TARGET) gmon.out call_tree.txt call_tree.png
