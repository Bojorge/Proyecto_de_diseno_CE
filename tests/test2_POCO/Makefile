# Compilador
CXX = g++

# Opciones de compilación
CXXFLAGS = -std=c++17 -Wall -I/usr/include/poco

# Opciones de enlace
LIBS = -L/usr/lib/x86_64-linux-gnu -lPocoFoundation -lPocoUtil

# Archivos fuente
SRCS_MANAGER = shared_memory_manager.cpp
SRCS_CLIENT = shared_memory_client.cpp
SRCS_HANDLER = MySharedMemoryHandler.cpp
OBJS_MANAGER = $(SRCS_MANAGER:.cpp=.o)
OBJS_CLIENT = $(SRCS_CLIENT:.cpp=.o)
OBJS_HANDLER = $(SRCS_HANDLER:.cpp=.o)

# Nombre de los ejecutables
TARGET_MANAGER = shared_memory_manager
TARGET_CLIENT = shared_memory_client

# Regla por defecto
all: $(TARGET_MANAGER) $(TARGET_CLIENT)

# Regla para crear el ejecutable del gestor de memoria compartida
$(TARGET_MANAGER): $(OBJS_MANAGER) $(OBJS_HANDLER)
	$(CXX) $(CXXFLAGS) $(OBJS_MANAGER) $(OBJS_HANDLER) -o $(TARGET_MANAGER) $(LIBS)

# Regla para crear el ejecutable del cliente de memoria compartida
$(TARGET_CLIENT): $(OBJS_CLIENT) $(OBJS_HANDLER)
	$(CXX) $(CXXFLAGS) $(OBJS_CLIENT) $(OBJS_HANDLER) -o $(TARGET_CLIENT) $(LIBS)

# Regla para compilar archivos fuente a archivos objeto
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Regla para limpiar los archivos generados
clean:
	rm -f $(TARGET_MANAGER) $(TARGET_CLIENT) $(OBJS_MANAGER) $(OBJS_CLIENT) $(OBJS_HANDLER)
