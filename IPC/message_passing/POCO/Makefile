# Variables
CXX = g++
CXXFLAGS = -std=c++17 -Wall -I/usr/include/poco # Ajusta la ruta si es necesario
LDFLAGS = -lPocoNet -lPocoFoundation
SRCS = client.cpp server.cpp sockets.cpp
OBJS = $(SRCS:.cpp=.o)
EXEC_CLIENT = client
EXEC_SERVER = server

# Regla principal para construir los ejecutables
all: $(EXEC_CLIENT) $(EXEC_SERVER)

# Regla para construir el ejecutable client
$(EXEC_CLIENT): client.o sockets.o
	$(CXX) $(CXXFLAGS) -o $(EXEC_CLIENT) client.o sockets.o $(LDFLAGS)

# Regla para construir el ejecutable server
$(EXEC_SERVER): server.o sockets.o
	$(CXX) $(CXXFLAGS) -o $(EXEC_SERVER) server.o sockets.o $(LDFLAGS)

# Regla para construir el archivo objeto de client
client.o: client.cpp sockets.hpp
	$(CXX) $(CXXFLAGS) -c client.cpp

# Regla para construir el archivo objeto de server
server.o: server.cpp sockets.hpp
	$(CXX) $(CXXFLAGS) -c server.cpp

# Regla para construir el archivo objeto de sockets
sockets.o: sockets.cpp sockets.hpp
	$(CXX) $(CXXFLAGS) -c sockets.cpp

# Regla para limpiar los archivos generados
clean:
	rm -f $(OBJS) $(EXEC_CLIENT) $(EXEC_SERVER)

run: $(TARGETS)
	gnome-terminal -- bash -c "./server; exec bash" & \
	gnome-terminal -- bash -c "./client; exec bash" &
