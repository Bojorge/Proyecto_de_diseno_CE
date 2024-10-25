# Variables
CXX := g++
#CXXFLAGS := -std=c++17 -I/opt/xilinx/xrt/include
#LDFLAGS := -L/opt/xilinx/xrt/lib -lxrt_core -lxrt_cu -lxrt_kernel -lpthread -ldl

CXXFLAGS := -std=c++17 -I/usr/include/xrt
LDFLAGS := -L/usr/lib/xrt -lxrt_core -lxrt_cu -lxrt_kernel -lpthread -ldl

SRC := matvecmul_xrt.cpp
OBJ := $(SRC:.cpp=.o)
EXEC := matvecmul_xrt

# Default target
all: $(EXEC)

# Compile the application
$(EXEC): $(OBJ)
	$(CXX) -o $@ $(OBJ) $(LDFLAGS)

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# Clean build artifacts
clean:
	rm -f $(OBJ) $(EXEC)

