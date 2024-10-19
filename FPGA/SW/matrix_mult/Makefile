# Variables
CXX := g++
CXXFLAGS := -std=c++17 -I/opt/xilinx/xrt/include
LDFLAGS := -L/opt/xilinx/xrt/lib -lxrt_core -lxrt_cu -lxrt_kernel -lpthread -ldl
SRC := matrix_mult_xrt.cpp
OBJ := $(SRC:.cpp=.o)
EXEC := matrix_mult_xrt

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

