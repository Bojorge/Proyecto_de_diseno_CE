CXX := g++
CXXFLAGS := -std=c++17 -I/usr/include/xrt
LDFLAGS := -L/usr/lib/xrt -lxrt_core -lxrt_cu -lxrt_kernel -lpthread -ldl
SRC := matrix_mult_xrt.cpp
OBJ := $(SRC:.cpp=.o)
EXEC := matrix_mult_xrt

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(EXEC)
