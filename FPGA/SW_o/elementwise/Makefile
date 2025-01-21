CXX := g++
CXXFLAGS := -std=c++17 -I/usr/include/xrt -I/usr/include/xrt/deprecated
#LDFLAGS := -L/usr/lib/xrt -lxrt_core -lxrt_cu -lxrt_kernel -lpthread -ldl
LDFLAGS = -L/usr/lib/xrt -lxrt_core -lxrt_coreutil -lxrt++ -lxrt_swemu
SRC := elementwise_xrt.cpp
OBJ := $(SRC:.cpp=.o)
EXEC := elementwise_xrt

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(EXEC)

