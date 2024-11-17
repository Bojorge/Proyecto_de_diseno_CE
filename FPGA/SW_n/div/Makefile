CXX := g++
CXXFLAGS := -std=c++17 -I/usr/include/xrt
#LDFLAGS := -L/usr/lib/xrt -lxrt_core -lxrt++ -lpthread -ldl
LDFLAGS = -L/usr/lib/xrt -lxrt_core -lxrt_coreutil -lxrt++ -lxrt_swemu
SRC := div_xrt.cpp
OBJ := $(SRC:.cpp=.o)
EXEC := div

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) -o $@ $(OBJ) $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJ) $(EXEC)
