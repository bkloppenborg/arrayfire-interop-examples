
LIBS=-lOpenCL -lafopencl
INCLUDES=-I/usr/local/include
CXX=/usr/bin/g++ $(COMPILER_OPTIONS)
COMPILER_OPTIONS=-std=c++11 -g

all: opencl-from-af af-from-opencl

opencl-from-af: opencl-from-af.cpp
	$(CXX) opencl-from-af.cpp -o opencl-from-af $(INCLUDES) $(LIBS) $(LIB_PATHS)

af-from-opencl: af-from-opencl.cpp
	$(CXX) af-from-opencl.cpp -o af-from-opencl $(INCLUDES) $(LIBS) $(LIB_PATHS)

clean:
	rm opencl-from-af af-from-opencl
