LIBS=-lOpenCL -lafopencl -lGL -lglfw
INCLUDES=-I/usr/local/include
CXX=/usr/bin/g++ $(COMPILER_OPTIONS)
COMPILER_OPTIONS=-std=c++11 -g

all: opengl-from-af 

opengl-from-af: opengl-from-af.cpp
	$(CXX) opengl-from-af.cpp -o opengl-from-af $(INCLUDES) $(LIBS) $(LIB_PATHS)

clean:
	rm opengl-from-af 
