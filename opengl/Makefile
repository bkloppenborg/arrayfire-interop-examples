INCLUDES=-I/usr/local/include

COMPILER_OPTIONS=-std=c++11 -g
CXX=/usr/bin/g++ $(COMPILER_OPTIONS)

COMMON_LIBS=-lGL -lglfw
CL_LIBS=-lOpenCL -lafopencl

NVCC := /usr/local/cuda/bin/nvcc $(COMPILER_OPTIONS)
CUDA_LIB_PATH := /usr/local/cuda/lib64/
CUDA_NVVM_PATH := /usr/local/cuda/nvvm/lib64/
CUDA_LDFLAGS   += -L$(CUDA_LIB_PATH) -L$(CUDA_NVVM_PATH) -lcudart -lafcuda -lnvvm -Xlinker -rpath,$(CUDA_LIB_PATH),-rpath,$(CUDA_NVVM_PATH)

all: opengl-opencl-from-af opengl-cuda-from-af

opengl-opencl-from-af: opengl-opencl-from-af.cpp afopengl-opencl.cpp
	$(CXX) opengl-opencl-from-af.cpp afopengl-opencl.cpp -o opengl-opencl-from-af $(INCLUDES) $(CL_LIBS) $(COMMON_LIBS) $(LIB_PATHS)

opengl-cuda-from-af: opengl-cuda-from-af.cpp afopengl-cuda.cpp
	$(NVCC) opengl-cuda-from-af.cpp afopengl-cuda.cpp -o opengl-cuda-from-af $(INCLUDES) $(CUDA_LDFLAGS) $(COMMON_LIBS) $(LIB_PATHS)

clean:
	rm opengl-opencl-from-af opengl-cuda-from-af
