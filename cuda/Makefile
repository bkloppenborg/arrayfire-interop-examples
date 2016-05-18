NVCC := /usr/local/cuda/bin/nvcc
CUDA_LIB_PATH := /usr/local/cuda/lib64/
CUDA_NVVM_PATH := /usr/local/cuda/nvvm/lib64/
LDFLAGS   += -L$(CUDA_LIB_PATH) -L$(CUDA_NVVM_PATH) -lcudart -lafcuda -lnvvm -Xlinker -rpath,$(CUDA_LIB_PATH),-rpath,$(CUDA_NVVM_PATH)
INCLUDES := -I/usr/local/include/

all: af-from-cuda cuda-from-af

cuda-from-af: cuda-from-af.cu
	${NVCC} -arch=sm_35 cuda-from-af.cu -o cuda-from-af $(LDFLAGS)

af-from-cuda: af-from-cuda.cu
	${NVCC} -arch=sm_35 af-from-cuda.cu -o af-from-cuda $(LDFLAGS)

clean:
	rm af-from-cuda cuda-from-af
