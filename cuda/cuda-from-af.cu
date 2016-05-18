// 1. Add includes
#include <arrayfire.h>
#include <af/cuda.h>

// A simple kernel to increment a value by 1.
__global__ void increment(float * values)
{
  int global_id = blockIdx.x * blockDim.x + threadIdx.x;
  values[global_id] += 1;
}

int main() {
    size_t num = 10;
    
    // Create ArrayFire array objects:
    af::array x = af::constant(0, num);

    // ... many ArrayFire operations here

    // 2. Ensure any JIT kernels have executed
    x.eval();
    af_print(x);

    // Run a custom CUDA kernel in the ArrayFire CUDA stream

    // 3. Obtain device pointers from ArrayFire array objects using
    //    the array::device() function:
    float *d_x = x.device<float>();

    // 4. Determine ArrayFire's CUDA stream
    int af_id = af::getDevice();
    int cuda_id = afcu::getNativeId(af_id);
    cudaStream_t af_cuda_stream = afcu::getStream(cuda_id);

    // 5. Set arguments and run your kernel in ArrayFire's stream
    //    Here launch with 10 blocks of 10 threads
    increment<<<1, num, 0, af_cuda_stream>>>(d_x);

    // 6. Return control of af::array memory to ArrayFire using
    //    the array::unlock() function:
    x.unlock();

    // ... resume ArrayFire operations
    af_print(x);

    // Because the device pointers, d_x and d_y, were returned to ArrayFire's
    // control by the unlock function, there is no need to free them using
    // cudaFree()

    return 0;
}
