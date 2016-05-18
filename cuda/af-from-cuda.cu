
// 1. Add includes
#include <arrayfire.h>
#include <af/cuda.h>

#include <vector>
using namespace std;

template <typename T>
void print_cuda_array(T * cuda_array, int length)
{
    vector<T> temp;
    temp.resize(length);
    cudaMemcpy(temp.data(), cuda_array, length * sizeof(T), cudaMemcpyDeviceToHost);
    for(int i = 0; i < temp.size(); i++)
        cout << i << " " << temp[i] << endl;
}

int main() {

    // Create and populate CUDA memory objects
    const int elements = 10;
    size_t size = elements * sizeof(float);
    vector<float> h_A(0, elements);
    float *cuda_A;
    cudaMalloc((void**) &cuda_A, size);
    cudaMemcpy(cuda_A, h_A.data(), size, cudaMemcpyHostToDevice);

    cout << "cuda_A input" << endl;
    print_cuda_array<float>(cuda_A, elements);

    // ... perform many CUDA operations here

    // 2. Finish any pending CUDA operations
    cudaDeviceSynchronize();

    // 3. Create ArrayFire arrays from existing CUDA pointers.
    //    Be sure to specify that the memory type is afDevice.
    af::array d_A(elements, cuda_A, afDevice);
    af_print(d_A);

    // NOTE: ArrayFire now manages cuda_A

    // 4. Perform operations on the ArrayFire Arrays.
    d_A = d_A * 2;

    // NOTE: ArrayFire does not perform the above transaction using
    // in-place memory, thus the pointers containing memory to d_A have
    // likely changed.

    // 5. Instruct ArrayFire to finish pending operations
    af::eval(d_A);
    af::sync();

    // 6. Get pointers to important memory objects.
    //    Once device is called, ArrayFire will not manage the memory.
    float * outputValue = d_A.device<float>();

    // 7. continue CUDA application as normal
    cout << "outputValue" << endl;
    print_cuda_array<float>(outputValue, elements);

    // 8. Free non-managed memroy
    //    We removed outputValue from ArrayFire's control, we need to free it
    cudaFree(outputValue);

    return 0;
}
