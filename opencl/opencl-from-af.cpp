#include "arrayfire.h"
#include "af/opencl.h"

#include <stdexcept>
#include <cstdio>

using namespace std;

#define CHECK_OPENCL_ERROR(actual, msg) \
    if(actual != CL_SUCCESS) \
    { \
        std::cout << msg << std::endl; \
        std::cout << "Location : " << __FILE__ << ":" << __LINE__<< std::endl; \
        exit(1); \
    }

int main(int argc, char** argv)
{
	char t_device_name[64] = {0};
	char t_device_platform[64] = {0};
	char t_device_toolkit[64] = {0};
	char t_device_compute[64] = {0};
	af::deviceInfo(t_device_name, t_device_platform, t_device_toolkit, t_device_compute);

    printf("Device name: %s\n", t_device_name);
    printf("Platform name: %s\n", t_device_platform);
    printf("Toolkit: %s\n", t_device_toolkit);
    printf("Compute version: %s\n", t_device_compute);

// --- begin interop snippit -- //

    size_t length = 10;

    // Create ArrayFire array objects:
    af::array A = af::randu(length, f32);
    af::array B = af::constant(0, length, f32);

    cout << "Initial arrays:" << endl;
    af_print(A);
    af_print(B);

    // ... additional ArrayFire operations here

    // 2. Obtain the device, context, and queue used by ArrayFire
    cl_context af_context = afcl::getContext();
    cl_device_id af_device_id = afcl::getDeviceId();
    cl_command_queue af_queue = afcl::getQueue();

    // 3. Obtain cl_mem references to af::array objects
    cl_mem * d_A = A.device<cl_mem>();
    cl_mem * d_B = B.device<cl_mem>();

    // 4. Load, build, and use your kernels.
    //    For the sake of readability, we have omitted error checking.
    const char * kernel_name = "copy_kernel";
    int status = CL_SUCCESS;

    // A simple copy kernel, uses C++11 syntax for multi-line strings.
    const char * source = R"(
        void __kernel
        copy_kernel(__global float * gA, __global float * gB)
        {
            int id = get_global_id(0);
            gB[id] = gA[id];
        }
    )";

    // Create the program
    cl_program program = clCreateProgramWithSource(af_context, 1, &source, NULL, &status);
	if (!program || status != CL_SUCCESS)
	{
		size_t length;
		char build_log[2048];
		//printf("%s\n", block_source);
		cout<< "Error: Failed to create compute program from source: " << kernel_name << endl;
		status = clGetProgramBuildInfo(program, af_device_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, &length);
		CHECK_OPENCL_ERROR(status, "clGetProgramBuildInfo failed.");
		printf("%s\n", build_log);

	    exit(1);
    }

	// Build the program executable
	status = clBuildProgram(program, 1, &af_device_id, NULL, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		size_t length;
		char build_log[2048];
		//printf("%s\n", block_source);
		cout << "Error: Failed to build compute program: " << kernel_name << endl;
		status = clGetProgramBuildInfo(program, af_device_id, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, &length);
		CHECK_OPENCL_ERROR(status, "clGetProgramBuildInfo failed.");
		printf("%s\n", build_log);

	    exit(1);
	}

    // Extract symbols
    cl_kernel kernel = clCreateKernel(program, kernel_name, &status);

    // 5. Set arguments and launch your kernels
    clSetKernelArg(kernel, 0, sizeof(cl_mem), d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), d_B);
	clEnqueueNDRangeKernel(af_queue, kernel, 1, NULL, &length, NULL, 0, NULL, NULL);

    // 6. Return control of af::array memory to ArrayFire
    A.unlock();
    B.unlock();

    // ... resume ArrayFire operations
    af_print(B);

    // Because the device pointers, d_x and d_y, were returned to ArrayFire's
    // control by the unlock function, there is no need to free them using
    // cudaFree()

    return 0;
}
