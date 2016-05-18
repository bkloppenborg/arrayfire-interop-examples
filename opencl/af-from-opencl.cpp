// 1. Add arrayfire.h and af/opencl.h to your application
#include "arrayfire.h"
#include "af/opencl.h"

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120    // Use OpenCL 1.2 specs (for NVIDIA)
#include <CL/cl2.hpp>

#include <stdexcept>
#include <cstdio>
#include <vector>

using namespace std;

int main(int argc, char** argv)
{
  cl::Context context(CL_DEVICE_TYPE_ALL);
  vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
  cl::Device device = devices[0];

  cl::CommandQueue queue(context, device);

  // Create a buffer of size 10 filled with zeros, copy it to the device
  int length = 10;
  vector<float> h_A(length, 0);
  cl::Buffer cl_A(context, CL_MEM_READ_WRITE, length * sizeof(float), h_A.data());

  // 2. Instruct OpenCL to complete its operations using clFinish (or similar)
  queue.finish();

  // 3. Instruct ArrayFire to use the user-created context
  // Create a device from the current OpenCL device + context + queue
  afcl::addDevice(device(), context(), queue());
  // Switch to the device
  afcl::setDevice(device(), context());

  // 3. Create ArrayFire arrays from OpenCL memory objects
  af::array af_A = afcl::array(length, cl_A(), f32, true);

  // 4. Perform ArrayFire operations on the Arrays
  af_A = af::randu(length);

  // 5. Instruct ArrayFire to finish operations using af::sync
  af::sync();

  // 6. Obtain cl_mem references for important memory
  cl_A = *af_A.device<cl_mem>();

  // 7. Continue your OpenCL application
  queue.enqueueReadBuffer(cl_A, CL_TRUE, 0, h_A.size() * sizeof(float), h_A.data());

  // Print out the results
  for(int i = 0; i < h_A.size(); i++)
    cout << i << " " << h_A[i] << endl;

  return 0;
}
