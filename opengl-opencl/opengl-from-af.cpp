#include "arrayfire.h"
// Add OpenCL interop header
#include "af/opencl.h"

// Pull in OpenCL
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120    // Use OpenCL 1.2 specs (for NVIDIA)
#include <CL/cl2.hpp>

#include <GLFW/glfw3.h>
#include <GL/glx.h>

#include <iostream>
using namespace std;

// Get the properties required to enable OpenCL-OpenGL interop on the specified
// platform
vector<cl_context_properties> get_interop_properties(cl_platform_id platform)
{
    // Allocate enough space for defining the parameters below:
    vector<cl_context_properties> properties(7);

#if defined (__APPLE__) || defined(MACOSX)	// Apple / OSX

    CGLContextObj context = CGLGetCurrentContext();
    CGLShareGroupObj share_group = CGLGetShareGroup(context);

    if(context != NULL && share_group != NULL)
    {
        properties[0] = CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE;
        properties[1] = (cl_context_properties) share_group;
        properties[2] = CL_CONTEXT_PLATFORM;
        properties[3] = (cl_context_properties) platform;
        properties[4] = 0;
    }

#elif defined WIN32 // Windows

    HGLRC WINAPI context = wglGetCurrentContext();
    HDC WINAPI dc = wglGetCurrentDC();

    if(context != NULL && dc != NULL)
    {
        properties[0] = CL_GL_CONTEXT_KHR;
        properties[1] = (cl_context_properties) context;
        properties[2] = CL_WGL_HDC_KHR;
        properties[3] = (cl_context_properties) dc;
        properties[4] = CL_CONTEXT_PLATFORM;
        properties[5] = (cl_context_properties) platform;
        properties[6] = 0;
    }

#else	// Linux

    GLXContext context = glXGetCurrentContext();
    Display * display = glXGetCurrentDisplay();

    if(context != NULL && display != NULL)
    {
        // Enable an OpenCL - OpenGL interop session.
        // This works for an X11 OpenGL session on Linux.
        properties[0] = CL_GL_CONTEXT_KHR;
        properties[1] = (cl_context_properties) context;
        properties[2] = CL_GLX_DISPLAY_KHR;
        properties[3] = (cl_context_properties) display;
        properties[4] = CL_CONTEXT_PLATFORM;
        properties[5] = (cl_context_properties) platform;
        properties[6] = 0;
    }

#endif

    return properties;
}

int main(int argc, char** argv)
{
    // 1. Create an OpenGL window
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit())
        return -1;

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);

    // 2. Construct an OpenCL context
    // a. Get properties for OpenCL-OpenGL interop on the current platform
    cl::Platform ocl_platform = cl::Platform::getDefault();
    vector<cl_context_properties> properties = get_interop_properties(ocl_platform());

    // b. Construct an OpenCL device corresponding to the interop device 
    size_t input_size = 8;
    size_t output_size = 0;
    cl_device_id devices[input_size];
    cl_int status = clGetGLContextInfoKHR(properties.data(), CL_DEVICES_FOR_GL_CONTEXT_KHR,
                                          input_size,
                                          devices, &output_size);
    // Assume the zeroth device matches, this may not always be true.
    cl::Device ocl_device = cl::Device(devices[0]);

    // Create OpenCL context and queue from the device and properties obtained above
    cl::Context ocl_context(ocl_device, properties.data());
    cl::CommandQueue queue(ocl_context, ocl_device);

    // 3. Register the device with ArrayFire
    // Create a device from the current OpenCL device + context + queue
    afcl::addDevice(ocl_device(), ocl_context(), queue());
    // Switch to the device
    afcl::setDevice(ocl_device(), ocl_context());

    // (optional) print out information about the device provided to ArrayFire
    char t_device_name[64] = {0};
    char t_device_platform[64] = {0};
    char t_device_toolkit[64] = {0};
    char t_device_compute[64] = {0};
    af::deviceInfo(t_device_name, t_device_platform, t_device_toolkit, t_device_compute);
    printf("Device name: %s\n", t_device_name);
    printf("Platform name: %s\n", t_device_platform);
    printf("Toolkit: %s\n", t_device_toolkit);
    printf("Compute version: %s\n", t_device_compute);





    // ArrayFire 3.3 has a bug which will double-free the OpenCL queue and context
    // if it is active during destruction. For the moment, the workaround is too
    // select the default ArrayFire device and manually remove the custom device:
    af::setDevice(0);
    afcl::deleteDevice(ocl_device(), ocl_context());

    return 0;
}
