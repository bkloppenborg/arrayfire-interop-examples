
/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * https://opensource.org/licenses
 ********************************************************/
#include "defines.h"

#include <CL/cl2.hpp>

#include "arrayfire.h"
#include "af/opencl.h"
#include "afopengl.h"

#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
using namespace std;

#include "clerrors.h"

#include <GL/glx.h>
// Get the properties required to enable OpenCL-OpenGL interop on the specified
// platform
vector<cl_context_properties> getInteropProperties(cl_platform_id platform)
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

void afInteropInit()
{
    
    //
    // 2. Construct an OpenCL context from the OpenGL context
    //
    // a. Get properties for OpenCL-OpenGL interop on the current platform
    cl::Platform ocl_platform = cl::Platform::getDefault();
    vector<cl_context_properties> properties = getInteropProperties(ocl_platform());

    // b. Construct an OpenCL device corresponding to the interop device 
    size_t input_size = 8;
    size_t output_size = 0;
    cl_device_id devices[input_size];
    cl_int status = clGetGLContextInfoKHR(properties.data(), CL_DEVICES_FOR_GL_CONTEXT_KHR,
                                          input_size,
                                          devices, &output_size);

    // Assume the zeroth device matches, this may not always be true, but is probably ok
    // for this example.
    cl::Device device = cl::Device(devices[0]);

    // Create OpenCL context and queue from the device and properties obtained above
    cl::Context context(device, properties.data());
    cl::CommandQueue queue(context, device);

    //
    // 3. Register the device with ArrayFire
    //
    // Create a device from the current OpenCL device + context + queue
    afcl::addDevice(device(), context(), queue());
    // Switch to the device
    afcl::setDevice(device(), context());

}


void afInteropTerminate()
{

    // ArrayFire 3.3 has a bug which will double-free the OpenCL queue and context
    // if it is active during destruction. For the moment, the workaround is too
    // select the default ArrayFire device and manually remove the custom device:
    cl_device_id device = afcl::getDeviceId();
    cl_context context = afcl::getContext();
    af::setDevice(0);
    afcl::deleteDevice(device, context);
}


/// Copies an OpenCL buffer to a (mapped) OpenGL buffer with no offsets.
void copyToGLBuffer(
    compute_resource_ptr src,
    graphics_resource_ptr gl_dest,
    size_t size,
    size_t src_offset,
    size_t dest_offset)
{
    cl_event waitEvent;
    cl_command_queue queue = afcl::getQueue();

    cl_mem *t_src = (cl_mem*) src;
    cl_mem *t_gl_dest = (cl_mem*) gl_dest;

    glFinish();
    OPENCL(clEnqueueAcquireGLObjects(queue, 1, t_gl_dest, 0, NULL, &waitEvent),
           "clEnqueueAcquireGLObjects failed");
    OPENCL(clWaitForEvents(1, &waitEvent), "clWaitForEvents failed");

    OPENCL(clEnqueueCopyBuffer(queue, *t_src, *t_gl_dest,
                               src_offset, dest_offset, size, 0, NULL, NULL),
           "clEnqueueCopyBuffer failed");

    OPENCL(clEnqueueReleaseGLObjects(queue, 1, t_gl_dest, 0, NULL, &waitEvent),
           "clEnqueueReleaseGLObjects failed");
    OPENCL(clWaitForEvents(1, &waitEvent), "clWaitForEvents failed");
}


void copyToGLBuffer(
    af::array & src,
    graphics_resource_ptr gl_dest,
    size_t size,
    size_t src_offset,
    size_t dest_offset)
{
    // Get a compute resource pointer. The specific underlying type
    // doesn't matter as all of the following functions use void*
    compute_resource_ptr af_src = (compute_resource_ptr) src.device<cl_mem>();
    copyToGLBuffer(af_src, gl_dest, size, src_offset, dest_offset);

    // Be sure to return control of memory to ArrayFire!
    src.unlock();
}

void copyFromGLBuffer(
    graphics_resource_ptr gl_src,
    compute_resource_ptr dest,
    size_t size,
    size_t src_offset,
    size_t dest_offset)
{
    cl_event waitEvent;
    cl_command_queue queue = afcl::getQueue();

    cl_mem * t_gl_src = (cl_mem*) gl_src;
    cl_mem * t_dest = (cl_mem*) dest;

    glFinish();
    OPENCL(clEnqueueAcquireGLObjects(queue, 1, t_gl_src, 0, NULL, &waitEvent),
           "clEnqueueAcquireGLObjects failed");
    OPENCL(clWaitForEvents(1, &waitEvent), "clWaitForEvents failed");

    OPENCL(clEnqueueCopyBuffer(queue, *t_gl_src, *t_dest,
                               src_offset, dest_offset, size, 0, NULL, NULL),
           "clEnqueueCopyBuffer failed");

    OPENCL(clEnqueueReleaseGLObjects(queue, 1, t_gl_src, 0, NULL, &waitEvent),
           "clEnqueueReleaseGLObjects failed");
    OPENCL(clWaitForEvents(1, &waitEvent), "clWaitForEvents failed");
}

void copyFromGLBuffer(
    graphics_resource_ptr gl_src,
    af::array & dest,
    size_t size,
    size_t src_offset,
    size_t dest_offset)
{
    // Get a compute resource pointer. The specific underlying type
    // doesn't matter as all of the following functions use void*
    compute_resource_ptr af_dest = (compute_resource_ptr) dest.device<cl_mem>();

    copyFromGLBuffer(gl_src, af_dest, size, src_offset, dest_offset);
    // Be sure to return control of memory to ArrayFire!
    dest.unlock();
}

void
createBuffer(GLuint& buffer,
              GLenum buffer_target,
              const unsigned size,
              GLenum buffer_usage,
              graphics_resource_ptr & compute_ptr,
//              cl_mem_flags flags,
              const void* data)
{
    cl_mem * cl_buffer = new cl_mem();
    int status = CL_SUCCESS;
    cl_context context = afcl::getContext();

    glGenBuffers(1, &buffer);
    glBindBuffer(buffer_target, buffer);
    glBufferData(buffer_target, size, data, buffer_usage);

    // Create the OpenCL buffer
    *cl_buffer = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE,
                                                        buffer, &status);
    OPENCL(status, "clCreateFromGLBuffer failed");
    compute_ptr = cl_buffer;

    glBindBuffer(buffer_target, 0);
}

void
createRenderbuffer(GLuint& buffer,
                    GLenum format,
                    const unsigned int width,
                    const unsigned int height,
                    graphics_resource_ptr & compute_ptr)
{
    cl_mem * cl_buffer = new cl_mem();
    int status = CL_SUCCESS;
    cl_context context = afcl::getContext();

    glGenRenderbuffers(1, &buffer);
    glBindRenderbuffer(GL_RENDERBUFFER, buffer);
    glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);

    // Create the OpenCL buffer
    *cl_buffer = clCreateFromGLRenderbuffer(context, CL_MEM_READ_WRITE,
                                        buffer, &status);
    OPENCL(status, "clCreateFromGLBuffer failed");
    compute_ptr = cl_buffer;

    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}


void
deleteBuffer(GLuint buffer,
              GLuint buffer_target,
              graphics_resource_ptr gl_resource)
{
    cl_mem * cl_buffer = (cl_mem *) gl_resource;
    OPENCL(clReleaseMemObject(*cl_buffer), "clReleaseMemObject failed");
    if (buffer_target == GL_RENDERBUFFER) {
        glBindRenderbuffer(buffer_target, buffer);
        glDeleteRenderbuffers(1, &buffer);
        buffer = 0;
    } else {
        glBindBuffer(buffer_target, buffer);
        glDeleteRenderbuffers(1, &buffer);
        buffer = 0;
    }
}
