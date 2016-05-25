#ifndef CL_GL_H
#define CL_GL_H

#if defined (__APPLE__) || defined(MACOSX)	// Apple / OSX
#elif defined WIN32 // Windows
#else	// Linux

#include <CL/cl2.hpp>
#include <GL/glx.h>

#endif

#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
using namespace std;

#include "clerrors.h"


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


// Creates an OpenGL buffer and an OpenCL reference to the same.
void
create_buffer(GLuint& buffer,
              GLenum buffer_target,
              const unsigned size,
              GLenum buffer_usage,
              cl_context context,
              cl_mem& cl_buffer,
              cl_mem_flags flags,
              const void* data = NULL)
{
    int status = CL_SUCCESS;

    glGenBuffers(1, &buffer);
    glBindBuffer(buffer_target, buffer);
    glBufferData(buffer_target, size, data, buffer_usage);

    // Create the OpenCL buffer
    cl_buffer = clCreateFromGLBuffer(context, flags, buffer, &status);
    OPENCL(status, "clCreateFromGLBuffer failed");

    glBindBuffer(buffer_target, 0);
}

void
delete_buffer(GLuint buffer,
              GLuint buffer_target,
              cl_mem cl_buffer)
{
    OPENCL(clReleaseMemObject(cl_buffer), "clReleaseMemObject failed");
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

/// Copies an OpenCL buffer to a (mapped) OpenGL buffer with no offsets.
void copy_to_gl_buffer(
    cl_command_queue queue,
    cl_mem src,
    cl_mem dest,
    const unsigned size,
    size_t src_offset = 0,
    size_t dest_offset = 0)
{
    cl_event waitEvent;

    glFinish();
    OPENCL(clEnqueueAcquireGLObjects(queue, 1, &dest, 0, NULL, &waitEvent),
           "clEnqueueAcquireGLObjects failed");
    OPENCL(clWaitForEvents(1, &waitEvent), "clWaitForEvents failed");

    OPENCL(clEnqueueCopyBuffer(queue, src, dest,
                               src_offset, dest_offset, size, 0, NULL, NULL),
           "clEnqueueCopyBuffer failed");

    OPENCL(clEnqueueReleaseGLObjects(queue, 1, &dest, 0, NULL, &waitEvent),
           "clEnqueueReleaseGLObjects failed");
    OPENCL(clWaitForEvents(1, &waitEvent), "clWaitForEvents failed");
}

#endif // CL_GL_H
