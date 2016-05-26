
#if defined (__APPLE__) || defined(MACOSX)	// Apple / OSX
#elif defined WIN32 // Windows
#else	// Linux

#include <CL/cl2.hpp>

#endif

#include "arrayfire.h"
#include "af/opencl.h"
#include "afopengl.h"

#include <vector>
#include <stdexcept>
#include <iostream>
#include <string>
using namespace std;

#include "clerrors.h"


/// Copies an OpenCL buffer to a (mapped) OpenGL buffer with no offsets.
void copy_to_gl_buffer(
    af_graphics_t src,
    af_graphics_t gl_dest,
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

void copy_from_gl_buffer(
    af_graphics_t gl_src,
    af_graphics_t dest,
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

/// Creates an OpenGL buffer and an af_graphics_t reference to the same
///
/// In the OpenCL backend, af_graphics_t is a `cl_mem*`
void
create_buffer(GLuint& buffer,
              GLenum buffer_target,
              const unsigned size,
              GLenum buffer_usage,
              af_graphics_t& cl_buffer,
//              cl_mem_flags flags,
              const void* data)
{
    cl_mem * t_cl_buffer = new cl_mem();
    int status = CL_SUCCESS;
    cl_context context = afcl::getContext();

    glGenBuffers(1, &buffer);
    glBindBuffer(buffer_target, buffer);
    glBufferData(buffer_target, size, data, buffer_usage);

    // Create the OpenCL buffer
    *t_cl_buffer = clCreateFromGLBuffer(context, CL_MEM_READ_WRITE,
                                                        buffer, &status);
    OPENCL(status, "clCreateFromGLBuffer failed");
    cl_buffer = t_cl_buffer;

    glBindBuffer(buffer_target, 0);
}

void
delete_buffer(GLuint buffer,
              GLuint buffer_target,
              af_graphics_t cl_buffer)
{
    cl_mem * t_cl_buffer = (cl_mem *) cl_buffer;
    OPENCL(clReleaseMemObject(*t_cl_buffer), "clReleaseMemObject failed");
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
