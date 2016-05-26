#include "afopengl.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <iostream>
using namespace std;

#define CUDA(x) do {                                                    \
        cudaError_t err = (x);                                          \
        if(cudaSuccess != err) {                                        \
            fprintf(stderr, "CUDA Error in %s:%d: %s \nReturned: %s.\n", \
                    __FILE__, __LINE__, #x, cudaGetErrorString(err) );  \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while(0)

void afInteropInit() {};
void afInteropTerminate() {};

void
unmap_resource(cudaGraphicsResource_t cuda_resource,
               bool is_mapped)
{
    if (is_mapped) {
        CUDA(cudaGraphicsUnmapResources(1, &cuda_resource));
        is_mapped = false;
    }
}

// Gets the device pointer from the mapped resource
// Sets is_mapped to true
void copy_to_gl_buffer(af_graphics_t src,
                       af_graphics_t gl_dest,
                       size_t size,
                       size_t src_offset,
                       size_t dst_offset)
{
    cudaGraphicsResource_t * t_gl_dest = (cudaGraphicsResource_t*) gl_dest;

    CUDA(cudaGraphicsMapResources(1, t_gl_dest));

    bool is_mapped = true;
    void * opengl_ptr = NULL;
    CUDA(cudaGraphicsResourceGetMappedPointer((void**)&opengl_ptr, (size_t*)&size,
                                              *t_gl_dest));
    CUDA(cudaMemcpy(opengl_ptr, src, size, cudaMemcpyDeviceToDevice));

    unmap_resource(*t_gl_dest, is_mapped);
}

// Gets the device pointer from the mapped resource
// Sets is_mapped to true
void copy_from_gl_buffer(af_graphics_t gl_src,
                       af_graphics_t dest,
                       size_t size,
                       size_t src_offset,
                       size_t dst_offset)
{
    cudaGraphicsResource_t * t_gl_src = (cudaGraphicsResource_t*) gl_src;

    CUDA(cudaGraphicsMapResources(1, t_gl_src));

    bool is_mapped = true;
    void * opengl_ptr = NULL;
    CUDA(cudaGraphicsResourceGetMappedPointer((void**)&opengl_ptr, (size_t*)&size,
                                              *t_gl_src));
    CUDA(cudaMemcpy(dest, opengl_ptr, size, cudaMemcpyDeviceToDevice));

    unmap_resource(*t_gl_src, is_mapped);
}

/// Creates an OpenGL buffer with a corresponding af_graphics_t
///
/// af_graphics_t for the CUDA backend represents a `cudaGraphicsResource_t *`
void
create_buffer(GLuint& buffer,
              GLenum buffer_target,
              const unsigned size,
              GLenum buffer_usage,
              af_graphics_t & cuda_buffer,
//              GLenum buffer_usage,
              const void* data)
{
    cudaGraphicsResource_t * t_cuda_resource = new cudaGraphicsResource_t();

    glGenBuffers(1, &buffer);
    glBindBuffer(buffer_target, buffer);
    glBufferData(buffer_target, size, data, buffer_usage);
    CUDA(cudaGraphicsGLRegisterBuffer(t_cuda_resource,
                                      buffer, cudaGraphicsRegisterFlagsNone));

    cuda_buffer = t_cuda_resource;

    glBindBuffer(buffer_target, 0);
}

void
create_renderbuffer(GLuint& buffer,
                    GLenum format,
                    const unsigned int width,
                    const unsigned int height,
                    af_graphics_t & cuda_resource)
{
    cudaGraphicsResource_t * t_cuda_resource = new cudaGraphicsResource_t();

    glGenRenderbuffers(1, &buffer);
    glBindRenderbuffer(GL_RENDERBUFFER, buffer);
    glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);

    CUDA(cudaGraphicsGLRegisterImage(t_cuda_resource, buffer, GL_RENDERBUFFER,
                                     cudaGraphicsRegisterFlagsNone));
    cuda_resource = t_cuda_resource;

    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

/// Deletes an OpenGL buffer and corresponding af_graphics_t object
void
delete_buffer(GLuint buffer,
              GLuint buffer_target,
              af_graphics_t cuda_resource)
{
    cudaGraphicsResource_t * t_cuda_resource = (cudaGraphicsResource_t*) cuda_resource;
    CUDA(cudaGraphicsUnregisterResource(*t_cuda_resource));
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
