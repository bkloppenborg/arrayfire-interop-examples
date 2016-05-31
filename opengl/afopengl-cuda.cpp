/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * https://opensource.org/licenses
 ********************************************************/

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

void copyToGLBuffer(compute_resource_ptr src,
                       graphics_resource_ptr gl_dest,
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

void copyToGLBuffer(
    af::array & src,
    graphics_resource_ptr gl_dest,
    size_t size,
    size_t src_offset,
    size_t dest_offset)
{
    // Get a compute resource pointer. The specific underlying type
    // doesn't matter as all of the following functions use void*
    compute_resource_ptr af_src = (compute_resource_ptr) src.device<float>();
    copyToGLBuffer(af_src, gl_dest, size, src_offset, dest_offset);

    // Be sure to return control of memory to ArrayFire!
    src.unlock();
}

void copyFromGLBuffer(graphics_resource_ptr gl_src,
                       compute_resource_ptr dest,
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

void copyFromGLBuffer(
    graphics_resource_ptr gl_src,
    af::array & dest,
    size_t size,
    size_t src_offset,
    size_t dest_offset)
{
    // Get a compute resource pointer. The specific underlying type
    // doesn't matter as all of the following functions use void*
    compute_resource_ptr af_dest = (compute_resource_ptr) dest.device<float>();

    copyFromGLBuffer(gl_src, af_dest, size, src_offset, dest_offset);
    // Be sure to return control of memory to ArrayFire!
    dest.unlock();
}


void
createBuffer(GLuint& buffer,
              GLenum buffer_target,
              const unsigned size,
              GLenum buffer_usage,
              compute_resource_ptr & compute_ptr,
//              GLenum buffer_usage,
              const void* data)
{
    cudaGraphicsResource_t * t_cuda_resource = new cudaGraphicsResource_t();

    glGenBuffers(1, &buffer);
    glBindBuffer(buffer_target, buffer);
    glBufferData(buffer_target, size, data, buffer_usage);
    CUDA(cudaGraphicsGLRegisterBuffer(t_cuda_resource,
                                      buffer, cudaGraphicsRegisterFlagsNone));

    compute_ptr = t_cuda_resource;

    glBindBuffer(buffer_target, 0);
}

void
createRenderbuffer(GLuint& buffer,
                    GLenum format,
                    const unsigned int width,
                    const unsigned int height,
                    graphics_resource_ptr & gl_resource)
{
    cudaGraphicsResource_t * t_cuda_resource = new cudaGraphicsResource_t();

    glGenRenderbuffers(1, &buffer);
    glBindRenderbuffer(GL_RENDERBUFFER, buffer);
    glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);

    CUDA(cudaGraphicsGLRegisterImage(t_cuda_resource, buffer, GL_RENDERBUFFER,
                                     cudaGraphicsRegisterFlagsNone));
    gl_resource = t_cuda_resource;

    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void
deleteBuffer(GLuint buffer,
              GLuint buffer_target,
              graphics_resource_ptr gl_resource)
{
    cudaGraphicsResource_t * t_cuda_resource = (cudaGraphicsResource_t*) gl_resource;
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
