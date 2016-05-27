#pragma once

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>

/** A backend-agnostic pointer to a compute memory resource originating from an OpenGL resource.
    For example:
        cudaGraphicsResource_t* from cudaGraphicsGLRegister[Buffer|Image]
        cl_mem* from clCreateFromGL[Buffer|RenderBuffer|Texture]
    An example of creating a graphics_resource_ptr can be found in create_buffer.
 */
typedef void* graphics_resource_ptr;

/** A backend-agnostic pointer to a compute memory resource. For example:
        CUDA device pointer, like float*, int* from cudaMalloc.
        A cl_mem* from OpenCL's clCreateBuffer

 */
typedef void* compute_resource_ptr;  //

void afInteropInit();
void afInteropTerminate();

void copy_to_gl_buffer(
    compute_resource_ptr src,
    graphics_resource_ptr gl_dest,
    size_t size,
    size_t src_offset = 0,
    size_t dest_offset = 0);

void copy_from_gl_buffer(
    graphics_resource_ptr gl_src,
    compute_resource_ptr dest,
    size_t size,
    size_t src_offset = 0,
    size_t dest_offset = 0);

void
create_buffer(GLuint& buffer,
              GLenum buffer_target,
              const unsigned size,
              GLenum buffer_usage,
              graphics_resource_ptr & compute_ptr,
//              af_graphics_flags flags,
              const void* data = NULL);

void
create_renderbuffer(GLuint& buffer,
                    GLenum format,
                    const unsigned int width,
                    const unsigned int height,
                    graphics_resource_ptr & compute_ptr
//                    af_graphics_flags flags
    );

void
delete_buffer(GLuint buffer,
              GLuint buffer_target,
              graphics_resource_ptr gl_resource);
