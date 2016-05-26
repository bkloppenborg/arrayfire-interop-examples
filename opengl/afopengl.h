#pragma once

#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>

typedef void * af_graphics_t;

void afInteropInit();
void afInteropTerminate();

void copy_to_gl_buffer(
    af_graphics_t src,
    af_graphics_t gl_dest,
    size_t size,
    size_t src_offset = 0,
    size_t dest_offset = 0);

void copy_from_gl_buffer(
    af_graphics_t gl_src,
    af_graphics_t dest,
    size_t size,
    size_t src_offset = 0,
    size_t dest_offset = 0);

void
create_buffer(GLuint& buffer,
              GLenum buffer_target,
              const unsigned size,
              GLenum buffer_usage,
              af_graphics_t & af_buffer,
//              af_graphics_flags flags,
              const void* data = NULL);

void
create_renderbuffer(GLuint& buffer,
                    GLenum format,
                    const unsigned int width,
                    const unsigned int height,
                    af_graphics_t & cuda_resource
//                    af_graphics_flags flags
    );

void
delete_buffer(GLuint buffer,
              GLuint buffer_target,
              af_graphics_t af_buffer);


