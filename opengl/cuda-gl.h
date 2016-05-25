#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define CUDA(x) do {                                                    \
        cudaError_t err = (x);                                          \
        if(cudaSuccess != err) {                                        \
            fprintf(stderr, "CUDA Error in %s:%d: %s \nReturned: %s.\n", \
                    __FILE__, __LINE__, #x, cudaGetErrorString(err) );  \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while(0)


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
template<typename T>
void copy_from_device_pointer(cudaGraphicsResource_t cuda_resource,
                              T& d_ptr,
                              GLuint buffer_target,
                              const unsigned size)
{
    CUDA(cudaGraphicsMapResources(1, &cuda_resource));
    bool is_mapped = true;
    if (buffer_target == GL_RENDERBUFFER) {
        cudaArray* array_ptr = NULL;
        CUDA(cudaGraphicsSubResourceGetMappedArray(&array_ptr, cuda_resource, 0, 0));
        CUDA(cudaMemcpyToArray(array_ptr, 0, 0, d_ptr, size, cudaMemcpyDeviceToDevice));
    } else {
        T* opengl_ptr = NULL;
        CUDA(cudaGraphicsResourceGetMappedPointer((void**)&opengl_ptr, (size_t*)&size, cuda_resource));
        CUDA(cudaMemcpy(opengl_ptr, d_ptr, size, cudaMemcpyDeviceToDevice));
    }
    unmap_resource(cuda_resource, is_mapped);
}

// Gets the device pointer from the mapped resource
// Sets is_mapped to true
template<typename T>
void copy_to_device_pointer(cudaGraphicsResource_t cuda_resource,
                            T& d_ptr,
                            GLuint buffer_target,
                            const unsigned size)
{
    cudaGraphicsMapResources(1, &cuda_resource);
    bool is_mapped = true;
    if (GL_RENDERBUFFER == buffer_target) {
        cudaArray* array_ptr;
        CUDA(cudaGraphicsSubResourceGetMappedArray(&array_ptr, cuda_resource, 0, 0));
        CUDA(cudaMemcpyFromArray(d_ptr, array_ptr, 0, 0, size, cudaMemcpyDeviceToDevice));
    } else {
        T* opengl_ptr = NULL;
        CUDA(cudaGraphicsResourceGetMappedPointer((void**)&opengl_ptr, (size_t*)&size, cuda_resource));
        CUDA(cudaMemcpy(d_ptr, opengl_ptr, size, cudaMemcpyDeviceToDevice));
    }
    unmap_resource(cuda_resource, is_mapped);
}

void
create_buffer(GLuint& buffer,
              GLenum buffer_target,
              cudaGraphicsResource_t* cuda_resource,
              const unsigned size,
              GLenum buffer_usage,
              const void* data = NULL)
{
    glGenBuffers(1, &buffer);
    glBindBuffer(buffer_target, buffer);
    glBufferData(buffer_target, size, data, buffer_usage);
    CUDA(cudaGraphicsGLRegisterBuffer(cuda_resource, buffer, cudaGraphicsRegisterFlagsNone));
    glBindBuffer(buffer_target, 0);
}

void
create_buffer(GLuint& buffer,
              GLenum format,
              const unsigned width,
              const unsigned height,
              cudaGraphicsResource_t* cuda_resource = NULL)
{
    glGenRenderbuffers(1, &buffer);
    glBindRenderbuffer(GL_RENDERBUFFER, buffer);
    glRenderbufferStorage(GL_RENDERBUFFER, format, width, height);
    if(cuda_resource != NULL)
        CUDA(cudaGraphicsGLRegisterImage(cuda_resource, buffer, GL_RENDERBUFFER, cudaGraphicsRegisterFlagsNone));
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void
delete_buffer(GLuint buffer,
              GLuint buffer_target,
              cudaGraphicsResource_t cuda_resource)
{
    CUDA(cudaGraphicsUnregisterResource(cuda_resource));
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
