/*
MIT License

Copyright (c) 2016 Brian Kloppenborg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef CL_ERRORS_H
#define CL_ERRORS_H

#ifndef __OPENCL_CL_H
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#endif

// C interface
static const char * get_error_string(cl_int error);

/// Macro which checks and marks the source of an OpenCL error.
#define OPENCL(cl_status, message) \
    if(cl_status != CL_SUCCESS) \
    { \
        const char * errorString = get_error_string(cl_status);\
        printf("Error: %s \nError Code: %s\n", message, errorString);\
        printf("Location: %s:%i\n", __FILE__, __LINE__);\
        exit(EXIT_FAILURE);                             \
    }

/** Convert OpenCL error codes to const char* */
#define STRING_CASE(ARG) case ARG: return #ARG;
static const char * get_error_string(cl_int error)
{
    switch(error)
    {
        // from cl.h
        STRING_CASE(CL_SUCCESS);
        STRING_CASE(CL_DEVICE_NOT_FOUND);
        STRING_CASE(CL_DEVICE_NOT_AVAILABLE);
        STRING_CASE(CL_COMPILER_NOT_AVAILABLE);
        STRING_CASE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
        STRING_CASE(CL_OUT_OF_RESOURCES);
        STRING_CASE(CL_OUT_OF_HOST_MEMORY);
        STRING_CASE(CL_PROFILING_INFO_NOT_AVAILABLE);
        STRING_CASE(CL_MEM_COPY_OVERLAP);
        STRING_CASE(CL_IMAGE_FORMAT_MISMATCH);
        STRING_CASE(CL_IMAGE_FORMAT_NOT_SUPPORTED);
        STRING_CASE(CL_BUILD_PROGRAM_FAILURE);
        STRING_CASE(CL_MAP_FAILURE);
        STRING_CASE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
        STRING_CASE(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
#ifdef CL_VERSION_1_2
        STRING_CASE(CL_COMPILE_PROGRAM_FAILURE);
        STRING_CASE(CL_LINKER_NOT_AVAILABLE);
        STRING_CASE(CL_LINK_PROGRAM_FAILURE);
        STRING_CASE(CL_DEVICE_PARTITION_FAILED);
        STRING_CASE(CL_KERNEL_ARG_INFO_NOT_AVAILABLE);
#endif
        STRING_CASE(CL_INVALID_VALUE);
        STRING_CASE(CL_INVALID_DEVICE_TYPE);
        STRING_CASE(CL_INVALID_PLATFORM);
        STRING_CASE(CL_INVALID_DEVICE);
        STRING_CASE(CL_INVALID_CONTEXT);
        STRING_CASE(CL_INVALID_QUEUE_PROPERTIES);
        STRING_CASE(CL_INVALID_COMMAND_QUEUE);
        STRING_CASE(CL_INVALID_HOST_PTR);
        STRING_CASE(CL_INVALID_MEM_OBJECT);
        STRING_CASE(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR);
        STRING_CASE(CL_INVALID_IMAGE_SIZE);
        STRING_CASE(CL_INVALID_SAMPLER);
        STRING_CASE(CL_INVALID_BINARY);
        STRING_CASE(CL_INVALID_BUILD_OPTIONS);
        STRING_CASE(CL_INVALID_PROGRAM);
        STRING_CASE(CL_INVALID_PROGRAM_EXECUTABLE);
        STRING_CASE(CL_INVALID_KERNEL_NAME);
        STRING_CASE(CL_INVALID_KERNEL_DEFINITION);
        STRING_CASE(CL_INVALID_KERNEL);
        STRING_CASE(CL_INVALID_ARG_INDEX);
        STRING_CASE(CL_INVALID_ARG_VALUE);
        STRING_CASE(CL_INVALID_ARG_SIZE);
        STRING_CASE(CL_INVALID_KERNEL_ARGS);
        STRING_CASE(CL_INVALID_WORK_DIMENSION);
        STRING_CASE(CL_INVALID_WORK_GROUP_SIZE);
        STRING_CASE(CL_INVALID_WORK_ITEM_SIZE);
        STRING_CASE(CL_INVALID_GLOBAL_OFFSET);
        STRING_CASE(CL_INVALID_EVENT_WAIT_LIST);
        STRING_CASE(CL_INVALID_EVENT);
        STRING_CASE(CL_INVALID_OPERATION);
        STRING_CASE(CL_INVALID_GL_OBJECT);
        STRING_CASE(CL_INVALID_BUFFER_SIZE);
        STRING_CASE(CL_INVALID_MIP_LEVEL);
        STRING_CASE(CL_INVALID_GLOBAL_WORK_SIZE);
#ifdef CL_VERSION_1_2
        STRING_CASE(CL_INVALID_PROPERTY);
        STRING_CASE(CL_INVALID_IMAGE_DESCRIPTOR);
        STRING_CASE(CL_INVALID_COMPILER_OPTIONS);
        STRING_CASE(CL_INVALID_LINKER_OPTIONS);
        STRING_CASE(CL_INVALID_DEVICE_PARTITION_COUNT);
#endif // CL_VERSION_1_2
#ifdef CL_VERSION_2_0
        STRING_CASE(CL_INVALID_PIPE_SIZE);
        STRING_CASE(CL_INVALID_DEVICE_QUEUE);
         #endif // CL_VERSION_2_0

        // from cl_d3d10.h
#ifdef __OPENCL_CL_D3D10_H
        STRING_CASE(CL_INVALID_D3D10_DEVICE_KHR);
        STRING_CASE(CL_INVALID_D3D10_RESOURCE_KHR);
        STRING_CASE(CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR);
        STRING_CASE(CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR);
#endif

         // from cl_d3d11.h
#ifdef __OPENCL_CL_D3D11_H
        STRING_CASE(CL_INVALID_D3D11_DEVICE_KHR);
        STRING_CASE(CL_INVALID_D3D11_RESOURCE_KHR);
        STRING_CASE(CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR);
        STRING_CASE(CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR);
#endif

         // from cl_d3d.h
#ifdef __OPENCL_CL_DX9_MEDIA_SHARING_H
        STRING_CASE(CL_INVALID_DX9_MEDIA_ADAPTER_KHR);
        STRING_CASE(CL_INVALID_DX9_MEDIA_SURFACE_KHR);
        STRING_CASE(CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR);
        STRING_CASE(CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR);
#endif

        // from cl_egl.h
#ifdef __OPENCL_CL_EGL_H
        STRING_CASE(CL_INVALID_EGL_OBJECT_KHR);
        STRING_CASE(CL_EGL_RESOURCE_NOT_ACQUIRED_KHR);
#endif

        // from cl_ext.h
#ifdef __CL_EXT_H
        STRING_CASE(CL_PLATFORM_NOT_FOUND_KHR);
#ifdef CL_VERSION_1_1
        STRING_CASE(CL_DEVICE_PARTITION_FAILED_EXT);
        STRING_CASE(CL_INVALID_PARTITION_COUNT_EXT);
        STRING_CASE(CL_INVALID_PARTITION_NAME_EXT);
#endif // CL_VERSION_1_1
#endif // __OPENCL_EXT_H

        // from cl_gh.h
#ifdef __OPENCL_CL_GL_H
        STRING_CASE(CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR);
#endif

    default:
        char message[] = "Unknown error code";
        char * output = (char*) malloc(sizeof(message) + 10);
        sprintf(output, "%s %d", message, error);
        return output;
    }
}

#ifdef __cplusplus

#include <cstring>

static std::string getErrorString(cl_int error)
{
    return std::string(get_error_string(error));
}

#endif // __cplusplus

#endif // CL_ERRORS_H
