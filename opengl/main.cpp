/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * https://opensource.org/licenses
 ********************************************************/

#include "defines.h"

// Pull in OpenGL
#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT 1
#include <GLFW/glfw3.h>
#include <GL/glext.h>

#include "arrayfire.h"
#if defined(AF_CUDA_INTEROP)
#include "af/cuda.h"
#elif defined(AF_OPENCL_INTEROP)
#include "af/opencl.h"
#endif
#include "afopengl.h"

#include <iostream>
using namespace std;

#include "gl-shaders.h" // source code for our (simple) shaders

#define TIMESTEP (0.01)

int main(int argc, char** argv)
{
    //
    // 1. Create an OpenGL window
    //
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

    //
    // 2. Initialize OpenGL interop.
    //    For CUDA, this could involve selecting the device corresponding to the OpenGL
    //    device above. For OpenCL, this process is quite involved. We suggest you consult
    //    the code found in afopengl-cuda.cpp and afopengl-opencl.cpp for further details.
    afInteropInit();

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

    // ArrayFire 3.3 has a bug which changes the active context when a device is added
    // to the OpenCL backend regardless of whether or not setDevice is called.
    // Thus we need to re-activate the OpenGL context before continuing.
    glfwMakeContextCurrent(window);

    //
    // 3. Create OpenGL resources and assign them to cl_mem references as needed.
    //
    // Create a Vertex Array Object
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLfloat vertices[] = {
        0.0f,  0.5,
        0.5f, -0.5f,
        -0.5f, -0.5f
    };
    GLfloat colors [] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };

    //
    // 3a. Create OpenGL buffers and the corresponding CUDA/OpenCL references to the
    //     OpenGL memory. The create_buffer function wraps the corresponding CUDA
    //     calls to generate cudaGraphicsResource_t* and OpenCL cl_mem* pointers.
    GLuint vertex_b;
    graphics_resource_ptr vertex_cl;
    create_buffer(vertex_b, GL_ARRAY_BUFFER, sizeof(vertices), GL_DYNAMIC_DRAW,
                  vertex_cl, vertices);

    GLuint colors_b;
    graphics_resource_ptr colors_cl;
    create_buffer(colors_b, GL_ARRAY_BUFFER, sizeof(colors), GL_DYNAMIC_DRAW,
                  colors_cl, colors);

    // Load the shaders.
    GLuint vertexShader, fragmentShader, shaderProgram;
    GLint positionAttrib, colorAttrib;
    initShaders(vertexShader, fragmentShader, shaderProgram, positionAttrib, colorAttrib,
                vertex_b, colors_b);

    //
    // 4. Use ArrayFire in your application
    //
    af::array af_vertices_t0 = af::array(2, 3, vertices);
    af::array af_vertices;
    af::array af_colors = af::array(3, 3, colors);

    float t = 0;
    while(!glfwWindowShouldClose(window)) {
        af_vertices = af_vertices_t0 * (cos(t) +1);

        //
        // 5. Finish any pending ArrayFire operations
        //
        af_vertices.eval();

        // 6. Copy data to/from OpenGL memory.
        //    ArrayFire cannot use OpenGL memory as an endpoint for its operations, thus
        //    the programmer must initiate device-to-device memory transfer operations.
        //

        // 6a. Copy data from an af::array to an OpenGL buffer:
        //  i. Obtain pointers to the underlying ArrayFire memory:
#if defined(AF_CUDA_INTEROP)
        compute_resource_ptr d_vertices = af_vertices.device<float>();
#elif defined(AF_OPENCL_INTEROP)
        compute_resource_ptr d_vertices = af_vertices.device<cl_mem>();
#endif
 
        //  ii. Copy data using copy_to_gl_buffer.
        //      This function handles all of the necessary operations including glFinish()
        //      mapping/unmapping buffers, and transferring the data.
        copy_to_gl_buffer(d_vertices, vertex_cl, 6 * sizeof(float));

        af_vertices = af::constant(0, 2, 3);
        af_print(af_vertices);

        // 6b. Copy data to an af::array from an OpenGL buffer:
        //  i. Obtain pointers to the underlying ArrayFire memory
#if defined(AF_CUDA_INTEROP)
        d_vertices = af_vertices.device<float>();
#elif defined(AF_OPENCL_INTEROP)
        d_vertices = af_vertices.device<cl_mem>();
#endif
        //  ii. Copy data using copy_from_gl_buffer.
        //      This function handles all of the necessary operations including glFinish()
        //      mapping/unmapping buffers, and transferring the data.
        copy_from_gl_buffer(vertex_cl, d_vertices, 6 * sizeof(float));
        af_print(af_vertices);

        //
        // 7. Continue with normal OpenGL operations
        //
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw 3 vertices from the VBO
        glDrawArrays(GL_TRIANGLES, 0, 3);

        // Swap buffers and poll
        glfwSwapBuffers(window);
        glfwPollEvents();

        t += TIMESTEP;
    }

    //
    // 8. Clean up OpenGL resources
    //

    delete_buffer(colors_b, GL_ARRAY_BUFFER, colors_cl);
    delete_buffer(vertex_b, GL_ARRAY_BUFFER, vertex_cl);

    glDeleteVertexArrays(1, &vao);

    glDeleteProgram(shaderProgram);
    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    // 9. Turn off OpenGL interop
    afInteropTerminate();
    glfwTerminate();

    return 0;
}
