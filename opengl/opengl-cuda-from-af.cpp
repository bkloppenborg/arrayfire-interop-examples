
#define TIMESTEP (0.01)
// Pull in OpenGL
#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT 1
#include <GLFW/glfw3.h>
#include <GL/glext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "arrayfire.h"
// Add OpenCL interop header
#include "af/cuda.h"

#include <iostream>
using namespace std;

#include "gl-shaders.h" // source code for our (simple) shaders
#include "cuda-gl.h"

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

//    // (optional) print out information about the device provided to ArrayFire
//    char t_device_name[64] = {0};
//    char t_device_platform[64] = {0};
//    char t_device_toolkit[64] = {0};
//    char t_device_compute[64] = {0};
//    af::deviceInfo(t_device_name, t_device_platform, t_device_toolkit, t_device_compute);
//    printf("Device name: %s\n", t_device_name);
//    printf("Platform name: %s\n", t_device_platform);
//    printf("Toolkit: %s\n", t_device_toolkit);
//    printf("Compute version: %s\n", t_device_compute);

    //
    // 2. Create OpenGL resources and assign them to cudaGraphics_t as needed.
    //
    // Create a Vertex Array Object
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    GLfloat vertices[] = {
        0.0f,  0.5f,
        0.5f, -0.5f,
        -0.5f, -0.5f
    };
    GLfloat colors [] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 1.0f
    };

    // Create a Vertex Buffer Object and copy the vertex data to it
    GLuint vertex_b;
    cudaGraphicsResource_t vertex_cuda;
    create_buffer(vertex_b, GL_ARRAY_BUFFER, &vertex_cuda, sizeof(vertices), GL_DYNAMIC_DRAW,
                  vertices);

    GLuint colors_b;
    cudaGraphicsResource_t colors_cuda;
    create_buffer(colors_b, GL_ARRAY_BUFFER, &colors_cuda, sizeof(colors), GL_DYNAMIC_DRAW,
                  colors);

    // Load the shaders
    GLuint vertexShader, fragmentShader, shaderProgram;
    GLint positionAttrib, colorAttrib;
    initShaders(vertexShader, fragmentShader, shaderProgram, positionAttrib, colorAttrib,
                vertex_b, colors_b);

    //
    // 3. Use ArrayFire in your application
    //
    af::array af_vertices_t0 = af::array(2, 3, vertices);
    af::array af_vertices;
    af::array af_colors = af::array(3, 3, colors);

    float t = 0;
    while(!glfwWindowShouldClose(window)) {
        af_vertices = af_vertices_t0 * (cos(t) +1);

        //
        // 4. Finish any pending ArrayFire operations
        //
        af_vertices.eval();

        //
        // 5. Obtain device pointers for af::array objects, copy values to OpenGL resources
        //    Be sure to cudaGraphicsMapResources and cudaGraphicsUnmapResources.
        // 
        float * d_vertices = af_vertices.device<float>();
        copy_from_device_pointer(vertex_cuda, d_vertices, GL_ARRAY_BUFFER, 6 * sizeof(float));

        //
        // 6. Continue with normal OpenGL operations
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
    // 6. Clean up cudaGraphicsResource_t and OpenGL resources.
    //

    delete_buffer(colors_b, GL_ARRAY_BUFFER, colors_cuda);
    delete_buffer(vertex_b, GL_ARRAY_BUFFER, vertex_cuda);

    glDeleteVertexArrays(1, &vao);

    glDeleteProgram(shaderProgram);
    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    // 7. Shut down OpenGL
    glfwTerminate();

    return 0;
}
