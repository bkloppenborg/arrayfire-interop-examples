
#define TIMESTEP (0.01)

// Pull in OpenCL
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120    // Use OpenCL 1.2 specs (for NVIDIA)
#include <CL/cl2.hpp>

// Pull in OpenGL
#define GL_GLEXT_PROTOTYPES
#define GLFW_INCLUDE_GLEXT 1
#include <GLFW/glfw3.h>
#include <GL/glext.h>

#include "arrayfire.h"
// Add OpenCL interop header
#include "af/opencl.h"
#include "cl-gl.h"
#include "clerrors.h"

#include <iostream>
using namespace std;

#include "gl-shaders.h" // source code for our (simple) shaders

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
    // 2. Construct an OpenCL context from the OpenGL context
    //
    // a. Get properties for OpenCL-OpenGL interop on the current platform
    cl::Platform ocl_platform = cl::Platform::getDefault();
    vector<cl_context_properties> properties = get_interop_properties(ocl_platform());

    // b. Construct an OpenCL device corresponding to the interop device 
    size_t input_size = 8;
    size_t output_size = 0;
    cl_device_id devices[input_size];
    cl_int status = clGetGLContextInfoKHR(properties.data(), CL_DEVICES_FOR_GL_CONTEXT_KHR,
                                          input_size,
                                          devices, &output_size);

    // Assume the zeroth device matches, this may not always be true, but is probably ok
    // for this example.
    cl::Device device = cl::Device(devices[0]);

    // Create OpenCL context and queue from the device and properties obtained above
    cl::Context context(device, properties.data());
    cl::CommandQueue queue(context, device);

    //
    // 3. Register the device with ArrayFire
    //
    // Create a device from the current OpenCL device + context + queue
    afcl::addDevice(device(), context(), queue());
    // Switch to the device
    afcl::setDevice(device(), context());

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
    // regardless of whether or not setDevice is called. So we need to re-activate the
    // OpenGL context before continuing.
    glfwMakeContextCurrent(window);

    //
    // 4. Create OpenGL resources and assign them to cl_mem references as needed.
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

    // Create a Vertex Buffer Object and copy the vertex data to it
    GLuint vertex_b;
    cl_mem vertex_cl;
    create_buffer(vertex_b, GL_ARRAY_BUFFER, sizeof(vertices), GL_DYNAMIC_DRAW,
                  context(), vertex_cl, CL_MEM_WRITE_ONLY, vertices);

    GLuint colors_b;
    cl_mem colors_cl;
    create_buffer(colors_b, GL_ARRAY_BUFFER, sizeof(colors), GL_DYNAMIC_DRAW,
                  context(), colors_cl, CL_MEM_WRITE_ONLY, colors);

    // Load the shaders
    GLuint vertexShader, fragmentShader, shaderProgram;
    GLint positionAttrib, colorAttrib;
    initShaders(vertexShader, fragmentShader, shaderProgram, positionAttrib, colorAttrib,
                vertex_b, colors_b);

    //
    // 5. Use ArrayFire in your application
    //
    af::array af_vertices_t0 = af::array(2, 3, vertices);
    af::array af_vertices;
    af::array af_colors = af::array(3, 3, colors);

    float t = 0;
    while(!glfwWindowShouldClose(window)) {
        af_vertices = af_vertices_t0 * (cos(t) +1);

        //
        // 6. Finish any pending ArrayFire operations
        //
        af_vertices.eval();

        //
        // 7. Obtain device pointers for af::array objects, copy values to OpenGL resources
        //    Be sure to acquire and release the OpenGL objects.
        // 
        cl_mem * d_vertices = af_vertices.device<cl_mem>();
        copy_to_gl_buffer(queue(), *d_vertices, vertex_cl, 6 * sizeof(float));

        //
        // 8. Continue with normal OpenGL operations
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
    // 9. Clean up cudaGraphicsResource_t and OpenGL resources.
    //

    delete_buffer(colors_b, GL_ARRAY_BUFFER, colors_cl);
    delete_buffer(vertex_b, GL_ARRAY_BUFFER, vertex_cl);

    glDeleteVertexArrays(1, &vao);

    glDeleteProgram(shaderProgram);
    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    // ArrayFire 3.3 has a bug which will double-free the OpenCL queue and context
    // if it is active during destruction. For the moment, the workaround is too
    // select the default ArrayFire device and manually remove the custom device:
    af::setDevice(0);
    afcl::deleteDevice(device(), context());

    // 10. Shut down OpenGL
    glfwTerminate();

    return 0;
}
