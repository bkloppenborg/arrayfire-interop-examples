/*******************************************************
 * Copyright (c) 2016, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * https://opensource.org/licenses
 ********************************************************/

/*
Portions of this code come from the OpenGL tutorials found at
https://github.com/Overv/Open.GL

and are licensed under the Creative Commons Attribute-ShareAlike 4.0
International license:

http://creativecommons.org/licenses/by-sa/4.0/
 */

// Shader sources
const char* vertexSource =
    "#version 150 core\n"
    "in vec2 position;"
    "in vec3 color;"
    "out vec3 Color;"
    "void main()"
    "{"
    "    Color = color;"
    "    gl_Position = vec4(position, 0.0, 1.0);"
    "}";

const char* fragmentSource =
    "#version 150 core\n"
    "in vec3 Color;"
    "out vec4 outColor;"
    "void main()"
    "{"
    "    outColor = vec4(Color, 1.0);"
    "}";

void initShaders(GLuint & vertexShader, GLuint & fragmentShader,
                 GLuint & shaderProgram,
                 GLint & positionAttrib, GLint & colorAttrib,
                 GLint positionBuffer, GLint colorBuffer)
{
    // Create and compile the vertex shader
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    // Create and compile the fragment shader
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

    // Link the vertex and fragment shader into a shader program
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glBindFragDataLocation(shaderProgram, 0, "outColor");
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

    // Specify the layout of the vertex data
    positionAttrib = glGetAttribLocation(shaderProgram, "position");
    glEnableVertexAttribArray(positionAttrib);
    glBindBuffer(GL_ARRAY_BUFFER, positionBuffer);
    glVertexAttribPointer(positionAttrib, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), 0);

    colorAttrib = glGetAttribLocation(shaderProgram, "color");
    glEnableVertexAttribArray(colorAttrib);
    glBindBuffer(GL_ARRAY_BUFFER, colorBuffer);
    glVertexAttribPointer(colorAttrib, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
}
