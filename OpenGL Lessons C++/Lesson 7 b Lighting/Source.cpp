#define _CRT_SECURE_NO_WARNINGS
#include <cstdio> // or other headers
#include <GL/glew.h>    // OpenGL Extension Wrangler (loads OpenGL functions)
#include <GLFW/glfw3.h> // GLFW (creates a window and handles input)
#include <iostream>     // For error messages
#include <glm/glm.hpp>  // GLM for matrix math
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // stb_image for loading images
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" // stb_image_write for saving images

// Vertex Shader Source Code
const char* vertexShaderSource = R"(
#version 330 core
layout(location = 0) in vec3 aPos;  // Vertex position attribute
layout(location = 1) in vec3 aNormal; // Vertex normal attribute
layout(location = 2) in vec2 aTexCoord; // Texture coordinate attribute
out vec3 FragPos; // Pass fragment position to fragment shader
out vec3 Normal;  // Pass normal to fragment shader
out vec2 TexCoord; // Pass texture coordinates to fragment shader
uniform mat4 model; // Model matrix
uniform mat4 view;  // View matrix
uniform mat4 projection; // Projection matrix
void main() {
    FragPos = vec3(model * vec4(aPos, 1.0)); // Transform vertex position
    Normal = mat3(transpose(inverse(model))) * aNormal; // Transform normal
    TexCoord = aTexCoord; // Pass texture coordinates
    gl_Position = projection * view * model * vec4(aPos, 1.0); // Final vertex position
}
)";

// Fragment Shader Source Code
const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor; // Output color
in vec3 FragPos;    // Fragment position from vertex shader
in vec3 Normal;     // Normal from vertex shader
in vec2 TexCoord;   // Texture coordinates from vertex shader
uniform sampler2D ourTexture; // Texture sampler
// Directional light
struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};
uniform DirLight dirLight;
// Point light
struct PointLight {
    vec3 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
};
uniform PointLight pointLight;
// Spotlight
struct SpotLight {
    vec3 position;
    vec3 direction;
    float cutOff;
    float outerCutOff;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float constant;
    float linear;
    float quadratic;
};
uniform SpotLight spotLight;
// Function to calculate directional light
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir) {
    vec3 lightDir = normalize(-light.direction);
    // Diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // Specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    // Combine results
    vec3 ambient = light.ambient * texture(ourTexture, TexCoord).rgb;
    vec3 diffuse = light.diffuse * diff * texture(ourTexture, TexCoord).rgb;
    vec3 specular = light.specular * spec * texture(ourTexture, TexCoord).rgb;
    return (ambient + diffuse + specular);
}
// Function to calculate point light
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    // Diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // Specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    // Attenuation
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    // Combine results
    vec3 ambient = light.ambient * texture(ourTexture, TexCoord).rgb;
    vec3 diffuse = light.diffuse * diff * texture(ourTexture, TexCoord).rgb;
    vec3 specular = light.specular * spec * texture(ourTexture, TexCoord).rgb;
    ambient *= attenuation;
    diffuse *= attenuation;
    specular *= attenuation;
    return (ambient + diffuse + specular);
}
// Function to calculate spotlight
vec3 CalcSpotLight(SpotLight light, vec3 normal, vec3 fragPos, vec3 viewDir) {
    vec3 lightDir = normalize(light.position - fragPos);
    // Diffuse shading
    float diff = max(dot(normal, lightDir), 0.0);
    // Specular shading
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    // Attenuation
    float distance = length(light.position - fragPos);
    float attenuation = 1.0 / (light.constant + light.linear * distance + light.quadratic * (distance * distance));
    // Spotlight intensity
    float theta = dot(lightDir, normalize(-light.direction));
    float epsilon = light.cutOff - light.outerCutOff;
    float intensity = clamp((theta - light.outerCutOff) / epsilon, 0.0, 1.0);
    // Combine results
    vec3 ambient = light.ambient * texture(ourTexture, TexCoord).rgb;
    vec3 diffuse = light.diffuse * diff * texture(ourTexture, TexCoord).rgb;
    vec3 specular = light.specular * spec * texture(ourTexture, TexCoord).rgb;
    ambient *= attenuation * intensity;
    diffuse *= attenuation * intensity;
    specular *= attenuation * intensity;
    return (ambient + diffuse + specular);
}
void main() {
    // Properties
    vec3 norm = normalize(Normal);
    vec3 viewDir = normalize(-FragPos); // The camera is at (0, 0, 0) in view space
    // Directional light
    vec3 result = CalcDirLight(dirLight, norm, viewDir);
    // Point light
    result += CalcPointLight(pointLight, norm, FragPos, viewDir);
    // Spotlight
    result += CalcSpotLight(spotLight, norm, FragPos, viewDir);
    FragColor = vec4(result, 1.0);
}
)";

// Function to handle errors
void errorCallback(int error, const char* description) {
    std::cerr << "Error: " << description << std::endl;
}

// Camera settings
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f); // Camera position
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f); // Camera direction
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f); // Camera up vector
float yaw = -90.0f;   // Yaw angle (left/right)
float pitch = 0.0f;   // Pitch angle (up/down)
float lastX = 400.0f; // Last mouse X position
float lastY = 300.0f; // Last mouse Y position
bool firstMouse = true; // First mouse movement

// Mouse callback function
void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }
    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // Reversed since y-coordinates go from bottom to top
    lastX = xpos;
    lastY = ypos;
    float sensitivity = 0.1f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;
    yaw += xoffset;
    pitch += yoffset;
    // Clamp pitch to avoid flipping
    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;
    // Update camera direction
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

// Save screenshot function
void saveScreenshot(const char* filename, int width, int height) {
    unsigned char* pixels = new unsigned char[width * height * 3]; // RGB format
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

    // Flip the image vertically (OpenGL reads pixels bottom-to-top)
    for (int y = 0; y < height / 2; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                std::swap(pixels[(y * width + x) * 3 + c], pixels[((height - y - 1) * width + x) * 3 + c]);
            }
        }
    }

    // Save the image using stb_image_write
    stbi_write_png(filename, width, height, 3, pixels, width * 3);

    delete[] pixels;
    std::cout << "Screenshot saved as " << filename << std::endl;
}

// Process keyboard input for camera movement and screenshot
void processInput(GLFWwindow* window, int windowWidth, int windowHeight) {
    float cameraSpeed = 0.01f;

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        cameraPos += cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        cameraPos -= cameraSpeed * cameraFront;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;

    // Check for Ctrl+S
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS && glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
        saveScreenshot("screenshot.png", windowWidth, windowHeight);
    }
}

int main() {
    // Step 1: Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Set error callback
    glfwSetErrorCallback(errorCallback);

    // Step 2: Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Core profile

    // Step 3: Create a window
    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL Lighting", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Set mouse callback
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // Hide and capture mouse

    // Step 4: Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Print OpenGL version
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    // Step 5: Define vertex data for a cube
    float vertices[] = {
        // Positions          // Normals           // Texture Coords
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f, 1.0f
    };

    // Step 6: Set up VBO, VAO, and texture coordinates
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Normal attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Texture coordinate attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // Unbind VBO and VAO (optional)
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);

    // Step 7: Load and create a texture
    unsigned int texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Set texture wrapping and filtering options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Load image using stb_image
    int width, height, nrChannels;
    unsigned char* data = stbi_load("Cherno.png", &width, &height, &nrChannels, 0);
    if (data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else {
        std::cerr << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);

    // Step 8: Create and compile shaders
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
    glCompileShader(vertexShader);

    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
        std::cerr << "Vertex Shader Compilation Failed: " << infoLog << std::endl;
    }

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
    glCompileShader(fragmentShader);

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
        std::cerr << "Fragment Shader Compilation Failed: " << infoLog << std::endl;
    }

    // Step 9: Create a shader program and link shaders
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader Program Linking Failed: " << infoLog << std::endl;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Step 10: Enable depth testing
    glEnable(GL_DEPTH_TEST);

    // Step 11: Set up lighting
    glm::vec3 dirLightDirection = glm::vec3(-0.2f, -1.0f, -0.3f);
    glm::vec3 dirLightAmbient = glm::vec3(0.2f, 0.2f, 0.2f);
    glm::vec3 dirLightDiffuse = glm::vec3(0.5f, 0.5f, 0.5f);
    glm::vec3 dirLightSpecular = glm::vec3(1.0f, 1.0f, 1.0f);

    glm::vec3 pointLightPosition = glm::vec3(1.2f, 1.0f, 2.0f);
    glm::vec3 pointLightAmbient = glm::vec3(0.2f, 0.2f, 0.2f);
    glm::vec3 pointLightDiffuse = glm::vec3(0.5f, 0.5f, 0.5f);
    glm::vec3 pointLightSpecular = glm::vec3(1.0f, 1.0f, 1.0f);
    float pointLightConstant = 1.0f;
    float pointLightLinear = 0.09f;
    float pointLightQuadratic = 0.032f;

    glm::vec3 spotLightPosition = cameraPos;
    glm::vec3 spotLightDirection = cameraFront;
    float spotLightCutOff = glm::cos(glm::radians(12.5f));
    float spotLightOuterCutOff = glm::cos(glm::radians(17.5f));
    glm::vec3 spotLightAmbient = glm::vec3(0.2f, 0.2f, 0.2f);
    glm::vec3 spotLightDiffuse = glm::vec3(0.5f, 0.5f, 0.5f);
    glm::vec3 spotLightSpecular = glm::vec3(1.0f, 1.0f, 1.0f);
    float spotLightConstant = 1.0f;
    float spotLightLinear = 0.09f;
    float spotLightQuadratic = 0.032f;

    // Step 12: Main loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window, 800, 600);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f); // Dark cyan background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);
        glBindTexture(GL_TEXTURE_2D, texture);

        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);

        glm::mat4 model = glm::mat4(1.0f); // Identity matrix
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glUniform3fv(glGetUniformLocation(shaderProgram, "dirLight.direction"), 1, glm::value_ptr(dirLightDirection));
        glUniform3fv(glGetUniformLocation(shaderProgram, "dirLight.ambient"), 1, glm::value_ptr(dirLightAmbient));
        glUniform3fv(glGetUniformLocation(shaderProgram, "dirLight.diffuse"), 1, glm::value_ptr(dirLightDiffuse));
        glUniform3fv(glGetUniformLocation(shaderProgram, "dirLight.specular"), 1, glm::value_ptr(dirLightSpecular));

        glUniform3fv(glGetUniformLocation(shaderProgram, "pointLight.position"), 1, glm::value_ptr(pointLightPosition));
        glUniform3fv(glGetUniformLocation(shaderProgram, "pointLight.ambient"), 1, glm::value_ptr(pointLightAmbient));
        glUniform3fv(glGetUniformLocation(shaderProgram, "pointLight.diffuse"), 1, glm::value_ptr(pointLightDiffuse));
        glUniform3fv(glGetUniformLocation(shaderProgram, "pointLight.specular"), 1, glm::value_ptr(pointLightSpecular));
        glUniform1f(glGetUniformLocation(shaderProgram, "pointLight.constant"), pointLightConstant);
        glUniform1f(glGetUniformLocation(shaderProgram, "pointLight.linear"), pointLightLinear);
        glUniform1f(glGetUniformLocation(shaderProgram, "pointLight.quadratic"), pointLightQuadratic);

        glUniform3fv(glGetUniformLocation(shaderProgram, "spotLight.position"), 1, glm::value_ptr(cameraPos));
        glUniform3fv(glGetUniformLocation(shaderProgram, "spotLight.direction"), 1, glm::value_ptr(cameraFront));
        glUniform1f(glGetUniformLocation(shaderProgram, "spotLight.cutOff"), spotLightCutOff);
        glUniform1f(glGetUniformLocation(shaderProgram, "spotLight.outerCutOff"), spotLightOuterCutOff);
        glUniform3fv(glGetUniformLocation(shaderProgram, "spotLight.ambient"), 1, glm::value_ptr(spotLightAmbient));
        glUniform3fv(glGetUniformLocation(shaderProgram, "spotLight.diffuse"), 1, glm::value_ptr(spotLightDiffuse));
        glUniform3fv(glGetUniformLocation(shaderProgram, "spotLight.specular"), 1, glm::value_ptr(spotLightSpecular));
        glUniform1f(glGetUniformLocation(shaderProgram, "spotLight.constant"), spotLightConstant);
        glUniform1f(glGetUniformLocation(shaderProgram, "spotLight.linear"), spotLightLinear);
        glUniform1f(glGetUniformLocation(shaderProgram, "spotLight.quadratic"), spotLightQuadratic);

        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 36);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Step 13: Clean up
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}