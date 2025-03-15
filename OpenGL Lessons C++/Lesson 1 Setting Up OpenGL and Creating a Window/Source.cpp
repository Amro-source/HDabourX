#include <GL/glew.h>    // OpenGL Extension Wrangler (loads OpenGL functions)
#include <GLFW/glfw3.h> // GLFW (creates a window and handles input)
#include <iostream>     // For error messages

// Function to handle errors
void errorCallback(int error, const char* description) {
    std::cerr << "Error: " << description << std::endl;
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
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // Core profile (no legacy functions)

    // Step 3: Create a window
    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL Window", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Step 4: Initialize GLEW
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Print OpenGL version
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    // Step 5: Main loop
    while (!glfwWindowShouldClose(window)) {
        // Clear the screen with a color
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f); // Set clear color (dark cyan)
        glClear(GL_COLOR_BUFFER_BIT);          // Clear the color buffer

        // Swap buffers (double buffering)
        glfwSwapBuffers(window);

        // Poll for events (e.g., window close, keyboard input)
        glfwPollEvents();
    }

    // Step 6: Clean up
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}