#include "pch.h"
#include <iostream>
extern "C" {
    __declspec(dllexport) int add(int a, int b) {
        return a + b;
    }

    __declspec(dllexport) void print_message(const char* message) {
        std::cout << "Message from C++: " << message << std::endl;
    }

    __declspec(dllexport) double multiply(double a, double b) {
        return a * b;
    }
}
