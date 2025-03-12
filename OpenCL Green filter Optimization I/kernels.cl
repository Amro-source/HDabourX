__kernel void example_kernel(__global const float* input, __global float* output) {
    int gid = get_global_id(0);
    output[gid] = input[gid] * 2.0f;
}