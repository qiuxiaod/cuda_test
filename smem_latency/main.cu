#include <iostream>
#include <cuda_runtime.h>

// Define the size of the shared memory array
#define SHARED_MEM_SIZE 32
#define THD_NUM 32

// CUDA kernel to perform memory loads from shared memory and measure latency
__global__ void benchmarkSLMLoadLatency(unsigned long long *d_start, unsigned long long *d_end, int* data) {
    // Declare shared memory
    __shared__ int sharedMem[SHARED_MEM_SIZE];

    // Initialize shared memory
    int tid = threadIdx.x;
    for (int i = tid; i < SHARED_MEM_SIZE; i += THD_NUM) {
        sharedMem[i] = data[i];
    }
    __syncthreads();
    int value = data[0];
    // Measure the start time
    unsigned long long start = clock64();

    // Perform memory loads from shared memory
    

    for (int i = 0; i < SHARED_MEM_SIZE; i += THD_NUM) {
        value += sharedMem[i];
    }

    // Measure the end time
    unsigned long long end = clock64();

    // Write the start and end times to global memory
    if (tid == 0) {
        d_start[0] = start;
        d_end[0] = end;
        data[0] = value;
    }
}

int main() {
    // Allocate memory on the device for start and end times
    unsigned long long *d_start, *d_end;
    int *d_data;
    cudaMalloc(&d_data, sizeof(int) * SHARED_MEM_SIZE);
    cudaMalloc(&d_start, sizeof(unsigned long long));
    cudaMalloc(&d_end, sizeof(unsigned long long));

    // Launch the kernel
    benchmarkSLMLoadLatency<<<1, THD_NUM>>>(d_start, d_end, d_data);

    // Copy the start and end times back to the host
    unsigned long long h_start, h_end;
    cudaMemcpy(&h_start, d_start, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_end, d_end, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Calculate the latency in clock cycles
    double latency = (h_end - h_start) / double(SHARED_MEM_SIZE / THD_NUM);

    // Print the result
    std::cout << "SLM Load Latency: " << latency << " clock cycles" << std::endl;

    // Clean up
    cudaFree(d_start);
    cudaFree(d_end);

    return 0;
}

// Compile with: nvcc -o slm_test slm_test.cu