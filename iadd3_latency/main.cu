#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

#define SHARED_MEM_SIZE 32
#define THD_NUM 32
#define WARP_SIZE 32

#ifndef REP
#define REP 8
#endif

// CUDA kernel to perform memory loads from shared memory and measure latency
__global__ void benchmarkIADDLatency(unsigned long long *d_start, unsigned long long *d_end, uint32_t* data) {
    // Declare shared memory
    // __shared__ uint32_t sharedMem[SHARED_MEM_SIZE];
    int tid = threadIdx.x;
    // int warp_id = tid / WARP_SIZE;
    // if(warp_id == 0){
    //     tmem_allocate(sharedMem, 512);
    // }
    // __syncthreads();

    // uint32_t tmem_ptr = sharedMem[0];
    // uint32_t tmem_ptr1 = tmem_ptr + 128;
    
    uint32_t val_array[REP];

    #pragma unroll
    for(int i = 0; i < REP; i++){
        val_array[i] = data[tid + i * THD_NUM];
    }
    
    // tmem_st_32dp32bNx<REP>(tmem_ptr, val_array);
    // tmem_st_32dp32bNx<REP>(tmem_ptr1, val_array);

    // fence_view_async_tmem_store();

    __syncthreads();
    unsigned long long start[2];
    unsigned long long end[2];
    // uint32_t val_array_tmp[REP];
    uint32_t dst = 0;
    __threadfence();

    __syncthreads();
    __syncwarp();
    start[0] = clock64();

    #pragma unroll
    for(int i = 0; i < REP; i++){
        dst += val_array[i];
    }

    end[0] = clock64();
    // if(warp_id < 4){

    // }


    __syncthreads();

    // if(warp_id == 0){
    //     tmem_free(tmem_ptr, 512);
    // }
    // __syncthreads();

    // Write the start and end times to global memory
    if (tid == 0) {
        d_start[0] = start[0];
        d_end[0] = end[0];
        data[0] = dst;
    }

    // #pragma unroll
    // for(int i = 0; i < REP; i++){
    //     data[tid + i * THD_NUM] = val_array[i];
    // }
}

int main() {
    // Allocate memory on the device for start and end times
    unsigned long long *d_start, *d_end;
    uint32_t *d_data;
    cudaMalloc(&d_data, sizeof(uint32_t) * THD_NUM * REP);
    cudaMalloc(&d_start, sizeof(unsigned long long) * 2);
    cudaMalloc(&d_end, sizeof(unsigned long long) * 2);

    // Launch the kernel
    benchmarkIADDLatency<<<1, THD_NUM>>>(d_start, d_end, d_data);

    // Copy the start and end times back to the host
    unsigned long long* h_start = (unsigned long long*)malloc(sizeof(unsigned long long) * 2);
    unsigned long long* h_end = (unsigned long long*)malloc(sizeof(unsigned long long) * 2);

    cudaMemcpy(h_start, d_start, sizeof(unsigned long long) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_end, d_end, sizeof(unsigned long long) * 2, cudaMemcpyDeviceToHost);


    // Calculate the latency in clock cycles
    double latency = (h_end[0] - h_start[0]);
    // Print the result
    std::cout << "IADD Latency: " << latency << " clock cycles" << std::endl;

    // Clean up
    cudaFree(d_start);
    cudaFree(d_end);
    free(h_start);
    free(h_end);
    cudaFree(d_data);

    return 0;
}