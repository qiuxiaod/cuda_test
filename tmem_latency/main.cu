#include <iostream>
#include "../inline_ptx_func.hpp"
#include <cuda_runtime.h>

#define SHARED_MEM_SIZE (8192 + 256)
#define WARP_SIZE 32

#ifndef THD_NUM
#define THD_NUM 128
#endif

#ifndef REP
#define REP 8
#endif

#ifndef TEST_MODE
#define TEST_MODE 0
#endif

// CUDA kernel to perform memory loads from shared memory and measure latency
__global__ void benchmarkTMEMLoadLatency(unsigned long long *d_start, unsigned long long *d_end, uint32_t* data) {
    // Declare shared memory
    __shared__ uint32_t sharedMem[SHARED_MEM_SIZE];
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    if(warp_id == 0){
        tmem_allocate(sharedMem, 512);
    }
    __syncthreads();

    uint32_t tmem_ptr = sharedMem[0];
    uint32_t tmem_ptr1 = tmem_ptr + 128;
    
    uint32_t val_array[REP];

    #pragma unroll
    for(int i = 0; i < REP; i++){
        val_array[i] = data[tid + i * THD_NUM];
    }
    
    tmem_st_32dp32bNx<REP>(tmem_ptr, val_array);
    tmem_st_32dp32bNx<REP>(tmem_ptr1, val_array);

    fence_view_async_tmem_store();

    __syncthreads();
    unsigned long long start[2];
    unsigned long long end[2];
    uint32_t val_array_tmp[REP];

#if TEST_MODE == 0
        __syncthreads();
        __syncwarp();
        start[0] = clock64();

        tmem_ld_32dp32bNx<REP>(tmem_ptr, val_array_tmp);
        fence_view_async_tmem_load();

        val_array[0] += val_array_tmp[0];

        end[0] = clock64();
#elif TEST_MODE == 1
        __syncthreads();
        __syncwarp();
        start[0] = clock64();

        tmem_st_32dp32bNx<REP>(tmem_ptr, val_array);
        fence_view_async_tmem_store();
        tmem_ld_32dp32bNx<REP>(tmem_ptr, val_array_tmp);
        fence_view_async_tmem_load();

        val_array[0] += val_array_tmp[0];
        end[0] = clock64();

#elif TEST_MODE == 2
        uint32_t val_array_tmp1[REP];
        __syncthreads();
        __syncwarp();
        start[0] = clock64();

        tmem_st_32dp32bNx<REP>(tmem_ptr, val_array);
        tmem_ld_32dp32bNx<REP>(tmem_ptr1, val_array_tmp1);
        fence_view_async_tmem_store();
        tmem_ld_32dp32bNx<REP>(tmem_ptr, val_array_tmp);
        fence_view_async_tmem_load();

        val_array[0] += val_array_tmp[0] + val_array_tmp1[0];
        end[0] = clock64();

#elif TEST_MODE == 3
        uint32_t val_array_tmp1[REP];
        __syncthreads();
        __syncwarp();
        start[0] = clock64();

        tmem_ld_32dp32bNx<REP>(tmem_ptr1, val_array_tmp1);
        tmem_ld_32dp32bNx<REP>(tmem_ptr, val_array_tmp);
        fence_view_async_tmem_load();

        val_array[0] += val_array_tmp[0] + val_array_tmp1[0];
        end[0] = clock64();
#elif TEST_MODE == 4
        uint32_t val_array_tmp1[REP];
        
        __syncthreads();
        if(warp_id < 4){
            __syncwarp();
            start[0] = clock64();

            tmem_ld_32dp32bNx<REP>(tmem_ptr1, val_array_tmp1);
            tmem_ld_32dp32bNx<REP>(tmem_ptr, val_array_tmp);
            fence_view_async_tmem_load();

            val_array[0] += val_array_tmp[0] + val_array_tmp1[0];
            end[0] = clock64();
        }else{
            // different fpu code seems will cause a small difference with small REP. For big REP, seems pretty stable
            float* val_array_f32 = (float*)val_array;
            for(int i = 0; i < REP; i++){
                val_array_f32[i] += val_array_f32[i];
            }
            for(int i = 0; i < REP; i++){
                val_array_f32[i] *= val_array_f32[i];
            }
            for(int i = 0; i < REP; i++){
                val_array_f32[i] -= val_array_f32[i];
            }
            for(int i = 0; i < REP; i++){
                val_array_f32[i] *= val_array_f32[i];
            }
            for(int i = 0; i < REP; i++){
                val_array_f32[i] -= val_array_f32[i];
            }
            for(int i = 0; i < REP; i++){
                val_array_f32[i] *= val_array_f32[i];
            }
           
            // data[0] = (uint32_t)val_array_f32[0];
        }
#elif TEST_MODE == 5
        uint32_t val_array_tmp1[REP];
        __syncthreads();
        __syncwarp();
        constexpr int M = 128;
        constexpr int N = 128;
        constexpr int K = 64; // wg_k
        if(tid == 0){
            InstrDescriptor idesc = make_instr_desc<M, N>();
            SmemDescriptor desc_a = make_smem_desc<M, K>((uint16_t*) sharedMem);
            SmemDescriptor desc_b = make_smem_desc<M, K>((uint16_t*) (sharedMem + 4096));
            amma_fp16bf16_ss<M, N>(uint64_t(desc_a), uint64_t (desc_b), tmem_ptr, uint32_t(idesc));
            amma_fp16bf16_ss<M, N>(uint64_t(desc_a), uint64_t (desc_b), tmem_ptr, uint32_t(idesc));
            amma_fp16bf16_ss<M, N>(uint64_t(desc_a), uint64_t (desc_b), tmem_ptr, uint32_t(idesc));
            amma_fp16bf16_ss<M, N>(uint64_t(desc_a), uint64_t (desc_b), tmem_ptr, uint32_t(idesc));
        }
        __syncwarp();
        start[0] = clock64();

        tmem_ld_32dp32bNx<REP>(tmem_ptr1, val_array_tmp1);
        tmem_ld_32dp32bNx<REP>(tmem_ptr, val_array_tmp);
        fence_view_async_tmem_load();

        val_array[0] += val_array_tmp[0] + val_array_tmp1[0];
        end[0] = clock64();
#endif
    // if(warp_id < 4){

    // }


    __syncthreads();

    if(warp_id == 0){
        tmem_free(tmem_ptr, 512);
    }
    __syncthreads();

    // Write the start and end times to global memory
    if (tid == 0) {
        d_start[0] = start[0];
        d_end[0] = end[0];
    }

    #pragma unroll
    for(int i = 0; i < REP; i++){
        data[tid + i * THD_NUM] = val_array[i];
    }

}

int main() {
    // Allocate memory on the device for start and end times
    unsigned long long *d_start, *d_end;
    uint32_t *d_data;
    cudaMalloc(&d_data, sizeof(uint32_t) * THD_NUM * REP);
    cudaMalloc(&d_start, sizeof(unsigned long long) * 2);
    cudaMalloc(&d_end, sizeof(unsigned long long) * 2);

    // Launch the kernel
    benchmarkTMEMLoadLatency<<<1, THD_NUM>>>(d_start, d_end, d_data);

    // Copy the start and end times back to the host
    unsigned long long* h_start = (unsigned long long*)malloc(sizeof(unsigned long long) * 2);
    unsigned long long* h_end = (unsigned long long*)malloc(sizeof(unsigned long long) * 2);

    cudaMemcpy(h_start, d_start, sizeof(unsigned long long) * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_end, d_end, sizeof(unsigned long long) * 2, cudaMemcpyDeviceToHost);

    #if TEST_MODE == 0
        // Calculate the latency in clock cycles
        double latency = (h_end[0] - h_start[0]);
        // Print the result
        std::cout << "TMEM Load[0] Latency: " << latency << " clock cycles" << std::endl;
    #elif TEST_MODE == 1
        double latency = (h_end[0] - h_start[0]);
        std::cout << "TMEM Store[0] + Load[0] Latency: " << latency << " clock cycles" << std::endl;
    #elif TEST_MODE == 2
        double latency = (h_end[0] - h_start[0]);
        std::cout << "TMEM Store[0] + Load[1] + Load[0] Latency: " << latency << " clock cycles" << std::endl;
    #elif TEST_MODE == 3
        double latency = (h_end[0] - h_start[0]);
        std::cout << "TMEM Load[0] + Load[1] Latency: " << latency << " clock cycles" << std::endl;
    #elif TEST_MODE == 4
        double latency = (h_end[0] - h_start[0]);
        std::cout << "TMEM FPU + Load[0] + Load[1] Latency: " << latency << " clock cycles" << std::endl;
    #elif TEST_MODE == 5
        double latency = (h_end[0] - h_start[0]);
        std::cout << "TMEM AMMA + Load[0] + Load[1] Latency: " << latency << " clock cycles" << std::endl;
    #endif
    // Clean up
    cudaFree(d_start);
    cudaFree(d_end);
    free(h_start);
    free(h_end);
    cudaFree(d_data);

    return 0;
}