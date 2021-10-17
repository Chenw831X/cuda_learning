#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <cstdio>

#include "addArray.cuh"

__global__ void addArray(int *a, int *b, int *c, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= n){
        return;
    }
    c[idx] = a[idx] + b[idx];
}

void func(int *a, int *b, int *c, int n){
    int dev = 0;
    cudaSetDevice(dev);

    int *a_d = NULL;
    int *b_d = NULL;
    int *c_d = NULL;
    CHECK(cudaMalloc((int**)&a_d, n*sizeof(int)));
    CHECK(cudaMalloc((int**)&b_d, n*sizeof(int)));
    CHECK(cudaMalloc((int**)&c_d, n*sizeof(int)));

    CHECK(cudaMemcpy(a_d, a, n*sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b, n*sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid((n-1)/block.x+1);
    addArray<<<grid, block>>>(a_d, b_d, c_d, n);
    cudaDeviceSynchronize();
    CHECK(cudaMemcpy(c, c_d, n*sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    cudaDeviceReset();
}