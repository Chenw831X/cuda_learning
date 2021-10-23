#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <cstdio>

#include "test.cuh"


__global__ void addMatrix(double *a, int row, int col){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= row){
        return;
    }
    for(int i=0; i<col; ++i){
        a[idx+i*row] += 0.5;
    }
}

void func(Eigen::MatrixXd &a){
    int dev = 0;
    cudaSetDevice(dev);

    int row = a.rows(), col = a.cols();
    int size = a.size();
    double *a_h = a.data();
    double *a_d = NULL;
    CHECK(cudaMalloc((double**)&a_d, size*sizeof(double)));
    CHECK(cudaMemcpy(a_d, a_h, size*sizeof(double), cudaMemcpyHostToDevice));

    dim3 block(1024);
    dim3 grid((row-1)/block.x+1);
    addMatrix<<<grid, block>>>(a_d, row, col);
    cudaDeviceSynchronize();

    CHECK(cudaMemcpy(a_h, a_d, size*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(a_d);
    cudaDeviceReset();
}
