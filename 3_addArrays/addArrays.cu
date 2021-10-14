#include <stdio.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

void initArray(double* ip,int size){
    time_t t;
    srand((unsigned )time(&t));
    for(int i=0;i<size;i++){
        ip[i]=(float)(rand()&0xffff)/1000.0f;
    }
}

void checkResult(double *a, double *b, const int size){
    double eps = 1e-8;
    int flag = 1;
    for(int i = 0; i<size; ++i){
        if(abs(a[i] - b[i]) > eps){
            flag = 0;
            break;
        }
    }
    if(flag){
        printf("Check result success!\n");
    }
    else{
        printf("Check result fail!\n");
    }
}

void addArraysOnCPU(double *a, double *b, double *res, const int size){
    for(int i=0; i<size; i+=4){
        res[i] = a[i] + b[i];
        res[i+1] = a[i+1] + b[i+1];
        res[i+2] = a[i+2] + b[i+2];
        res[i+3] = a[i+3] + b[i+3];
    }
}

__global__ void addArraysOnGPU(double *a, double *b, double *res){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    res[id] = a[id] + b[id];
}

int main(int argc, char **argv){
    int dev = 0;
    cudaSetDevice(dev);

    int n = (1<<24);
    printf("Vector size: %d\n", n);
    int nByte = sizeof(double) * n;
    double *a_h = (double*)malloc(nByte);
    double *b_h = (double*)malloc(nByte);
    double *res_h = (double*)malloc(nByte);
    double *res_fromGPU_h = (double*)malloc(nByte);
    memset(res_h, 0, nByte);
    memset(res_fromGPU_h, 0, nByte);

    double *a_d, *b_d, *res_d;
    CHECK(cudaMalloc((double**)&a_d, nByte));
    CHECK(cudaMalloc((double**)&b_d, nByte));
    CHECK(cudaMalloc((double**)&res_d, nByte));

    initArray(a_h, n);
    initArray(b_h, n);

    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    dim3 block(1<<10);
    dim3 grid((n-1)/block.x+1);
    
    double gpuStart = cpuSecond();
    addArraysOnGPU<<<grid, block>>>(a_d, b_d, res_d);
    CHECK(cudaDeviceSynchronize());
    double gpuEnd = cpuSecond();
    printf("Execution configuration <<<%d, %d>>>\n", grid.x, block.x);
    CHECK(cudaMemcpy(res_fromGPU_h, res_d, nByte, cudaMemcpyDeviceToHost));

    double cpuStart = cpuSecond();
    addArraysOnCPU(a_h, b_h, res_h, n);
    double cpuEnd = cpuSecond();
    checkResult(res_h, res_fromGPU_h, n);

    printf("GPU time elapsed: %f\n", gpuEnd - gpuStart);
    printf("CPU time elapsed: %f\n", cpuEnd - cpuStart);
    
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);
    free(a_h);
    free(b_h);
    free(res_h);
    free(res_fromGPU_h);

    return 0;
}