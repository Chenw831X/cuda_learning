#include <cuda_runtime.h>
#include <stdio.h>
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

void addMatrixOnCPU(double *a, double *b, double *c, int size){
    for(int i=0; i<size; ++i){
        c[i] = a[i] + b[i];
    }
}

__global__ void addMatrixOnGPU(double *a, double *b, double *c, int nx, int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int id = ix + iy * ny;
    if(ix<nx && iy<ny){
        c[id] = a[id] + b[id];
    }
}

int main(int argc, char **argv){
    int dev = 0;
    cudaSetDevice(dev);

    int nx = (1<<12);
    int ny = (1<<12);
    int nxy = (nx * ny);
    int nByte = sizeof(double) * nxy;

    double *a_h = (double*)malloc(nByte);
    double *b_h = (double*)malloc(nByte);
    double *res_h = (double*)malloc(nByte);
    double *res_fromGPU_h = (double*)malloc(nByte);
    initArray(a_h, nxy);
    initArray(b_h, nxy);

    double *a_d = NULL;
    double *b_d = NULL;
    double *res_d = NULL;
    CHECK(cudaMalloc((double**)&a_d, nByte));
    CHECK(cudaMalloc((double**)&b_d, nByte));
    CHECK(cudaMalloc((double**)&res_d, nByte));

    CHECK(cudaMemcpy(a_d, a_h, nByte, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d, b_h, nByte, cudaMemcpyHostToDevice));

    double cpuStart = cpuSecond();
    addMatrixOnCPU(a_h, b_h, res_h, nxy);
    double cpuEnd = cpuSecond();
    printf("CPU: time elapsed: %f\n", cpuEnd - cpuStart);

    // 2d block and 2d grid
    int dimx = 32;
    int dimy = 32;
    dim3 block0(dimx, dimy);
    dim3 grid0((nx-1)/dimx+1, (ny-1)/dimy+1);
    double gpuStart = cpuSecond();
    addMatrixOnGPU<<<grid0, block0>>>(a_d, b_d, res_d, nx, ny);
    CHECK(cudaDeviceSynchronize());
    double gpuEnd = cpuSecond();
    printf("GPU: execution configuration <<<(%d, %d), (%d, %d)>>>, time elapsed %f\n",
        grid0.x, grid0.y, block0.x, block0.y, gpuEnd-gpuStart);
    CHECK(cudaMemcpy(res_fromGPU_h, res_d, nByte, cudaMemcpyDeviceToHost));
    checkResult(res_h, res_fromGPU_h, nxy);

    //1d grid and 1d block
    dimx = 32;
    dim3 block1(dimx);
    dim3 grid1((nxy-1)/block1.x+1);
    gpuStart = cpuSecond();
    addMatrixOnGPU<<<grid1, block1>>>(a_d, b_d, res_d, nxy, 1);
    CHECK(cudaDeviceSynchronize());
    gpuEnd = cpuSecond();
    printf("GPU: execution configuration <<<(%d, %d), (%d, %d)>>>, time elapsed %f\n",
        grid1.x, grid1.y, block1.x, block1.y, gpuEnd-gpuStart);
    CHECK(cudaMemcpy(res_fromGPU_h, res_d, nByte, cudaMemcpyDeviceToHost));
    checkResult(res_h, res_fromGPU_h, nxy);
    
    //2d grid and 1d block
    dimx = 32;
    dim3 block2(dimx);
    dim3 grid2((nx-1)/block2.x+1, ny);
    gpuStart = cpuSecond();
    addMatrixOnGPU<<<grid2, block2>>>(a_d, b_d, res_d, nx, ny);
    CHECK(cudaDeviceSynchronize());
    gpuEnd = cpuSecond();
    printf("GPU: execution configuration <<<(%d, %d), (%d, %d)>>>, time elapsed %f\n",
        grid2.x, grid2.y, block2.x, block2.y, gpuEnd-gpuStart);
    CHECK(cudaMemcpy(res_fromGPU_h, res_d, nByte, cudaMemcpyDeviceToHost));
    checkResult(res_h, res_fromGPU_h, nxy);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);
    free(a_h);
    free(b_h);
    free(res_h);
    free(res_fromGPU_h);
    cudaDeviceReset();
    return 0;
}
