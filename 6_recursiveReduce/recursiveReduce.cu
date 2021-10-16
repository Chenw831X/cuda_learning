#include <cuda_runtime.h>
#include <stdio.h>
#include "helper_cuda.h"

void initArray_Int(int* ip, int size){
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i<size; i++){
        ip[i] = int(rand()&0xff);
    }
}

__global__ void warmup(int *in, int *out, const int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size){
        return;
    }
    int tid = threadIdx.x;
    int *data = in + blockIdx.x * blockDim.x;

    for(int stride=1; stride<blockDim.x; stride<<=1){
        if(tid%(2*stride)==0){
            data[tid] += data[tid+stride];
        }
        // synchronize within block
        __syncthreads();
    }

    if(tid==0){
        out[blockIdx.x] = data[0];
    }
}

__global__ void reduceNeighbored(int *in, int *out, const int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size){
        return;
    }
    int tid = threadIdx.x;
    int *data = in + blockIdx.x * blockDim.x;

    for(int stride=1; stride<blockDim.x; stride<<=1){
        if(tid%(2*stride)==0){
            data[tid] += data[tid+stride];
        }
        // synchronize within block
        __syncthreads();
    }

    if(tid==0){
        out[blockIdx.x] = data[0];
    }
}

__global__ void reduceNeighboredLess(int *in ,int *out, const int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx>=size){
        return;
    }
    int tid = threadIdx.x;
    int *data = in + blockIdx.x * blockDim.x;

    for(int stride=1; stride<blockDim.x; stride<<=1){
        int index = tid * (2 * stride);
        if(index < blockDim.x){
            data[index] += data[index+stride];
        }
        __syncthreads();
    }

    if(tid==0){
        out[blockIdx.x] = data[0];
    }
}

__global__ void reduceInterleaved(int *in, int *out, const int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= size){
        return;
    }
    int tid = threadIdx.x;
    int *data = in + blockIdx.x * blockDim.x;

    for(int stride=blockDim.x/2; stride>0; stride>>=1){
        if(tid<stride){
            data[tid] += data[tid+stride];
        }
        __syncthreads();
    }

    if(tid==0){
        out[blockIdx.x] = data[0];
    }
}

__global__ void reduceUnroll2(int *in, int *out, const int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 2;
    if(idx >= size){
        return;
    }
    int tid = threadIdx.x;
    int *data = in + blockIdx.x * blockDim.x * 2;
    if(idx+blockDim.x < size){
        data[tid] += data[tid+blockDim.x];
    }
    __syncthreads();

    for(int stride=blockDim.x/2; stride>0; stride>>=1){
        if(tid < stride){
            data[tid] += data[tid + stride];
        }
        __syncthreads();
    }

    if(tid==0){
        out[blockIdx.x] = data[0];
    }
}

__global__ void reduceUnroll4(int *in, int *out, const int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 4;
    if(idx >= size){
        return;
    }
    int tid = threadIdx.x;
    int *data = in + blockIdx.x * blockDim.x * 4;
    if(idx+blockDim.x*3 < size){
        data[tid] += data[tid+blockDim.x];
        data[tid] += data[tid+blockDim.x*2];
        data[tid] += data[tid+blockDim.x*3];
    }
    __syncthreads();

    for(int stride=blockDim.x/2; stride>0; stride>>=1){
        if(tid < stride){
            data[tid] += data[tid + stride];
        }
        __syncthreads();
    }

    if(tid==0){
        out[blockIdx.x] = data[0];
    }
}

__global__ void reduceUnroll8(int *in, int *out, const int size){
    int idx = threadIdx.x + blockIdx.x * blockDim.x * 8;
    if(idx >= size){
        return;
    }
    int tid = threadIdx.x;
    int *data = in + blockIdx.x * blockDim.x * 8;
    if(idx+blockDim.x*7 < size){
        data[tid] += data[tid+blockDim.x];
        data[tid] += data[tid+blockDim.x*2];
        data[tid] += data[tid+blockDim.x*3];
        data[tid] += data[tid+blockDim.x*4];
        data[tid] += data[tid+blockDim.x*5];
        data[tid] += data[tid+blockDim.x*6];
        data[tid] += data[tid+blockDim.x*7];
    }
    __syncthreads();

    for(int stride=blockDim.x/2; stride>0; stride>>=1){
        if(tid < stride){
            data[tid] += data[tid + stride];
        }
        __syncthreads();
    }

    if(tid==0){
        out[blockIdx.x] = data[0];
    }
}

int main(int argc, char **argv){
    int dev = 0;
    cudaSetDevice(dev);

    int size = (1<<24);
    printf("Array size: %d\n", size);

    // execution configuration
    int blocksize = 1024;
    if(argc > 1){
        blocksize = atoi(argv[1]);
    }
    dim3 block(blocksize);
    dim3 grid((size-1)/block.x+1);

    // allocate host memory
    size_t nByte = size * sizeof(int);
    int *idata_h = (int*)malloc(nByte);
    int *odata_h = (int*)malloc(grid.x*sizeof(int));
    initArray_Int(idata_h, size);

    // allocate device memory
    int *idata_d = NULL;
    int *odata_d = NULL;
    CHECK(cudaMalloc((int**)&idata_d, nByte));
    CHECK(cudaMalloc((int**)&odata_d, grid.x*sizeof(int)));

    // CPU
    int cpuSum = 0;
    double cpuStart = cpuSecond();
    for(int i=0; i<size; ++i){
        cpuSum += idata_h[i];
    }
    double cpuEnd = cpuSecond();
    printf("CPU: elapsed time %f\n", cpuEnd-cpuStart);

    //GPU
    // kernel1: warmup
    CHECK(cudaMemcpy(idata_d, idata_h, nByte, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    double gpuStart = cpuSecond();
    warmup<<<grid, block>>>(idata_d, odata_d, size);
    cudaDeviceSynchronize();
    double gpuEnd = cpuSecond();
    CHECK(cudaMemcpy(odata_h, odata_d, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    int gpuSum = 0;
    for(int i=0; i<grid.x; ++i){
        gpuSum += odata_h[i];
    }
    if(gpuSum == cpuSum){
        printf("Check sum success!    ");
    }
    else{
        printf("Check sum fail!    ");
    }
    printf("GPU warm up: <<<%d, %d>>>, elapsed time %f\n", grid.x, block.x, gpuEnd-gpuStart);

    // kernel2: reduceNeighbored
    CHECK(cudaMemcpy(idata_d, idata_h, nByte, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    gpuStart = cpuSecond();
    reduceNeighbored<<<grid, block>>>(idata_d, odata_d, size);
    cudaDeviceSynchronize();
    gpuEnd = cpuSecond();
    CHECK(cudaMemcpy(odata_h, odata_d, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    gpuSum = 0;
    for(int i=0; i<grid.x; ++i){
        gpuSum += odata_h[i];
    }
    if(gpuSum == cpuSum){
        printf("Check sum success!    ");
    }
    else{
        printf("Check sum fail!    ");
    }
    printf("GPU reduceNeighbored: <<<%d, %d>>>, elapsed time %f\n", grid.x, block.x, gpuEnd-gpuStart);

    // kernel3: reduceNeighboredLess
    CHECK(cudaMemcpy(idata_d, idata_h, nByte, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    gpuStart = cpuSecond();
    reduceNeighboredLess<<<grid, block>>>(idata_d, odata_d, size);
    cudaDeviceSynchronize();
    gpuEnd = cpuSecond();
    CHECK(cudaMemcpy(odata_h, odata_d, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    gpuSum = 0;
    for(int i=0; i<grid.x; ++i){
        gpuSum += odata_h[i];
    }
    if(gpuSum == cpuSum){
        printf("Check sum success!    ");
    }
    else{
        printf("Check sum fail!    ");
    }
    printf("GPU reduceNeighboredLess: <<<%d, %d>>>, elapsed time %f\n", grid.x, block.x, gpuEnd-gpuStart);

    // kernel4: reduceInterleaved
    CHECK(cudaMemcpy(idata_d, idata_h, nByte, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    gpuStart = cpuSecond();
    reduceInterleaved<<<grid, block>>>(idata_d, odata_d, size);
    cudaDeviceSynchronize();
    gpuEnd = cpuSecond();
    CHECK(cudaMemcpy(odata_h, odata_d, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    gpuSum = 0;
    for(int i=0; i<grid.x; ++i){
        gpuSum += odata_h[i];
    }
    if(gpuSum == cpuSum){
        printf("Check sum success!    ");
    }
    else{
        printf("Check sum fail!    ");
    }
    printf("GPU reduceInterleaved: <<<%d, %d>>>, elapsed time %f\n", grid.x, block.x, gpuEnd-gpuStart);

    // kernel5: reduceUnroll2
    CHECK(cudaMemcpy(idata_d, idata_h, nByte, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    gpuStart = cpuSecond();
    reduceUnroll2<<<grid.x/2, block>>>(idata_d, odata_d, size);
    cudaDeviceSynchronize();
    gpuEnd = cpuSecond();
    CHECK(cudaMemcpy(odata_h, odata_d, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    gpuSum = 0;
    for(int i=0; i<grid.x/2; ++i){
        gpuSum += odata_h[i];
    }
    if(gpuSum == cpuSum){
        printf("Check sum success!    ");
    }
    else{
        printf("Check sum fail!    ");
    }
    printf("GPU reduceUnroll2: <<<%d, %d>>>, elapsed time %f\n", grid.x/2, block.x, gpuEnd-gpuStart);

    // kernel6: reduceUnroll4
    CHECK(cudaMemcpy(idata_d, idata_h, nByte, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    gpuStart = cpuSecond();
    reduceUnroll4<<<grid.x/4, block>>>(idata_d, odata_d, size);
    cudaDeviceSynchronize();
    gpuEnd = cpuSecond();
    CHECK(cudaMemcpy(odata_h, odata_d, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    gpuSum = 0;
    for(int i=0; i<grid.x/4; ++i){
        gpuSum += odata_h[i];
    }
    if(gpuSum == cpuSum){
        printf("Check sum success!    ");
    }
    else{
        printf("Check sum fail!    ");
    }
    printf("GPU reduceUnroll4: <<<%d, %d>>>, elapsed time %f\n", grid.x/4, block.x, gpuEnd-gpuStart);

    // kernel7: reduceUnroll8
    CHECK(cudaMemcpy(idata_d, idata_h, nByte, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    gpuStart = cpuSecond();
    reduceUnroll8<<<grid.x/8, block>>>(idata_d, odata_d, size);
    cudaDeviceSynchronize();
    gpuEnd = cpuSecond();
    CHECK(cudaMemcpy(odata_h, odata_d, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    gpuSum = 0;
    for(int i=0; i<grid.x/8; ++i){
        gpuSum += odata_h[i];
    }
    if(gpuSum == cpuSum){
        printf("Check sum success!    ");
    }
    else{
        printf("Check sum fail!    ");
    }
    printf("GPU reduceUnroll8: <<<%d, %d>>>, elapsed time %f\n", grid.x/8, block.x, gpuEnd-gpuStart);

    CHECK(cudaFree(idata_d));
    CHECK(cudaFree(odata_d));
    free(idata_h);
    free(odata_h);
    cudaDeviceReset();
    return 0;
}