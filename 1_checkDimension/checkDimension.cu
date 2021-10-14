#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void){
    printf("threadIdx: (%d, %d, %d)    blockIdx: (%d, %d, %d)    \
        blockDim: (%d, %d, %d)    gridDim: (%d, %d, %d)\n",
        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x,
        blockIdx.y, blockIdx.z, blockDim.x, blockDim.y,
        blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main(int argc, char **argv){
    int n = 6;
    dim3 block(3);
    dim3 grid((n-1)/block.x+1);
    printf("grid: (%d, %d, %d)\n", grid.x, grid.y, grid.z);
    printf("block: (%d, %d, %d)\n", block.x, block.y, block.z);
    
    checkIndex<<<grid, block>>>();

    cudaDeviceReset();
    return 0;
}