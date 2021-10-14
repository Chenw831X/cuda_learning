#include <cuda_runtime.h>
#include <stdio.h>
#include "helper_cuda.h"

int main(int argc, char **argv){
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceReset();
    return 0;
}