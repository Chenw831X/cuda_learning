#include <stdio.h>
#include "helper_cuda.h"

__global__ void hello_world(void){
    printf("GPU: hello world\n");
}

int main(int argc, char **argv){
    printf("CPU: hello world\n");
    double tBegin = cpuSecond();
    hello_world<<<1, 10>>>();
    cudaDeviceReset();
    double tElapsed = cpuSecond() - tBegin;
    printf("Time elapsed: %f\n", tElapsed);
    return 0;
}