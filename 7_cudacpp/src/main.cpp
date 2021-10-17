#include <cstdio>
#include "addArray.cuh"

int main(int argc, char** argv){
    int a[10], b[10], c[10];
    for(int i=0; i<10; ++i){
        a[i] = i;
        b[i] = i;
    }
    func(a, b, c, 10);

    printf("c:");
    for(int i=0; i<10; ++i){
        printf(" %d", c[i]);
    }
    printf("\n");

    return 0;
}