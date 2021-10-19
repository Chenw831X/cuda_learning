#include <iostream>
#include <cstdio>
#include <Eigen/Eigen>

#include "test.cuh"

int main(int argc, char** argv){
    /*
    Eigen::VectorXd a(3);
    a << 1, 2, 3;
    double *b = a.data();
    a(0) = 0;
    b[1] = -1;
    std::cout << "a:" << std::endl << a << std::endl;
    std::cout << "b:" << std::endl;
    for(int i=0; i<3; ++i){
        std::cout << b[i] << std::endl;
    }
    */
    Eigen::MatrixXd a2(2, 2);
    a2 << 1, 2, 3, 4;
    double *b2 = a2.transpose().data();

    std::cout << "a2:" << std::endl << a2 << std::endl;
    std::cout << "b2:" << std::endl;
    for(int i=0; i<2; ++i){
        for(int j=0; j<2; ++j){
            std::cout << b2[i*2+j] << std::endl;
        }
    }

    return 0;
}