#include <iostream>
#include <cstdio>
#include <Eigen/Eigen>

#include "test.cuh"

int main(int argc, char** argv){
    Eigen::MatrixXd a(3, 3);
    a << 1, 1, 1, 2, 2, 2, 3, 3, 3;
    std::cout << a << std::endl << std::endl;

    func(a);
    std::cout << a << std::endl << std::endl;

    return 0;
}
