cmake_minimum_required(VERSION 3.9)
project(checkDimension LANGUAGES CUDA)

add_executable(checkDimension checkDimension.cu)
include_directories(/home/cw/MyCode/cuda_learning/include)