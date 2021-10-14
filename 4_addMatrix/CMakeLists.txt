cmake_minimum_required(VERSION 3.9)
project(addMatrix LANGUAGES CUDA)

add_executable(addMatrix addMatrix.cu)
include_directories(/home/cw/MyCode/cuda_learning/include)