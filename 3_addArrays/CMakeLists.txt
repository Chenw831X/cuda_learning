cmake_minimum_required(VERSION 3.9)
project(addArrays LANGUAGES CUDA)

add_executable(addArrays addArrays.cu)
include_directories(/home/cw/MyCode/cuda_learning/include)