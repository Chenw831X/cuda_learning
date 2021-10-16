cmake_minimum_required(VERSION 3.9)
project(cudaInfo LANGUAGES CUDA)

add_executable(cudaInfo cudaInfo.cu)
include_directories(/home/cw/MyCode/cuda_learning/include)