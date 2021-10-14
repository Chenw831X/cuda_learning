cmake_minimum_required(VERSION 3.9)
project(hello_world LANGUAGES CUDA)

add_executable(hello_world helloworld.cu)
include_directories(/home/cw/MyCode/cuda_learning/include)