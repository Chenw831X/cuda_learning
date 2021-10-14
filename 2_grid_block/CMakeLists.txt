cmake_minimum_required(VERSION 3.9)
project(grid_block LANGUAGES CUDA)

add_executable(grid_block grid_block.cu)
include_directories(/home/cw/MyCode/cuda_learning/include)