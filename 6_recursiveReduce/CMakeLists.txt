cmake_minimum_required(VERSION 3.9)
project(recursiveReduce LANGUAGES CUDA)

add_executable(recursiveReduce recursiveReduce.cu)
include_directories(../include)