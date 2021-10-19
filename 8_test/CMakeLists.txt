# cmake_minimum_required(VERSION 3.9)
# project(cudacpp LANGUAGES CXX CUDA)

# file(GLOB SRCFILES
#     "src/*.cu"
# )
# add_library(${PROJECT_NAME}_dev ${SRCFILES})
# target_include_directories(${PROJECT_NAME}_dev PUBLIC
#     "src"
# )

# add_executable(${PROJECT_NAME}_bin "src/main.cpp")
# target_link_libraries(${PROJECT_NAME}_bin PUBLIC ${PROJECT_NAME}_dev)
# include_directories(../include)

cmake_minimum_required(VERSION 3.9)
project(cudacpp LANGUAGES CXX CUDA)

add_executable(cudacpp "src/main.cpp" "src/test.cu")
include_directories("src" "../include" "/usr/local/include/eigen-3.4.0")