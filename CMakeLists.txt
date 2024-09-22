cmake_minimum_required(VERSION 3.10)

project(plain_nn VERSION 1.0 LANGUAGES CXX DESCRIPTION "A simple neural network library")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR}/include)

# GLOB all subdirectories inside src
# file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(plain_nn SHARED
    ${PROJECT_SOURCE_DIR}/plain_nn/src/data_loaders/mnist_dataloader.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/layers/activation_fncs/activation_fncs.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/layers/activation_fncs/none.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/layers/activation_fncs/relu.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/layers/activation_fncs/sigmoid.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/layers/activation_fncs/tanh.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/layers/dense.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/layers/initialization.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/layers/input.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/layers/layers.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/lr_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/model_storage.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/plain_nn.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/tensor.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/utils.cpp
    ${PROJECT_SOURCE_DIR}/plain_nn/src/image_utils.cpp
)
# set output directory for mnist_cpp to bin folder
# set_target_properties(mnist_cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
target_include_directories(plain_nn PUBLIC ${PROJECT_SOURCE_DIR}/plain_nn/include)
target_include_directories(plain_nn PUBLIC ${PROJECT_SOURCE_DIR}/plain_nn/include/stb_image)
target_include_directories(plain_nn PUBLIC ${PROJECT_SOURCE_DIR}/plain_nn/include/plain_nn)
target_include_directories(plain_nn PUBLIC ${PROJECT_SOURCE_DIR}/plain_nn/include/plain_nn/data_loaders)
target_include_directories(plain_nn PUBLIC ${PROJECT_SOURCE_DIR}/plain_nn/include/plain_nn/layers)
target_include_directories(plain_nn PUBLIC ${PROJECT_SOURCE_DIR}/plain_nn/include/plain_nn/layers/activation_fncs)

add_subdirectory(${PROJECT_SOURCE_DIR}/live_demo)
add_subdirectory(${PROJECT_SOURCE_DIR}/examples)

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/test)