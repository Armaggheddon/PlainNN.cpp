cmake_minimum_required(VERSION 3.10)

project(mnist_cpp VERSION 1.0 LANGUAGES CXX DESCRIPTION "A simple C++ implementation of a neural network for the MNIST dataset")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


if(CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR}/include)

# GLOB all subdirectories inside src
# file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(mnist_cpp SHARED
    ${PROJECT_SOURCE_DIR}/src/model/data_loaders/mnist_dataloader.cpp
    ${PROJECT_SOURCE_DIR}/src/model/layers/activation_fncs/activation_fncs.cpp
    ${PROJECT_SOURCE_DIR}/src/model/layers/activation_fncs/none.cpp
    ${PROJECT_SOURCE_DIR}/src/model/layers/activation_fncs/relu.cpp
    ${PROJECT_SOURCE_DIR}/src/model/layers/activation_fncs/sigmoid.cpp
    ${PROJECT_SOURCE_DIR}/src/model/layers/activation_fncs/tanh.cpp
    ${PROJECT_SOURCE_DIR}/src/model/layers/dense.cpp
    ${PROJECT_SOURCE_DIR}/src/model/layers/initialization.cpp
    ${PROJECT_SOURCE_DIR}/src/model/layers/input.cpp
    ${PROJECT_SOURCE_DIR}/src/model/layers/layers.cpp
    ${PROJECT_SOURCE_DIR}/src/model/lr_scheduler.cpp
    ${PROJECT_SOURCE_DIR}/src/model/model_storage.cpp
    ${PROJECT_SOURCE_DIR}/src/model/model.cpp
    ${PROJECT_SOURCE_DIR}/src/model/tensor.cpp
    ${PROJECT_SOURCE_DIR}/src/model/utils.cpp
    ${PROJECT_SOURCE_DIR}/src/model/image_utils.cpp
)
# set output directory for mnist_cpp to bin folder
# set_target_properties(mnist_cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
target_include_directories(mnist_cpp PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(mnist_cpp PUBLIC ${PROJECT_SOURCE_DIR}/include/stb_image)
target_include_directories(mnist_cpp PUBLIC ${PROJECT_SOURCE_DIR}/include/model)
target_include_directories(mnist_cpp PUBLIC ${PROJECT_SOURCE_DIR}/include/model/data_loaders)
target_include_directories(mnist_cpp PUBLIC ${PROJECT_SOURCE_DIR}/include/model/layers)
target_include_directories(mnist_cpp PUBLIC ${PROJECT_SOURCE_DIR}/include/model/layers/activation_fncs)

add_subdirectory(${PROJECT_SOURCE_DIR}/live_demo)
add_subdirectory(${PROJECT_SOURCE_DIR}/examples)

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/test)