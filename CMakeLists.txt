cmake_minimum_required(VERSION 3.10)

project(mnist_cpp VERSION 0.1 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(SOURCES
    src/main.cpp
    src/activation.cpp
    src/dense.cpp
    src/initialization.cpp
    src/input.cpp
    src/model.cpp
    src/MNISTDataLoader.cpp
)

add_executable(mnist_cpp ${SOURCES})

target_include_directories(mnist_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include)