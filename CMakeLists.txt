cmake_minimum_required(VERSION 3.10)

project(mnist_cpp VERSION 0.1 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# include_directories(${CMAKE_SOURCE_DIR}/include)

# GLOB all subdirectories inside include

# include_directories(${CMAKE_SOURCE_DIR}/include)

# GLOB all subdirectories inside src
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
# set(GLOB_RECURSE INCLUDES ${PROJECT_SOURCE_DIR}/include/*.h)

# set(GLOB SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

add_executable(mnist_cpp ${SOURCES})
# target_include_directories(mnist_cpp PRIVATE ${INCLUDES})
target_include_directories(mnist_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(mnist_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include/stb_image)
target_include_directories(mnist_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include/model)
target_include_directories(mnist_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include/model/layers)