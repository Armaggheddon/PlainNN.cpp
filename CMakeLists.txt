cmake_minimum_required(VERSION 3.10)

project(mnist_cpp VERSION 0.1 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenMP
find_package(OpenMP)

if(OpenMp_FOUND)
    message(STATUS "OpenMP found")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(STATUS "OpenMP not found")
endif()

# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(${CMAKE_SOURCE_DIR}/include)

# GLOB all subdirectories inside src
file(GLOB_RECURSE SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_executable(mnist_cpp ${SOURCES})
# set output directory for mnist_cpp to bin folder
# set_target_properties(mnist_cpp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
target_include_directories(mnist_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(mnist_cpp PRIVATE ${PROJECT_SOURCE_DIR}/include/stb_image)

enable_testing()
add_subdirectory(${PROJECT_SOURCE_DIR}/test)