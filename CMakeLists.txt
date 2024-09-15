cmake_minimum_required(VERSION 3.10)

project(mnist_cpp VERSION 0.1 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# if(NOT CMAKE_BUILD_TYPE)
#     set(CMAKE_BUILD_TYPE Release)
# endif()

# set(CMAKE_CXX_FLAGS "-Wall -Wextra")
# set(CMAKE_CXX_FLAGS_DEBUG "-g")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3")


# Find OpenMP
find_package(OpenMP)

if(OpenMp_CXX_FOUND)
    message(STATUS "OpenMP found")
    target_link_libraries(mnist_cpp PUBLIC OpenMP::OpenMP_CXX)
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