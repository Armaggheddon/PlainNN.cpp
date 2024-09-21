include_directories(${CMAKE_SOURCE_DIR}/include)

project(live_demo VERSION 1.0 LANGUAGES CXX)

# GLOB all subdirectories inside src
# file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(live_demo live_demo.cpp)

target_link_libraries(live_demo plain_nn)