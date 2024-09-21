project(create_model VERSION 1.0 LANGUAGES CXX)
project(load_save_model VERSION 1.0 LANGUAGES CXX)
project(train_model VERSION 1.0 LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(create_model create_model.cpp)
add_executable(load_save_model load_save_model.cpp)
add_executable(train_model train_model.cpp)

target_link_libraries(create_model plain_nn)
target_link_libraries(load_save_model plain_nn)
target_link_libraries(train_model plain_nn)