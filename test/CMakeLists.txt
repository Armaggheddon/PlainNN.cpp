# tests must return 0 to pass, 1 to fail

set(GRAYSCALE_IMAGE_NAME "grayscale_image.png")
set(RGB_IMAGE_NAME "rgb_image.png")

# TEST WRITE GRAYSCALE
add_executable( img_utils_test_write_gray image_utils/test_write_grayscale.cpp ${CMAKE_SOURCE_DIR}/src/model/image_utils.cpp)
target_include_directories(img_utils_test_write_gray PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(img_utils_test_write_gray PRIVATE ${CMAKE_SOURCE_DIR}/include/stb_image)
add_test( NAME img_utils_test_write_gray COMMAND img_utils_test_write_gray ${GRAYSCALE_IMAGE_NAME} --output-on-failure)

# TEST READ GRAYSCALE
add_executable( img_utils_test_read_gray image_utils/test_read_grayscale.cpp ${CMAKE_SOURCE_DIR}/src/model/image_utils.cpp)
target_include_directories(img_utils_test_read_gray PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(img_utils_test_read_gray PRIVATE ${CMAKE_SOURCE_DIR}/include/stb_image)
add_test( NAME img_utils_test_read_gray COMMAND img_utils_test_read_gray ${GRAYSCALE_IMAGE_NAME} --output-on-failure)

set_tests_properties(img_utils_test_write_gray img_utils_test_read_gray PROPERTIES RUN_SERIAL TRUE)

# TEST WRITE RGB
add_executable( img_utils_test_write_rgb image_utils/test_write_rgb.cpp ${CMAKE_SOURCE_DIR}/src/model/image_utils.cpp)
target_include_directories(img_utils_test_write_rgb PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(img_utils_test_write_rgb PRIVATE ${CMAKE_SOURCE_DIR}/include/stb_image)
add_test( NAME img_utils_test_write_rgb COMMAND img_utils_test_write_rgb ${RGB_IMAGE_NAME} --output-on-failure)

# TEST READ RGB
add_executable( img_utils_test_read_rgb image_utils/test_read_rgb.cpp ${CMAKE_SOURCE_DIR}/src/model/image_utils.cpp)
target_include_directories(img_utils_test_read_rgb PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(img_utils_test_read_rgb PRIVATE ${CMAKE_SOURCE_DIR}/include/stb_image)
add_test( NAME img_utils_test_read_rgb COMMAND img_utils_test_read_rgb ${RGB_IMAGE_NAME} --output-on-failure)

set_tests_properties(img_utils_test_write_rgb img_utils_test_read_rgb PROPERTIES RUN_SERIAL TRUE)