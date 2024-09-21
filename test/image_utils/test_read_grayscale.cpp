#include "image_utils.hpp"

#include <iostream>
#include <vector>

#define TEST_SUCCESS 0
#define TEST_FAIL 1

int main(int argc, char *argv[]){

    // parse command line file name
    if(argc < 2){
        std::cout << "No file name provided" << std::endl;
        return TEST_FAIL;
    }

    std::string file_name = argv[1];

    std::vector<std::vector<unsigned char> > gray_img;
    int width, height;
    int result = load_grayscale(file_name, gray_img, width, height);

    if(result == 0){
        std::cout << "Error loading image: " << file_name << std::endl;
        return TEST_FAIL;
    }

    if(gray_img.size() != static_cast<size_t>(width) || gray_img[0].size() != static_cast<size_t>(height)) {
        std::cout << "Image vector dimension does not match with returned dimension" << std::endl;
        return TEST_FAIL;
    }

    return TEST_SUCCESS;

}