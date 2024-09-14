#include "image_utils.h"

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

    // create a 128x128 grayscale image with all 0s
    std::vector<std::vector<unsigned char> > gray_img(128, std::vector<unsigned char>(128, 0));
    int result = save_grayscale(file_name, gray_img);

    if(result == 0){
        std::cout << "Error saving image: " << file_name << std::endl;
        return TEST_FAIL;
    }
    
    return TEST_SUCCESS;

}