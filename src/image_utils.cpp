#include "image_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>
#include <string>
#include <iostream>
#include <ctime>

int load_grayscale(
    const std::string image_path,
    std::vector<std::vector<unsigned char> > &image,
    int &width,
    int &height
){
    int channels = 0;
    unsigned char *data = stbi_load(image_path.c_str(), &width, &height, &channels, 1);
    if(data == nullptr){
        std::cerr << "Error loading image: " << image_path << std::endl;
        return 0;
    }

    image.resize(height, std::vector<unsigned char>(width));
    
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            image[i][j] = data[i * width + j];
        }
    }

    stbi_image_free(data);
    return 1;
}

int load_rgb(
    const std::string image_path,
    std::vector<std::vector<std::vector<unsigned char> > > &image,
    int &width,
    int &height
){
    int channels = 0;
    unsigned char *data = stbi_load(image_path.c_str(), &width, &height, &channels, 3);
    if(data == nullptr){
        std::cerr << "Error loading image: " << image_path << std::endl;
        return 0;
    }

    image.resize(height, std::vector<std::vector<unsigned char> >(width, std::vector<unsigned char>(3)));
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            for(int k=0; k<3; k++){
                image[i][j][k] = data[(i * width + j) * 3 + k];
            }
        }
    }
    // std::copy(data, data + width * height * 3, image.begin());
    stbi_image_free(data);
    return 1;
}

int save_grayscale(
    const std::string image_path,
    const std::vector<std::vector<unsigned char> > &image
){
    int extension_idx = image_path.find_last_of(".");
    if(extension_idx == std::string::npos){
        std::cerr << "Error: invalid image path" << std::endl;
        return 0;
    }

    unsigned char *data = new unsigned char[image.size() * image[0].size()];
    
    for(int i=0; i<image.size(); i++){
        for(int j=0; j<image[0].size(); j++){
            data[i * image[0].size() + j] = image[i][j];
        }
    }

    std::string extension = image_path.substr(extension_idx + 1);
    int result = 0;
    if(extension.compare("jpg") == 0 || extension.compare("jpeg") == 0){
        result = stbi_write_jpg(image_path.c_str(), image[0].size(), image.size(), 1, data, 95);
    } else if(extension.compare("png") == 0){
        result = stbi_write_png(image_path.c_str(), image[0].size(), image.size(), 1, data, image[0].size()*sizeof(unsigned char));
    } else if(extension.compare("bmp") == 0){
        result = stbi_write_bmp(image_path.c_str(), image[0].size(), image.size(), 1, data);
    } else {
        std::cerr << "Error: invalid image extension" << std::endl;
    }

    return result;
}

int save_rgb(
    const std::string image_path,
    const std::vector<std::vector<std::vector<unsigned char> > > &image
){

    // get the pointer to the correct function based on the image type
    int extension_idx = image_path.find_last_of(".");
    if(extension_idx == std::string::npos){
        std::cerr << "Error: invalid image path" << std::endl;
        return 0;
    }

    std::vector<unsigned char> data(image.size() * image[0].size() * 3);
    for(int i=0; i<image.size(); i++){
        for(int j=0; j<image[0].size(); j++){
            for(int k=0; k<3; k++){
                data[(i * image[0].size() + j) * 3 + k] = image[i][j][k];
            }
        }
    }
    
    std::string extension = image_path.substr(extension_idx + 1);
    int result = 0;
    if(extension.compare("jpg") == 0 || extension.compare("jpeg") == 0){
        result = stbi_write_jpg(image_path.c_str(), image[0].size(), image.size(), 3, data.data(), 95);
    } else if(extension.compare("png") == 0){
        result = stbi_write_png(image_path.c_str(), image[0].size(), image.size(), 3, data.data(), image[0].size() * 3);
    } else if(extension.compare("bmp") == 0){
        result = stbi_write_bmp(image_path.c_str(), image[0].size(), image.size(), 3, data.data());
    } else {
        std::cerr << "Error: invalid image extension" << std::endl;
    }

    return result;
}
