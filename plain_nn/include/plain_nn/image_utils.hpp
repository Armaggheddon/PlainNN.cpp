#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <vector>
#include <string>

// returns 1 on success, 0 on failure
int load_grayscale(
    const std::string image_path,
    std::vector<std::vector<unsigned char> > &image,
    int &width,
    int &height);

// returns 1 on success, 0 on failure
int load_rgb(
    const std::string image_path,
    std::vector<std::vector<std::vector<unsigned char> > > &image,
    int &width,
    int &height);

// returns != 0 on success, 0 on failure
int save_grayscale(
    const std::string image_path,
    const std::vector<std::vector<unsigned char> > &image);

// returns != 0 on success, 0 on failure
int save_rgb(
    const std::string image_path,
    const std::vector<std::vector<std::vector<unsigned char> > > &image);


#endif // IMAGE_UTILS_H