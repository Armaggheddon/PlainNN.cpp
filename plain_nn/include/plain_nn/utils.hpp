#ifndef PLAIN_NN_UTILS_H
#define PLAIN_NN_UTILS_H

#include "tensor.hpp"
#include <string>
#include <algorithm>

/**
 * @brief Convert a string to lower case
 * 
 * @param str The string to convert
 * @return std::string The converted string
 */
std::string string_to_lower(std::string str);

/**
 * @brief Compute the softmax of a tensor
 * 
 * @param input The input tensor
 * @return Tensor The softmax of the input tensor
 */
Tensor softmax(Tensor& input);

#endif // PLAIN_NN_UTILS_H