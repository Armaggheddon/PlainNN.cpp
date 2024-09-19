#include "utils.h"

#include <algorithm>
#include <string>
#include <cmath>

std::string string_to_lower(std::string str){
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return std::tolower(c); });
    return str;
};

Tensor softmax(Tensor& input){
    Tensor result(input.shape());
    double sum = 0;
    for(int i = 0; i < input.size(); i++){
        sum += std::exp(input[i]);
    }

    for(int i = 0; i < input.size(); i++){
        result[i] = std::exp(input[i]) / sum;
    }

    return result;
};