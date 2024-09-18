#include "utils.h"

#include <algorithm>
#include <string>

std::string string_to_lower(std::string str){
    std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c){ return std::tolower(c); });
    return str;
};