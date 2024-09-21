#include "activation_fncs.hpp"

#include <cmath>

Tanh::Tanh(){
    this->fn_type = ActivationType::TANH;
}

double Tanh::forward(const double input){
    return std::tanh(input);
}

double Tanh::backward(const double input){
    return 1 - pow(tanh(input), 2);
}