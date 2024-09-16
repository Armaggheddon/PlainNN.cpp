#include "sigmoid.h"

#include <cmath>

Sigmoid::Sigmoid(){
    this->fn_type = ActivationType::SIGMOID;
}

double Sigmoid::forward(const double input){
    return 1 / (1 + std::exp(-input));
}

double Sigmoid::backward(const double input){
    return input * (1 - input);
}